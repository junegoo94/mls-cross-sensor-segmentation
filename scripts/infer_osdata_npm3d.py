"""Predict a preprocessed point cloud with a Paris-Lille-3D model and write labelled LAZ files."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import laspy
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, POINTCEPT)
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn

CLASS_NAMES = ["Ground", "Building", "Pole", "Bollard", "Trash_can",
               "Barrier", "Pedestrian", "Car", "Natural"]
CLASS_COLORS = np.array([
    [0.50, 0.50, 0.50],
    [0.00, 0.00, 1.00],
    [0.99, 0.01, 1.00],
    [0.99, 0.50, 0.03],
    [1.00, 1.00, 0.04],
    [0.55, 0.27, 0.07],
    [1.00, 0.00, 0.00],
    [0.00, 1.00, 1.00],
    [0.06, 0.50, 0.00],
])
MAX_POINTS = 600000

def build_transform(feat_keys):
    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.05, hash_type="fnv",
             mode="train", return_grid_coord=True, return_inverse=True),
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "inverse"),
             feat_keys=feat_keys),
    ])

@torch.no_grad()
def predict_points(coord, strength, model, tf, device):
    n = coord.shape[0]
    try:
        data = {"coord": coord.copy(), "strength": strength.copy()}
        data = tf(data)
        inverse = data.pop("inverse")
        batch = point_collate_fn([data])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        out = model(batch)
        logits = out["seg_logits"] if "seg_logits" in out else out["seg_logit"]
        voxel_pred = logits.argmax(1).cpu().numpy()
        pred = voxel_pred[inverse.cpu().numpy()]
        return pred.astype(np.int16)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if n < 20000:
            raise
        mid = np.median(coord[:, 0])
        m = coord[:, 0] < mid
        out = np.empty(n, dtype=np.int16)
        out[m] = predict_points(coord[m], strength[m], model, tf, device)
        out[~m] = predict_points(coord[~m], strength[~m], model, tf, device)
        return out

def viz_topview(xy, labels, out_png, title, max_pts=1500000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if xy.shape[0] > max_pts:
        sel = np.random.choice(xy.shape[0], max_pts, replace=False)
        xy, labels = xy[sel], labels[sel]
    colors = CLASS_COLORS[labels]
    span_x = xy[:, 0].max() - xy[:, 0].min()
    span_y = xy[:, 1].max() - xy[:, 1].min()
    fig, ax = plt.subplots(figsize=(14, 14 * span_y / max(span_x, 1)))
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=0.4, marker=".", linewidths=0)
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    present = np.unique(labels)
    ax.legend(handles=[Patch(color=CLASS_COLORS[c], label=f"{c}-{CLASS_NAMES[c]}")
                       for c in present], loc="upper right", markerscale=2, fontsize=9)
    plt.tight_layout(); plt.savefig(out_png, dpi=90); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--tag", required=True, help="output tag, e.g. coord or coord-intensity")
    ap.add_argument("--data_root", default=f"{ROOT}/data/osdata")
    ap.add_argument("--out_dir", default=f"{ROOT}/exp/osdata_infer")
    args = ap.parse_args()

    device = "cuda"
    cfg = Config.fromfile(args.config)
    feat_keys = tuple(cfg.feat_keys)
    print(f">> model={args.tag} feat_keys={feat_keys}")

    model = build_model(cfg.model)
    sd = torch.load(args.weight, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if "state_dict" in sd else sd
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f">> weight loaded: {args.weight}")

    tf = build_transform(feat_keys)
    meta = json.load(open(os.path.join(args.data_root, "meta.json")))
    os.makedirs(args.out_dir, exist_ok=True)

    for stem, info in meta["files"].items():
        print(f"\n===== {stem} ({args.tag}) =====")
        offset = np.array(info["offset"], dtype=np.float64)
        chunks = sorted(glob.glob(os.path.join(args.data_root, "infer", f"{stem}_*")))
        vox_coord, vox_pred = [], []
        for i, ch in enumerate(chunks):
            coord = np.load(os.path.join(ch, "coord.npy"))
            strength = np.load(os.path.join(ch, "strength.npy"))
            pred = predict_points(coord, strength, model, tf, device)
            vox_coord.append(coord.astype(np.float64) + offset)
            vox_pred.append(pred)
            if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                print(f"   chunk {i+1}/{len(chunks)}")
        vox_coord = np.concatenate(vox_coord)
        vox_pred = np.concatenate(vox_pred)
        print(f"   voxel preds: {vox_pred.shape[0]:,}")

        las = laspy.read(info["source"])
        full_xyz = np.stack([np.asarray(las.x), np.asarray(las.y),
                             np.asarray(las.z)], axis=1)
        print(f"   backprojecting to full-res {full_xyz.shape[0]:,} pts (KDTree)...")
        tree = cKDTree(vox_coord)
        _, nn = tree.query(full_xyz, k=1, workers=-1)
        full_pred = vox_pred[nn]

        out_laz = os.path.join(args.out_dir, f"{stem}_pred_{args.tag}.laz")
        las.classification = full_pred.astype(np.uint8)
        las.write(out_laz)
        print(f"   saved LAZ: {out_laz}")

        out_png = os.path.join(args.out_dir, f"{stem}_pred_{args.tag}.png")
        viz_topview(full_xyz[:, :2], full_pred, out_png,
                    f"{stem}  [{args.tag}]")
        print(f"   saved PNG: {out_png}")

        uq, cnt = np.unique(full_pred, return_counts=True)
        print("   predicted class distribution:")
        for u, c in zip(uq, cnt):
            print(f"     {u}-{CLASS_NAMES[u]:<14} {c:>12,} ({100*c/full_pred.shape[0]:5.2f}%)")

    print("\n>> DONE")

if __name__ == "__main__":
    main()
