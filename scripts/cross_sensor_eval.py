"""Shared-5 mIoU of one model on another sensor's validation split."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
import os, glob, argparse, numpy as np, torch
import sys; sys.path.insert(0, POINTCEPT)
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn

SHARED = ["Ground", "Building", "Natural", "Pole", "Car"]
TOR2S = {0:0, 2:2, 3:1, 5:3, 6:4}
NPM2S = {0:0, 1:1, 2:3, 7:4, 8:2}
MAPS = {"tor": TOR2S, "npm": NPM2S}

def to_shared(arr, mapping):
    out = np.full(arr.shape, -1, dtype=np.int64)
    for k, v in mapping.items():
        out[arr == k] = v
    return out

def build_tf(feat_keys):
    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.05, hash_type="fnv",
             mode="train", return_grid_coord=True, return_inverse=True),
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "inverse"), feat_keys=feat_keys),
    ])

@torch.no_grad()
def predict(coord, strength, model, tf, device):
    data = {"coord": coord.copy(), "strength": strength.copy()}
    data = tf(data); inv = data.pop("inverse")
    batch = point_collate_fn([data])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(batch)
    logits = out.get("seg_logits", out.get("seg_logit"))
    return logits.argmax(1).cpu().numpy()[inv.cpu().numpy()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cfg", required=True)
    ap.add_argument("--src_weight", required=True)
    ap.add_argument("--src_space", required=True, choices=["tor", "npm"])
    ap.add_argument("--tgt_val", required=True)
    ap.add_argument("--tgt_space", required=True, choices=["tor", "npm"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda"

    cfg = Config.fromfile(args.src_cfg)
    feat_keys = tuple(cfg.feat_keys)
    model = build_model(cfg.model)
    sd = torch.load(args.src_weight, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if "state_dict" in sd else sd
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True)
    model = model.to(device).eval()
    tf = build_tf(feat_keys)

    src_map, tgt_map = MAPS[args.src_space], MAPS[args.tgt_space]
    nC = len(SHARED)
    I = np.zeros(nC, np.int64); U = np.zeros(nC, np.int64)
    scenes = sorted(glob.glob(os.path.join(args.tgt_val, "*")))
    for i, sc in enumerate(scenes):
        coord = np.load(f"{sc}/coord.npy"); strength = np.load(f"{sc}/strength.npy")
        gt = to_shared(np.load(f"{sc}/segment.npy").astype(np.int64), tgt_map)
        pred_raw = predict(coord, strength, model, tf, device)
        pred = to_shared(pred_raw, src_map)
        m = gt != -1; ps, gs = pred[m], gt[m]
        for c in range(nC):
            I[c] += int(((ps == c) & (gs == c)).sum())
            U[c] += int(((ps == c) | (gs == c)).sum())
        if (i + 1) % 10 == 0 or i + 1 == len(scenes):
            print(f"   {i+1}/{len(scenes)} scenes")
    iou = np.where(U > 0, I / np.maximum(U, 1), np.nan)
    print(f"\n=== {args.tag} ===")
    for c in range(nC):
        print(f"   {SHARED[c]:<10} IoU {iou[c]:.4f}")
    print(f"   >>> mIoU(shared5) = {np.nanmean(iou):.4f}")

if __name__ == "__main__":
    main()
