"""
학습된 NPM3D PTv3 모델로 OS_data 추론 (9-class, pedestrian 포함) → 라벨링된 LAZ + top-view PNG 저장.

사용:
  python infer_osdata.py --config <cfg.py> --weight <model_best.pth> --tag coord

처리 흐름:
  전처리된 voxel 청크(data/osdata/infer/<file>_xYY) 마다
   → 학습과 동일 transform(CenterShift+GridSample) → 모델 forward → voxel 예측
   → 파일 단위로 모아서 원본 full-res LAZ 점에 KDTree NN 역매핑
   → classification 필드에 예측 클래스 넣어 LAZ 저장 + top-view 시각화 PNG
대용량 청크는 OOM 시 x중앙값 기준 재귀 공간분할로 처리.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import laspy
from scipy.spatial import cKDTree

from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn

CLASS_NAMES = ["Ground", "Building", "Pole", "Bollard", "Trash_can",
               "Barrier", "Pedestrian", "Car", "Natural"]
# NPM3D 9-class 색 (RGB 0-1). Pedestrian 은 눈에 띄게 빨강.
CLASS_COLORS = np.array([
    [0.50, 0.50, 0.50],  # Ground - gray
    [0.00, 0.00, 1.00],  # Building - blue
    [0.99, 0.01, 1.00],  # Pole - magenta
    [0.99, 0.50, 0.03],  # Bollard - orange
    [1.00, 1.00, 0.04],  # Trash_can - yellow
    [0.55, 0.27, 0.07],  # Barrier - brown
    [1.00, 0.00, 0.00],  # Pedestrian - RED (강조)
    [0.00, 1.00, 1.00],  # Car - cyan
    [0.06, 0.50, 0.00],  # Natural - green
])
MAX_POINTS = 600000  # 청크 forward 상한 (초과 시 분할)


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
    """coord(N,3 float32, offset frame), strength(N,1) → 점별 라벨(N,). OOM시 분할."""
    n = coord.shape[0]
    try:
        data = {"coord": coord.copy(), "strength": strength.copy()}
        data = tf(data)
        inverse = data.pop("inverse")
        batch = point_collate_fn([data])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        # fp32 추론 (fp16/autocast 에선 spconv inference 튜너가 실패함)
        out = model(batch)
        logits = out["seg_logits"] if "seg_logits" in out else out["seg_logit"]
        voxel_pred = logits.argmax(1).cpu().numpy()      # (M,)
        pred = voxel_pred[inverse.cpu().numpy()]          # (N,)
        return pred.astype(np.int16)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if n < 20000:
            raise
        # x 중앙값으로 2분할 재귀
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
    ap.add_argument("--tag", required=True, help="출력 구분용 (예: coord, coord-intensity)")
    ap.add_argument("--data_root", default="/home/jmgoo/projects/os_seg/data/osdata")
    ap.add_argument("--out_dir", default="/home/jmgoo/projects/os_seg/exp/osdata_infer")
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
            coord = np.load(os.path.join(ch, "coord.npy"))      # offset frame, f32
            strength = np.load(os.path.join(ch, "strength.npy"))
            pred = predict_points(coord, strength, model, tf, device)
            vox_coord.append(coord.astype(np.float64) + offset)  # 원본 frame
            vox_pred.append(pred)
            if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                print(f"   chunk {i+1}/{len(chunks)}")
        vox_coord = np.concatenate(vox_coord)
        vox_pred = np.concatenate(vox_pred)
        print(f"   voxel preds: {vox_pred.shape[0]:,}")

        # full-res LAZ 로드 + KDTree NN 역매핑
        las = laspy.read(info["source"])
        full_xyz = np.stack([np.asarray(las.x), np.asarray(las.y),
                             np.asarray(las.z)], axis=1)
        print(f"   backprojecting to full-res {full_xyz.shape[0]:,} pts (KDTree)...")
        tree = cKDTree(vox_coord)
        _, nn = tree.query(full_xyz, k=1, workers=-1)
        full_pred = vox_pred[nn]

        # LAZ 저장 (classification 필드에 예측 클래스)
        out_laz = os.path.join(args.out_dir, f"{stem}_pred_{args.tag}.laz")
        las.classification = full_pred.astype(np.uint8)
        las.write(out_laz)
        print(f"   saved LAZ: {out_laz}")

        # 시각화
        out_png = os.path.join(args.out_dir, f"{stem}_pred_{args.tag}.png")
        viz_topview(full_xyz[:, :2], full_pred, out_png,
                    f"{stem}  [{args.tag}]")
        print(f"   saved PNG: {out_png}")

        # 클래스 분포 리포트
        uq, cnt = np.unique(full_pred, return_counts=True)
        print("   예측 분포:")
        for u, c in zip(uq, cnt):
            print(f"     {u}-{CLASS_NAMES[u]:<14} {c:>12,} ({100*c/full_pred.shape[0]:5.2f}%)")

    print("\n>> DONE")


if __name__ == "__main__":
    main()
