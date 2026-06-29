"""OS_data 예측 LAZ(classification 필드)에 '공통 색상표'로 RGB를 새겨 새 폴더에 저장.

목적: Toronto3D-학습 모델과 NPM3D-학습 모델의 OS 예측을 나란히 비교.
규칙: 두 데이터셋에서 의미가 같은 클래스(Ground/Building/Natural/Pole/Car)는 같은 색,
      한쪽에만 있는 클래스(Road_markings, Utility_line, Fence / Bollard, Trash_can,
      Barrier, Pedestrian)는 서로 다른 고유 색.

입력:
  exp/osdata_infer/*_pred_*.laz        (Toronto, 8 classes)
  exp/osdata_infer_npm3d/*_pred_*.laz  (NPM3D,  9 classes)
출력:
  exp/osdata_infer_colored/toronto/<name>_rgb.laz
  exp/osdata_infer_colored/npm3d/<name>_rgb.laz
  exp/osdata_infer_colored/color_legend.txt
"""
import os
import glob
import numpy as np
import laspy

OUT_ROOT = "/home/jmgoo/projects/os_seg/exp/osdata_infer_colored"

# ---- 공통 의미→색 (0-255). 같은 의미 = 같은 색, 전부 서로 구별되는 색 ----
SEMANTIC_COLOR = {
    "Ground":        (128, 128, 128),  # gray   (공통)
    "Building":      (  0,   0, 230),  # blue   (공통)
    "Natural":       ( 30, 140,  30),  # green  (공통)
    "Pole":          (240,  50, 230),  # magenta(공통)
    "Car":           (245, 130,  30),  # orange (공통)
    "Road_markings": (  0, 230, 230),  # cyan   (Toronto 전용)
    "Utility_line":  (160, 230,  40),  # yellow-green (Toronto 전용)
    "Fence":         (240, 225,  30),  # yellow (Toronto 전용)
    "Bollard":       (150,  60, 200),  # purple (NPM3D 전용)
    "Trash_can":     (240, 140, 200),  # pink   (NPM3D 전용)
    "Barrier":       (140,  70,  20),  # brown  (NPM3D 전용)
    "Pedestrian":    (230,  20,  20),  # red    (NPM3D 전용)
    "Fence_Barrier": (140,  70,  20),  # brown  (JOINT: Fence+Barrier 병합)
}

# 각 데이터셋의 class id -> 의미 이름 (= 색)
TORONTO_NAMES = ["Ground", "Road_markings", "Natural", "Building",
                 "Utility_line", "Pole", "Car", "Fence"]
NPM3D_NAMES = ["Ground", "Building", "Pole", "Bollard", "Trash_can",
               "Barrier", "Pedestrian", "Car", "Natural"]
JOINT_NAMES = ["Ground", "Building", "Natural", "Pole", "Car", "Fence_Barrier",
               "Road_markings", "Utility_line", "Bollard", "Trash_can", "Pedestrian"]

SOURCES = {
    "toronto": ("/home/jmgoo/projects/os_seg/exp/osdata_infer", TORONTO_NAMES),
    "npm3d":   ("/home/jmgoo/projects/os_seg/exp/osdata_infer_npm3d", NPM3D_NAMES),
    "joint":   ("/home/jmgoo/projects/os_seg/exp/osdata_infer_joint", JOINT_NAMES),
}


def build_lut(names):
    """class id -> (r,g,b) uint8 LUT."""
    lut = np.zeros((len(names), 3), dtype=np.uint8)
    for i, n in enumerate(names):
        lut[i] = SEMANTIC_COLOR[n]
    return lut


def color_file(path, names, out_path):
    las = laspy.read(path)
    cls = np.asarray(las.classification).astype(np.int64)
    lut = build_lut(names)
    cls = np.clip(cls, 0, len(names) - 1)
    rgb8 = lut[cls]                      # (N,3) uint8
    rgb16 = (rgb8.astype(np.uint16) * 257)  # 8bit -> 16bit (255->65535)
    las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    las.write(out_path)
    # 분포 요약
    uq, cnt = np.unique(cls, return_counts=True)
    parts = [f"{names[u]}={c}" for u, c in zip(uq, cnt)]
    return las.header.point_count, parts


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for src, (folder, names) in SOURCES.items():
        out_dir = os.path.join(OUT_ROOT, src)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(folder, "*_pred_*.laz")))
        print(f"\n===== {src}: {len(files)} files =====")
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(out_dir, f"{stem}_rgb.laz")
            n, parts = color_file(f, names, out_path)
            print(f"  {stem}  ({n:,} pts)")
            print(f"     -> {os.path.relpath(out_path, OUT_ROOT)}")

    # legend 저장
    leg = os.path.join(OUT_ROOT, "color_legend.txt")
    with open(leg, "w") as fp:
        fp.write("Unified color legend (RGB 0-255) — same meaning = same color\n")
        fp.write("=" * 56 + "\n")
        shared = ["Ground", "Building", "Natural", "Pole", "Car"]
        fp.write("\n[Shared by both datasets]\n")
        for k in shared:
            fp.write(f"  {k:<14} RGB {SEMANTIC_COLOR[k]}\n")
        fp.write("\n[Toronto3D only]\n")
        for k in ["Road_markings", "Utility_line", "Fence"]:
            fp.write(f"  {k:<14} RGB {SEMANTIC_COLOR[k]}\n")
        fp.write("\n[NPM3D only]\n")
        for k in ["Bollard", "Trash_can", "Barrier", "Pedestrian"]:
            fp.write(f"  {k:<14} RGB {SEMANTIC_COLOR[k]}\n")
    print(f"\n>> legend: {leg}")
    print(f">> output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
