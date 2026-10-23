"""Write the unified class colours into predicted LAZ files."""
import os
import glob
import numpy as np
import laspy

OUT_ROOT = "/home/jmgoo/projects/os_seg/exp/final_results/osdata_infer_colored"

SEMANTIC_COLOR = {
    "Ground":        (128, 128, 128),
    "Building":      (  0,   0, 230),
    "Natural":       ( 30, 140,  30),
    "Pole":          (240,  50, 230),
    "Car":           (245, 130,  30),
    "Road_markings": (  0, 230, 230),
    "Utility_line":  (160, 230,  40),
    "Fence":         (240, 225,  30),
    "Bollard":       (150,  60, 200),
    "Trash_can":     (240, 140, 200),
    "Barrier":       (140,  70,  20),
    "Pedestrian":    (230,  20,  20),
    "Fence_Barrier": (140,  70,  20),
}

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
    "all3":    ("/home/jmgoo/projects/os_seg/exp/osdata_infer_all3", JOINT_NAMES),
}

def build_lut(names):
    lut = np.zeros((len(names), 3), dtype=np.uint8)
    for i, n in enumerate(names):
        lut[i] = SEMANTIC_COLOR[n]
    return lut

def color_file(path, names, out_path):
    las = laspy.read(path)
    cls = np.asarray(las.classification).astype(np.int64)
    lut = build_lut(names)
    cls = np.clip(cls, 0, len(names) - 1)
    rgb8 = lut[cls]
    rgb16 = (rgb8.astype(np.uint16) * 257)
    las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    las.write(out_path)
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
