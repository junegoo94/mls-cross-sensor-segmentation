"""Joint Toronto3D + Paris-Lille-3D training set in the unified label space."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
import os
import glob
import json
import numpy as np

OS_SEG = ROOT
TOR = f"{OS_SEG}/data/toronto3d"
NPM = f"{OS_SEG}/data/npm3d"
OUT = f"{OS_SEG}/data/joint"

UNIFIED = ["Ground", "Building", "Natural", "Pole", "Car", "Fence_Barrier",
           "Road_markings", "Utility_line", "Bollard", "Trash_can", "Pedestrian"]

TOR_MAP = {0: 0, 1: 6, 2: 2, 3: 1, 4: 7, 5: 3, 6: 4, 7: 5}
NPM_MAP = {0: 0, 1: 1, 2: 3, 3: 8, 4: 9, 5: 5, 6: 10, 7: 4, 8: 2}

def remap_segment(seg, mapping):
    out = np.full(seg.shape, -1, dtype=np.int32)
    for src, dst in mapping.items():
        out[seg == src] = dst
    return out

def process(src_root, mapping, prefix):
    n = {"train": 0, "val": 0}
    hist = np.zeros(len(UNIFIED), dtype=np.int64)
    for split in ("train", "val"):
        for scene in sorted(glob.glob(os.path.join(src_root, split, "*"))):
            name = f"{prefix}_{os.path.basename(scene)}"
            dst = os.path.join(OUT, split, name)
            os.makedirs(dst, exist_ok=True)
            for asset in ("coord.npy", "strength.npy"):
                s = os.path.join(scene, asset)
                d = os.path.join(dst, asset)
                if os.path.exists(s):
                    if os.path.islink(d) or os.path.exists(d):
                        os.remove(d)
                    os.symlink(os.path.abspath(s), d)
            seg = np.load(os.path.join(scene, "segment.npy"))
            rseg = remap_segment(seg, mapping)
            np.save(os.path.join(dst, "segment.npy"), rseg)
            for c in range(len(UNIFIED)):
                hist[c] += int((rseg == c).sum())
            n[split] += 1
    return n, hist

def main():
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
    print(">> Toronto3D ...")
    n_tor, h_tor = process(TOR, TOR_MAP, "tor")
    print(f"   scenes: {n_tor}")
    print(">> NPM3D ...")
    n_npm, h_npm = process(NPM, NPM_MAP, "npm")
    print(f"   scenes: {n_npm}")

    total = {k: n_tor[k] + n_npm[k] for k in n_tor}
    hist = h_tor + h_npm
    grand = hist.sum()
    print("\n>> joint class distribution (train+val valid points):")
    for c, nm in enumerate(UNIFIED):
        src = []
        if c in TOR_MAP.values():
            src.append("Tor")
        if c in NPM_MAP.values():
            src.append("NPM")
        print(f"   {c:2d} {nm:<14} {hist[c]:>13,}  {100*hist[c]/grand:5.2f}%  [{'+'.join(src)}]")

    meta = {
        "unified_names": UNIFIED, "num_classes": len(UNIFIED), "ignore_index": -1,
        "toronto_map": TOR_MAP, "npm3d_map": NPM_MAP,
        "n_scenes": total,
        "note": "coord/strength symlinked from data/toronto3d & data/npm3d (already z-scored). "
                "Fence(Toronto)+Barrier(NPM3D) merged to id5. NPM3D has no Road_markings/Utility_line; "
                "Toronto has no Bollard/Trash_can/Pedestrian.",
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n>> total scenes: {total}")
    print(f">> output: {OUT}")

if __name__ == "__main__":
    main()
