"""Remap each source to the unified label space and build the leave-one-sensor-out folds."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
import os, glob, numpy as np

OS_SEG = ROOT
UNIFIED = ["Ground","Building","Natural","Pole","Car","Fence_Barrier",
           "Road_markings","Utility_line","Bollard","Trash_can","Pedestrian"]
TOR_MAP = {0:0,1:6,2:2,3:1,4:7,5:3,6:4,7:5}
NPM_MAP = {0:0,1:1,2:3,3:8,4:9,5:5,6:10,7:4,8:2}

def remap_segment(seg, mapping):
    out = np.full(seg.shape, -1, dtype=np.int32)
    for s, d in mapping.items():
        out[seg == s] = d
    return out

def build_unified(src_root, mapping, out_root):
    n = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for scene in sorted(glob.glob(os.path.join(src_root, split, "*"))):
            dst = os.path.join(out_root, split, os.path.basename(scene))
            os.makedirs(dst, exist_ok=True)
            for asset in ("coord.npy", "strength.npy"):
                s = os.path.join(scene, asset)
                if os.path.exists(s):
                    d = os.path.join(dst, asset)
                    if os.path.islink(d) or os.path.exists(d):
                        os.remove(d)
                    os.symlink(os.path.abspath(s), d)
            seg = np.load(os.path.join(scene, "segment.npy"))
            np.save(os.path.join(dst, "segment.npy"), remap_segment(seg, mapping))
            n[split] += 1
    print(f"  {out_root}: train {n['train']} / val {n['val']}")
    return n

def link_scenes(sources, out_root):
    n = {"train": 0, "val": 0}
    for split in ("train", "val"):
        outdir = os.path.join(out_root, split)
        os.makedirs(outdir, exist_ok=True)
        for prefix, root in sources:
            for scene in sorted(glob.glob(os.path.join(root, split, "*"))):
                dst = os.path.join(outdir, f"{prefix}_{os.path.basename(scene)}")
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst) if os.path.islink(dst) else None
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(scene), dst)
                n[split] += 1
    print(f"  {out_root}: train {n['train']} / val {n['val']}")
    return n

def main():
    print(">> 1) unified dataset per sensor")
    build_unified(f"{OS_SEG}/data/toronto3d", TOR_MAP, f"{OS_SEG}/data/uni_toronto")
    build_unified(f"{OS_SEG}/data/npm3d",     NPM_MAP, f"{OS_SEG}/data/uni_npm3d")
    WHU = f"{OS_SEG}/data/whu3d"

    print(">> 2) LOSO folds (train/val = the two training sensors)")
    link_scenes([("npm", f"{OS_SEG}/data/uni_npm3d"), ("whu", WHU)],
                f"{OS_SEG}/data/loso_fold1_test_toronto")
    link_scenes([("tor", f"{OS_SEG}/data/uni_toronto"), ("whu", WHU)],
                f"{OS_SEG}/data/loso_fold2_test_npm3d")
    print("  fold3 (test WHU) uses data/joint")

    print("\n>> held-out evaluation sets:")
    print("  fold1 -> data/uni_toronto/val")
    print("  fold2 -> data/uni_npm3d/val")
    print("  fold3 -> data/whu3d/val")
    print(">> done.")

if __name__ == "__main__":
    main()
