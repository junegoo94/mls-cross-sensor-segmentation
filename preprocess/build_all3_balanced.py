"""Balanced three-source training set for the deployment model (all3-AB-coord):
the same rule as build_loso_balanced.py (178 train / 13 val chunks per sensor, seed 42),
with all three labelled sensors. Output: data/all3_balanced/{train,val} (symlinks)."""
import os, numpy as np
from build_loso_balanced import OS_SEG, UNI, N_TR, N_VA, sample

def main():
    out = os.path.join(OS_SEG, "data", "all3_balanced"); rng = np.random.RandomState(42); cnt = {}
    for split, k in (("train", N_TR), ("val", N_VA)):
        outdir = os.path.join(out, split); os.makedirs(outdir, exist_ok=True); cnt[split] = 0
        for s in ("tor", "npm", "whu"):
            for scene in sample(UNI[s], split, k, rng):
                dst = os.path.join(outdir, f"{s}_{os.path.basename(scene)}")
                if os.path.islink(dst): os.remove(dst)
                if not os.path.exists(dst): os.symlink(os.path.abspath(scene), dst)
                cnt[split] += 1
    print(f"all3_balanced: train {cnt['train']} / val {cnt['val']}")

if __name__ == "__main__":
    main()
