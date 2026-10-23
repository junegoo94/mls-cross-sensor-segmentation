"""Balanced LOSO folds with the same number of chunks per sensor."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
import os, glob, numpy as np

OS_SEG = ROOT
UNI = {"tor": f"{OS_SEG}/data/uni_toronto",
       "npm": f"{OS_SEG}/data/uni_npm3d",
       "whu": f"{OS_SEG}/data/whu3d"}
N_TR = 178
N_VA = 13

FOLDS = {
    "loso1b_test_toronto": ["npm", "whu"],
    "loso2b_test_npm3d":   ["tor", "whu"],
    "loso3b_test_whu":     ["tor", "npm"],
}

def sample(sensor_root, split, k, rng):
    scenes = sorted(glob.glob(os.path.join(sensor_root, split, "*")))
    if len(scenes) <= k:
        return scenes
    idx = rng.choice(len(scenes), k, replace=False)
    return [scenes[i] for i in sorted(idx)]

def main():
    for fold, sensors in FOLDS.items():
        out = os.path.join(OS_SEG, "data", fold)
        rng = np.random.RandomState(42)
        cnt = {"train": 0, "val": 0}
        for split, k in (("train", N_TR), ("val", N_VA)):
            outdir = os.path.join(out, split)
            os.makedirs(outdir, exist_ok=True)
            for s in sensors:
                for scene in sample(UNI[s], split, k, rng):
                    dst = os.path.join(outdir, f"{s}_{os.path.basename(scene)}")
                    if os.path.islink(dst):
                        os.remove(dst)
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(scene), dst)
                    cnt[split] += 1
        print(f"{fold} ({'+'.join(sensors)}): train {cnt['train']} / val {cnt['val']}")
    print(">> done. held-out eval: uni_toronto/val, uni_npm3d/val, whu3d/val")

if __name__ == "__main__":
    main()
