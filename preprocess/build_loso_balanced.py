"""균형 LOSO 데이터셋: 각 센서를 동일 chunk 수로 서브샘플(불균형 confound 제거).
train N_TR/센서, val N_VA/센서 (랜덤, 고정 seed). held-out 평가셋은 별도(전체 사용).

산출: data/loso1b_test_toronto, loso2b_test_npm3d, loso3b_test_whu  (각 {train,val})
"""
import os, glob, numpy as np

OS_SEG = "/home/jmgoo/projects/os_seg"
UNI = {"tor": f"{OS_SEG}/data/uni_toronto",
       "npm": f"{OS_SEG}/data/uni_npm3d",
       "whu": f"{OS_SEG}/data/whu3d"}
N_TR = 178   # 센서당 train chunk (NPM3D=최소)
N_VA = 13    # 센서당 val chunk

FOLDS = {
    "loso1b_test_toronto": ["npm", "whu"],   # held-out Toronto
    "loso2b_test_npm3d":   ["tor", "whu"],   # held-out NPM3D
    "loso3b_test_whu":     ["tor", "npm"],   # held-out WHU
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
    print(">> done. held-out eval: uni_toronto/val, uni_npm3d/val, whu3d/val (전체)")


if __name__ == "__main__":
    main()
