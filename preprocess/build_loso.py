"""Leave-One-Sensor-Out(LOSO) 데이터셋 구축.
3센서(Toronto/NPM3D/WHU)를 통합 11클래스로 정렬 → fold별 2센서 학습셋 구성.

산출:
  data/uni_toronto/{train,val}   (Toronto -> unified, symlink coord/strength + remap segment)
  data/uni_npm3d/{train,val}     (NPM3D   -> unified)
  data/loso_fold1_test_toronto/{train,val}  (train/val = NPM3D + WHU)   held-out eval = uni_toronto/val
  data/loso_fold2_test_npm3d/{train,val}    (train/val = Toronto + WHU) held-out eval = uni_npm3d/val
  (fold3 test WHU = 기존 data/joint;                                    held-out eval = data/whu3d/val)

* fold의 val = '학습 센서들의 val' (in-domain, model_best 선택용; held-out 누출 방지).
* WHU(data/whu3d)는 이미 unified. scene 이름 충돌 방지 위해 sensor prefix 부여.
"""
import os, glob, numpy as np

OS_SEG = "/home/jmgoo/projects/os_seg"
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
    """센서 하나를 unified로: coord/strength symlink, segment remap 저장."""
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
    """sources = [(sensor_prefix, unified_root)] 를 fold 폴더로 symlink (scene dir 통째)."""
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
    print(">> 1) 센서별 unified 데이터셋")
    build_unified(f"{OS_SEG}/data/toronto3d", TOR_MAP, f"{OS_SEG}/data/uni_toronto")
    build_unified(f"{OS_SEG}/data/npm3d",     NPM_MAP, f"{OS_SEG}/data/uni_npm3d")
    WHU = f"{OS_SEG}/data/whu3d"

    print(">> 2) LOSO fold 폴더 (train/val = 학습 2센서)")
    # fold1: test Toronto -> train on NPM3D + WHU
    link_scenes([("npm", f"{OS_SEG}/data/uni_npm3d"), ("whu", WHU)],
                f"{OS_SEG}/data/loso_fold1_test_toronto")
    # fold2: test NPM3D -> train on Toronto + WHU
    link_scenes([("tor", f"{OS_SEG}/data/uni_toronto"), ("whu", WHU)],
                f"{OS_SEG}/data/loso_fold2_test_npm3d")
    # fold3: test WHU -> train on Toronto + NPM3D = 기존 data/joint (이미 존재)
    print("  fold3 (test WHU) = 기존 data/joint 사용 (재생성 불필요)")

    print("\n>> held-out 평가셋:")
    print("  fold1 -> data/uni_toronto/val")
    print("  fold2 -> data/uni_npm3d/val")
    print("  fold3 -> data/whu3d/val")
    print(">> done.")


if __name__ == "__main__":
    main()
