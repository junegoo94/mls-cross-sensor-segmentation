"""
WHU-Urban3D (city A, MLS) h5 -> Pointcept DefaultDataset 포맷 전처리.
Toronto3D/NPM3D와 '완전히 동일한' 파이프라인 (센서 간 공정 비교 목적):
  - 0.05m voxel 다운샘플 (voxel당 첫 실점, 평균 안 냄)
  - 라벨: WHU raw 구조ID -> 공식 sem_map(0-17) -> 우리 통합 11클래스(joint와 동일) / 나머지 ignore(-1)
  - 좌표: 글로벌 offset(min) 차감 후 float32
  - intensity: per-scene z-score (WHU는 16-bit 800~65533 → z-score가 스케일 흡수)
  - split: scene 단위 (val = 아래 VAL_SCENES), train=50m/25m 겹침청크, val=50m 비겹침
색상(color) 없음 (NPM3D와 동일 조건).

입력:  data/whu3d_download/extracted/mls/h5/<scene>.h5  (coords/intensity/number_returns/semantics/instances)
출력:  data/whu3d/{train,val}/<scene>_xNN_yMM/{coord,strength,segment}.npy + meta.json
"""
import os, json, glob, argparse, numpy as np, h5py

# ---- 통합 11클래스 (joint/Toronto+NPM3D와 동일 순서·의미) ----
CLASS_NAMES = ["Ground","Building","Natural","Pole","Car","Fence_Barrier",
               "Road_markings","Utility_line","Bollard","Trash_can","Pedestrian"]
IGNORE_INDEX = -1

# WHU 공식 18클래스(0-17) -> 우리 통합 11클래스 (-1=ignore)
#  0 others   -> ignore        9 board     -> ignore
#  1 tree     -> Natural(2)    10 driveway  -> Ground(0)
#  2 nd.way   -> Ground(0)     11 roadmark  -> Road_markings(6)
#  3 building -> Building(1)   12 car       -> Car(4)
#  4 box      -> ignore        13 person    -> Pedestrian(10)
#  5 light    -> Pole(3)       14 sig.light -> Pole(3)   (Toronto 컨벤션: 신호등=Pole)
#  6 tel.pole -> Pole(3)       15 detector  -> ignore
#  7 mun.pole -> Pole(3)       16 fence     -> Fence_Barrier(5)
#  8 low veg. -> Natural(2)    17 wire      -> Utility_line(7)
MAP18 = np.array([-1, 2, 0, 1, -1, 3, 3, 3, 2, -1, 0, 6, 4, 10, 3, -1, 5, 7], dtype=np.int64)

# 우리 다운로드에서 라벨 있는 29 scene 중 val 홀드아웃 (라벨 풍부 + ID 범위 분산)
VAL_SCENES = ['0444', '2022', '2447', '4629', '6037', '8048']


def build_raw_lut():
    """WHU raw 구조ID -> 통합 11클래스(-1) LUT. 공식 sem_map 사용."""
    from pywhu3d.configs.mls_config import sem_map  # raw_id -> 0..17
    lut = {}
    for raw, c18 in sem_map.items():
        lut[int(raw)] = int(MAP18[c18])
    return lut  # dict; 없는 raw는 -1 처리


def remap_label(raw_sem, lut):
    seg = np.full(raw_sem.shape[0], IGNORE_INDEX, dtype=np.int64)
    for raw, uni in lut.items():
        if uni >= 0:
            seg[raw_sem == raw] = uni
    return seg


PED = 10
def correct_ped_mislabels(coord, seg, ins, zthr=2.5):
    """QC 정정: pedestrian instance 중 수직높이(Z-span) > zthr(m) 인 것은
    기하적으로 사람일 수 없는 오라벨(예: 건물벽) → ignore(-1)로 제외.
    (WHU-Urban3D scene 4629 등에서 확인된 person 오라벨 교정. 육안+렌더 검증.)"""
    pm = seg == PED
    n_ins = n_pts = 0
    for iid in np.unique(ins[pm]):
        m = pm & (ins == iid)
        z = coord[m, 2]
        if z.max() - z.min() > zthr:
            seg[m] = IGNORE_INDEX
            n_ins += 1; n_pts += int(m.sum())
    return seg, n_ins, n_pts


def voxel_downsample(coord, grid_size):
    """voxel당 첫 점 인덱스 (실점/라벨 보존). Toronto/NPM3D와 동일."""
    if grid_size is None or grid_size <= 0:
        return np.arange(coord.shape[0])
    key = np.floor((coord - coord.min(0)) / grid_size).astype(np.int64)
    mx = key.max(0) + 1
    kid = key[:, 0] * (mx[1] * mx[2]) + key[:, 1] * mx[2] + key[:, 2]
    _, idx = np.unique(kid, return_index=True)
    return np.sort(idx)


def chunk_indices(coord_xy, size, stride, min_points):
    """size x size 윈도우(겹침) 청크 인덱스. Toronto/NPM3D와 동일."""
    mins = coord_xy.min(0)
    cell = np.floor((coord_xy - mins) / stride).astype(np.int64)
    nb = int(np.ceil(size / stride))
    BIG = cell[:, 1].max() + 2
    cid = cell[:, 0] * BIG + cell[:, 1]
    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    uniq, start = np.unique(cid_s, return_index=True)
    bounds = np.append(start, len(cid_s))
    groups = {int(u): order[bounds[i]:bounds[i + 1]] for i, u in enumerate(uniq)}
    chunks = []
    for u in uniq:
        cx, cy = int(u) // BIG, int(u) % BIG
        cand = []
        for dx in range(nb):
            for dy in range(nb):
                key = (cx + dx) * BIG + (cy + dy)
                if key in groups:
                    cand.append(groups[key])
        if not cand:
            continue
        cand = np.concatenate(cand)
        ox = mins[0] + cx * stride
        oy = mins[1] + cy * stride
        xy = coord_xy[cand]
        m = (xy[:, 0] >= ox) & (xy[:, 0] < ox + size) & \
            (xy[:, 1] >= oy) & (xy[:, 1] < oy + size)
        idx = cand[m]
        if idx.shape[0] >= min_points:
            chunks.append((f"x{cx:02d}_y{cy:02d}", idx))
    return chunks


def save_scene(out_dir, coord, strength, segment):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "coord.npy"), coord.astype(np.float32))
    np.save(os.path.join(out_dir, "strength.npy"), strength.astype(np.float32))
    np.save(os.path.join(out_dir, "segment.npy"), segment.astype(np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root",
                    default="/home/jmgoo/projects/os_seg/data/whu3d_download/extracted/mls/h5")
    ap.add_argument("--output_root",
                    default="/home/jmgoo/projects/os_seg/data/whu3d")
    ap.add_argument("--chunk_size", type=float, default=50.0)
    ap.add_argument("--chunk_stride", type=float, default=25.0)
    ap.add_argument("--min_points", type=int, default=10000)
    ap.add_argument("--grid_size", type=float, default=0.05)
    args = ap.parse_args()

    lut = build_raw_lut()
    files = sorted(glob.glob(os.path.join(args.input_root, "*.h5")))

    # 1) 라벨 있는 scene만 로드 + voxel + zscore + 글로벌 offset 계산
    print(">> reading WHU MLS scenes ...")
    data = {}
    gmin = np.array([np.inf, np.inf, np.inf])
    strength_stats = {}
    for f in files:
        sc = os.path.splitext(os.path.basename(f))[0]
        with h5py.File(f, "r") as h:
            raw_sem = h["semantics"][:].astype(np.int64)
            if np.unique(raw_sem).size <= 1:      # test set = 라벨 없음 → skip
                print(f"   {sc}: no labels, skip")
                continue
            coord = h["coords"][:].astype(np.float64)
            strength = h["intensity"][:].astype(np.float32).reshape(-1, 1)
            ins = h["instances"][:].astype(np.int64)
        seg = remap_label(raw_sem, lut)
        seg, nbi, nbp = correct_ped_mislabels(coord, seg, ins)   # QC: 건물-오라벨 person 제외
        if nbi:
            print(f"   {sc}: QC ped-mislabel fix — {nbi} instance / {nbp:,} pts -> ignore")
        keep = voxel_downsample(coord, args.grid_size)
        coord, strength, seg = coord[keep], strength[keep], seg[keep]
        finite = np.isfinite(strength)
        mu = float(strength[finite].mean()); sd = float(strength[finite].std()) + 1e-6
        strength = np.where(finite, strength, mu)
        strength = ((strength - mu) / sd).astype(np.float32)
        strength_stats[sc] = {"mean": mu, "std": sd}
        data[sc] = (coord, strength, seg)
        gmin = np.minimum(gmin, coord.min(0))
        print(f"   {sc}: kept {coord.shape[0]:,} pts  valid(seg>=0)={int((seg>=0).sum()):,}  "
              f"intensity mu={mu:.0f} sd={sd:.0f}")
    print(f">> global offset (subtract): {gmin}")
    print(f">> labeled scenes: {len(data)}  (val={[s for s in VAL_SCENES if s in data]})")

    # 2) offset 적용 + scene 단위 split 저장
    n_scenes = {"train": 0, "val": 0}
    for sc, (coord, strength, seg) in data.items():
        coord = (coord - gmin).astype(np.float32)
        split = "val" if sc in VAL_SCENES else "train"
        if split == "train":
            chunks = chunk_indices(coord[:, :2], args.chunk_size, args.chunk_stride, args.min_points)
        else:
            chunks = chunk_indices(coord[:, :2], args.chunk_size, args.chunk_size, 1000)
        print(f">> {sc}: {len(chunks)} {split} chunks")
        for tag, idx in chunks:
            out = os.path.join(args.output_root, split, f"{sc}_{tag}")
            save_scene(out, coord[idx], strength[idx], seg[idx])
            n_scenes[split] += 1

    # 3) meta
    os.makedirs(args.output_root, exist_ok=True)
    meta = {
        "global_offset": gmin.tolist(),
        "strength_norm": "per-scene z-score ((x-mean)/std)",
        "strength_stats": strength_stats,
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "val_scenes": VAL_SCENES,
        "grid_size": args.grid_size,
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "min_points": args.min_points,
        "n_scenes": n_scenes,
        "source": args.input_root,
        "label_map": "WHU raw sem_map(0-17) -> unified11 via MAP18; sig.light&light->Pole (Toronto conv.)",
    }
    with open(os.path.join(args.output_root, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f">> done. scenes: {n_scenes}")
    print(f">> output: {args.output_root}")


if __name__ == "__main__":
    main()
