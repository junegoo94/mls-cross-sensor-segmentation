"""
OS_data (Ordnance Survey, *.laz) -> Pointcept DefaultDataset 포맷 (추론용).

Toronto3D 전처리와 동일한 정규화를 적용해 도메인 정합:
  - 좌표: 파일별 min offset 차감 (BNG float32 정밀도)  [모델은 CenterShift로 재중심화]
  - intensity: per-cloud z-score ((x-mean)/std)         [Toronto3D 타일과 동일]
  - 0.05m voxel 다운샘플 + 50m 비겹침 청크

출력:
  <output_root>/infer/<stem>_xYY_yZZ/{coord,strength}.npy
  <output_root>/meta.json   # 파일별 offset, intensity mean/std, voxel→fullres 복원용 정보

* OS_data 는 GT 라벨이 없으므로 segment 저장 안 함 (추론 전용).
* 전체 해상도 예측 복원은 추론 스크립트에서 원본 LAZ 와 NN 매칭으로 처리.
"""
import os
import json
import argparse
import numpy as np
import laspy

GRID = 0.05
CHUNK = 50.0


def voxel_downsample(coord, grid_size):
    key = np.floor((coord - coord.min(0)) / grid_size).astype(np.int64)
    mx = key.max(0) + 1
    kid = key[:, 0] * (mx[1] * mx[2]) + key[:, 1] * mx[2] + key[:, 2]
    _, idx = np.unique(kid, return_index=True)
    return np.sort(idx)


def chunk_indices(coord_xy, size):
    """비겹침(stride=size) 청크 인덱스."""
    mins = coord_xy.min(0)
    cell = np.floor((coord_xy - mins) / size).astype(np.int64)
    BIG = cell[:, 1].max() + 2
    cid = cell[:, 0] * BIG + cell[:, 1]
    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    uniq, start = np.unique(cid_s, return_index=True)
    bounds = np.append(start, len(cid_s))
    chunks = []
    for i, u in enumerate(uniq):
        idx = order[bounds[i]:bounds[i + 1]]
        cx, cy = int(u) // BIG, int(u) % BIG
        if idx.shape[0] >= 100:
            chunks.append((f"x{cx:02d}_y{cy:02d}", idx))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", default="/home/jmgoo/OS_data/ordnance_survey")
    ap.add_argument("--files", nargs="+",
                    default=["soton_l1_5cm_bng_442_111_shift.laz",
                             "soton_l2_bng_442_111.laz"])
    ap.add_argument("--output_root", default="/home/jmgoo/projects/os_seg/data/osdata")
    ap.add_argument("--grid_size", type=float, default=GRID)
    ap.add_argument("--chunk_size", type=float, default=CHUNK)
    args = ap.parse_args()

    meta = {"grid_size": args.grid_size, "chunk_size": args.chunk_size, "files": {}}
    infer_dir = os.path.join(args.output_root, "infer")
    os.makedirs(infer_dir, exist_ok=True)
    total_scenes = 0

    for fname in args.files:
        stem = os.path.splitext(fname)[0]
        path = os.path.join(args.input_root, fname)
        print(f"\n>> reading {fname} ...")
        las = laspy.read(path)
        coord = np.stack([np.asarray(las.x), np.asarray(las.y),
                          np.asarray(las.z)], axis=1).astype(np.float64)
        strength = np.asarray(las.intensity).astype(np.float32)
        n_full = coord.shape[0]
        print(f"   full points: {n_full:,}")

        # voxel 다운샘플
        keep = voxel_downsample(coord, args.grid_size)
        coord = coord[keep]
        strength = strength[keep]
        print(f"   after {args.grid_size}m voxel: {coord.shape[0]:,}")

        # offset (파일별 min)
        offset = coord.min(0)
        coord = (coord - offset).astype(np.float32)

        # intensity per-cloud z-score (NaN 안전)
        finite = np.isfinite(strength)
        mu = float(strength[finite].mean())
        sd = float(strength[finite].std()) + 1e-6
        strength = np.where(finite, strength, mu)
        strength = ((strength - mu) / sd).astype(np.float32).reshape(-1, 1)

        # 50m 비겹침 청크 저장
        chunks = chunk_indices(coord[:, :2], args.chunk_size)
        print(f"   {len(chunks)} chunks")
        for tag, idx in chunks:
            out = os.path.join(infer_dir, f"{stem}_{tag}")
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, "coord.npy"), coord[idx])
            np.save(os.path.join(out, "strength.npy"), strength[idx])
            total_scenes += 1

        meta["files"][stem] = {
            "source": path,
            "n_full": int(n_full),
            "n_voxel": int(coord.shape[0]),
            "offset": offset.tolist(),
            "intensity_mean": mu,
            "intensity_std": sd,
            "n_chunks": len(chunks),
        }

    with open(os.path.join(args.output_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n>> done. {total_scenes} infer scenes -> {infer_dir}")
    print(f">> meta: {os.path.join(args.output_root, 'meta.json')}")


if __name__ == "__main__":
    main()
