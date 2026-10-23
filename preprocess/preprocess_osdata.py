"""Unlabelled OS LAZ files to Pointcept format for inference, with the same pipeline as the sources."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
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
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--files", nargs="+", default=None, help="file names in input_root (default: every .las/.laz)")
    ap.add_argument("--output_root", default=f"{ROOT}/data/osdata")
    ap.add_argument("--grid_size", type=float, default=GRID)
    ap.add_argument("--chunk_size", type=float, default=CHUNK)
    args = ap.parse_args()

    meta = {"grid_size": args.grid_size, "chunk_size": args.chunk_size, "files": {}}
    infer_dir = os.path.join(args.output_root, "infer")
    os.makedirs(infer_dir, exist_ok=True)
    total_scenes = 0

    files = args.files or sorted(f for f in os.listdir(args.input_root) if f.lower().endswith((".las", ".laz")))
    for fname in files:
        stem = os.path.splitext(fname)[0]
        path = os.path.join(args.input_root, fname)
        print(f"\n>> reading {fname} ...")
        las = laspy.read(path)
        coord = np.stack([np.asarray(las.x), np.asarray(las.y),
                          np.asarray(las.z)], axis=1).astype(np.float64)
        strength = np.asarray(las.intensity).astype(np.float32)
        n_full = coord.shape[0]
        print(f"   full points: {n_full:,}")

        keep = voxel_downsample(coord, args.grid_size)
        coord = coord[keep]
        strength = strength[keep]
        print(f"   after {args.grid_size}m voxel: {coord.shape[0]:,}")

        offset = coord.min(0)
        coord = (coord - offset).astype(np.float32)

        finite = np.isfinite(strength)
        mu = float(strength[finite].mean())
        sd = float(strength[finite].std()) + 1e-6
        strength = np.where(finite, strength, mu)
        strength = ((strength - mu) / sd).astype(np.float32).reshape(-1, 1)

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
