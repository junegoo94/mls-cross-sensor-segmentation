"""Toronto3D to Pointcept format: 5 cm voxels, per-cloud intensity z-score, 50 m chunks."""
import os
import json
import argparse
import numpy as np
from plyfile import PlyData

CLASS_NAMES = ["Ground", "Road_markings", "Natural", "Building",
               "Utility_line", "Pole", "Car", "Fence"]
IGNORE_INDEX = -1

SPLIT_MAP = {"L001": "train", "L002": "val", "L003": "train", "L004": "train"}

def read_tile(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    coord = np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                      np.asarray(v["z"])], axis=1).astype(np.float64)
    color = np.stack([np.asarray(v["red"]), np.asarray(v["green"]),
                      np.asarray(v["blue"])], axis=1).astype(np.uint8)
    strength = np.asarray(v["scalar_Intensity"]).astype(np.float32).reshape(-1, 1)
    raw_label = np.asarray(v["scalar_Label"]).astype(np.int64)
    return coord, color, strength, raw_label

def remap_label(raw_label):
    seg = np.full(raw_label.shape[0], IGNORE_INDEX, dtype=np.int32)
    for k in range(1, 9):
        seg[raw_label == k] = k - 1
    return seg

def voxel_downsample(coord, grid_size):
    if grid_size is None or grid_size <= 0:
        return np.arange(coord.shape[0])
    key = np.floor((coord - coord.min(0)) / grid_size).astype(np.int64)
    mx = key.max(0) + 1
    kid = key[:, 0] * (mx[1] * mx[2]) + key[:, 1] * mx[2] + key[:, 2]
    _, idx = np.unique(kid, return_index=True)
    return np.sort(idx)

def save_scene(out_dir, coord, color, strength, segment):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "coord.npy"), coord.astype(np.float32))
    np.save(os.path.join(out_dir, "color.npy"), color.astype(np.uint8))
    np.save(os.path.join(out_dir, "strength.npy"), strength.astype(np.float32))
    np.save(os.path.join(out_dir, "segment.npy"), segment.astype(np.int32))

def chunk_indices(coord_xy, size, stride, min_points):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root",
                    default="/home/jmgoo/toronto3D_JM/initial_point_clouds")
    ap.add_argument("--output_root",
                    default="/home/jmgoo/projects/os_seg/data/toronto3d")
    ap.add_argument("--chunk_size", type=float, default=50.0)
    ap.add_argument("--chunk_stride", type=float, default=25.0)
    ap.add_argument("--min_points", type=int, default=10000)
    ap.add_argument("--grid_size", type=float, default=0.05,
                    help="voxel size in metres (0 disables downsampling)")
    args = ap.parse_args()

    tiles = ["L001", "L002", "L003", "L004"]
    paths = {t: os.path.join(args.input_root, f"{t}.ply") for t in tiles}

    print(">> reading tiles ...")
    data = {}
    gmin = np.array([np.inf, np.inf, np.inf])
    strength_stats = {}
    for t in tiles:
        coord, color, strength, raw = read_tile(paths[t])
        seg = remap_label(raw)
        keep = voxel_downsample(coord, args.grid_size)
        coord, color, strength, seg = coord[keep], color[keep], strength[keep], seg[keep]
        finite = np.isfinite(strength)
        n_bad = int((~finite).sum())
        mu = float(strength[finite].mean())
        sd = float(strength[finite].std()) + 1e-6
        strength = np.where(finite, strength, mu)
        strength = ((strength - mu) / sd).astype(np.float32)
        strength_stats[t] = {"mean": mu, "std": sd, "n_nonfinite": n_bad}
        data[t] = (coord, color, strength, seg)
        gmin = np.minimum(gmin, coord.min(0))
        print(f"   {t}: kept {coord.shape[0]:,} pts (grid={args.grid_size}m)  "
              f"valid(seg>=0)={int((seg >= 0).sum()):,}  intensity mu={mu:.1f} sd={sd:.1f}")
    print(f">> global offset (subtract): {gmin}")

    n_scenes = {"train": 0, "val": 0}
    for t in tiles:
        coord, color, strength, seg = data[t]
        coord = (coord - gmin).astype(np.float32)
        split = SPLIT_MAP[t]
        split_dir = os.path.join(args.output_root, split)

        if split == "train":
            chunks = chunk_indices(coord[:, :2], args.chunk_size,
                                   args.chunk_stride, args.min_points)
        else:
            chunks = chunk_indices(coord[:, :2], args.chunk_size,
                                   args.chunk_size, 1000)
        print(f">> {t}: {len(chunks)} {split} chunks")
        for tag, idx in chunks:
            out = os.path.join(split_dir, f"{t}_{tag}")
            save_scene(out, coord[idx], color[idx], strength[idx], seg[idx])
            n_scenes[split] += 1

    os.makedirs(args.output_root, exist_ok=True)
    meta = {
        "global_offset": gmin.tolist(),
        "strength_norm": "per-tile z-score ((x-mean)/std)",
        "strength_stats": strength_stats,
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "split_map": SPLIT_MAP,
        "grid_size": args.grid_size,
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "min_points": args.min_points,
        "n_scenes": n_scenes,
        "source": args.input_root,
    }
    with open(os.path.join(args.output_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f">> done. scenes: {n_scenes}")
    print(f">> output: {args.output_root}")

if __name__ == "__main__":
    main()
