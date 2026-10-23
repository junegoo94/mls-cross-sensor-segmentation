"""Print basic statistics of a raw tile: extent, point count, and voxel occupancy."""
import sys
import numpy as np
from plyfile import PlyData

path = sys.argv[1]

print(f"reading {path} ...")
ply = PlyData.read(path)
v = ply["vertex"].data
print("fields:", v.dtype.names)
n = v.shape[0]
print("num points:", n)

x = np.asarray(v["x"]); y = np.asarray(v["y"]); z = np.asarray(v["z"])
print("\n== coord range ==")
for name, a in [("x", x), ("y", y), ("z", z)]:
    print(f"  {name}: min={a.min():.3f} max={a.max():.3f} span={a.max()-a.min():.3f}")

intn = np.asarray(v["scalar_Intensity"]).astype(np.float64)
print("\n== intensity ==")
print(f"  min={intn.min():.3f} max={intn.max():.3f} mean={intn.mean():.3f} "
      f"p1={np.percentile(intn,1):.1f} p99={np.percentile(intn,99):.1f}")

rgb = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], 1)
print("\n== rgb ==")
print(f"  dtype-ish range: min={rgb.min()} max={rgb.max()}")

lab = np.asarray(v["scalar_Label"]).astype(np.int64)
print("\n== label histogram ==")
uq, cnt = np.unique(lab, return_counts=True)
names = ["Unclassified","Ground","Road_markings","Natural","Building",
         "Utility_line","Pole","Car","Fence"]
for u, c in zip(uq, cnt):
    nm = names[u] if 0 <= u < len(names) else "?"
    print(f"  {u:>3} {nm:<14} {c:>12,}  ({100*c/n:5.2f}%)")

print("\n== voxel occupancy for several grid sizes ==")
coord = np.stack([x, y, z], 1)
coord = coord - coord.min(0)
for g in [0.02, 0.03, 0.05, 0.08, 0.10]:
    keys = np.floor(coord / g).astype(np.int64)
    h = keys[:,0]*73856093 ^ keys[:,1]*19349663 ^ keys[:,2]*83492791
    nu = np.unique(h).size
    print(f"  grid={g:>4}m -> voxels={nu:>12,}  kept={100*nu/n:5.1f}%")
