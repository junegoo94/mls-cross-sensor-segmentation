"""Held-out shared-5 evaluation of one model (voxel level, no TTA), same protocol as
loso_eval_ab.py / uda_eval.py. Prints one JSON line and optionally appends it to --out."""
import os, glob, json, argparse, numpy as np, torch, sys
sys.path.insert(0, "/home/jmgoo/projects/Pointcept")
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn
ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True); ap.add_argument("--weight", required=True)
ap.add_argument("--val", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", default=None)
a = ap.parse_args()
SHARED = ["Ground", "Building", "Natural", "Pole", "Car"]
def to_s(x):
    o = np.full(x.shape, -1, np.int64)
    for k in range(5): o[x == k] = k
    return o
cfg = Config.fromfile(a.cfg); fk = tuple(cfg.feat_keys)
m = build_model(cfg.model); sd = torch.load(a.weight, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=True); m = m.cuda().eval()
tf = Compose([dict(type="CenterShift", apply_z=True),
              dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
              dict(type="CenterShift", apply_z=False), dict(type="ToTensor"),
              dict(type="Collect", keys=("coord", "grid_coord", "inverse"), feat_keys=fk)])
@torch.no_grad()
def pred(c, s):
    d = {"coord": c.copy(), "strength": s.copy()}; d = tf(d); inv = d.pop("inverse").cpu().numpy()
    b = point_collate_fn([d]); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    return m(b).get("seg_logits").argmax(1).cpu().numpy()[inv]
I = np.zeros(5, np.int64); U = np.zeros(5, np.int64)
for sc in sorted(glob.glob(f"{a.val}/*")):
    c = np.load(f"{sc}/coord.npy"); s = np.load(f"{sc}/strength.npy"); g = to_s(np.load(f"{sc}/segment.npy").astype(np.int64))
    p = to_s(pred(c, s)); mk = g != -1
    for k in range(5):
        I[k] += int(((p == k) & (g == k) & mk).sum()); U[k] += int((((p == k) | (g == k)) & mk).sum())
iou = I / np.maximum(U, 1)
rec = dict(tag=a.tag, cfg=a.cfg, weight=a.weight, val=a.val, miou=float(iou.mean()), iou={n: float(v) for n, v in zip(SHARED, iou)})
print(json.dumps(rec), flush=True)
if a.out: open(a.out, "a").write(json.dumps(rec) + "\n")
