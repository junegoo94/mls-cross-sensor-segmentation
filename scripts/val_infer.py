"""Predict a source's own validation split and save predicted and ground-truth LAZ files."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
os.chdir(ROOT)
import os, glob, argparse, numpy as np, torch, laspy
import sys; sys.path.insert(0, POINTCEPT)
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn

SEMANTIC_COLOR = {
  "Ground":(128,128,128), "Building":(0,0,230), "Natural":(30,140,30),
  "Pole":(240,50,230), "Car":(245,130,30), "Road_markings":(0,230,230),
  "Utility_line":(160,230,40), "Fence":(240,225,30), "Bollard":(150,60,200),
  "Trash_can":(240,140,200), "Barrier":(140,70,20), "Pedestrian":(230,20,20),
}
def _cols(names):
    return np.array([SEMANTIC_COLOR[n] for n in names], np.float64)/255.0

DS = {
  "toronto": dict(
    val="data/toronto3d/val", meta="data/toronto3d/meta.json",
    names=["Ground","Road_markings","Natural","Building","Utility_line","Pole","Car","Fence"],
    variants={
      "coord":           ("configs/semseg-pt-v3-coord.py",           "exp/toronto3d-ptv3-coord/model/model_best.pth"),
      "coord-intensity": ("configs/semseg-pt-v3-coord-intensity.py", "exp/toronto3d-ptv3-coord-intensity/model/model_best.pth"),
    }),
  "npm3d": dict(
    val="data/npm3d/val", meta="data/npm3d/meta.json",
    names=["Ground","Building","Pole","Bollard","Trash_can","Barrier","Pedestrian","Car","Natural"],
    variants={
      "coord":           ("configs/semseg-pt-v3-npm3d-coord.py",           "exp/npm3d-ptv3-coord/model/model_best.pth"),
      "coord-intensity": ("configs/semseg-pt-v3-npm3d-coord-intensity.py", "exp/npm3d-ptv3-coord-intensity/model/model_best.pth"),
    }),
}
for _d in DS.values(): _d["colors"] = _cols(_d["names"])
IGNORE_COL = np.array([0,0,0])

def build_tf(feat_keys):
    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.05, hash_type="fnv",
             mode="train", return_grid_coord=True, return_inverse=True),
        dict(type="CenterShift", apply_z=False),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord","grid_coord","inverse"), feat_keys=feat_keys),
    ])

@torch.no_grad()
def predict(coord, strength, model, tf, device):
    data = {"coord": coord.copy(), "strength": strength.copy()}
    data = tf(data); inv = data.pop("inverse")
    batch = point_collate_fn([data])
    batch = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
    out = model(batch)
    logits = out.get("seg_logits", out.get("seg_logit"))
    return logits.argmax(1).cpu().numpy()[inv.cpu().numpy()]

def write_laz(path, xyz, rgb01):
    h = laspy.LasHeader(point_format=2, version="1.2")
    mins = xyz.min(0)
    h.offsets = np.floor(mins); h.scales = [0.001,0.001,0.001]
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:,0], xyz[:,1], xyz[:,2]
    rgb16 = (np.clip(rgb01,0,1)*65535).astype(np.uint16)
    las.red, las.green, las.blue = rgb16[:,0], rgb16[:,1], rgb16[:,2]
    las.write(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["toronto","npm3d"])
    ap.add_argument("--variant", default="coord", choices=["coord","coord-intensity"])
    ap.add_argument("--out_root", default="exp/final_results/val_infer")
    args = ap.parse_args()
    d = DS[args.dataset]; device="cuda"
    cfg_path, weight = d["variants"][args.variant]
    import json; offset = np.array(json.load(open(d["meta"]))["global_offset"])
    names, colors = d["names"], d["colors"]; nC=len(names)

    cfg = Config.fromfile(cfg_path); feat_keys=tuple(cfg.feat_keys)
    model = build_model(cfg.model)
    sd = torch.load(weight, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if "state_dict" in sd else sd
    model.load_state_dict({k.replace("module.",""):v for k,v in sd.items()}, strict=True)
    model = model.to(device).eval()
    tf = build_tf(feat_keys)
    print(f">> {args.dataset} / {args.variant}  feat={feat_keys}")

    XYZ,PRED,GT = [],[],[]
    I=np.zeros(nC,np.int64); U=np.zeros(nC,np.int64)
    scenes = sorted(glob.glob(os.path.join(d["val"],"*")))
    for i,sc in enumerate(scenes):
        coord=np.load(f"{sc}/coord.npy"); strength=np.load(f"{sc}/strength.npy")
        gt=np.load(f"{sc}/segment.npy").astype(np.int64)
        pred=predict(coord, strength, model, tf, device)
        XYZ.append(coord.astype(np.float64)+offset); PRED.append(pred); GT.append(gt)
        m=gt!=-1
        for c in range(nC):
            I[c]+=int(((pred==c)&(gt==c)&m).sum()); U[c]+=int((((pred==c)|(gt==c))&m).sum())
        if (i+1)%5==0 or i+1==len(scenes): print(f"   {i+1}/{len(scenes)}")
    XYZ=np.concatenate(XYZ); PRED=np.concatenate(PRED); GT=np.concatenate(GT)

    out=os.path.join(args.out_root, args.dataset, args.variant); os.makedirs(out, exist_ok=True)
    write_laz(f"{out}/{args.dataset}_val_pred.laz", XYZ, colors[PRED])
    gt_rgb=np.where((GT>=0)[:,None], colors[np.clip(GT,0,nC-1)], IGNORE_COL)
    write_laz(f"{out}/{args.dataset}_val_gt.laz", XYZ, gt_rgb)

    iou=np.where(U>0, I/np.maximum(U,1), np.nan)
    valid=GT!=-1
    acc=float((PRED[valid]==GT[valid]).mean())
    print(f"\n=== {args.dataset} val / {args.variant} ({XYZ.shape[0]:,} pts, {len(scenes)} scenes) ===")
    for c in range(nC): print(f"   {names[c]:<13} IoU {iou[c]:.4f}")
    print(f"   >>> mIoU {np.nanmean(iou):.4f}   overall acc {acc:.4f}")
    print(f">> saved: {out}/{args.dataset}_val_pred.laz  &  _gt.laz")

if __name__ == "__main__":
    main()
