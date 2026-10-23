"""In-domain WHU-Urban3D evaluation on all validation scenes after the label correction (in-domain table), with a pedestrian height check."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
os.chdir(ROOT)
import os, glob, numpy as np, torch, sys
sys.path.insert(0, POINTCEPT)
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn
NAMES=["Ground","Building","Natural","Pole","Car","Fence_Barrier","Road_markings","Utility_line","Bollard","Trash_can","Pedestrian"]
PED=10
def build(cfg_p,w):
    cfg=Config.fromfile(cfg_p); fk=tuple(cfg.feat_keys)
    m=build_model(cfg.model); sd=torch.load(w,map_location="cpu",weights_only=False); sd=sd.get("state_dict",sd)
    m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()},strict=True); m=m.cuda().eval()
    tf=Compose([dict(type="CenterShift",apply_z=True),dict(type="GridSample",grid_size=0.05,hash_type="fnv",mode="train",return_grid_coord=True,return_inverse=True),dict(type="CenterShift",apply_z=False),dict(type="ToTensor"),dict(type="Collect",keys=("coord","grid_coord","inverse"),feat_keys=fk)])
    return m,tf
@torch.no_grad()
def predict(m,tf,coord,strength):
    d={"coord":coord.copy(),"strength":strength.copy()}; d=tf(d); inv=d.pop("inverse").cpu().numpy()
    b=point_collate_fn([d]); b={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
    return m(b).get("seg_logits").argmax(1).cpu().numpy()[inv]

print("=== QC: pedestrian height span per validation scene (>3 m = suspected label error) ===")
from collections import defaultdict
zc=defaultdict(list)
for f in sorted(glob.glob("data/whu3d/val/*")):
    sc=os.path.basename(f).rsplit('_x',1)[0]
    c=np.load(f"{f}/coord.npy"); g=np.load(f"{f}/segment.npy")
    if (g==PED).any(): zc[sc].append(c[g==PED,2])
for sc in sorted(zc):
    z=np.concatenate(zc[sc]); span=z.max()-z.min()
    print(f"   {sc}: ped Z-span {span:5.1f}m  {'SUSPECT' if span>3 else 'OK'}")

for tag,cfg_p,w in [("coord","configs/semseg-pt-v3-whu-coord.py","exp/whu-ptv3-coord/model/model_best.pth"),
                    ("coord+int","configs/semseg-pt-v3-whu-coord-intensity.py","exp/whu-ptv3-coord-intensity/model/model_best.pth")]:
    m,tf=build(cfg_p,w); nC=len(NAMES); I=np.zeros(nC,np.int64); U=np.zeros(nC,np.int64)
    for f in sorted(glob.glob("data/whu3d/val/*")):
        sc=os.path.basename(f).rsplit('_x',1)[0]
        c=np.load(f"{f}/coord.npy"); s=np.load(f"{f}/strength.npy"); g=np.load(f"{f}/segment.npy").astype(np.int64)
        p=predict(m,tf,c,s); mask=g!=-1
        for cc in range(nC):
            I[cc]+=int(((p==cc)&(g==cc)&mask).sum()); U[cc]+=int((((p==cc)|(g==cc))&mask).sum())
    iou=np.where(U>0,I/np.maximum(U,1),np.nan)
    present=[cc for cc in range(nC) if U[cc]>0]
    print(f"\n=== WHU-Urban3D validation, all scenes ({len(glob.glob('data/whu3d/val/*'))} chunks) - {tag} ===")
    for cc in range(nC):
        if U[cc]>0: print(f"   {NAMES[cc]:<14} {iou[cc]:.3f}")
    print(f"   >>> mIoU(present {len(present)}) {np.nanmean(iou[present]):.3f}   Pedestrian {iou[PED]:.3f}")
    del m; torch.cuda.empty_cache()
