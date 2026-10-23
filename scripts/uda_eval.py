"""Held-out evaluation of the WHU-Urban3D self-training model."""
import os
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTCEPT = os.environ.setdefault("POINTCEPT_ROOT", os.path.join(os.path.dirname(ROOT), "Pointcept"))
os.chdir(ROOT)
import glob, numpy as np, torch, sys
sys.path.insert(0, POINTCEPT)
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn
SHARED=["Ground","Building","Natural","Pole","Car"]; UNI2S={0:0,1:1,2:2,3:3,4:4}
def to_s(a):
    o=np.full(a.shape,-1,np.int64)
    for k,v in UNI2S.items(): o[a==k]=v
    return o
cfg=Config.fromfile("configs/semseg-pt-v3-uda-whu-coord.py"); fk=tuple(cfg.feat_keys)
m=build_model(cfg.model); sd=torch.load("exp/uda-whu-coord/model/model_best.pth",map_location="cpu",weights_only=False)
sd=sd.get("state_dict",sd); m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()},strict=True); m=m.cuda().eval()
tf=Compose([dict(type="CenterShift",apply_z=True),dict(type="GridSample",grid_size=0.05,hash_type="fnv",mode="train",return_grid_coord=True,return_inverse=True),
    dict(type="CenterShift",apply_z=False),dict(type="ToTensor"),dict(type="Collect",keys=("coord","grid_coord","inverse"),feat_keys=fk)])
@torch.no_grad()
def pred(c,s):
    d={"coord":c.copy(),"strength":s.copy()}; d=tf(d); inv=d.pop("inverse").cpu().numpy()
    b=point_collate_fn([d]); b={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
    return m(b).get("seg_logits").argmax(1).cpu().numpy()[inv]
nS=5; I=np.zeros(nS,np.int64); U=np.zeros(nS,np.int64)
for sc in sorted(glob.glob("data/whu3d/val/*")):
    c=np.load(f"{sc}/coord.npy"); s=np.load(f"{sc}/strength.npy"); g=to_s(np.load(f"{sc}/segment.npy").astype(np.int64))
    p=to_s(pred(c,s)); mask=g!=-1
    for k in range(nS): I[k]+=int(((p==k)&(g==k)&mask).sum()); U[k]+=int((((p==k)|(g==k))&mask).sum())
iou=np.where(U>0,I/np.maximum(U,1),np.nan)
print(f"UDA self-training -> held-out WHU: shared-5 mIoU {np.nanmean(iou):.3f}  [{' '.join(f'{SHARED[c][:3]}{iou[c]:.2f}' for c in range(nS))}]")
print("reference: best single source (Tor->WHU) 0.639 | DG joint A+B 0.688")
