"""Full 3x3 cross-sensor transfer matrix (shared-5 mIoU)."""
import os, glob, numpy as np, torch, sys
sys.path.insert(0,"/home/jmgoo/projects/Pointcept")
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn

SHARED=["Ground","Building","Natural","Pole","Car"]
TOR2S={0:0,2:2,3:1,5:3,6:4}
NPM2S={0:0,1:1,2:3,7:4,8:2}
WHU2S={0:0,1:1,2:2,3:3,4:4}
MAP={"tor":TOR2S,"npm":NPM2S,"whu":WHU2S}

MODELS={
 "Toronto":("configs/semseg-pt-v3-coord.py","exp/toronto3d-ptv3-coord/model/model_best.pth","tor"),
 "NPM3D":  ("configs/semseg-pt-v3-npm3d-coord.py","exp/npm3d-ptv3-coord/model/model_best.pth","npm"),
 "WHU":    ("configs/semseg-pt-v3-whu-coord.py","exp/whu-ptv3-coord/model/model_best.pth","whu"),
}
VALS={
 "Toronto":("data/toronto3d/val","tor"),
 "NPM3D":  ("data/npm3d/val","npm"),
 "WHU":    ("data/whu3d/val","whu"),
}

def to_shared(a,m):
    o=np.full(a.shape,-1,np.int64)
    for k,v in m.items(): o[a==k]=v
    return o
def build_tf(fk):
    return Compose([dict(type="CenterShift",apply_z=True),
        dict(type="GridSample",grid_size=0.05,hash_type="fnv",mode="train",return_grid_coord=True,return_inverse=True),
        dict(type="CenterShift",apply_z=False),dict(type="ToTensor"),
        dict(type="Collect",keys=("coord","grid_coord","inverse"),feat_keys=fk)])
@torch.no_grad()
def predict(coord,strength,model,tf):
    d={"coord":coord.copy(),"strength":strength.copy()}; d=tf(d); inv=d.pop("inverse").cpu().numpy()
    b=point_collate_fn([d]); b={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
    return model(b).get("seg_logits").argmax(1).cpu().numpy()[inv]

def load(cfg_p,w):
    cfg=Config.fromfile(cfg_p); fk=tuple(cfg.feat_keys)
    m=build_model(cfg.model); sd=torch.load(w,map_location="cpu",weights_only=False); sd=sd.get("state_dict",sd)
    m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()},strict=True); return m.cuda().eval(),build_tf(fk)

nS=len(SHARED); mat={}
for mn,(cfg_p,w,mspace) in MODELS.items():
    model,tf=load(cfg_p,w); pmap=MAP[mspace]
    for vn,(vdir,vspace) in VALS.items():
        gmap=MAP[vspace]; I=np.zeros(nS,np.int64); U=np.zeros(nS,np.int64)
        for sc in sorted(glob.glob(f"{vdir}/*")):
            coord=np.load(f"{sc}/coord.npy"); strg=np.load(f"{sc}/strength.npy"); gt=np.load(f"{sc}/segment.npy").astype(np.int64)
            pr=to_shared(predict(coord,strg,model,tf),pmap); g=to_shared(gt,gmap); mask=g!=-1
            for c in range(nS):
                I[c]+=int(((pr==c)&(g==c)&mask).sum()); U[c]+=int((((pr==c)|(g==c))&mask).sum())
        iou=np.where(U>0,I/np.maximum(U,1),np.nan); mat[(mn,vn)]=(np.nanmean(iou),iou)
        print(f"  {mn:>8} -> {vn:<8}: shared-5 mIoU {np.nanmean(iou):.3f}   [{' '.join(f'{SHARED[c][:3]}{iou[c]:.2f}' for c in range(nS))}]",flush=True)
    del model; torch.cuda.empty_cache()

print("\n===== 3x3 NoAdapt shared-5 mIoU (row=train model, col=test sensor) =====")
hdr="train\\test"; print(f"{hdr:>10} "+" ".join(f"{v:>9}" for v in VALS))
for mn in MODELS:
    print(f"{mn:>10} "+" ".join(f"{mat[(mn,vn)][0]:>9.3f}"+("*" if mn==vn else " ") for vn in VALS))
print("(* = in-domain diagonal)")
