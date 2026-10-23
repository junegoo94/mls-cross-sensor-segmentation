"""Pseudo-label WHU-Urban3D training scenes with the LOSO fold-3 model."""
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
THR=0.9
cfg=Config.fromfile("configs/semseg-pt-v3-loso3b-AB-coord.py"); fk=tuple(cfg.feat_keys)
m=build_model(cfg.model); sd=torch.load("exp/loso3b-AB-coord/model/model_best.pth",map_location="cpu",weights_only=False)
sd=sd.get("state_dict",sd); m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()},strict=True); m=m.cuda().eval()
tf=Compose([dict(type="CenterShift",apply_z=True),dict(type="GridSample",grid_size=0.05,hash_type="fnv",mode="train",return_grid_coord=True,return_inverse=True),
    dict(type="CenterShift",apply_z=False),dict(type="ToTensor"),dict(type="Collect",keys=("coord","grid_coord","inverse"),feat_keys=fk)])
@torch.no_grad()
def predict(coord,strength):
    d={"coord":coord.copy(),"strength":strength.copy()}; d=tf(d); inv=d.pop("inverse").cpu().numpy()
    b=point_collate_fn([d]); b={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
    p=torch.softmax(m(b).get("seg_logits"),1); mx,pr=p.max(1)
    return pr.cpu().numpy()[inv], mx.cpu().numpy()[inv]

scenes=sorted(glob.glob("data/whu3d/train/*"))
rng=np.random.RandomState(42); pick=[scenes[i] for i in sorted(rng.choice(len(scenes),178,replace=False))]
out_root="data/whu_pseudo/train"; os.makedirs(out_root,exist_ok=True)
tot=0; kept=0
for i,sc in enumerate(pick):
    coord=np.load(f"{sc}/coord.npy"); strength=np.load(f"{sc}/strength.npy")
    pr,mx=predict(coord,strength)
    pseudo=np.where(mx>=THR, pr, -1).astype(np.int32)
    tot+=len(pseudo); kept+=int((pseudo>=0).sum())
    dst=os.path.join(out_root,f"whu_{os.path.basename(sc)}"); os.makedirs(dst,exist_ok=True)
    for a in ("coord.npy","strength.npy"):
        d=os.path.join(dst,a);
        if os.path.islink(d) or os.path.exists(d): os.remove(d)
        os.symlink(os.path.abspath(f"{sc}/{a}"),d)
    np.save(os.path.join(dst,"segment.npy"),pseudo)
    if (i+1)%40==0: print(f"  {i+1}/178  kept so far {100*kept/max(tot,1):.1f}%",flush=True)
print(f">> done. {len(pick)} scenes, pseudo-labels kept {100*kept/tot:.1f}% (conf>={THR})")
