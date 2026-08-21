import os, glob, argparse, numpy as np, torch, sys
sys.path.insert(0,"/home/jmgoo/projects/Pointcept")
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn
ap=argparse.ArgumentParser()
ap.add_argument("--src_cfg",required=True); ap.add_argument("--src_weight",required=True)
ap.add_argument("--tgt_train",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--n",type=int,default=178); ap.add_argument("--thr",type=float,default=0.9)
a=ap.parse_args()
cfg=Config.fromfile(a.src_cfg); fk=tuple(cfg.feat_keys)
m=build_model(cfg.model); sd=torch.load(a.src_weight,map_location="cpu",weights_only=False); sd=sd.get("state_dict",sd)
m.load_state_dict({k.replace("module.",""):v for k,v in sd.items()},strict=True); m=m.cuda().eval()
tf=Compose([dict(type="CenterShift",apply_z=True),dict(type="GridSample",grid_size=0.05,hash_type="fnv",mode="train",return_grid_coord=True,return_inverse=True),
    dict(type="CenterShift",apply_z=False),dict(type="ToTensor"),dict(type="Collect",keys=("coord","grid_coord","inverse"),feat_keys=fk)])
@torch.no_grad()
def predict(c,s):
    d={"coord":c.copy(),"strength":s.copy()}; d=tf(d); inv=d.pop("inverse").cpu().numpy()
    b=point_collate_fn([d]); b={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
    p=torch.softmax(m(b).get("seg_logits"),1); mx,pr=p.max(1); return pr.cpu().numpy()[inv],mx.cpu().numpy()[inv]
scenes=sorted(glob.glob(f"{a.tgt_train}/*")); rng=np.random.RandomState(42)
pick=[scenes[i] for i in sorted(rng.choice(len(scenes),min(a.n,len(scenes)),replace=False))]
os.makedirs(a.out,exist_ok=True); tot=kept=0
for sc in pick:
    coord=np.load(f"{sc}/coord.npy"); strength=np.load(f"{sc}/strength.npy")
    pr,mx=predict(coord,strength); pseudo=np.where(mx>=a.thr,pr,-1).astype(np.int32)
    tot+=len(pseudo); kept+=int((pseudo>=0).sum())
    dst=os.path.join(a.out,os.path.basename(sc)); os.makedirs(dst,exist_ok=True)
    for asset in ("coord.npy","strength.npy"):
        d=os.path.join(dst,asset)
        if os.path.islink(d) or os.path.exists(d): os.remove(d)
        os.symlink(os.path.abspath(f"{sc}/{asset}"),d)
    np.save(os.path.join(dst,"segment.npy"),pseudo)
print(f">> {a.out}: {len(pick)} scenes, pseudo 채택 {100*kept/tot:.1f}% (conf>={a.thr})")
