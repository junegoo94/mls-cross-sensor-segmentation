"""LOSO held-out 평가: 각 fold 모델을 '학습에 안 쓴 센서' val에 평가 (shared-5, voxel, no TTA).
모든 fold 모델·held-out val은 unified 11클래스 → shared-5 매핑 동일.
single-source baseline(3x3에서)과 비교해 '센서 다양성(A)이 미지 센서 일반화를 돕나' 판정."""
import os, glob, numpy as np, torch, sys
sys.path.insert(0,"/home/jmgoo/projects/Pointcept")
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import point_collate_fn

SHARED=["Ground","Building","Natural","Pole","Car"]
UNI2S={0:0,1:1,2:2,3:3,4:4}   # unified 11 -> shared-5 (나머지 ignore)

# single-source cross-sensor (3x3에서): held-out별 [소스1점, 소스2점]
#  Toronto: NPM->Tor .680, WHU->Tor .636 | NPM3D: Tor->NPM .854, WHU->NPM .573 | WHU: Tor->WHU .639, NPM->WHU .599
SINGLE={"Toronto":[0.680,0.636], "NPM3D":[0.854,0.573], "WHU":[0.639,0.599]}
FOLDS={  # fold: (cfg, weight, held_out_val_dir, held_out_name)
 "loso1":("configs/semseg-pt-v3-loso1-coord.py","exp/loso1-coord/model/model_best.pth","data/uni_toronto/val","Toronto"),
 "loso2":("configs/semseg-pt-v3-loso2-coord.py","exp/loso2-coord/model/model_best.pth","data/uni_npm3d/val","NPM3D"),
 "loso3":("configs/semseg-pt-v3-loso3-coord.py","exp/loso3-coord/model/model_best.pth","data/whu3d/val","WHU"),
}

def to_shared(a):
    o=np.full(a.shape,-1,np.int64)
    for k,v in UNI2S.items(): o[a==k]=v
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

nS=len(SHARED); results={}
for fold,(cfg_p,w,vdir,name) in FOLDS.items():
    if not os.path.exists(w): print(f"[skip] {fold}: weight 없음 {w}"); continue
    model,tf=load(cfg_p,w); I=np.zeros(nS,np.int64); U=np.zeros(nS,np.int64)
    for sc in sorted(glob.glob(f"{vdir}/*")):
        coord=np.load(f"{sc}/coord.npy"); strg=np.load(f"{sc}/strength.npy"); gt=np.load(f"{sc}/segment.npy").astype(np.int64)
        pr=to_shared(predict(coord,strg,model,tf)); g=to_shared(gt); mask=g!=-1
        for c in range(nS):
            I[c]+=int(((pr==c)&(g==c)&mask).sum()); U[c]+=int((((pr==c)|(g==c))&mask).sum())
    iou=np.where(U>0,I/np.maximum(U,1),np.nan); m=np.nanmean(iou); results[name]=(m,iou)
    print(f"{fold} (A) -> held-out {name}: shared-5 mIoU {m:.3f}  [{' '.join(f'{SHARED[c][:3]}{iou[c]:.2f}' for c in range(nS))}]",flush=True)
    del model; torch.cuda.empty_cache()

# 세 baseline: worst / avg(현실) / best(oracle) single-source  vs  A
print("\n===== LOSO held-out shared-5 mIoU: single-source baselines vs A =====")
print(f"{'held-out':>10}{'worst':>8}{'avg*':>8}{'best(oracle)':>13}{'A':>8}{'A-avg':>8}{'A-best':>8}")
def agg(f):
    return np.mean([f(name) for name in results])
for name,(m,iou) in results.items():
    ss=SINGLE[name]; wo,av,be=min(ss),np.mean(ss),max(ss)
    print(f"{name:>10}{wo:>8.3f}{av:>8.3f}{be:>13.3f}{m:>8.3f}{m-av:>+8.3f}{m-be:>+8.3f}")
if results:
    A=np.mean([m for m,_ in results.values()])
    WO=agg(lambda n:min(SINGLE[n])); AV=agg(lambda n:np.mean(SINGLE[n])); BE=agg(lambda n:max(SINGLE[n]))
    print(f"{'MEAN':>10}{WO:>8.3f}{AV:>8.3f}{BE:>13.3f}{A:>8.3f}{A-AV:>+8.3f}{A-BE:>+8.3f}")
print("* avg = 현실 baseline(라벨없는 타겟은 best 못 고름). best = oracle(타겟라벨 필요).")
