"""Raw sampling statistics per sensor: density, nearest-neighbour spacing, and 5 cm voxel survival."""
import numpy as np, json, os, glob, time
from scipy.spatial import cKDTree
from plyfile import PlyData
import h5py, laspy
OUT="/home/jmgoo/projects/os_seg/exp/final_results/sampling_stats"; os.makedirs(OUT,exist_ok=True)
GRID=0.05; CELL=1.0; NWIN=4; WIN=60.0; MARGIN=3.0; NQ=100_000
rng=np.random.RandomState(0)

def file_stats(xyz):
    n=len(xyz); mn=xyz.min(0); q=xyz-mn
    ij=np.floor(q[:,:2]/CELL).astype(np.int64); area=len(np.unique(ij[:,0]*(1<<32)+ij[:,1]))
    v=np.floor(q/GRID).astype(np.int64); nvox=len(np.unique((v[:,0]*(1<<21)+v[:,1])*(1<<21)+v[:,2]))
    nn=[]
    for _ in range(NWIN):
        for _try in range(20):
            c=q[rng.randint(n),:2]; m=(np.abs(q[:,0]-c[0])<WIN/2)&(np.abs(q[:,1]-c[1])<WIN/2)
            if m.sum()>50_000: break
        sub=q[m]; tree=cKDTree(sub)
        inner=(np.abs(sub[:,0]-c[0])<WIN/2-MARGIN)&(np.abs(sub[:,1]-c[1])<WIN/2-MARGIN)
        idx=np.where(inner)[0]
        if len(idx)==0: continue
        idx=rng.choice(idx,min(NQ,len(idx)),replace=False)
        d,_=tree.query(sub[idx],k=2); nn.append(d[:,1])
    nn=np.concatenate(nn) if nn else np.array([np.nan])
    return dict(n=int(n),area_m2=int(area),density=n/area,voxel_survival=nvox/n,
                pts_per_voxel=n/nvox,nn_med_cm=float(np.median(nn)*100),
                nn_p25_cm=float(np.percentile(nn,25)*100),nn_p75_cm=float(np.percentile(nn,75)*100),
                nn_samples=nn.astype(np.float32))

def ply_xyz(p):
    v=PlyData.read(p)["vertex"].data
    return np.stack([np.asarray(v["x"],np.float64),np.asarray(v["y"],np.float64),np.asarray(v["z"],np.float64)],1)
def h5_xyz(p):
    with h5py.File(p,"r") as h: return h["coords"][:].astype(np.float64)
def laz_xyz(p):
    l=laspy.read(p); return np.stack([np.asarray(l.x),np.asarray(l.y),np.asarray(l.z)],1)

SETS={
 "Toronto3D":(sorted(glob.glob("/home/jmgoo/toronto3D_JM/initial_point_clouds/L00*.ply")),ply_xyz),
 "Paris-Lille-3D":(sorted(glob.glob("/home/jmgoo/paris_lille_3d/training_10_classes/*.ply")),ply_xyz),
 "WHU-Urban3D":(sorted(glob.glob("/home/jmgoo/projects/os_seg/data/whu3d_download/extracted/mls/h5/*.h5")),h5_xyz),
 "OS-Southampton":(sorted(glob.glob("/home/jmgoo/OS_data/ordnance_survey/*.laz")),laz_xyz),
}
summary={}
for name,(files,reader) in SETS.items():
    per=[]; pooled=[]; t0=time.time()
    for f in files:
        try: xyz=reader(f)
        except Exception as e: print(f"[{name}] skip {os.path.basename(f)}: {e}",flush=True); continue
        s=file_stats(xyz); pooled.append(s.pop("nn_samples")); s["file"]=os.path.basename(f); per.append(s)
        print(f"[{name}] {s['file']}: n={s['n']:,} area={s['area_m2']:,} m2 dens={s['density']:.0f}/m2 "
              f"nn_med={s['nn_med_cm']:.2f}cm surv={100*s['voxel_survival']:.1f}%  ({time.time()-t0:.0f}s)",flush=True)
        del xyz
    N=sum(p["n"] for p in per); A=sum(p["area_m2"] for p in per); NV=sum(p["n"]*p["voxel_survival"] for p in per)
    nn=np.concatenate(pooled)
    summary[name]=dict(files=per,n=N,area_m2=A,density=N/A,voxel_survival=NV/N,pts_per_voxel=N/NV,
        nn_med_cm=float(np.median(nn)*100),nn_p25_cm=float(np.percentile(nn,25)*100),nn_p75_cm=float(np.percentile(nn,75)*100))
    json.dump(summary,open(f"{OUT}/sampling_stats.json","w"),indent=1)
    print(f"==> {name}: n={N/1e6:.1f}M area={A/1e3:.1f}k m2 density={N/A:.0f} pts/m2 nn_med={summary[name]['nn_med_cm']:.2f} cm "
          f"(IQR {summary[name]['nn_p25_cm']:.2f}-{summary[name]['nn_p75_cm']:.2f}) survival={100*NV/N:.1f}% pts/voxel={N/NV:.2f}",flush=True)
with open(f"{OUT}/sampling_stats_rows.tex","w") as fo:
    for k,s in summary.items():
        fo.write(f"{k} & {s['n']/1e6:.1f} & {s['area_m2']/1e3:.1f} & {s['density']:.0f} & {s['nn_med_cm']:.1f} ({s['nn_p25_cm']:.1f}--{s['nn_p75_cm']:.1f}) & {s['pts_per_voxel']:.2f} & {100*s['voxel_survival']:.0f}\\\\\n")
print("DONE",flush=True)
