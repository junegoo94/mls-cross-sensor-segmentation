"""Export WHU-Urban3D scene 4629 before and after the pedestrian height correction."""
import numpy as np, h5py, os
from plyfile import PlyData, PlyElement
from pywhu3d.configs.mls_config import sem_map
MAP18=np.array([-1,2,0,1,-1,3,3,3,2,-1,0,6,4,10,3,-1,5,7])
COL=np.array([[128,128,128],[0,0,230],[30,140,30],[240,50,230],[245,130,30],
              [140,70,20],[0,230,230],[160,230,40],[150,60,200],[240,140,200],[230,20,20]],np.uint8)
IGN=np.array([15,15,15],np.uint8)
with h5py.File("data/whu3d_download/extracted/mls/h5/4629.h5",'r') as h:
    xyz=h['coords'][:].astype(np.float64); raw=h['semantics'][:].astype(np.int64); ins=h['instances'][:].astype(np.int64)
lut={int(r):int(MAP18[c]) for r,c in sem_map.items()}
uni=np.full(len(raw),-1,np.int64)
for r,u in lut.items(): uni[raw==r]=u
pm=uni==10
for iid in np.unique(ins[pm]):
    m=pm&(ins==iid); z=xyz[m,2]
    if z.max()-z.min()>2.5: uni[m]=-1
rgb=np.where((uni>=0)[:,None], COL[np.clip(uni,0,10)], IGN)
off=xyz.min(0); P=(xyz-off).astype(np.float32)
dt=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),('corrected_class','f4'),('is_pedestrian','f4')]
v=np.empty(len(P),dtype=dt)
v['x'],v['y'],v['z']=P[:,0],P[:,1],P[:,2]; v['red'],v['green'],v['blue']=rgb[:,0],rgb[:,1],rgb[:,2]
v['corrected_class']=uni.astype(np.float32); v['is_pedestrian']=(uni==10).astype(np.float32)
os.makedirs("exp/whu_qc",exist_ok=True); out="exp/whu_qc/4629_corrected.ply"
PlyData([PlyElement.describe(v,'vertex')],text=False).write(out)
print(f"{out}: {len(P):,} pts, {os.path.getsize(out)/1e6:.0f}MB")
print(f"  pedestrian (corrected)={100*(uni==10).mean():.2f}%  ignore={100*(uni==-1).mean():.2f}%")
print("  In CloudCompare (RGB): black = removed facade strip, red = real pedestrians.")
