"""OS 예측 비교: NPM3D-only(old) vs JOINT+aug(new). 같은 점 순서라 점대점 비교.
- old가 Building이라 한 점을 new가 무엇으로 바꿨나(차→Car, 인도→Ground) 정량화
- 차/인도 오분류 많은 50m 구간 자동 선택 → top + side(elevation) before/after 크롭."""
import numpy as np, laspy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

STEM = "soton_l2_bng_442_111"
OLD = f"/home/jmgoo/projects/os_seg/exp/osdata_infer_npm3d/{STEM}_pred_coord.laz"   # 9-class
NEW = f"/home/jmgoo/projects/os_seg/exp/osdata_infer_joint/{STEM}_pred_coord.laz"   # 11-class
OUT = "/home/jmgoo/projects/os_seg/notes/figs/joint_vs_npm3d_coord.png"

# 통합(unified) 색 0-1, id 0-10
UNAME = ["Ground","Building","Natural","Pole","Car","Fence_Barrier",
         "Road_markings","Utility_line","Bollard","Trash_can","Pedestrian"]
UCOL = np.array([[.5,.5,.5],[0,0,.9],[.12,.55,.12],[.94,.2,.9],[.96,.51,.12],
                 [.55,.27,.08],[0,.9,.9],[.63,.9,.16],[.59,.24,.78],[.94,.55,.78],[.9,.08,.08]])
# NPM3D 9-class id -> unified id
NPM2U = {0:0,1:1,2:3,3:8,4:9,5:5,6:10,7:4,8:2}

print(">> loading ...")
las=laspy.read(OLD); xyz=np.stack([np.asarray(las.x),np.asarray(las.y),np.asarray(las.z)],1)
old9=np.asarray(las.classification)
old=np.array([NPM2U[c] for c in range(9)],dtype=np.int64)[old9]  # -> unified
new=np.asarray(laspy.read(NEW).classification).astype(np.int64)   # already unified
assert old.shape==new.shape==(xyz.shape[0],)
print(f"   {xyz.shape[0]:,} pts")

# --- old=Building 점들을 new가 어떻게 바꿨나 ---
mb = old==1
print(f"\n>> old(NPM3D-only) 가 Building 이라 한 점: {mb.sum():,}")
uq,cnt=np.unique(new[mb],return_counts=True)
for u,c in sorted(zip(uq,cnt),key=lambda t:-t[1]):
    print(f"   -> new: {UNAME[u]:<13} {c:>10,} ({100*c/mb.sum():4.1f}%)")

# --- 차/인도 fix 신호: old=Building & new in {Car, Ground} ---
fix_car = mb & (new==4)
fix_grd = mb & (new==0)
print(f"\n>> old-Building -> new-Car  (차 옆면 교정 후보): {fix_car.sum():,}")
print(f">> old-Building -> new-Ground(인도/바닥 교정 후보): {fix_grd.sum():,}")

# --- 교정 신호 많은 50m 셀 자동 선택 ---
size=50.0; mins=xyz[:,:2].min(0)
cell=np.floor((xyz[:,:2]-mins)/size).astype(int)
score_mask = fix_car | fix_grd
from collections import Counter
cc=Counter(map(tuple, cell[score_mask]))
best=max(cc, key=cc.get)
print(f"\n>> best crop cell {best}: 교정후보 {cc[best]:,}점")
ox,oy=mins[0]+best[0]*size, mins[1]+best[1]*size
m=(xyz[:,0]>=ox)&(xyz[:,0]<ox+size)&(xyz[:,1]>=oy)&(xyz[:,1]<oy+size)
cx,cy,cz=xyz[m,0],xyz[m,1],xyz[m,2]; co,cn=old[m],new[m]
print(f"   crop pts {m.sum():,}")

def scat(ax,a,b,lab,view):
    if view=="top": X,Y=a,b
    else: X,Y=a,b
    order=np.argsort(b) if view=="top" else np.argsort(-cz)  # 위/높이 정렬
    ax.scatter(X,Y,c=UCOL[(co if lab=='old' else cn)],s=1.2,marker=".",linewidths=0)

present=np.unique(np.concatenate([co,cn]))
fig,axes=plt.subplots(2,2,figsize=(15,12))
# top view (x,y)
axes[0,0].scatter(cx,cy,c=UCOL[co],s=1.2,marker=".",lw=0); axes[0,0].set_title("OLD (NPM3D-only)  —  TOP view",fontsize=13)
axes[0,1].scatter(cx,cy,c=UCOL[cn],s=1.2,marker=".",lw=0); axes[0,1].set_title("NEW (JOINT+aug)  —  TOP view",fontsize=13)
for ax in (axes[0,0],axes[0,1]): ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
# side view (x,z) — 수직면(차옆/벽) 보임
axes[1,0].scatter(cx,cz,c=UCOL[co],s=1.2,marker=".",lw=0); axes[1,0].set_title("OLD — SIDE (x-z) elevation",fontsize=13)
axes[1,1].scatter(cx,cz,c=UCOL[cn],s=1.2,marker=".",lw=0); axes[1,1].set_title("NEW — SIDE (x-z) elevation",fontsize=13)
for ax in (axes[1,0],axes[1,1]): ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("z (height)")
fig.legend(handles=[Patch(color=UCOL[c],label=f"{c}-{UNAME[c]}") for c in present],
           loc="lower center",ncol=6,fontsize=10)
fig.suptitle(f"OS {STEM} coord — crop {best}  (Car=orange, Building=blue, Pedestrian=red)",fontsize=14)
plt.tight_layout(rect=[0,0.04,1,0.97]); plt.savefig(OUT,dpi=110); print("\nsaved",OUT)
