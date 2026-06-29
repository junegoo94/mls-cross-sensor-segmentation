"""PPT용 데이터분석/결과 그림 생성 → notes/figs/"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jmgoo/projects/os_seg/notes/figs"
os.makedirs(OUT, exist_ok=True)
NAMES = ["Ground", "Road_mark", "Natural", "Building", "Utility", "Pole", "Car", "Fence"]
COL = np.array([[.5,.5,.5],[.13,1,1],[.06,.5,0],[0,0,1],[.13,1,.02],[.99,.01,1],[.99,.5,.03],[1,1,.04]])

# ---------- 1) Toronto3D train 클래스 분포 ----------
dist = [42.69, 2.13, 6.78, 26.82, 0.87, 1.09, 6.35, 0.52]  # % (학습셋, 겹침 포함)
fig, ax = plt.subplots(figsize=(9, 4.5))
b = ax.bar(NAMES, dist, color=COL)
ax.set_ylabel("비율 (%)"); ax.set_title("Toronto3D 학습셋 클래스 분포 (심한 불균형)")
for r, v in zip(b, dist):
    ax.text(r.get_x()+r.get_width()/2, v+0.4, f"{v}%", ha="center", fontsize=9)
ax.set_ylim(0, 48); plt.tight_layout(); plt.savefig(f"{OUT}/fig1_toronto_classdist.png", dpi=120); plt.close()

# ---------- 2) Voxel 다운샘플 효과 ----------
tiles = ["L001", "L002", "L003", "L004"]
raw = [21.57, 10.28, 39.72, 6.75]      # M
vox = [4.88, 4.99, 6.08, 4.07]         # M
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(x-w/2, raw, w, label="원본", color="#888")
ax.bar(x+w/2, vox, w, label="0.05m voxel", color="#2c7fb8")
for i in range(4):
    ax.text(i-w/2, raw[i]+0.5, f"{raw[i]:.1f}M", ha="center", fontsize=8)
    ax.text(i+w/2, vox[i]+0.5, f"{vox[i]:.1f}M", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(tiles); ax.set_ylabel("점 수 (백만)")
ax.set_title("Voxel 다운샘플 (0.05m) — 밀도 균일화\nL003 39.7M→6.1M (28M 핫스팟 붕괴)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig2_voxel_downsample.png", dpi=120); plt.close()

# ---------- 3) intensity 도메인 갭 (raw 샘플 + z-score) ----------
import laspy
from plyfile import PlyData
print("sampling intensity ...")
ply = PlyData.read("/home/jmgoo/toronto3D_JM/initial_point_clouds/L002.ply")
tor = np.asarray(ply["vertex"].data["scalar_Intensity"]).astype(np.float64)
tor = tor[np.isfinite(tor)]
las = laspy.read("/home/jmgoo/OS_data/ordnance_survey/soton_l2_bng_442_111.laz")
osd = np.asarray(las.intensity).astype(np.float64)
# 샘플
rng = np.random.default_rng(0)
tor_s = rng.choice(tor, min(400000, tor.size), replace=False)
os_s = rng.choice(osd, min(400000, osd.size), replace=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.hist(tor_s, bins=80, range=(0,150), alpha=.6, density=True, label=f"Toronto3D (μ={tor.mean():.1f})", color="C0")
ax.hist(os_s, bins=80, range=(0,150), alpha=.6, density=True, label=f"OS_data (μ={osd.mean():.1f})", color="C3")
ax.set_title("raw intensity — 센서 도메인 갭"); ax.set_xlabel("intensity (0-255)"); ax.legend(); ax.set_ylabel("density")
ax = axes[1]
tz = (tor_s-tor.mean())/tor.std(); oz = (os_s-osd.mean())/osd.std()
ax.hist(tz, bins=80, range=(-3,4), alpha=.6, density=True, label="Toronto3D (z-score)", color="C0")
ax.hist(oz, bins=80, range=(-3,4), alpha=.6, density=True, label="OS_data (z-score)", color="C3")
ax.set_title("per-cloud z-score 후 — 분포 정렬됨"); ax.set_xlabel("z-score"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_intensity_domaingap.png", dpi=120); plt.close()

# ---------- 4) 모델 비교 (per-class IoU) ----------
iou_c = [0.948, 0.001, 0.962, 0.925, 0.856, 0.780, 0.901, 0.337]
iou_i = [0.959, 0.559, 0.962, 0.914, 0.860, 0.802, 0.921, 0.296]
x = np.arange(8); w = 0.4
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(x-w/2, iou_c, w, label="coord-only (mIoU 0.714)", color="#9ecae1")
ax.bar(x+w/2, iou_i, w, label="coord+intensity (mIoU 0.784)", color="#fd8d3c")
ax.annotate("Road_markings\n0.001→0.559", (1, 0.56), (1.4, 0.62),
            arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("IoU"); ax.set_ylim(0,1.05)
ax.set_title("Toronto3D val — 클래스별 IoU (coord vs coord+intensity)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig4_model_compare.png", dpi=120); plt.close()

# ---------- 5) OS_data 예측 분포 (l1) ----------
pc = [46.30, 0.00, 15.35, 32.09, 0.02, 0.58, 2.33, 3.33]
pi = [45.35, 0.51, 15.50, 33.32, 0.01, 0.62, 2.59, 2.09]
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.bar(x-w/2, pc, w, label="coord-only", color="#9ecae1")
ax.bar(x+w/2, pi, w, label="coord+intensity", color="#fd8d3c")
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("예측 비율 (%)")
ax.set_title("OS_data (soton_l1) 예측 클래스 분포 — 두 모델 거의 동일 (road_markings 부재)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig5_os_pred_dist.png", dpi=120); plt.close()

print("saved figs to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
