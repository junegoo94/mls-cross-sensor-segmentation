"""리포트 PDF용 figure 일괄 생성 (영문 라벨) → notes/figs/.
Toronto3D + NPM3D 분석/결과 차트. 학습곡선/OS예측 top-view 는 exp/ 에 이미 존재."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jmgoo/projects/os_seg/notes/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=130)
    plt.close(fig)
    print("saved", name)


# ============================ Toronto3D ============================
TOR = ["Ground", "Road_mark", "Natural", "Building", "Utility", "Pole", "Car", "Fence"]
TOR_COL = np.array([[.5,.5,.5],[.13,1,1],[.06,.5,0],[0,0,1],[.13,1,.02],[.99,.01,1],[.99,.5,.03],[1,1,.04]])

# 1) Toronto train 클래스 분포
dist = [42.69, 2.13, 6.78, 26.82, 0.87, 1.09, 6.35, 0.52]
fig, ax = plt.subplots(figsize=(9, 4.2))
b = ax.bar(TOR, dist, color=TOR_COL, edgecolor="k", linewidth=0.4)
ax.set_ylabel("Share (%)"); ax.set_title("Toronto3D Training-set Class Distribution (severe imbalance)")
for r, v in zip(b, dist):
    ax.text(r.get_x()+r.get_width()/2, v+0.5, f"{v}%", ha="center", fontsize=8.5)
ax.set_ylim(0, 48); save(fig, "tor_classdist.png")

# 2) Voxel 다운샘플 효과
tiles = ["L001", "L002", "L003", "L004"]
raw = [21.57, 10.28, 39.72, 6.75]; vox = [4.88, 4.99, 6.08, 4.07]
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(x-w/2, raw, w, label="Raw", color="#888")
ax.bar(x+w/2, vox, w, label="0.05 m voxel", color="#2c7fb8")
for i in range(4):
    ax.text(i-w/2, raw[i]+0.5, f"{raw[i]:.1f}", ha="center", fontsize=8)
    ax.text(i+w/2, vox[i]+0.5, f"{vox[i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(tiles); ax.set_ylabel("Points (M)")
ax.set_title("Voxel Downsampling Equalizes Density (Toronto3D)"); ax.legend()
save(fig, "tor_voxel.png")

# 3) intensity 도메인 갭
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(2); w = 0.35
ax.bar(x-w/2, [15, 31], w, yerr=[15, 27], capsize=5, label="mean (±std)",
       color=["#2c7fb8", "#d95f0e"])
ax.set_xticks(x); ax.set_xticklabels(["Toronto3D", "OS_data"])
ax.set_ylabel("Intensity (0-255)")
ax.set_title("Intensity Domain Gap → per-cloud z-score")
ax.text(0, 16, "~15", ha="center"); ax.text(1, 32, "~31", ha="center")
save(fig, "tor_intensity_gap.png")

# 4) Toronto per-class IoU (coord vs coord+intensity)
iou_c = [0.948, 0.001, 0.962, 0.925, 0.856, 0.780, 0.901, 0.337]
iou_ci = [0.959, 0.559, 0.962, 0.914, 0.860, 0.802, 0.921, 0.296]
x = np.arange(8); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(x-w/2, iou_c, w, label="coord-only (mIoU 0.714)", color="#7fb3d5")
ax.bar(x+w/2, iou_ci, w, label="coord+intensity (mIoU 0.784)", color="#e67e22")
ax.set_xticks(x); ax.set_xticklabels(TOR, rotation=20, ha="right")
ax.set_ylabel("IoU"); ax.set_ylim(0, 1.05)
ax.set_title("Toronto3D Val Per-class IoU — intensity gain is all Road_markings")
ax.annotate("+0.558", (1, 0.56), textcoords="offset points", xytext=(0, 8),
            ha="center", color="#c0392b", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
save(fig, "tor_perclass_iou.png")

# ============================ NPM3D ============================
NPM = ["Ground", "Building", "Pole", "Bollard", "Trash", "Barrier", "Pedestrian", "Car", "Natural"]
NPM_COL = np.array([[.5,.5,.5],[0,0,1],[.99,.01,1],[.99,.5,.03],[1,1,.04],
                    [.55,.27,.07],[1,0,0],[0,1,1],[.06,.5,0]])

# 5) NPM3D train 클래스 분포 (log y, 희소 클래스 가시화)
ndist = [40.67, 34.76, 0.55, 0.05, 0.26, 3.36, 0.17, 3.07, 17.10]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
b = ax.bar(NPM, ndist, color=NPM_COL, edgecolor="k", linewidth=0.4)
ax.set_yscale("log"); ax.set_ylabel("Share (%, log)")
ax.set_title("NPM3D Training-set Class Distribution (Pedestrian only 0.17%)")
for r, v in zip(b, ndist):
    ax.text(r.get_x()+r.get_width()/2, v*1.1, f"{v}%", ha="center", fontsize=8)
# pedestrian 강조
b[6].set_edgecolor("red"); b[6].set_linewidth(2)
save(fig, "npm_classdist.png")

# 6) NPM3D per-class IoU (coord vs coord+intensity)  [TTA test]
niou_c = [0.982, 0.986, 0.768, 0.840, 0.804, 0.680, 0.916, 0.979, 0.894]
niou_ci = [0.977, 0.985, 0.858, 0.863, 0.818, 0.766, 0.811, 0.987, 0.890]
x = np.arange(9); w = 0.38
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.bar(x-w/2, niou_c, w, label="coord-only (mIoU 0.872)", color="#7fb3d5")
ax.bar(x+w/2, niou_ci, w, label="coord+intensity (mIoU 0.884)", color="#e67e22")
ax.set_xticks(x); ax.set_xticklabels(NPM, rotation=20, ha="right")
ax.set_ylabel("IoU"); ax.set_ylim(0, 1.08)
ax.set_title("NPM3D Val Per-class IoU — Pedestrian is geometric (coord-only wins)")
ax.annotate("0.916 vs 0.811", (6, 0.916), textcoords="offset points", xytext=(0, 10),
            ha="center", color="#c0392b", fontweight="bold", fontsize=9)
ax.legend(loc="lower left", fontsize=9)
save(fig, "npm_perclass_iou.png")

# 7) OS_data pedestrian 검출 수
labels = ["soton_l1", "soton_l2"]
ped_c = [14575, 41352]; ped_ci = [25989, 35063]
x = np.arange(2); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.3))
b1 = ax.bar(x-w/2, ped_c, w, label="coord-only", color="#7fb3d5")
b2 = ax.bar(x+w/2, ped_ci, w, label="coord+intensity", color="#e67e22")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+600,
                f"{int(r.get_height()):,}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Predicted pedestrian points")
ax.set_title("OS_data Pedestrian Detections (Toronto model: 0 — class absent)")
ax.legend()
save(fig, "npm_os_pedestrian.png")

print(">> all figures ->", OUT)
