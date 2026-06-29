"""colored 예측(LAZ RGB)의 라벨↔색 범례 → 1-page PDF(+PNG).
3개 모델(Toronto / NPM3D / JOINT) class id → 색. 색은 scripts/color_predictions.py 와 동일."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

C = {  # 0-255
    "Ground": (128,128,128), "Building": (0,0,230), "Natural": (30,140,30),
    "Pole": (240,50,230), "Car": (245,130,30), "Road_markings": (0,230,230),
    "Utility_line": (160,230,40), "Fence": (240,225,30), "Bollard": (150,60,200),
    "Trash_can": (240,140,200), "Barrier": (140,70,20), "Pedestrian": (230,20,20),
    "Fence_Barrier": (140,70,20),
}
def rgb(n): return tuple(c/255 for c in C[n])

TORONTO = ["Ground","Road_markings","Natural","Building","Utility_line","Pole","Car","Fence"]
NPM3D   = ["Ground","Building","Pole","Bollard","Trash_can","Barrier","Pedestrian","Car","Natural"]
JOINT   = ["Ground","Building","Natural","Pole","Car","Fence_Barrier",
           "Road_markings","Utility_line","Bollard","Trash_can","Pedestrian"]
SHARED  = {"Ground","Building","Natural","Pole","Car"}
NAVY = "#14304a"

fig = plt.figure(figsize=(14, 7.6), dpi=200); fig.patch.set_facecolor("white")
ax = fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)

ax.add_patch(Rectangle((0,0.90),1,0.10,color=NAVY))
ax.text(0.025,0.945,"OS_data Prediction — Label / Color Legend", color="white",
        fontsize=18, fontweight="bold", va="center")
ax.text(0.025,0.862,"exp/osdata_infer_colored/{toronto,npm3d,joint}/  ·  RGB baked into LAZ (Colors→RGB in CloudCompare)",
        fontsize=9.5, color="#444", va="center")

def panel(x0, title, names, sub):
    ax.text(x0, 0.80, title, fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(x0, 0.776, sub, fontsize=8.5, color="#666", style="italic")
    y = 0.735
    for i, n in enumerate(names):
        ax.add_patch(Rectangle((x0, y-0.020), 0.035, 0.034, facecolor=rgb(n),
                     edgecolor="#333", lw=0.5))
        ped = (n == "Pedestrian")
        ax.text(x0+0.045, y, f"{i:>2}  {n}{'  ◄' if ped else ''}", fontsize=10.5, va="center",
                fontweight=("bold" if ped else "normal"),
                color=("#c0392b" if ped else "#1a1a1a"))
        y -= 0.052

panel(0.025, "Toronto3D model  (8)", TORONTO, "classification id → class")
panel(0.36,  "NPM3D model  (9)", NPM3D, "classification id → class")
panel(0.69,  "JOINT model  (11)  ★ best", JOINT, "Toronto+NPM3D, domain-adapted")

# 하단 노트
ax.add_patch(Rectangle((0.025,0.045),0.95,0.075, facecolor="#f4f7fa", edgecolor="#ccc"))
ax.text(0.04,0.10,"Same meaning = same color across all three models:", fontsize=9.5, fontweight="bold")
sx = 0.04
for n in ["Ground","Building","Natural","Pole","Car"]:
    ax.add_patch(Rectangle((sx,0.058),0.026,0.024, facecolor=rgb(n), edgecolor="#333", lw=0.5))
    ax.text(sx+0.032,0.070,n,fontsize=8.5,va="center"); sx += 0.12
ax.text(0.04,0.03,"Pedestrian = red ◄ (NPM3D & JOINT only).   JOINT merges Fence+Barrier (brown) and adds domain adaptation (A+B).",
        fontsize=8.5, color="#555")

fig.savefig("/home/jmgoo/projects/os_seg/exp/osdata_infer_colored/color_legend.pdf")
fig.savefig("/home/jmgoo/projects/os_seg/exp/osdata_infer_colored/color_legend.png", dpi=160)
print(">> saved color_legend.pdf / .png (3 models)")
