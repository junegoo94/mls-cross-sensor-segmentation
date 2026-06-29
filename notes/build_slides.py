"""8-page 16:9 슬라이드 PDF 생성 (matplotlib 직접 렌더 → 표 안 깨짐).
모든 시각화는 notes/report_figures/ 의 PNG 사용. 영문."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

BASE = "/home/jmgoo/projects/os_seg"
RF = f"{BASE}/notes/report_figures"
OUT = f"{BASE}/notes/slides_en.pdf"
PNGDIR = f"{BASE}/notes/slides_png"
os.makedirs(PNGDIR, exist_ok=True)
_N = [0]


def emit(pdf, fig):
    _N[0] += 1
    fig.savefig(f"{PNGDIR}/slide_{_N[0]}.png", dpi=150, facecolor="white")
    pdf.savefig(fig)
    plt.close(fig)

NAVY = "#14304a"; ORANGE = "#e67e22"; GRAY = "#555555"; RED = "#c0392b"
W, H = 13.333, 7.5  # 16:9


def new_slide(pdf, title, idx, subtitle=None):
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor("white")
    # title bar
    bar = fig.add_axes([0, 0.88, 1, 0.12]); bar.axis("off")
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, color=NAVY))
    bar.text(0.018, 0.55, title, color="white", fontsize=21, fontweight="bold",
             va="center", ha="left")
    if subtitle:
        bar.text(0.018, 0.16, subtitle, color="#c9d6e3", fontsize=11, va="center")
    bar.text(0.982, 0.5, f"{idx}/8", color="#9fb3c8", fontsize=12, va="center", ha="right")
    return fig


def bullets(fig, rect, lines, fs=13, lh=None):
    ax = fig.add_axes(rect); ax.axis("off")
    n = len(lines); lh = lh or (1.0 / max(n, 1))
    y = 1.0
    for ln in lines:
        if ln == "":
            y -= lh * 0.5; continue
        sub = ln.startswith("  ")
        txt = ln.strip()
        mark = "–" if sub else "•"
        color = GRAY if sub else "#1a1a1a"
        x = 0.05 if sub else 0.0
        # bold 처리: **...**
        weight = "bold" if txt.startswith("**") else "normal"
        txt = txt.replace("**", "")
        ax.text(x, y, f"{mark}  {txt}", fontsize=fs - (1 if sub else 0),
                va="top", ha="left", color=color, fontweight=weight, wrap=True)
        y -= lh
    return ax


def image(fig, rect, path, title=None):
    ax = fig.add_axes(rect); ax.axis("off")
    if not os.path.exists(path):
        ax.text(0.5, 0.5, f"[missing]\n{os.path.basename(path)}", ha="center", va="center",
                fontsize=8, color="red"); return ax
    ax.imshow(mpimg.imread(path))
    if title:
        ax.set_title(title, fontsize=9.5, color=GRAY)
    return ax


def table(fig, rect, col_labels, cell_text, col_w=None, fs=11, highlight_row=None):
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                 loc="center", colWidths=col_w)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.55)
    for (r, c), cell in t.get_cells().items() if hasattr(t, "get_cells") else t.get_celld().items():
        cell.set_edgecolor("#aaaaaa")
        if r == 0:
            cell.set_facecolor(NAVY); cell.set_text_props(color="white", fontweight="bold")
        elif highlight_row is not None and r == highlight_row:
            cell.set_facecolor("#fdebd0")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f7fa")
    return ax


def caption(fig, x, y, text):
    fig.text(x, y, text, fontsize=8.5, color=GRAY, style="italic", ha="center")


# ===================================================================
pdf = PdfPages(OUT)

# ---- Slide 1: Title ----
fig = plt.figure(figsize=(W, H), dpi=200); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0.62), 1, 0.38, color=NAVY))
ax.text(0.5, 0.86, "Cross-Domain Point Cloud", ha="center", fontsize=33, fontweight="bold", color="white")
ax.text(0.5, 0.775, "Semantic Segmentation with PTv3", ha="center", fontsize=33, fontweight="bold", color="white")
ax.text(0.5, 0.685, "Transfer from Toronto3D & Paris-Lille-3D (NPM3D)  →  OS_data (Ordnance Survey MLS)",
        ha="center", fontsize=14, color="#c9d6e3")
image(fig, [0.30, 0.07, 0.40, 0.50], f"{RF}/fig9f_os_l2_npm3d_coord_pedestrian.png")
ax.text(0.5, 0.045, "OS_data soton_l2 — NPM3D PTv3 prediction (class 6 = Pedestrian, red)",
        ha="center", fontsize=9.5, color=GRAY, style="italic")
emit(pdf, fig)

# ---- Slide 2: Motivation & Data ----
fig = new_slide(pdf, "Motivation & Data", 2, "Label an RGB-less outdoor MLS cloud with no ground truth")
bullets(fig, [0.045, 0.18, 0.40, 0.66], [
    "**Goal: segment OS_data (Southampton street MLS)",
    "  RGB-less, intensity only — no labels available",
    "",
    "**Strategy: train on a labeled source, transfer",
    "  Toronto3D — has Road_markings, NO pedestrian",
    "  NPM3D — has Pedestrian (key for people)",
    "",
    "**Both sources are RGB-less MLS → match OS sensor",
    "**Compare coord-only vs coord+intensity",
], fs=13.5)
table(fig, [0.49, 0.30, 0.49, 0.50],
      ["", "Toronto3D", "NPM3D", "OS_data"],
      [["Type", "MLS", "MLS", "MLS (target)"],
       ["RGB", "yes*", "no", "no"],
       ["Classes", "8", "9", "— (infer)"],
       ["Labels", "yes", "yes", "none"],
       ["Role", "source", "source", "target"]],
      col_w=[0.16, 0.20, 0.20, 0.22], fs=11)
caption(fig, 0.735, 0.25, "*RGB present but dropped to match OS_data's RGB-less condition")
emit(pdf, fig)

# ---- Slide 3: Preprocessing ----
fig = new_slide(pdf, "Preprocessing Pipeline", 3, "Density + intensity normalization for domain alignment")
bullets(fig, [0.045, 0.30, 0.36, 0.54], [
    "**0.05 m voxel downsample",
    "  first-point per voxel (NOT averaged)",
    "  keeps real coord + intensity + label",
    "",
    "**Per-cloud intensity z-score",
    "  closes sensor intensity gap",
    "",
    "**50 m chunks (overlap train / tiled val)",
], fs=13)
image(fig, [0.42, 0.12, 0.29, 0.66], f"{RF}/fig4_voxel_downsample.png")
image(fig, [0.70, 0.12, 0.29, 0.66], f"{RF}/fig3b_intensity_domain_gap.png")
emit(pdf, fig)

# ---- Slide 4: Toronto3D results ----
fig = new_slide(pdf, "Toronto3D Results — Key Finding", 4, "Intensity helps exactly one class: Road_markings")
image(fig, [0.035, 0.44, 0.31, 0.40], f"{RF}/fig5a_toronto_coord_curves.png")
image(fig, [0.035, 0.05, 0.31, 0.40], f"{RF}/fig5b_toronto_intensity_curves.png")
image(fig, [0.36, 0.20, 0.42, 0.60], f"{RF}/fig6_toronto_perclass_iou.png")
bullets(fig, [0.80, 0.28, 0.19, 0.52], [
    "**mIoU",
    "  coord: 0.714",
    "  +intensity: 0.784",
    "",
    "**+0.07 mIoU is",
    "  ALL Road_markings",
    "  (0.00 → 0.56 IoU)",
    "",
    "rest ~unchanged",
], fs=11.5)
emit(pdf, fig)

# ---- Slide 5: Toronto -> OS inference ----
fig = new_slide(pdf, "Toronto3D → OS_data Transfer", 5, "Coherent segmentation, but pedestrians have nowhere to go")
image(fig, [0.04, 0.05, 0.36, 0.78], f"{RF}/fig7a_os_l1_toronto_coord.png", "soton_l1 (Toronto coord-only)")
image(fig, [0.40, 0.05, 0.36, 0.78], f"{RF}/fig7b_os_l2_toronto_coord.png", "soton_l2 (Toronto coord-only)")
bullets(fig, [0.78, 0.32, 0.21, 0.48], [
    "**Road / building /",
    "**vegetation / car",
    "  cleanly separated",
    "",
    "**Problem:",
    "  no Pedestrian class",
    "  → people fall into",
    "  Pole / Car",
], fs=12)
emit(pdf, fig)

# ---- Slide 6: NPM3D introduction ----
fig = new_slide(pdf, "Recovering Pedestrians: NPM3D", 6, "Paris-Lille-3D adds a Pedestrian class — but it is rare")
bullets(fig, [0.045, 0.24, 0.40, 0.60], [
    "**NPM3D = Paris-Lille-3D (Velodyne HDL-32 MLS)",
    "  Paris + Lille streets, RGB-less",
    "",
    "**10 coarse → 9 valid classes, incl. Pedestrian",
    "**Same pipeline as Toronto (0.05 m, z-score, 50 m)",
    "**178 train / 13 val scenes (val = Lille2)",
    "",
    "**Pedestrian is only 0.17 % of points",
    "  rare-class challenge (log scale →)",
], fs=13)
image(fig, [0.47, 0.13, 0.51, 0.66], f"{RF}/fig9a_npm3d_classdist.png")
emit(pdf, fig)

# ---- Slide 7: NPM3D results ----
fig = new_slide(pdf, "NPM3D Results", 7, "Pedestrian is purely geometric — coord-only wins it")
image(fig, [0.035, 0.44, 0.30, 0.40], f"{RF}/fig9b_npm3d_coord_curves.png")
image(fig, [0.035, 0.05, 0.30, 0.40], f"{RF}/fig9c_npm3d_intensity_curves.png")
image(fig, [0.35, 0.22, 0.42, 0.58], f"{RF}/fig9d_npm3d_perclass_iou.png")
table(fig, [0.785, 0.52, 0.20, 0.24],
      ["", "coord", "+int"],
      [["mIoU", "0.872", "0.884"],
       ["Pedestrian", "0.916", "0.811"]],
      col_w=[0.46, 0.27, 0.27], fs=11, highlight_row=2)
bullets(fig, [0.79, 0.30, 0.20, 0.16], [
    "intensity helps Pole /",
    "Barrier / Bollard",
    "pedestrian = pure shape",
], fs=11)
emit(pdf, fig)

# ---- Slide 8: OS pedestrian detection + conclusion ----
fig = new_slide(pdf, "OS_data Pedestrian Detection & Summary", 8, "Pedestrians now appear — the Toronto-only model detected none")
image(fig, [0.04, 0.12, 0.40, 0.64], f"{RF}/fig9e_os_pedestrian_counts.png")
image(fig, [0.44, 0.06, 0.30, 0.76], f"{RF}/fig9f_os_l2_npm3d_coord_pedestrian.png")
bullets(fig, [0.755, 0.20, 0.24, 0.60], [
    "**Summary",
    "  Toronto: clean transfer,",
    "  no pedestrian",
    "  NPM3D: pedestrian",
    "  recovered on OS",
    "",
    "**intensity = 1 class only",
    "  (Road_mark / Pole-Barrier)",
    "**pedestrian = geometry",
    "",
    "  no OS GT → counts,",
    "  not accuracy",
], fs=11)
emit(pdf, fig)

pdf.close()
print(">> slides ->", OUT)
