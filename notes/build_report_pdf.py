"""report_en.md → figure 삽입 → HTML → PDF (xhtml2pdf).
모든 그래프/학습곡선/OS예측 top-view 를 본문 해당 페이지 끝에 캡션과 함께 삽입."""
import os
import re
import markdown
from xhtml2pdf import pisa

BASE = "/home/jmgoo/projects/os_seg"
RF = f"{BASE}/notes/report_figures"
MD = f"{BASE}/notes/report_en.md"
OUT_PDF = f"{BASE}/notes/report_en.pdf"

# 페이지 번호 -> [(이미지경로, 캡션), ...]  (각 ## Page N 섹션 끝에 삽입)
# 이미지는 모두 notes/report_figures/ 에 모아둔 파일을 참조.
FIGS = {
    3: [(f"{RF}/fig3a_toronto_classdist.png", "Fig 3a. Toronto3D training-set class distribution (severe imbalance)."),
        (f"{RF}/fig3b_intensity_domain_gap.png", "Fig 3b. Intensity domain gap between Toronto3D and OS_data → per-cloud z-score.")],
    4: [(f"{RF}/fig4_voxel_downsample.png", "Fig 4. 0.05 m voxel downsampling equalizes the uneven MLS density.")],
    5: [(f"{RF}/fig5a_toronto_coord_curves.png", "Fig 5a. Toronto3D coord-only training curves (loss / val mIoU)."),
        (f"{RF}/fig5b_toronto_intensity_curves.png", "Fig 5b. Toronto3D coord+intensity training curves.")],
    6: [(f"{RF}/fig6_toronto_perclass_iou.png", "Fig 6. Toronto3D val per-class IoU — the entire intensity gain is Road_markings.")],
    7: [(f"{RF}/fig7a_os_l1_toronto_coord.png", "Fig 7a. OS_data soton_l1 prediction (Toronto coord-only), top view."),
        (f"{RF}/fig7b_os_l2_toronto_coord.png", "Fig 7b. OS_data soton_l2 prediction (Toronto coord-only), top view.")],
    9: [(f"{RF}/fig9a_npm3d_classdist.png", "Fig 9a. NPM3D training-set class distribution (log scale; Pedestrian only 0.17%)."),
        (f"{RF}/fig9b_npm3d_coord_curves.png", "Fig 9b. NPM3D coord-only training curves."),
        (f"{RF}/fig9c_npm3d_intensity_curves.png", "Fig 9c. NPM3D coord+intensity training curves."),
        (f"{RF}/fig9d_npm3d_perclass_iou.png", "Fig 9d. NPM3D val per-class IoU — Pedestrian is geometric, coord-only wins."),
        (f"{RF}/fig9e_os_pedestrian_counts.png", "Fig 9e. OS_data predicted pedestrian counts (Toronto model detected none)."),
        (f"{RF}/fig9f_os_l2_npm3d_coord_pedestrian.png", "Fig 9f. OS_data soton_l2 prediction (NPM3D coord-only) — class 6 = Pedestrian (red).")],
}


def fig_md(items):
    out = []
    for path, cap in items:
        if not os.path.exists(path):
            print("  WARNING missing:", path)
            continue
        out.append(f'\n\n<div class="fig"><img src="{path}"/>'
                   f'<div class="cap">{cap}</div></div>\n')
    return "".join(out)


def inject(md_text):
    # "## Page N — ..." 기준으로 분할, 각 섹션 끝에 해당 figure 삽입
    parts = re.split(r'(?=^## Page )', md_text, flags=re.M)
    rebuilt = []
    for seg in parts:
        m = re.match(r'## Page (\d+)', seg)
        if m:
            n = int(m.group(1))
            if n in FIGS:
                seg = seg.rstrip() + "\n" + fig_md(FIGS[n]) + "\n"
        rebuilt.append(seg)
    return "".join(rebuilt)


CSS = """
@page { size: A4 portrait; margin: 1.6cm 1.4cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.5pt; color: #1a1a1a; line-height: 1.4; }
h1 { font-size: 17pt; color: #14304a; }
h2 { font-size: 13pt; color: #14304a; border-top: 2px solid #14304a; padding-top: 6px;
     margin-top: 14px; page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 10.5pt; color: #555; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8.6pt; }
th, td { border: 0.6px solid #999; padding: 3px 5px; text-align: left; }
th { background: #e8eef4; }
strong { color: #b32d00; }
.fig { margin: 10px 0; text-align: center; }
.fig img { max-width: 100%; }
.cap { font-size: 8pt; color: #555; font-style: italic; margin-top: 2px; }
hr { border: none; }
code { background: #f0f0f0; padding: 0 2px; font-size: 8.5pt; }
"""


def link_callback(uri, rel):
    # 절대경로 그대로 사용
    if os.path.isabs(uri) and os.path.exists(uri):
        return uri
    return uri


def main():
    md_text = open(MD).read()
    md_text = inject(md_text)
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    # 디버그용 HTML 저장
    open(f"{BASE}/notes/report_en.html", "w").write(html)
    with open(OUT_PDF, "wb") as f:
        status = pisa.CreatePDF(html, dest=f, link_callback=link_callback)
    print("PDF errors:" if status.err else "PDF OK ->", OUT_PDF)
    return status.err


if __name__ == "__main__":
    raise SystemExit(main())
