"""Share of labelled points per shared-5 class in each source (class balance table)."""
import os, glob, numpy as np
ROOT = os.environ.setdefault("OS_SEG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCES = {"Toronto3D": "data/uni_toronto", "Paris-Lille-3D": "data/uni_npm3d", "WHU-Urban3D": "data/whu3d"}
SHARED = ["Ground", "Building", "Natural", "Pole", "Car"]
for name, rel in SOURCES.items():
    cnt = np.zeros(11, np.int64)
    for f in glob.glob(os.path.join(ROOT, rel, "*", "*", "segment.npy")):
        s = np.load(f).astype(np.int64); s = s[(s >= 0) & (s < 11)]
        cnt += np.bincount(s, minlength=11)
    share = 100 * cnt / max(cnt.sum(), 1)
    print(f"{name:15s} " + "  ".join(f"{c} {share[i]:.2f}%" for i, c in enumerate(SHARED)))
