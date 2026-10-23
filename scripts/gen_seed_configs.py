"""Copy the LOSO Recipe B and self-training configs with a pinned seed."""
import os, sys
seeds = [int(s) for s in sys.argv[1:]] or [1, 2]
os.makedirs("configs/seeds", exist_ok=True)
F = {1: "toronto", 2: "npm3d", 3: "whu"}
for s in seeds:
    for f, X in F.items():
        src = open(f"configs/semseg-pt-v3-loso{f}b-AB-coord.py").read()
        open(f"configs/seeds/loso{f}b-AB-coord-s{s}.py", "w").write(src + f"\n# seed replicate\nseed = {s}\n")
        src = open(f"configs/semseg-pt-v3-uda-{X}-coord.py").read()
        old = f'data_root = "/home/jmgoo/projects/os_seg/data/uda_{X}"'
        assert old in src, old
        src = src.replace(old, f'data_root = "/home/jmgoo/projects/os_seg/data/seeds/uda_{X}_s{s}"')
        open(f"configs/seeds/uda-{X}-coord-s{s}.py", "w").write(src + f"\n# seed replicate\nseed = {s}\n")
print("seed configs written for", seeds)
