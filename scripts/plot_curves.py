"""Pointcept train.log 파싱 → train loss / val mIoU 곡선 PNG 저장."""
import re
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jmgoo/projects/os_seg/exp/toronto3d-ptv3-coord"
log = os.path.join(exp, "train.log")
out = os.path.join(exp, "curves.png")
name = os.path.basename(exp)

train_loss = []   # (epoch, iter_global, loss)
epoch_loss = {}   # epoch -> [losses]
val = []          # (epoch, mIoU, mAcc, allAcc)

cur_epoch = 0
for line in open(log):
    m = re.search(r"Train: \[(\d+)/\d+\]\[(\d+)/(\d+)\].*loss: ([\d.]+)", line)
    if m:
        ep, it, nit, loss = int(m[1]), int(m[2]), int(m[3]), float(m[4])
        cur_epoch = ep
        gi = (ep - 1) * nit + it
        train_loss.append((gi, loss))
        epoch_loss.setdefault(ep, []).append(loss)
        continue
    m = re.search(r"Val result: mIoU/macroF1/mAcc/mPrecision/allAcc "
                  r"(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)", line)
    if m:
        val.append((cur_epoch, float(m[1]), float(m[3]), float(m[5])))

print(f"parsed: {len(train_loss)} train iters, {len(val)} val evals")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1) train loss
ax = axes[0]
if train_loss:
    gi, ls = zip(*train_loss)
    ax.plot(gi, ls, color="#bbb", lw=0.6, label="per-iter")
    eps = sorted(epoch_loss)
    nit = max(re.findall(r"/(\d+)\]", "") or [1])  # unused
    em = [np.mean(epoch_loss[e]) for e in eps]
    # epoch 평균을 global iter 위치에 대략 매핑
    ax.plot([e * len(epoch_loss[e]) for e in eps], em,
            color="C0", lw=2, marker="o", ms=3, label="per-epoch mean")
ax.set_xlabel("iteration"); ax.set_ylabel("train loss")
ax.set_title(f"{name}\nTrain loss"); ax.grid(alpha=0.3); ax.legend()

# 2) val metrics
ax = axes[1]
if val:
    ve, vmiou, vmacc, vacc = zip(*val)
    ax.plot(ve, vmiou, "o-", color="C1", label="mIoU")
    ax.plot(ve, vmacc, "s-", color="C2", label="mAcc")
    ax.plot(ve, vacc, "^-", color="C3", label="allAcc")
    best = max(range(len(vmiou)), key=lambda i: vmiou[i])
    ax.scatter([ve[best]], [vmiou[best]], color="red", zorder=5, s=60)
    ax.annotate(f"best mIoU={vmiou[best]:.3f}\n@ep{ve[best]}",
                (ve[best], vmiou[best]), textcoords="offset points",
                xytext=(8, -4), fontsize=9, color="red")
ax.set_xlabel("epoch"); ax.set_ylabel("metric")
ax.set_title("Validation"); ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(out, dpi=110)
print("saved:", out)
if val:
    print(f"final val: mIoU={val[-1][1]:.4f} mAcc={val[-1][2]:.4f} allAcc={val[-1][3]:.4f}")
    bi = max(range(len(val)), key=lambda i: val[i][1])
    print(f"best val: mIoU={val[bi][1]:.4f} @ epoch {val[bi][0]}")
