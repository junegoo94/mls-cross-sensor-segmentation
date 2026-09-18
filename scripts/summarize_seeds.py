"""Mean +/- std over training seeds for the LOSO Recipe-B (DG) and self-training (UDA)
held-out shared-5 mIoU, and for the two headline gains (+0.079 DG over avg-single,
+0.024 UDA over DG). Seed 0 = the published single-seed run (loso_eval_ab.log /
Table 5); seeds 1,2 come from exp/seeds/results.jsonl written by run_seeds.sh."""
import json, numpy as np, sys, collections
RES = "/home/jmgoo/projects/os_seg/exp/seeds/results.jsonl"
AVG_SINGLE = {"Toronto": 0.658, "NPM3D": 0.714, "WHU": 0.619}           # realistic baseline (Table 5)
PUB = {"dg": {"Toronto": 0.706, "NPM3D": 0.836, "WHU": 0.688},          # published seed-0 (Table 5)
       "uda": {"Toronto": 0.725, "NPM3D": 0.869, "WHU": 0.707}}
runs = collections.defaultdict(dict)   # runs[(kind, seed)][fold] = miou
for line in open(RES):
    r = json.loads(line); t = r["tag"]
    if t.startswith("sanity"): continue
    kind, fold, s = t.split("_"); runs[(kind, int(s[1:]))][fold] = r["miou"]
for kind in ("dg", "uda"): runs[(kind, 0)] = dict(PUB[kind])
def complete(kind): return sorted(s for (k, s) in runs if k == kind and len(runs[(k, s)]) == 3)
seeds = sorted(set(complete("dg")) & set(complete("uda")))
print(f"complete seeds (DG and UDA on all three folds): {seeds}")
if not seeds: sys.exit(0)
folds = ["Toronto", "NPM3D", "WHU"]; avg = np.mean(list(AVG_SINGLE.values()))
rows = []
for s in seeds:
    dg = np.array([runs[("dg", s)][f] for f in folds]); ud = np.array([runs[("uda", s)][f] for f in folds])
    rows.append((s, dg, ud))
    print(f"seed {s}: DG {dg.round(3).tolist()} mean {dg.mean():.3f} | UDA {ud.round(3).tolist()} mean {ud.mean():.3f} "
          f"| gains: DG-avg {dg.mean()-avg:+.3f}  UDA-DG {ud.mean()-dg.mean():+.3f}  DG/UDA {dg.mean()/ud.mean():.3f}")
DG = np.array([r[1] for r in rows]); UD = np.array([r[2] for r in rows])
def ms(x): return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"
print("\n=== per held-out sensor (mean +/- std over seeds, ddof=1) ===")
for i, f in enumerate(folds): print(f"  {f:8s} DG {ms(DG[:, i])}   UDA {ms(UD[:, i])}")
print("=== headline ===")
print(f"  DG mean       {ms(DG.mean(1))}")
print(f"  UDA mean      {ms(UD.mean(1))}")
print(f"  DG - avg      {ms(DG.mean(1) - avg)}")
print(f"  UDA - DG      {ms(UD.mean(1) - DG.mean(1))}")
print(f"  DG / UDA      {ms(DG.mean(1) / UD.mean(1))}")
json.dump({"seeds": seeds, "dg": DG.tolist(), "uda": UD.tolist(), "avg_single_mean": avg},
          open("/home/jmgoo/projects/os_seg/exp/seeds/summary.json", "w"), indent=1)
