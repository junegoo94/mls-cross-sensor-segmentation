# Paper Plan — Label-free Cross-Sensor Domain Adaptation for MLS Point-Cloud Semantic Segmentation

## 1. Story / Contribution
**Title (draft):** *Cross-Sensor Generalization of MLS Point-Cloud Semantic Segmentation: How LiDAR Scan Pattern Drives Domain Shift and Cheap Label-Free Fixes.*

**Claim:** Outdoor MLS semseg models overfit the *source LiDAR's scan-pattern fingerprint* (beam count, vertical FOV, density gradient). We (i) quantify the cross-sensor degradation across **3 commercial sensors**, and (ii) show two **label-free** fixes — **(A) multi-sensor source training** and **(B) density/scan-pattern augmentation** — recover most of the gap, demonstrated on a real **national-survey (Ordnance Survey) Hesai** cloud.

**Why publishable (applied/remote-sensing angle):**
- Real heterogeneous commercial MLS sensors (Optech Maverick / Velodyne HDL-32E / Hesai PandarXT-32) incl. survey-grade OS data.
- Label-free adaptation (practical for industry — no target annotation).
- Systematic cross-sensor benchmark + ablations (the quantitative backbone).

**Honest novelty:** components (multi-dataset training ≈ Pointcept PPT; density aug ≈ LaserMix/PolarMix family) are known → frame as a **rigorous empirical cross-sensor study + real-world survey deployment**, not a new architecture.

## 2. Datasets (sensor-diverse)
| Dataset | Sensor | Beams | Type | Labels | Status |
|---|---|---|---|---|---|
| Toronto3D | Optech Maverick | 32-line | MLS | yes (8 cls) | ✅ have |
| NPM3D / Paris-Lille-3D | Velodyne HDL-32E | 32 | MLS | yes train (test hidden) | ✅ have |
| **OS_data (Southampton)** | **Hesai PandarXT-32** | 32 | MLS survey | **none** | ✅ have (real-world demo / target) |
| SemanticPOSS | Hesai Pandora | 40-line | driving | yes | ⬇ to add (Hesai-family → labeled proxy for OS) |
| SemanticKITTI | Velodyne HDL-64E | 64 | driving | yes | ⬇ optional (extra source diversity) |
| nuScenes-lidarseg | Velodyne 32 | 32 | driving | yes | ⬇ optional |

Common coarse label space (intersection across sets): **Ground, Building, Vegetation/Natural, Pole, Car, (Barrier/Fence), Pedestrian**. Map each dataset → this taxonomy.

## 3. Protocols
**P1 — Pairwise cross-sensor (RUNNABLE NOW, no new data).**
Train on one MLS sensor, test on the other's labeled val, shared classes.
- Toronto(Maverick) → NPM3D(Velodyne) val  and  NPM3D → Toronto val.
- Conditions: NoAdapt (source-only) vs **+B** (density aug).
- Gives the first quantitative cross-sensor numbers + B's effect. (A/joint needs a 3rd sensor to hold one out.)

**P2 — Leave-one-sensor-out (needs ≥3 labeled sensors).**
Train on N−1 sensors, test on held-out sensor's labeled val. Held-out = **SemanticPOSS (Hesai)** → labeled proxy for the OS target.
- Conditions: NoAdapt / +A (multi-source) / +B / **+A+B**.
- This is the core table that isolates A, B, A+B.

**P3 — Real-world deployment (qualitative + optional small GT).**
- OS_data: the road→Building fix (already shown), colored maps, pedestrian recovery.
- Optional: hand-label ~2–3 OS chunks → one real target mIoU point (strengthens, not required for P1/P2).

## 4. Methods compared
- **Source-only (NoAdapt)** — lower bound.
- **+B** — density/scan augmentation (RandomDropout; + ring-drop, LaserMix variants as ablation).
- **+A** — multi-sensor joint training (unified labels).
- **+A+B** — both (our full).
- **≥1 existing UDA baseline** (e.g., a feature-alignment or self-training method) for context.
- **Oracle (target-supervised)** — upper bound (where labels exist).

## 5. Metrics & rigor
- Per-class IoU + mIoU on shared classes; **cross-sensor gap** Δ = in-domain − cross-sensor.
- ≥3 seeds, report mean±std.
- Ablations: A alone / B alone / A+B; dropout ratio sweep; #source sensors (1→2→3); coord vs coord+intensity.
- Sensor-fingerprint analysis fig (beam/FOV/density of the 3 sensors) as the motivating figure.

## 6. Paper structure
1 Intro (sensor heterogeneity problem) · 2 Related (3D semseg, LiDAR UDA, multi-dataset) · 3 Sensor fingerprint analysis · 4 Method (A, B, unified labels) · 5 Cross-sensor benchmark (P1, P2 tables + ablations) · 6 Real-world OS deployment (P3) · 7 Conclusion.

## 7. Target venues
- Journal: **ISPRS J. P&RS**, **IEEE TGRS**, **Remote Sensing (MDPI)**, IEEE J-STARS.
- Faster: CVPR/ICCV workshop (large-scale 3D / autonomous driving) or ISPRS workshop.

## 8. Immediate next step (runnable now)
**P1 batch** with current Toronto + NPM3D data, shared label space:
1. Build shared-label datasets (remap both to common coarse classes).
2. Train: Toronto-only, Toronto-only+B, NPM3D-only, NPM3D-only+B (coord; ~37 min each).
3. Cross-evaluate on the other sensor's val → first cross-sensor mIoU table + B effect.
→ This produces the paper's first quantitative result with **zero new data / zero labeling**.
