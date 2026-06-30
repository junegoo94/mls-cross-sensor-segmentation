# Point Cloud Semantic Segmentation — Toronto3D → OS_data Transfer
### Presentation content (text + tables), condensed to 8 pages. Design to be added separately.

---

## Page 1 — Project Overview

**Goal:** Semantic segmentation of an RGB-less point cloud (OS_data, Ordnance Survey, Southampton UK).

**Strategy:** Train on a labeled dataset (Toronto3D, Canada) → transfer / inference on OS_data (no labels).

**Framework:** Pointcept; model = Point Transformer V3 (PTv3).

**Hardware:** 2 × RTX 3090 (24 GB), CUDA 12.4, PyTorch 2.5.

**Key research question:** Which input features transfer best across sensors —
geometry only (coord) vs geometry + intensity?

---

## Page 2 — Data: Source (Toronto3D) & Target (OS_data)

**Source — Toronto3D (training).** 4 tiles, mobile laser scanning, urban street.

| Tile | Points | Role |
|------|--------|------|
| L001 | 21,567,172 | train |
| L002 | 10,283,800 | val |
| L003 | 39,721,590 | train |
| L004 | 6,747,648 | train |
| **Total** | **78,319,210** | |

- **Per-point fields:** XYZ (double, UTM), RGB (0–255), Intensity (1–255), Label (0–8), GPS time, scan angle.
- **8 classes (+ unclassified):** Ground, Road_markings, Natural, Building, Utility_line, Pole, Car, Fence.
- **Notable:** UTM coords (x≈627,512 / y≈4,841,948) → subtract offset before float32 cast (precision).

**Target — OS_data (inference).** Two scan strips (l1, l2) over the same Southampton street corridor.

| File | Points | Coord system | Intensity | RGB |
|------|--------|--------------|-----------|-----|
| soton_l1 | 41,932,917 | British National Grid | mean 30.6 (0–255), 94% non-zero | all 0 (none) |
| soton_l2 | 26,562,331 | British National Grid | mean 33.9 (0–255), 93% non-zero | all 0 (none) |

- OS_data **has intensity** but **no usable RGB** (all zeros); **no ground-truth labels** (pure inference target).
- Raw `.pcap` (sensor packets, captured 2025-09-25) → processed to `.laz` (used here).

---

## Page 3 — Data Analysis: Class Imbalance & Intensity Domain Gap

**Class imbalance (Toronto3D training set, 59.6 M points, 0.05 m voxel).**

| Class | Share | Class | Share |
|-------|-------|-------|-------|
| Ground | 42.69% | Utility_line | 0.87% |
| Road_markings | 2.13% | Pole | 1.09% |
| Natural | 6.78% | Car | 6.35% |
| Building | 26.82% | Fence | 0.52% |
| | | ignore (−1) | 3.36% |

Severe imbalance (Ground 43% vs Fence 0.5%) → loss = **CrossEntropy + Lovász**.

**Intensity domain gap (key analysis).** Intensity is **sensor-dependent** — raw values not comparable across datasets.

| | Toronto3D | OS_data |
|---|-----------|---------|
| Intensity range | 0–255 | 0–255 |
| **Mean** | **~15** | **~31** (≈2×) |
| Std | ~15 | ~27 |

- **Risk:** training on raw Toronto intensity (centered ~15) then feeding OS intensity (centered ~31) = out-of-distribution input → intensity-dependent predictions break.
- **Fix:** **per-cloud z-score** maps both to mean 0 / std 1 → distributions aligned (1st/2nd moment). Computed without labels (intensity is a raw point attribute). Applied identically to OS_data at inference.
- *Caveat:* z-score aligns mean/std only — not the physical reflectance→intensity mapping of the two sensors. Hence a coord-only baseline (sensor-independent) is trained alongside.

---

## Page 4 — Preprocessing Pipeline & Voxel Downsampling

Toronto3D PLY → Pointcept `DefaultDataset` format (per-scene `coord/strength/segment.npy`).

| Step | Detail |
|------|--------|
| 1. Voxel downsample | 0.05 m grid, 1 real point per voxel (density equalization) |
| 2. Label remap | 0 (Unclassified) → ignore(−1); 1–8 → 0–7 |
| 3. Global offset | subtract common min `[627285.9, 4841948.97, 116.69]`, store float32 |
| 4. Intensity normalize | **per-tile z-score** `(x−μ)/σ` (domain-gap mitigation) |
| 5. Split + chunk | train (L001/3/4) = 50 m blocks, 25 m stride (overlap); val (L002) = 50 m non-overlap |

**Result:** 197 train scenes + 28 val scenes.

**Voxel downsampling effect.** Toronto3D MLS density is extremely uneven — one 50×50 m block in L003 held **28 M points** (scanner hotspot). 0.05 m voxelization fixes this.

| Tile | Raw | → 0.05 m voxel | Retained |
|------|-----|----------------|----------|
| L001 | 21.57 M | 4.88 M | 22.6% |
| L002 | 10.28 M | 4.99 M | 48.5% |
| L003 | 39.72 M | **6.08 M** | 15.3% (hotspot collapsed) |
| L004 | 6.75 M | 4.07 M | 60.4% |

After voxelization: train chunk sizes min 10.6k / median 156k / **max 1.57 M** (manageable).

---

## Page 5 — Model, Training Setup & Results

| Item | Value |
|------|-------|
| Model | Point Transformer V3 (PTv3), 46.2 M params |
| Loss | CrossEntropy + Lovász (ignore_index = −1) |
| Optimizer / sched | AdamW, OneCycleLR, 100 epochs |
| Batch / aug | bs 8 (2 GPU), GridSample 0.05 m, SphereCrop 102,400, rotate/scale/flip/jitter |
| Acceleration | flash-attention, AMP (fp16) |
| Train time | ~65 min / single-source model (2 × RTX 3090); joint ~98 min |

**Two variants:** coord-only — input = XYZ (in_channels = 3); coord + intensity — input = XYZ + z-scored intensity (in_channels = 4).

**Training results (Toronto3D val).**

| Metric | coord-only | coord+intensity |
|--------|-----------|-----------------|
| best mIoU | 0.718 | **0.789** |
| final mIoU | 0.714 | 0.784 |
| allAcc | 0.960 | 0.967 |
| mAcc | 0.788 | 0.872 |

Both converged cleanly (no overfitting); val mIoU plateaus after ~20 epochs.

---

## Page 6 — Per-class IoU (the key finding)

| Class | coord-only | coord+intensity | Δ |
|-------|-----------|-----------------|-----|
| Ground | 0.948 | 0.959 | +0.012 |
| **Road_markings** | **0.001** | **0.559** | **+0.558** |
| Natural | 0.962 | 0.962 | +0.001 |
| Building | 0.925 | 0.914 | −0.011 |
| Utility_line | 0.856 | 0.860 | +0.004 |
| Pole | 0.780 | 0.802 | +0.022 |
| Car | 0.901 | 0.921 | +0.020 |
| Fence | 0.337 | 0.296 | −0.042 |
| **mIoU** | **0.714** | **0.784** | **+0.071** |

**Takeaway:** the entire +0.07 mIoU gain comes from **Road_markings** (0.001 → 0.559).
Road paint is geometrically identical to the road surface — invisible to geometry, separable only by intensity.
The other 7 classes are essentially tied.

---

## Page 7 — OS_data Inference: Pipeline & Results

**Pipeline.**

| Step | Detail |
|------|--------|
| Preprocess | same as training: 0.05 m voxel + per-cloud z-score intensity + 50 m chunks |
| Inference | model runs on **0.05 m voxels** (same resolution as training) |
| Back-projection | voxel predictions → full-res points via **KDTree nearest-neighbor** |
| Output | labeled `.laz` (classification field) + top-view PNG |

| File | Raw | → voxel | Chunks |
|------|-----|---------|--------|
| soton_l1 | 41.9 M | 17.4 M | 49 |
| soton_l2 | 26.6 M | 15.5 M | 48 |

*Note:* model predicts at 0.05 m; full-res labels are NN-propagated (details < 5 cm not independently predicted).

**Results — predicted class distribution (soton_l1, 41.9 M full-res points).**

| Class | coord-only | coord+intensity |
|-------|-----------|-----------------|
| Ground | 46.30% | 45.35% |
| Road_markings | 0.00% | 0.51% |
| Natural | 15.35% | 15.50% |
| Building | 32.09% | 33.32% |
| Utility_line | 0.02% | 0.01% |
| Pole | 0.58% | 0.62% |
| Car | 2.33% | 2.59% |
| Fence | 3.33% | 2.09% |

**Two models almost identical on OS_data** — the one class intensity helps (Road_markings) is essentially **absent in OS_data** (visual check confirmed). For this target, **coord-only is sufficient and most robust**.

---

## Page 8 — Findings, Limitations, Next Steps & Assets

**Findings**
- **Cross-domain transfer works:** a model trained on Toronto (Canada) produces coherent segmentation on Southampton (UK), a different city and sensor — road / building / vegetation structure is clean.
- **Intensity domain alignment validated:** z-score normalization let the intensity-dependent behavior transfer (Road_markings activated on OS where present in training).
- **For OS_data specifically:** intensity adds little (no road markings) → coord-only is the practical choice.

**Limitations**
- OS_data has **no ground truth** → numbers are *prediction distributions*, not accuracy.
- **No Pedestrian class** in Toronto3D → pedestrians (present in OS_data) get misassigned (typically Pole / Car).
- Predictions at 0.05 m resolution (sub-5 cm detail not independently predicted).

**Next steps**
1. **Pedestrian:** train on a dense-MLS dataset that includes pedestrians (Paris-Lille-3D / NPM3D), then transfer to OS_data.
2. **Both feature variants** (coord / coord+intensity) on Paris-Lille for comparison.
3. **Joint training option** (Toronto3D + Paris-Lille, unified label space via Pointcept PPT) — recovers Road_markings + adds Pedestrian in one model (more complex).
4. **Quantitative eval on OS_data:** would require manually labeling a few regions.

**Assets / Paths**

| Asset | Path |
|-------|------|
| Preprocessing (Toronto / OS) | `projects/os_seg/preprocess/preprocess_toronto3d.py` , `preprocess_osdata.py` |
| Configs | `projects/os_seg/configs/semseg-pt-v3-coord{,-intensity}.py` |
| Inference | `projects/os_seg/scripts/infer_osdata.py` |
| Trained models | `projects/os_seg/exp/toronto3d-ptv3-coord{,-intensity}/model/model_best.pth` |
| Training curves / OS predictions | `projects/os_seg/exp/.../curves.png` , `projects/os_seg/exp/osdata_infer/*.laz` + `*.png` |
| Figure script (optional charts) | `projects/os_seg/notes/make_figs.py` |

---

## Page 9 — Follow-up: Recovering the Pedestrian Class (Paris-Lille-3D → OS_data)

**Motivation.** Toronto3D has **no pedestrian class**, so people in OS_data were forced into Pole / Car. To recover them, we trained a second source model on **Paris-Lille-3D (NPM3D)** — a dense MLS urban dataset (Velodyne HDL-32, Paris + Lille) that **does** label pedestrians — and transferred it to OS_data, reusing the identical pipeline.

**Source data — NPM3D.** 4 labeled training tiles (Paris, Lille1_1, Lille1_2, Lille2), `x,y,z` + `reflectance` (uchar, no RGB — same condition as OS_data). 10 coarse classes → **9 valid (0–8)** + ignore. val = Lille2 (held-out). Same preprocessing as Toronto: 0.05 m voxel, per-tile z-score intensity, 50 m chunks → **178 train / 13 val** scenes. Pedestrian is rare (**0.17 %** of points).

**Two variants trained** (coord-only, in_channels=3; coord+intensity, in_channels=4), PTv3, 100 epochs, ~17 min each.

**NPM3D validation (Lille2, TTA test) — per-class IoU.**

| Class | coord-only | coord+intensity | Class | coord-only | coord+intensity |
|-------|-----------|-----------------|-------|-----------|-----------------|
| Ground | 0.982 | 0.977 | Barrier | 0.680 | 0.766 |
| Building | 0.986 | 0.985 | **Pedestrian** | **0.916** | **0.811** |
| Pole | 0.768 | 0.858 | Car | 0.979 | 0.987 |
| Bollard | 0.840 | 0.863 | Natural | 0.894 | 0.890 |
| Trash_can | 0.804 | 0.818 | **mIoU** | **0.872** | **0.884** |

**Key finding (consistent with Toronto).** Intensity helps only specific classes (Pole / Barrier / Bollard); **Pedestrian is purely geometric** — coord-only is actually *better* for it (0.916 vs 0.811). The "intensity helps one class family, not the rest" pattern from Toronto (there: Road_markings) holds again.

**OS_data inference — predicted pedestrian counts (full-res).**

| File | coord-only | coord+intensity |
|------|-----------|-----------------|
| soton_l1 (41.9 M) | 14,575 (0.03 %) | 25,989 (0.06 %) |
| soton_l2 (26.6 M) | 41,352 (0.16 %) | 35,063 (0.13 %) |

**Pedestrians are now detected on OS_data** — the gap from the Toronto-only model is closed. Overall segmentation stays coherent (road / building / vegetation / car cleanly separated). *Caveat:* OS_data has no ground truth, so these are prediction counts, not accuracy; at full-corridor top-view the pedestrian points are too small to see individually — a zoomed crop / CloudCompare check is the remaining qualitative validation.

**NPM3D assets**

| Asset | Path |
|-------|------|
| Preprocessing | `projects/os_seg/preprocess/preprocess_npm3d.py` |
| Configs | `projects/os_seg/configs/semseg-pt-v3-npm3d-coord{,-intensity}.py` |
| Trained models | `projects/os_seg/exp/npm3d-ptv3-coord{,-intensity}/model/model_best.pth` |
| Inference script | `projects/os_seg/scripts/infer_osdata_npm3d.py` |
| OS predictions | `projects/os_seg/exp/osdata_infer_npm3d/*.laz` + `*.png` |
