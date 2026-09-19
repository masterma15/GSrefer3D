# Experimental results (GSrefer3D · data2)

> Last updated: 2026-09-19  
> Raw 2D: [`results_2d_eval.json`](results_2d_eval.json) · **3D instance:** [`results_gaussian_seg.json`](results_gaussian_seg.json)  
> Chart: [`../demo/teaser_gaussian_seg.png`](../demo/teaser_gaussian_seg.png) · pipeline: [`../demo/pipeline.png`](../demo/pipeline.png)

---

## Chronology (same order as [README](../README.md))

| Step | What | Where in this doc |
|------|------|-------------------|
| 1 | Depth ablation (z₀ source) | §1 |
| 2 | Manual OBB in CloudCompare | [`bbox_data2.json`](bbox_data2.json), [`bbox_labels/`](bbox_labels/) |
| 3 | Initial fusion (invdepth + snap) → mask seed + 469 SFT pack | README §3; train-data teaser |
| 4 | LoRA on `data2_location` | Training run (not tabulated here) |
| 5 | Base vs LoRA — in-domain 2D + hold-out tape | §2, §3 |
| 6 | RefSpatial-Expand-Bench (OOD) | §4 |
| 7 | Frustum-vote 3D instance vs hand OBB | §2b |

**Seed vs result:** **`fused.json` / `P_world`** = mask seed only. **`gaussian_seg.json`** = reported 3D instance. Do not use historical seed-point OBB tables as the 3D metric.

---

## How to read these tables

| Metric | Primary for README / interview? | Notes |
|--------|----------------------------------|-------|
| **2D median L2** (vs SFT GT) | **Yes** (VLM) | Normalized `(nx, ny)` Euclidean distance; lower is better |
| **centroid ∈ OBB** / **frac in OBB** | **Yes** (3D instance) | Vote centroid and selected-Gaussian precision vs hand OBB |
| **2D mask reprojection precision** | Secondary | Selected μ fall in that view’s SAM mask |
| **Expand Location %** | Out-of-domain | Report separately from in-domain data2 |
| **Fuse `P_world` ∈ OBB** | No (seed only) | Intermediate; not a reported 3D score |

**Reproduce 2D table:**

```bash
python bridge/eval_2d_vs_gt.py --out docs/results_2d_eval.json
```

GT: `training_data/data2_sft/location_point.json` (SFT views only, **n ≤ 72** per object).

---

## 1. Depth source ablation (D1)

Fixed referring pixel + camera; only the depth source changes → unproject `P_world` → **NN distance (m)** to `point_cloud.ply`.

| Depth source (median over 20 groups) | 3DGS | DAV2 raw | DAV2 affine | DAV2 inv |
|--------------------------------------|------|----------|-------------|----------|
| **NN distance to scene (m)** ↓ | **0.133** | 0.572 | 0.368 | 0.183 |

**Per-group win rate** (same 20 groups; lower NN wins):

| Comparison | Result |
|------------|--------|
| 3DGS vs DAV2 affine | **15 / 20** groups favor 3DGS |

**One-liner:** 20 unprojection groups — 3DGS render depth NN median **0.133 m** vs DAV2 affine **0.368 m** (3DGS better in 15/20).  
Details: [`depth_compare_batch.json`](depth_compare_batch.json) · figure: [`../demo/teaser_depth_ablation.png`](../demo/teaser_depth_ablation.png).

---

## 2. In-domain data2 · Base vs LoRA (2D vs GT) — main table

72-view render pack `3DGS/test2/` · Base = `RoboRefer-2B-SFT` API · LoRA = `RoboRefer-2B-SFT-data2-merged` API · Same fuse / unproject pipeline.

| Object | Group | run_id (suffix) | n | median L2↓ | mean L2 | %&lt;0.05 | support | Δ median (LoRA−Base) |
|--------|-------|-----------------|---|------------|---------|----------|---------|----------------------|
| Electric shaver | Base | `170540_4c3b9a32` | 42 | 0.0273 | 0.0483 | 85.7% | 24 | — |
| Electric shaver | LoRA | `143457_4c3b9a32` | 42 | **0.0085** | 0.0095 | 100% | 20 | **−0.0188** |
| Brown rabbit | Base | `171359_147bac82` | 60 | 0.0373 | 0.0656 | 68.3% | 42 | — |
| Brown rabbit | LoRA | `144845_147bac82` | 60 | **0.0065** | 0.0075 | 100% | 52 | **−0.0308** |
| Golden retriever | Base | `172627_f8dbfcc3` | 63 | 0.0615 | 0.0818 | 44.4% | 48 | — |
| Golden retriever | LoRA | `154013_f8dbfcc3` | 63 | **0.0083** | 0.0137 | 98.4% | 57 | **−0.0532** |
| Golden bowl | Base | `173958_8d83a715` | 53 | 0.0089 | 0.0087 | 100% | 36 | — |
| Golden bowl | LoRA | `154649_8d83a715` | 53 | **0.0029** | 0.0039 | 100% | 36 | **−0.0060** |
| Umbrella | Base | `174835_d7bab60f` | 52 | 0.0670 | 0.1546 | 34.6% | 11 | — |
| Umbrella | LoRA | `160219_d7bab60f` | 52 | **0.0106** | 0.0327 | 92.3% | 32 | **−0.0564** |
| Toy cake | Base | `175733_7dd80c38` | 50 | 0.0221 | 0.0651 | 72.0% | 28 | — |
| Toy cake | LoRA | `161222_7dd80c38` | 50 | **0.0142** | 0.0322 | 80.0% | 40 | **−0.0079** |
| Cookie bag | Base | `180800_e51c780a` | 39 | 0.0372 | 0.1433 | 59.0% | 17 | — |
| Cookie bag | LoRA | `162019_e51c780a` | 39 | **0.0147** | 0.0400 | 89.7% | 19 | **−0.0225** |
| Medicine bottle | Base | `182106_04149b86` | 39 | 0.0247 | 0.0635 | 92.3% | 33 | — |
| Medicine bottle | LoRA | `163004_04149b86` | 39 | **0.0065** | 0.0305 | 94.9% | 35 | **−0.0182** |
| Bracelet | Base | `001018_cb2e562f` | 36 | 0.0232 | 0.0712 | 80.6% | 22 | — |
| Bracelet | LoRA | `163546_cb2e562f` | 36 | **0.0182** | 0.0632 | 94.4% | 21 | **−0.0050** |
| Hair clip | Base | `183039_65bf02a5` | 35 | 0.0218 | 0.0575 | 80.0% | 22 | — |
| Hair clip | LoRA | `164312_65bf02a5` | 35 | **0.0104** | 0.0265 | 94.3% | 25 | **−0.0114** |

*L2 = normalized image-plane Euclidean distance; counted only for `parse_ok` views present in GT.*

**Summary (10/10 training objects):** LoRA **median L2 ≤ Base** on every object. Largest gains: **umbrella** (−0.056), **golden retriever** (−0.053), **brown rabbit** (−0.031).  
Overlays: `3DGS/test2/runs/<run_id>/overlays_rgb/view_XXX.png` (local; not in Git).

---

## 2b. 3D instance — frustum vote vs hand OBB

`P_world` is a mask seed only. GT = hand OBB in [`bbox_data2.json`](bbox_data2.json). Metrics from `gaussian_seg.json` (`--min-vote-frac 0.5`).

| Object | n selected | centroid ∈ OBB | frac in OBB | mask2d prec |
|--------|----------:|:--------------:|------------:|------------:|
| golden_bowl | 13903 | ✓ | 0.983 | 0.952 |
| bracelet | 3722 | ✓ | 1.000 | 0.916 |
| cookie_bag | 3001 | ✓ | 0.987 | 0.834 |
| golden_retriever | 25955 | ✓ | 0.999 | 0.938 |
| hair_clip | 3833 | ✓ | 0.975 | 0.871 |
| medicine_bottle | 9541 | ✓ | 1.000 | 0.940 |
| brown_rabbit | 103584 | ✓ | 0.996 | 0.931 |
| electric_shaver | 6958 | ✓ | 0.992 | 0.914 |
| toy_cake | 6451 | ✓ | 1.000 | 0.868 |
| umbrella | 19311 | ✓ | 1.000 | 0.836 |

**Summary (10/10):** centroid hit **100%** · mean frac in OBB **0.993** · mean view-mask precision **0.900**.  
JSON: [`results_gaussian_seg.json`](results_gaussian_seg.json) · figure: [`../demo/teaser_gaussian_seg.png`](../demo/teaser_gaussian_seg.png). Hair-clip fuse seed can sit ~10 cm outside the thin OBB; the vote centroid is inside.

SIBR orbit (voted Gaussians, official 3D viz): [`../demo/shaver.gif`](../demo/shaver.gif) · [`rabbit.gif`](../demo/rabbit.gif) · [`golden_retriever.gif`](../demo/golden_retriever.gif) · [`umbrella.gif`](../demo/umbrella.gif) · [`cake.gif`](../demo/cake.gif) · [`hair_clip.gif`](../demo/hair_clip.gif). Hold-out tape: [`../demo/double_sided_tape.gif`](../demo/double_sided_tape.gif). Do not use `teaser_3d_*.gif` (old seed-marker orbits).

Reproduce:

```powershell
python bridge/eval_gaussian_seg.py --obj-glob training_data/data2_* --output docs/results_gaussian_seg.json
```

SIBR: `python bridge/inject_gaussian_seg.py --obj-glob training_data/data2_*`

---

## 3. Double-sided tape — hold-out (not in 469 SFT)

Same 3DGS scene; **not** exported to `data2_sft`. Prompt: *clear double-sided adhesive tape on the desk*. Mask pack: `training_data/data2_tape/` (46 kept views). Seed for masks: LoRA `20260519_132142_6c883d56`.

**2D vs mask centroid** (same L2 as §2; **do not mix** into the 10-object table). JSON: [`results_2d_eval_tape.json`](results_2d_eval_tape.json).

| Group | run_id (suffix) | n | median L2↓ | mean L2 | %&lt;0.05 | support | Δ median (LoRA−Base) |
|-------|-----------------|---|------------|---------|----------|---------|----------------------|
| Base | `000313_6c883d56` | 46 | 0.0214 | 0.1666 | 56.5% | 15 | — |
| LoRA | `132142_6c883d56` | 46 | **0.0132** | 0.1528 | 63.0% | 17 | **−0.0082** |

Mean ≫ median: a few views are far off; report median as the primary 2D number.

```powershell
python bridge/eval_2d_vs_gt.py --tape-only --out docs/results_2d_eval_tape.json
```

Frustum vote vs hand OBB (hold-out, not in the 10-object 3D summary): centroid ∈ OBB **yes** · frac in OBB **0.958** · view-mask precision **0.821**. SIBR: `--iteration seg_double_sided_tape`. Orbit: [`../demo/double_sided_tape.gif`](../demo/double_sided_tape.gif). Overlay teaser: [`../demo/teaser_base_lora_tape.png`](../demo/teaser_base_lora_tape.png).

---

## 4. Out-of-domain · RefSpatial-Expand-Bench

| Task | n | Base | LoRA (data2) | Δ (LoRA−Base) |
|------|---|------|--------------|---------------|
| **Location** | 241 | **50.21%** | 45.64% | −4.57 pp |
| **Placement** | 200 | **48.50%** | 47.00% | −1.50 pp |

Base matches paper reproduction; LoRA reflects domain adaptation trade-off on out-of-domain data. **Do not mix** with in-domain 2D L2.

---
