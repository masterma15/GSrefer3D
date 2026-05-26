# Experimental results (GSrefer3D · data2)

> Last updated: 2026-05-26  
> Raw 2D metrics: [`results_2d_eval.json`](results_2d_eval.json) · Depth ablation: [`depth_compare_batch.json`](depth_compare_batch.json) · 3D OBB: [`results_3d_obb_hit.json`](results_3d_obb_hit.json)  
> Pipeline figure: [`../demo/pipeline.png`](../demo/pipeline.png)

---

## Chronology (same order as [README](../README.md))

| Step | What | Where in this doc |
|------|------|-------------------|
| 1 | Depth ablation (z₀ source) | §1 |
| 2 | Manual OBB in CloudCompare | [`bbox_data2.json`](bbox_data2.json), [`bbox_labels/`](bbox_labels/) |
| 3 | Initial fusion (invdepth + snap) → 469 SFT pack | README §3; train-data teaser |
| 4 | LoRA on `data2_location` | Training run (not tabulated here) |
| 5 | Base vs LoRA — in-domain 2D + hold-out tape | §2, §3 |
| 6 | RefSpatial-Expand-Bench (OOD) | §4 |
| 7 | Ray fusion → `fused_ray.json` + OBB hit vs manual box | §2b |

**Fusion naming:** **`fused.json`** = historical e2e (invdepth + snap). **`fused_ray.json`** = same predictions, ray depth pull-in (step 7 only).

---

## How to read these tables

| Metric | Primary for README / interview? | Notes |
|--------|----------------------------------|-------|
| **2D median L2** (vs SFT GT) | **Yes** | Normalized `(nx, ny)` Euclidean distance; lower is better |
| **% L2 &lt; 0.05** | Secondary | Rough “near GT” rate |
| **support** (fuse inlier count) | Secondary | Multi-view geometry consistency; **not** the same as 2D accuracy |
| **Expand Location %** | Out-of-domain | Report separately from in-domain data2 |
| **Depth NN median** | One-line ablation | Shows why unprojection uses 3DGS `depth_raw` |

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

## 2b. 3D OBB hit rate (manual CloudCompare OBB)

11 objects · GT = hand OBB in [`bbox_data2.json`](bbox_data2.json) · **Base** `fused.json` · **LoRA** `fused_ray.json` (ray depth re-fuse).

| Object | Base hit | LoRA hit | Base outside (m) | LoRA outside (m) |
|--------|:--------:|:--------:|------------------:|------------------:|
| Electric shaver | ✓ | ✓ | 0.000 | 0.000 |
| Brown rabbit | ✓ | ✓ | 0.000 | 0.000 |
| Golden retriever | ✓ | ✓ | 0.000 | 0.000 |
| Umbrella | ✓ | ✓ | 0.000 | 0.000 |
| Toy cake | ✓ | ✓ | 0.000 | 0.000 |
| Medicine bottle | ✓ | ✓ | 0.000 | 0.000 |
| Bracelet | ✓ | ✓ | 0.000 | 0.000 |
| Cookie bag | ✗ | ✓ | 0.006 | 0.000 |
| Golden bowl | ✗ | ✓ | 0.126 | 0.000 |
| Double-sided tape | ✗ | ✓ | 0.056 | 0.000 |
| Hair clip | ✗ | ✗ | 0.086 | 0.104 |

**Hit rate:** Base **63.6%** (7/11) · LoRA **90.9%** (10/11).  
Reproduce: `python bridge/eval_3d_obb_offset.py --refuse-lora-ray` · SIBR: `python bridge/inject_obb_compare.py --all-presets`  
JSON: [`results_3d_obb_hit.json`](results_3d_obb_hit.json) · [`results_3d_obb_offset.json`](results_3d_obb_offset.json)

---

## 3. Double-sided tape — excluded from data2 SFT (no GT)

Excluded from data2 SFT (469 samples); same 3DGS scene; qualitative overlay only. Not in `data2_sft` · Prompt: *clear double-sided adhesive tape on the desk* · 72 views.

| Group | run_id (suffix) | 2D GT | support | Notes |
|-------|-----------------|-------|---------|-------|
| Base | `000313_6c883d56` | — | 15 | Check `overlays_rgb/` qualitatively |
| LoRA | `132142_6c883d56` | — | 17 | Same |

No automatic 2D score; report as **qualitative overlay comparison** only. Teaser: [`../demo/teaser_base_lora_tape.png`](../demo/teaser_base_lora_tape.png).

---

## 4. Out-of-domain · RefSpatial-Expand-Bench

| Task | n | Base | LoRA (data2) | Δ (LoRA−Base) |
|------|---|------|--------------|---------------|
| **Location** | 241 | **50.21%** | 45.64% | −4.57 pp |
| **Placement** | 200 | **48.50%** | 47.00% | −1.50 pp |

Base matches paper reproduction; LoRA reflects domain adaptation trade-off on out-of-domain data. **Do not mix** with in-domain 2D L2.

---
