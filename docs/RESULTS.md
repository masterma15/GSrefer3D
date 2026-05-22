# Experimental results (GSrefer3D · data2)

> Last updated: 2026-05-21  
> Raw 2D metrics: [`results_2d_eval.json`](results_2d_eval.json) · Depth ablation: [`depth_compare_batch.json`](depth_compare_batch.json)  
> Pipeline figure: [`../demo/pipeline.png`](../demo/pipeline.png)

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

| Summary (20 groups) | 3DGS | DAV2 raw | DAV2 affine | DAV2 inv |
|---------------------|------|----------|-------------|----------|
| **median NN (m)** | **0.133** | 0.572 | 0.368 | 0.183 |
| Wins vs DAV2 affine | **15 / 20** | — | — | — |

**One-liner:** 20 unprojection groups — 3DGS render depth NN median **0.133 m** vs DAV2 affine **0.368 m** (3DGS better in 15/20).  
Details: [`depth_compare_batch.json`](depth_compare_batch.json).

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

**Summary (10/10 training objects):** LoRA **median L2 ≤ Base** on every object. Largest gains: **umbrella** (−0.056), **golden retriever** (−0.053), **brown rabbit** (−0.031).  
Overlays: `3DGS/test2/runs/<run_id>/overlays_rgb/view_XXX.png` (local; not in Git).

---

## 3. Double-sided tape — excluded from data2 SFT (no GT)

Excluded from data2 SFT (469 samples); same 3DGS scene; qualitative overlay only. Not in `data2_sft` · Prompt: *clear double-sided adhesive tape on the desk* · 72 views.

| Group | run_id (suffix) | 2D GT | support | Notes |
|-------|-----------------|-------|---------|-------|
| Base | `000313_6c883d56` | — | 15 | Check `overlays_rgb/` qualitatively |
| LoRA | `132142_6c883d56` | — | 17 | Same |

No automatic 2D score; report as **qualitative overlay comparison** only.

---

## 4. Out-of-domain · RefSpatial-Expand-Bench

| Task | n | Base | LoRA (data2) | Δ (LoRA−Base) |
|------|---|------|--------------|---------------|
| **Location** | 241 | **50.21%** | 45.64% | −4.57 pp |
| **Placement** | 200 | **48.50%** | 47.00% | −1.50 pp |

Base matches paper reproduction; LoRA reflects domain adaptation trade-off on out-of-domain data. **Do not mix** with in-domain 2D L2.

---

## 5. Copy-paste bullets (resume / README)

**In-domain fine-tuning (2D)**

```text
On 10 training object categories in data2, 2D referring error vs synthetic GT: LoRA median L2 ≤ Base on all objects (e.g. umbrella 0.067→0.011, golden retriever 0.062→0.008, normalized coords); same 72-view render and fusion pipeline.
```

**Depth ablation (one line)**

```text
20 unprojection groups: 3DGS render depth NN median 0.133 m vs DAV2 affine-aligned 0.368 m (15/20 groups favor 3DGS).
```

**Out-of-domain (optional)**

```text
RefSpatial-Expand Location: Base 50.21% (repro), data2 LoRA 45.64% (−4.57 pp, out-of-domain).
```

---

*L2 = normalized image-plane Euclidean distance; counted only for `parse_ok` views present in GT.*
