# Experimental results (GSrefer3D · data2)

> Last updated: 2026-10-28  
> **Primary 2D (ray-visible GT):** [`results_2d_eval_ray.json`](results_2d_eval_ray.json) · tape: [`results_2d_eval_tape_ray.json`](results_2d_eval_tape_ray.json)  
> Historical 2D (mask-centroid SFT): [`results_2d_eval.json`](results_2d_eval.json) · [`results_2d_eval_tape.json`](results_2d_eval_tape.json)  
> **3D instance:** [`results_gaussian_seg.json`](results_gaussian_seg.json) · **referring-point ablation:** [`ablation_text_only.json`](ablation_text_only.json)  
> Chart: [`../demo/teaser_gaussian_seg.png`](../demo/teaser_gaussian_seg.png) · [`../demo/teaser_text_only_ablation.png`](../demo/teaser_text_only_ablation.png) · pipeline: [`../demo/pipeline.png`](../demo/pipeline.png)

---

## Chronology (same order as [README](../README.md))

| Step | What | Where in this doc |
|------|------|-------------------|
| 1 | Depth ablation (z₀ source) | §1 |
| 2 | Manual OBB in CloudCompare | [`bbox_data2.json`](bbox_data2.json), [`bbox_labels/`](bbox_labels/) |
| 3 | Initial fusion → mask seed; later **ray-visible** SFT pack (629) | README; `data2_sft_ray` |
| 4 | LoRA on `data2_location_ray` | `RoboRefer-2B-SFT-data2-ray-merged` |
| 5 | Base vs LoRA_ray — in-domain 2D + hold-out tape | **§2 (primary)**, §3 |
| 5′ | Historical Base vs LoRA on mask-centroid 469 | §2-hist |
| 6 | RefSpatial-Expand-Bench (OOD; Base / old LoRA / **ray LoRA**) | §4 |
| 7 | Frustum-vote 3D instance vs hand OBB | §2b |
| 8 | Referring-point ablation (text-only boxes, same vote) | §5 |

**Seed vs result:** **`fused.json` / `P_world`** = mask seed only. **`gaussian_seg.json`** = reported 3D instance. Do not use historical seed-point OBB tables as the 3D metric.

**Do not mix** primary §2 (`data2_sft_ray` GT + visible-center prompts) with §2-hist (`data2_sft` GT).

---

## How to read these tables

| Metric | Primary for README / interview? | Notes |
|--------|----------------------------------|-------|
| **2D median L2** (vs ray-visible GT) | **Yes** (VLM) | Normalized `(nx, ny)` Euclidean; lower is better |
| **centroid ∈ OBB** / **frac in OBB** | **Yes** (3D instance) | Vote centroid and selected-Gaussian precision vs hand OBB |
| **2D mask reprojection precision** | Secondary | Selected μ fall in that view’s SAM mask |
| **Expand Location %** | Out-of-domain | Report separately from in-domain data2; §4 includes ray LoRA |
| **Fuse `P_world` ∈ OBB** | No (seed only) | Intermediate; not a reported 3D score |

**Reproduce primary 2D table:**

```bash
python bridge/eval_2d_vs_gt.py --table-json 3DGS/test2/runs/table_ray2d_base_vs_lora.json --out docs/results_2d_eval_ray.json
```

Default GT: `training_data/data2_sft_ray/location_point.json`. Tape GT: `training_data/data2_tape/question.json` (hold-out; prompt = visible-center; **not** in SFT).

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

## 2. In-domain data2 · Base vs LoRA_ray (2D vs ray-visible GT) — **primary**

72-view `3DGS/test2/` · Base = `RoboRefer-2B-SFT` · LoRA = `RoboRefer-2B-SFT-data2-ray-merged` · 
Prompt: *center of the visible part* · GT: `data2_sft_ray` (629; visible-centroid labels). Runs: `ray2d_base_*` / `ray2d_lora_ray_*`.

| Object | Group | run_id | n | median L2↓ | mean L2 | %&lt;0.05 | Δ median (LoRA−Base) |
|--------|-------|--------|---|------------|---------|----------|----------------------|
| Electric shaver | Base | `ray2d_base_shaver` | 64 | 0.0547 | 0.1660 | 46.9% | — |
| Electric shaver | LoRA_ray | `ray2d_lora_ray_shaver` | 64 | **0.0123** | 0.0595 | 81.2% | **-0.0424** |
| Brown rabbit | Base | `ray2d_base_rabbit` | 72 | 0.0466 | 0.1001 | 51.4% | — |
| Brown rabbit | LoRA_ray | `ray2d_lora_ray_rabbit` | 72 | **0.0074** | 0.0094 | 100.0% | **-0.0392** |
| Golden retriever | Base | `ray2d_base_golden_retriever` | 72 | 0.0716 | 0.0865 | 34.7% | — |
| Golden retriever | LoRA_ray | `ray2d_lora_ray_golden_retriever` | 72 | **0.0073** | 0.0094 | 98.6% | **-0.0643** |
| Golden bowl | Base | `ray2d_base_bowl` | 62 | 0.0117 | 0.0755 | 82.3% | — |
| Golden bowl | LoRA_ray | `ray2d_lora_ray_bowl` | 62 | **0.0031** | 0.0430 | 88.7% | **-0.0086** |
| Umbrella | Base | `ray2d_base_umbrella` | 64 | 0.1012 | 0.2327 | 29.7% | — |
| Umbrella | LoRA_ray | `ray2d_lora_ray_umbrella` | 64 | **0.0160** | 0.0919 | 81.2% | **-0.0852** |
| Toy cake | Base | `ray2d_base_toy_cake` | 62 | 0.0215 | 0.1063 | 61.3% | — |
| Toy cake | LoRA_ray | `ray2d_lora_ray_toy_cake` | 62 | **0.0145** | 0.0322 | 82.3% | **-0.0070** |
| Cookie bag | Base | `ray2d_base_cookie` | 58 | 0.1161 | 0.2712 | 36.2% | — |
| Cookie bag | LoRA_ray | `ray2d_lora_ray_cookie` | 58 | **0.0207** | 0.1032 | 70.7% | **-0.0954** |
| Medicine bottle | Base | `ray2d_base_medicine_bottle` | 62 | 0.0139 | 0.1205 | 77.4% | — |
| Medicine bottle | LoRA_ray | `ray2d_lora_ray_medicine_bottle` | 62 | **0.0088** | 0.0643 | 88.7% | **-0.0051** |
| Bracelet | Base | `ray2d_base_bracelet` | 52 | 0.0292 | 0.1521 | 65.4% | — |
| Bracelet | LoRA_ray | `ray2d_lora_ray_bracelet` | 52 | **0.0160** | 0.0929 | 76.9% | **-0.0132** |
| Hair clip | Base | `ray2d_base_hair_clip` | 61 | 0.0492 | 0.1721 | 50.8% | — |
| Hair clip | LoRA_ray | `ray2d_lora_ray_hair_clip` | 61 | **0.0132** | 0.1013 | 70.5% | **-0.0360** |

**Summary (10/10 training objects):** LoRA_ray **median L2 ≤ Base** on every object. Mean of per-object medians: **0.0516 → 0.0119**.

---

## 3. Double-sided tape — hold-out (not in 629 ray SFT) — **primary**

Same scene; **not** in `data2_sft_ray`. Prompt (updated): *center of the visible part of the roll of clear double-sided adhesive tape*. GT = mask centroid in `data2_tape/question.json` (**42** views). Predictions already used the visible-center prompt (`ray2d_*_tape`).

| Group | run_id | n | median L2↓ | mean L2 | %&lt;0.05 | Δ median |
|-------|--------|---|------------|---------|----------|----------|
| Base | `ray2d_base_tape` | 42 | 0.0186 | 0.1451 | 64.3% | — |
| LoRA_ray | `ray2d_lora_ray_tape` | 42 | **0.0135** | 0.1428 | 64.3% | **-0.0051** |

Mean ≫ median: outliers; report **median**. Hold-out Δ is smaller than the historical mask-centroid LoRA table; train/eval prompts and in-domain GT are now aligned on visible-center — prefer that design over chasing hold-out Δ.

JSON: [`results_2d_eval_tape_ray.json`](results_2d_eval_tape_ray.json) (also rows in [`results_2d_eval_ray.json`](results_2d_eval_ray.json)).

Frustum vote vs hand OBB (unchanged 3D hold-out): see historical §3 notes / `results_gaussian_seg` tape entry.

---

## 2-hist. In-domain data2 · Base vs LoRA (mask-centroid 469) — historical

> Superseded for interview/README primary numbers by **§2**. GT was `data2_sft` (469). Do not mix with ray table.


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

## 3-hist. Double-sided tape — hold-out (mask-centroid LoRA era) — historical

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

Success = predicted point falls inside the GT mask. Outputs: `RoboRefer-main/Evaluation/outputs/`.

| Task | n | Base | LoRA (469 / old) | LoRA_ray (629) | Δ ray−Base |
|------|---|------|------------------|----------------|------------|
| **Location** | 241 | **50.21%** | 45.64% | **49.38%** | −0.83 pp |
| **Placement** | 200 | **48.50%** | 47.00% | 39.50% | −9.00 pp |

- Base / old LoRA: historical runs (`RoboRefer-2B-SFT-Depth`, `RoboRefer-2B-data2-LoRA-Depth`).
- LoRA_ray: `RoboRefer-2B-data2-ray-LoRA-Depth` (API = `RoboRefer-2B-SFT-data2-ray-merged`, 2026-10-28).
- Location: ray LoRA ≈ Base（明显好于旧 469 LoRA 的 −4.57 pp）。Placement 掉得更多。**Do not mix** with in-domain 2D L2.

---

## 5. Referring-point ablation (text-only boxes, same vote)

Object-name text still goes to Grounding DINO. The 2D referring point is not used to drop distant boxes or to prompt SAM. Kept views, `min-vote-frac 0.5`, `T(z)` weights, and the depth slice match the baseline packs in `training_data/data2_*`. Text-only outputs are under `training_data/ablation_text_only/` and do not replace those packs.

Score the voted set by **in-box and out-of-box counts**, not by the in-box fraction. Hair clip’s fraction rises from 0.975 to 0.993 while the in-box count falls from 3738 to 3074. Counts are `round(n_selected × frac_selected_in_obb)` from [`ablation_text_only.json`](ablation_text_only.json).

| Object | In-box with point | In-box text only | Δ in-box | Out-of-box with point | Out-of-box text only |
|--------|------------------:|-----------------:|---------:|----------------------:|---------------------:|
| golden_bowl | 13670 | 13670 | 0 | 233 | 233 |
| bracelet | 3722 | 3478 | −244 | 0 | 0 |
| cookie_bag | 2961 | 2920 | −41 | 40 | 25 |
| golden_retriever | 25925 | 25523 | −402 | 30 | 13 |
| hair_clip | 3738 | 3074 | −664 | 95 | 22 |
| medicine_bottle | 9541 | 0 | −9541 | 0 | 0 |
| brown_rabbit | 103120 | 90722 | −12398 | 464 | 22484 |
| electric_shaver | 6901 | 6862 | −39 | 57 | 44 |
| toy_cake | 6451 | 9199 | +2748 | 0 | 93113 |
| umbrella | 19303 | 18105 | −1198 | 8 | 4 |
| double_sided_tape (hold-out) | 5043 | 0 | −5043 | 220 | 13703 |

Medicine bottle text-only selects 0 Gaussians. Its recorded `centroid_obb_hit: true` is the empty-set fallback to `P_world`, not an object centroid. Tape’s 13703 selected Gaussians are all outside the OBB. Toy cake’s centroid leaves the OBB. Figure: [`../demo/teaser_text_only_ablation.png`](../demo/teaser_text_only_ablation.png).

```powershell
python tools/plot_text_only_ablation.py
```

---
