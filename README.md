# GSrefer3D

**Language-guided 3D spatial referring**: multi-view **3D Gaussian Splatting** rendering → **RGB-D VLM** (2D point) → fused `P_world` as a **mask seed only** → Grounding DINO + SAM2 + frustum vote → **object Gaussians**.

Research integration repo — original code is mainly [`bridge/`](bridge/). Upstream [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [RoboRefer](https://github.com/Zhoues/RoboRefer) are **cloned locally**, not vendored in Git. Layout notes: [`docs/UPSTREAM_SETUP.md`](docs/UPSTREAM_SETUP.md).

## Pipeline

![GSrefer3D end-to-end pipeline](demo/pipeline.png)

**3DGS** (once per scene): multi-view photos → COLMAP → `train.py` → `point_cloud.ply`. Not part of the online command.

**One command** ([`bridge/run_bridge_e2e.py`](bridge/run_bridge_e2e.py)): trained 3DGS model + text prompt → per-object SFT pack (`question.json` + masks) + frustum-voted Gaussians + SIBR iteration. Internally: render → RoboRefer API → fuse **seed** → project / ray filter → DINO+SAM2 → refine → vote → inject. `--skip-render` reuses an existing view pack.

> **Note:** `P_world` / `fused.json` is only a mask seed. The official 3D output is the **voted Gaussian set** (`gaussian_seg.json`). Primary SFT is **`data2_sft_ray` (629, visible-centroid)**; historical 469 mask-centroid pack archived. Fuse CLI default is `depth_mode=ray`. Do not treat seed-point OBB hit tables as the 3D metric.

---

## Quick start

Two Python environments on purpose (cannot share one process): **`envGS`** on Windows for 3DGS + bridge; **`roborefer`** on WSL Ubuntu (or a cloud GPU) for the VLM API and DINO+SAM2.

| Role | Env | Where | Needs |
|------|-----|--------|--------|
| Render / fuse / filter / vote / inject | `envGS` | Windows | CUDA rasterizer, `scipy`, `plyfile`, `requests` |
| RoboRefer HTTP API | `roborefer` | WSL or AutoDL | ~10GB+ VRAM (4060 8GB will OOM; tunnel a 4090) |
| Grounding DINO + SAM2 masks | `roborefer` | WSL with CUDA | SAM2 + GroundingDINO weights |

Laptop 8GB GPU: run **only** `envGS` locally; keep the API on a remote 24GB box and SSH-tunnel port `25547`.

### 0. Clone this repo

```powershell
git clone https://github.com/masterma15/GSrefer3D.git
Set-Location GSrefer3D
```

### 1. `envGS` — 3DGS + bridge (Windows)

```powershell
cd 3DGS
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
conda env create -f environment-envGS.yml
conda activate envGS
```

Do **not** use upstream `gaussian-splatting/environment.yml` (Python 3.7). This project’s file is Python 3.9, PyTorch 2.4 + CUDA 11.8, and includes **scipy** (ray-filter KD-tree).

Compile CUDA extensions from `3DGS/gaussian-splatting/`:

```powershell
Set-Location gaussian-splatting
$env:DISTUTILS_USE_SDK = "1"
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
Set-Location ..
```

`3DGS/render.py` is already in this repo (custom multi-view RGB-D export). Optional accel rasterizer: copy [`patches/3dgs/gaussian_renderer__init__.py`](patches/3dgs/gaussian_renderer__init__.py) over `gaussian-splatting/gaussian_renderer/__init__.py`.

Train a scene once (example `data2`; `-r 4` = 1/4 resolution). Phone photos: strip EXIF first.

```powershell
magick mogrify -strip gaussian-splatting/data2/input/*.jpg
python convert.py -s gaussian-splatting/data2
python train.py -s gaussian-splatting/data2 -r 4
```

Output model: `3DGS/gaussian-splatting/output/data2/`. You can skip COLMAP/train if you already have `point_cloud/iteration_*/point_cloud.ply`.

SIBR viewer (optional): unzip [official binaries](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) into `3DGS/gaussian-splatting/viewers/`. Always launch from `viewers/bin/` or `sibr_system.dll` will not load.

### 2. `roborefer` — VLM API (WSL Ubuntu or Linux cloud)

```bash
# WSL, repo mounted e.g. /mnt/e/GSrefer3D
cd /mnt/e/GSrefer3D
git clone https://github.com/Zhoues/RoboRefer.git RoboRefer-main
cd RoboRefer-main
bash env_setup.sh roborefer
conda activate roborefer
```

Upstream install uses Python 3.10.14, PyTorch 2.5, FlashAttention 2.5.8 (Linux wheel). Then pin:

```bash
pip install huggingface-hub==0.28.1
# do not pip install -U huggingface_hub
```

Apply the four small edits in [`patches/roborefer/INTEGRATION.md`](patches/roborefer/INTEGRATION.md) (`datasets_mixture.py` identifier fix + `data2_location` register, `Trainer.log` signature, optional API default paths, single-GPU `SLURM_JOB_GPUS_PER_NODE=1`).

**Weights** (not in Git; put at repo root):

| File | Path | Source |
|------|------|--------|
| RoboRefer 2B SFT | `RoboRefer-2B-SFT/` | [Zhoues/RoboRefer-2B-SFT](https://huggingface.co/Zhoues/RoboRefer-2B-SFT) |
| Depth Anything V2 ViT-L | `weights/depth_anything_v2_vitl.pth` | [depth-anything/Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth) |
| SAM 2.1 Hiera-L | `weights/sam2.1_hiera_large.pt` | [facebook/sam2.1](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |
| Grounding DINO Swin-T | `weights/groundingdino_swint_ogc.pth` | [IDEA-Research release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| data2 ray LoRA (optional) | `RoboRefer-2B-SFT-data2-ray-merged/` | primary merged LoRA; historical: `…-data2-merged/` |

```bash
# from repo root (WSL or Linux)
mkdir -p weights
huggingface-cli download Zhoues/RoboRefer-2B-SFT --local-dir RoboRefer-2B-SFT
wget -O weights/depth_anything_v2_vitl.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
wget -O weights/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget -O weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 3. DINO + SAM2 inside `roborefer` (WSL)

Mask stage is invoked by the e2e command via `wsl bash` (Windows default). Install these **in WSL `roborefer`**, not in `envGS`.

```bash
conda activate roborefer
cd /mnt/e/GSrefer3D
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install addict timm yapf supervision pycocotools
# PYTHONPATH to GroundingDINO is enough; pip install -e . often fails in this env

git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..
```

### 4. Start RoboRefer API (keep this terminal open)

WSL / Linux:

```bash
conda activate roborefer
cd /mnt/e/GSrefer3D/RoboRefer-main/API
python api.py --port 25547 --host 0.0.0.0 \
  --depth_model_path /mnt/e/GSrefer3D/weights/depth_anything_v2_vitl.pth \
  --vlm_model_path /mnt/e/GSrefer3D/RoboRefer-2B-SFT
```

Cloud GPU (recommended if local VRAM &lt; 16GB), then on Windows:

```powershell
ssh -CNg -L 25547:127.0.0.1:25547 -p <port> root@<host>
```

Leave the tunnel up. The e2e client still uses `--url http://127.0.0.1:25547`.

### 5. Run the pipeline (Windows `envGS`)

`--prompt` is the VLM instruction. `--object` is the short DINO caption (not inferred). `--name` is the output folder slug. `--custom-views-out` is the multi-view pack (default `3DGS/test2`, overridable).

```powershell
conda activate envGS
Set-Location E:\GSrefer3D

# First object: render  + query + fuse seed + mask + vote + SIBR inject
python bridge/run_bridge_e2e.py `
  --model-path 3DGS/gaussian-splatting/output/data2 `
  --custom-views-out 3DGS/test2 `
  --prompt "Please point to the electric shaver on the desk." `
  --object "electric shaver" `
  --name data2_shaver `
  --url http://127.0.0.1:25547

# Same scene, new object: reuse the view pack (skip the expensive render)
python bridge/run_bridge_e2e.py `
  --model-path 3DGS/gaussian-splatting/output/data2 `
  --custom-views-out 3DGS/test2 `
  --prompt "Please point to the brown stuffed rabbit." `
  --object "plush rabbit" `
  --name data2_rabbit `
  --skip-render --url http://127.0.0.1:25547
```

Preflight: API down → exit **10**; WSL / mask weights missing → exit **11**. One bad view is skipped; the command only dies if a whole stage is empty (no query inliers, no masks, zero voted Gaussians).

| Flag | Meaning |
|------|---------|
| `--skip-render` | Same as `--from query`; require `rgb/`, `camera_params/`, `depth_raw/` under `--custom-views-out` |
| `--from fuse\|filter\|mask\|vote` | Resume after a crash |
| `--pause-after-mask` | Write review PNGs and stop; continue with `--from vote` |
| `--out-dir` | Override `training_data/<name>/` |
| `--bbox-key` | Key in `docs/bbox_data2.json`; if found, overlay OBB + run 3D eval |
| `--clean` | After success, delete overlays / review / rejected-view dumps only |
| `--no-use-wsl` | Run DINO+SAM in the current env (Linux with CUDA) |

Does **not** write `training_data/data2_sft/` (merge 10 objects later with `bridge/export_spatial_train.py` if you train).

**Outputs**

| Path | What |
|------|------|
| `training_data/<name>/` | SFT pack: `question.json`, `mask/`, `gaussian_seg.json` + `.ply` |
| `<views>/runs/<run_id>/` | `predictions.json`, `fused.json` (seed), overlays, `run_manifest.json` |
| `<model>/point_cloud/iteration_seg_<name>/` | SIBR recolor (does not touch `iteration_30000`) |

```powershell
Set-Location E:\GSrefer3D\3DGS\gaussian-splatting\viewers\bin
.\SIBR_gaussianViewer_app.exe -m "E:\GSrefer3D\3DGS\gaussian-splatting\output\data2" --iteration seg_data2_shaver
```

Magenta = selected ∩ OBB · lime = selected outside OBB · cyan = hand OBB · blue = seed (optional).

**Eval** (after packs exist):

```powershell
python bridge/eval_2d_vs_gt.py --out docs/results_2d_eval.json
python bridge/eval_gaussian_seg.py --obj-glob training_data/data2_* --output docs/results_gaussian_seg.json
```

---

## Results (experiment timeline)

Full tables and run IDs: **[`docs/RESULTS.md`](docs/RESULTS.md)**.

| Step | What | Key artifacts |
|------|------|----------------|
| **1** | Depth ablation (unproject z₀ only) | [`depth_compare_batch.json`](docs/depth_compare_batch.json) |
| **2** | Manual **OBB** in CloudCompare (11 objects) | [`bbox_data2.json`](docs/bbox_data2.json), [`docs/bbox_labels/`](docs/bbox_labels/) |
| **3** | Fusion → mask seed; **ray-visible** SFT pack (**629**) | `data2_sft_ray` |
| **4** | **LoRA** on `data2_location_ray` | `RoboRefer-2B-SFT-data2-ray-merged` |
| **5** | **Base vs LoRA_ray** — in-domain 2D + hold-out tape | [`results_2d_eval_ray.json`](docs/results_2d_eval_ray.json) |
| **6** | **RefSpatial-Expand-Bench** (OOD) | Location / Placement % |
| **7** | **Frustum-vote 3D instance** vs hand OBB | [`results_gaussian_seg.json`](docs/results_gaussian_seg.json) |
| **8** | **Referring-point ablation** (text-only DINO boxes, same vote) | [`ablation_text_only.json`](docs/ablation_text_only.json) |

---

### 1 · Depth ablation (why we use 3DGS `depth_raw`)

![Depth source ablation — median NN distance to 3DGS point cloud (lower is better)](demo/teaser_depth_ablation.png)

Fixed referring pixel + camera; only the **depth source for z₀** changes → unproject → NN distance to `point_cloud.ply` (**lower is better**). Does **not** include later ray refinement.

| Source | median NN (m) |
|--------|---------------|
| **3DGS `depth_raw`** | **0.133** |
| DAV2 affine | 0.368 |
| DAV2 raw | 0.572 |

**15/20** groups: 3DGS &lt; DAV2 affine. Numbers: [`docs/depth_compare_batch.json`](docs/depth_compare_batch.json).

---

### 2 · Manual 3D OBB anchors (CloudCompare)

On `point_cloud.ply` (data2), each of **11 objects** (10 training + **double-sided tape** hold-out) got an oriented box via **Cross Section → segment → Edit clipping box**. Parameters are in **[`docs/bbox_data2.json`](docs/bbox_data2.json)** (`center`, `width`, `half_extent`, `rotation_columns`). The 10 training boxes are the GT for frustum-vote 3D eval; the tape box is hold-out / qualitative only.

| Field | Meaning |
|-------|---------|
| `objects.<key>.obb` | Primary eval geometry |
| `screenshot` | CloudCompare viewport after fitting |
| `screenshot_obb_dialog` | Edit clipping box dialog (dimensions / rotation) |

**Examples (labeling screenshots in repo):**

| Object | Viewport | Edit dialog |
|--------|----------|-------------|
| Electric shaver | ![shaver OBB in CloudCompare](docs/bbox_labels/electric_shaver_obb_cloudcompare.png) | ![shaver OBB dialog](docs/bbox_labels/electric_shaver_obb_edit_dialog.png) |
| Double-sided tape (hold-out) | ![tape OBB in CloudCompare](docs/bbox_labels/double_sided_tape_obb_cloudcompare.png) | ![tape OBB dialog](docs/bbox_labels/double_sided_tape_obb_edit_dialog.png) |

All 11 objects: PNGs under [`docs/bbox_labels/`](docs/bbox_labels/) (paths listed per object in `bbox_data2.json`).

---

### 3 · Initial fusion → multi-view training data (469)

**Fusion policy at this stage:** raster **`expected_invdepth`** at the click + **snap fused point to nearest Gaussian**. The fused point is a **mask seed**, not the 3D answer.

Pipeline: seed **`P_world`** → [`gen_training_data.py`](bridge/gen_training_data.py) **project** → **ray occlusion filter** → **DINO + SAM2** → **mask-centroid refine** → export **`data2_sft`** (469 RGB-D Location tuples) and later **frustum vote**.

![Synthetic SFT labels — green = 3D projection, red = mask centroid after refine](demo/teaser_train_data.png)

| Panel | Object | `move` (px) |
|-------|--------|-------------|
| Top-left | Golden bowl | 118.5 |
| Top-right | Hair clip | 27.5 |
| Bottom-left | Electric shaver | 75.3 |
| Bottom-right | Umbrella | 197.2 |

---

### 4 · LoRA fine-tuning

**2B LoRA** (1 epoch) on mixture **`data2_location_ray`** (`training_data/data2_sft_ray/`, **629** RGB-D, visible-centroid labels; tape **excluded**) → merged **`RoboRefer-2B-SFT-data2-ray-merged`**. Base = **`RoboRefer-2B-SFT`**. Historical 469 / `data2-merged` kept for archive only.

---

### 5 · Base vs LoRA_ray — in-domain 2D and hold-out (**primary**)

Same **72-view** `test2` pack; prompt = *center of the visible part*; GT = `data2_sft_ray` (not the old 469 mask-centroid pack).

**In-domain (10 objects):** LoRA_ray **median L2 ≤ Base on all 10/10**. Mean of per-object medians **0.052 → 0.012**.

| Object | Base median L2 | LoRA_ray median L2 | Δ |
|--------|----------------|---------------------|---|
| Cookie bag | 0.116 | **0.021** | -0.095 |
| Umbrella | 0.101 | **0.016** | -0.085 |
| Golden retriever | 0.072 | **0.007** | -0.064 |
| Electric shaver | 0.055 | **0.012** | -0.042 |

Full table: [`docs/RESULTS.md`](docs/RESULTS.md) §2 · [`results_2d_eval_ray.json`](docs/results_2d_eval_ray.json).

**Overlays** (3 views × Base | LoRA; green = fuse inliers):

| Object | Figure |
|--------|--------|
| Umbrella | ![Umbrella — Base vs LoRA](demo/teaser_base_lora_umbrella.png) |
| Golden retriever | ![Golden retriever — Base vs LoRA](demo/teaser_base_lora_golden_retriever.png) |
| Brown rabbit | ![Brown rabbit — Base vs LoRA](demo/teaser_base_lora_rabbit.png) |
| Electric shaver | ![Electric shaver — Base vs LoRA](demo/teaser_base_lora_shaver.png) |

**Hold-out — double-sided tape** (not in **629** ray SFT; **not** in the 10-object 2D table). Prompt = visible-center (updated in `question.json`). GT = SAM2 mask centroid (`data2_tape/question.json`, **42** views).

| Group | n | median L2↓ | mean L2 | %&lt;0.05 | Δ median |
|-------|---|------------|---------|----------|----------|
| Base | 42 | 0.0186 | 0.1451 | 64.3% | — |
| LoRA_ray | 42 | **0.0135** | 0.1428 | 64.3% | **-0.0051** |

JSON: [`docs/results_2d_eval_tape_ray.json`](docs/results_2d_eval_tape_ray.json). (Historical mask-centroid LoRA tape: [`results_2d_eval_tape.json`](docs/results_2d_eval_tape.json).) Overlay:

![Tape — Base left, LoRA right](demo/teaser_base_lora_tape.png)

---

### 6 · RefSpatial-Expand-Bench (out-of-domain)

| Task | Base | LoRA (469) | LoRA_ray (629) | Δ ray−Base |
|------|------|------------|----------------|------------|
| **Location** | **50.21%** | 45.64% | **49.38%** | −0.83 pp |
| **Placement** | **48.50%** | 47.00% | 39.50% | −9.00 pp |

Domain-adapted LoRA improves **in-domain data2**; OOD Location ≈ Base for ray LoRA (better than old 469). Placement drops more. Report separately from §5.

---

### 7 · 3D instance (frustum vote vs manual OBB)

`P_world` is only a seed for Grounding DINO + SAM2 masks. The 3D result is the **voted Gaussian set**. Primary metrics: vote-centroid in OBB, fraction of selected Gaussians inside the box, 2D mask reprojection precision. The table below is the earlier fraction vote (`--min-vote-frac 0.5`) on the 10 training objects.

![Frustum-vote 3D instance vs hand OBB](demo/teaser_gaussian_seg.png)

| Metric | Value (10 training objects) |
|--------|-----------------------------|
| Vote centroid ∈ hand OBB | **10 / 10** |
| Mean fraction of selected Gaussians in OBB | **0.993** |
| Mean 2D mask reprojection precision | **0.900** |

Hair-clip seed can sit ~10 cm outside the thin OBB; the **vote centroid is inside**. Per-object table: [`docs/RESULTS.md`](docs/RESULTS.md) §2b · [`results_gaussian_seg.json`](docs/results_gaussian_seg.json).

The default vote is now a two-pass view-angle weighted score. Rendered views are chosen by farthest-point sampling on directions from the captured cameras to their mean position, so the views that enter an object's vote still lie on an arc. Each view gets a mass that shrinks when many views share its direction. A pair's weight is those two masses times the angle between the view rays. A hit inside the image adds that weight; a miss inside the image subtracts it; a projection outside the frame or within 8 pixels of the border does not vote. The first pass keeps Gaussians with score > 0.03 and paints that set white. The second pass renders a footprint (only the white set is opaque) and a stain (every Gaussian stays opaque; the white set is white and the rest black). A pixel with a high footprint and a low stain is treated as occluded and does not vote. A low footprint outside the mask still subtracts. The final keep is score > 0.15. Visible weight for both renders is 0.5. The data2 comparison below, and the exported data3 packs, used an earlier vote at this same final keep, before the soft mask edge.

On data2 the two votes were scored on the same masks and hand OBBs, 11 objects (the 10 training objects plus the tape hold-out). Both put the centroid inside the hand OBB on **11 / 11**. Mean in-box fraction stays **0.992**. Mean 2D mask reprojection precision goes from **0.788** to **0.804**. Umbrella goes from 0.655 to 0.778 (11402 → 19565 Gaussians, in-box fraction 1.000 → 0.998). The medicine bottle goes from 0.733 to 0.846. The cookie bag goes from 0.670 to 0.609, and its in-box fraction from 0.998 to 0.980.

![data2 mask precision, fraction 0.5 vs view-angle](demo/teaser_vote_compare_data2.png)

Per-object numbers: [`docs/results_vote_compare_data2.json`](docs/results_vote_compare_data2.json).

```powershell
python bridge/eval_gaussian_seg.py --obj-glob training_data/data2_* --output docs/results_gaussian_seg.json
python tools/plot_gaussian_seg_teaser.py
python tools/plot_vote_compare.py
python bridge/inject_gaussian_seg.py --obj-glob training_data/data2_*
```

SIBR: magenta = selected ∩ OBB · lime = selected \ OBB · cyan = hand OBB · optional blue = seed.

**SIBR orbit** (voted Gaussians vs hand OBB — these are the 3D result, not the old fused-point / seed-marker GIFs). GitHub will not reliably show large GIFs inside Markdown tables, so they are linked as images below. Files: [`demo/shaver.gif`](demo/shaver.gif) · [`rabbit.gif`](demo/rabbit.gif) · [`golden_retriever.gif`](demo/golden_retriever.gif) · [`umbrella.gif`](demo/umbrella.gif) · [`cake.gif`](demo/cake.gif) · [`hair_clip.gif`](demo/hair_clip.gif) · [`double_sided_tape.gif`](demo/double_sided_tape.gif).

**Electric shaver** · **Brown rabbit**

<p>
<img src="demo/shaver.gif" alt="electric shaver SIBR orbit" width="49%" />
<img src="demo/rabbit.gif" alt="brown rabbit SIBR orbit" width="49%" />
</p>

**Golden retriever** · **Umbrella**

<p>
<img src="demo/golden_retriever.gif" alt="golden retriever SIBR orbit" width="49%" />
<img src="demo/umbrella.gif" alt="umbrella SIBR orbit" width="49%" />
</p>

**Toy cake** · **Hair clip**

<p>
<img src="demo/cake.gif" alt="toy cake SIBR orbit" width="49%" />
<img src="demo/hair_clip.gif" alt="hair clip SIBR orbit" width="49%" />
</p>

**Hold-out** (not in 469 SFT / not in the 10-object 3D summary):

<p>
<img src="demo/double_sided_tape.gif" alt="double-sided tape SIBR orbit" width="49%" />
</p>

---

### 8 · Referring-point ablation (same vote, text-only boxes)

The 2D referring point is removed. The object name still goes to Grounding DINO. Kept views, `min-vote-frac 0.5`, `T(z)` weights, and the depth slice stay the same as the baseline masks in `training_data/data2_*`. Text-only masks live under `training_data/ablation_text_only/` and do not replace those baseline packs.

The comparison is the **count** of voted Gaussians inside the hand OBB, and the count outside it. The in-box fraction is not the score: it can rise while the in-box count falls (hair clip 0.975 → 0.993, in-box count 3738 → 3074).

![Referring-point ablation — in-box and out-of-box Gaussian counts](demo/teaser_text_only_ablation.png)

Counts below are `round(n_selected × frac_selected_in_obb)` from [`docs/ablation_text_only.json`](docs/ablation_text_only.json).

| Object | In-box with point | In-box text only | Δ in-box | Out-of-box with point | Out-of-box text only |
|--------|------------------:|-----------------:|---------:|----------------------:|---------------------:|
| Golden bowl | 13670 | 13670 | 0 | 233 | 233 |
| Bracelet | 3722 | 3478 | −244 | 0 | 0 |
| Cookie bag | 2961 | 2920 | −41 | 40 | 25 |
| Golden retriever | 25925 | 25523 | −402 | 30 | 13 |
| Hair clip | 3738 | 3074 | −664 | 95 | 22 |
| Medicine bottle | 9541 | **0** | −9541 | 0 | 0 |
| Brown rabbit | 103120 | 90722 | −12398 | 464 | 22484 |
| Electric shaver | 6901 | 6862 | −39 | 57 | 44 |
| Toy cake | 6451 | 9199 | +2748 | 0 | 93113 |
| Umbrella | 19303 | 18105 | −1198 | 8 | 4 |
| Tape (hold-out) | 5043 | **0** | −5043 | 220 | 13703 |

The point changes the voted set. Medicine bottle text-only selects **no** Gaussians (the JSON centroid hit on that row is the empty-set fallback to `P_world`, not an object centroid). Tape’s voted set is entirely outside the box (0 in-box, 13703 out-of-box) and the centroid leaves the OBB. Toy cake’s centroid also leaves the OBB: in-box count rises by 2748 while out-of-box count rises from 0 to 93113. Rabbit loses 12398 in-box Gaussians and gains 22020 outside the box. Hair clip, umbrella, golden retriever, and bracelet lose hundreds to about a thousand in-box Gaussians even though their in-box fraction stays near 1. Bowl is unchanged; cookie bag and the shaver move by only a few dozen.

```powershell
python tools/plot_text_only_ablation.py
```

---

## Repository layout (what is in Git)

| Path | In Git? | Role |
|------|---------|------|
| [`bridge/`](bridge/) | **Yes** | 2D→3D, fuse, e2e, eval, training export |
| [`docs/bbox_data2.json`](docs/bbox_data2.json) | **Yes** | Manual OBB parameters (11 objects) |
| [`docs/bbox_labels/*.png`](docs/bbox_labels/) | **Yes** | CloudCompare labeling screenshots |
| [`docs/results_*.json`](docs/) | **Yes** | 2D L2 / frustum-vote 3D / depth ablation |
| [`docs/ablation_text_only.json`](docs/ablation_text_only.json) | **Yes** | Referring-point ablation counts |
| [`demo/`](demo/) | **Partial** | Pipeline, 2D teasers, frustum-vote and referring-point charts, SIBR orbit GIFs (see `.gitignore`) |
| `3DGS/gaussian-splatting/`, `RoboRefer-main/` | **No** | Clone locally — [Quick start](#quick-start) |
| `training_data/`, `3DGS/test2/runs/` | **No** | Local experiments |

## What we changed upstream (short)

| Upstream | Shipped here |
|----------|----------------|
| 3DGS | `3DGS/render.py`, `patches/3dgs/` |
| RoboRefer | `patches/roborefer/INTEGRATION.md` |

## License

- **MIT** — `bridge/`, public `docs/` and `demo/` assets listed in `.gitignore`, `patches/`, `3DGS/render.py` ([LICENSE](LICENSE)).
- **Upstream** — see [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

Cite upstream 3DGS and RoboRefer. This repo is a student research workspace, not an official release of either project.

## Public data files

| File | Use |
|------|-----|
| [docs/UPSTREAM_SETUP.md](docs/UPSTREAM_SETUP.md) | Clone / weights layout (details in Quick start) |
| [docs/RESULTS.md](docs/RESULTS.md) | Full tables (chronological) |
| [docs/bbox_data2.json](docs/bbox_data2.json) | Manual OBB (11 objects) |
| [docs/bbox_labels/](docs/bbox_labels/) | CloudCompare labeling PNGs |
| [docs/depth_compare_batch.json](docs/depth_compare_batch.json) | Step 1 depth ablation |
| [docs/results_2d_eval_ray.json](docs/results_2d_eval_ray.json) | Step 5 primary in-domain 2D (ray GT) |
| [docs/results_2d_eval_tape_ray.json](docs/results_2d_eval_tape_ray.json) | Hold-out tape 2D (ray-era) |
| [docs/results_2d_eval.json](docs/results_2d_eval.json) | Historical in-domain 2D (469) |
| [docs/results_2d_eval_tape.json](docs/results_2d_eval_tape.json) | Historical hold-out tape |
| [docs/results_gaussian_seg.json](docs/results_gaussian_seg.json) | Step 7 frustum-vote 3D instance |
| [docs/ablation_text_only.json](docs/ablation_text_only.json) | Step 8 referring-point ablation |
