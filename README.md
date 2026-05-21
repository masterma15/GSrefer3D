# 3DGS-VLM

**Language-guided 3D spatial referring**: multi-view **3D Gaussian Splatting** rendering → **RGB-D VLM** (2D point) → geometry fusion → world-space **3D anchor**.

Research integration repo — original code is mainly [`bridge/`](bridge/). Upstream [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [RoboRefer](https://github.com/Zhoues/RoboRefer) are **cloned locally**, not vendored in full. See [docs/UPSTREAM_SETUP.md](docs/UPSTREAM_SETUP.md).

<!-- Uncomment when exported: ![pipeline](demo/pipeline.png) -->

## Highlights

| Item | Result |
|------|--------|
| Depth for 2D→3D unprojection | 3DGS render depth NN median **0.133 m** vs DAV2 affine **0.368 m** (20 groups, 15/20 wins) |
| `bridge/` pipeline | Cross-env 3DGS + HTTP VLM + RANSAC fuse → `fused.json` / SIBR markers |
| Synthetic SFT | **469** RGB-D Location samples, **10** object categories |
| 2B LoRA (data2) | Domain 2D median L2 ≤ Base on **10/10** training objects (e.g. umbrella **0.067→0.011**, norm. coords) |
| Hold-out | Double-sided tape (not in SFT) — qualitative overlay |
| RefSpatial-Expand | Base **50.21%** Location (repro.); data2 LoRA **45.64%** (out-of-domain) |

Full tables: [`docs/results_table.md`](docs/results_table.md) · Pipeline figure: [`docs/pipeline.md`](docs/pipeline.md)

## Repository layout (what is in Git)

| Path | In Git? | Role |
|------|---------|------|
| [`bridge/`](bridge/) | **Yes** | 2D→3D unproject, fuse, e2e, eval, training export |
| [`docs/`](docs/) | **Yes** | Results, resume bullets, experiment index |
| [`patches/`](patches/) | **Yes** | Small upstream diffs + integration notes |
| [`3DGS/render.py`](3DGS/render.py) | **Yes** | `--custom_views` RGB + `depth_raw` + cameras |
| [`3DGS/environment-envGS.yml`](3DGS/environment-envGS.yml) | **Yes** | Conda env hint |
| `3DGS/gaussian-splatting/` | **No** | Clone Inria 3DGS — [setup](docs/UPSTREAM_SETUP.md) |
| `RoboRefer-main/` | **No** | Clone RoboRefer — [patches](patches/roborefer/INTEGRATION.md) |
| `weights/`, `RoboRefer-2B-SFT/` | **No** | Download from Hugging Face |
| `training_data/`, `3DGS/test2/runs/` | **No** | Local experiments |

## Quick start

**Prerequisites:** clone 3DGS under `3DGS/gaussian-splatting/`, clone RoboRefer into `RoboRefer-main/`, download weights — see [docs/UPSTREAM_SETUP.md](docs/UPSTREAM_SETUP.md).

```powershell
# 1) envGS: multi-view render pack
cd 3DGS
python render.py -m gaussian-splatting/output/<scene> --custom_views --output_path ../test2

# 2) RoboRefer API (WSL/cloud), then from repo root:
python bridge/roborefer_client.py --root 3DGS/test2 --url http://127.0.0.1:25547 --prompt "Please point to ..."

# 3) Fuse + optional snap
python bridge/fuse_multiview.py --predictions 3DGS/test2/predictions.json --ply <point_cloud.ply> --output 3DGS/test2/fused.json

# Or one-shot:
python bridge/run_bridge_e2e.py --model-path 3DGS/gaussian-splatting/output/data2 --custom-views-out 3DGS/test2 --prompt "..." --snap --url http://127.0.0.1:25547
```

**2D eval vs synthetic GT:**

```bash
python bridge/eval_2d_vs_gt.py --out docs/results_2d_eval.json
```

## What we changed upstream (short)

| Upstream | Change size | Shipped in this repo |
|----------|-------------|---------------------|
| 3DGS | **Medium** — new `render.py`; small `gaussian_renderer` patch for accel | `3DGS/render.py`, `patches/3dgs/` |
| RoboRefer | **Small** — dataset register, trainer log(), API defaults, `ds_2d_*` rename | `patches/roborefer/INTEGRATION.md` only |

**Do not mirror full upstream trees on GitHub** (size, license, noise). Reviewers care about `bridge/` + reproducible setup doc.

## License

- **MIT** — `bridge/`, `docs/`, `patches/`, `3DGS/render.py`, and project README ([LICENSE](LICENSE)).
- **Upstream** — 3DGS (Inria non-commercial research license), RoboRefer and others — see [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

If you use this integration, cite the upstream 3DGS and RoboRefer papers. This repository is a student research workspace, not an official release of either project.

## Docs index

| Doc | Use |
|-----|-----|
| [docs/UPSTREAM_SETUP.md](docs/UPSTREAM_SETUP.md) | Clone & weights |
| [docs/results_table.md](docs/results_table.md) | Numbers for resume |
| [docs/RESUME_AND_INTERVIEW.md](docs/RESUME_AND_INTERVIEW.md) | Interview cheat sheet |
| [docs/pipeline.md](docs/pipeline.md) | Mermaid system diagram |
