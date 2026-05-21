# 3DGS integration changes

## 1. Multi-view render entry (required)

Copy or symlink the repo file:

```text
3DGS/render.py   →   <your-3dgs-workspace>/render.py
```

Run from `3DGS/` (parent of `gaussian-splatting/`):

```bash
python render.py -m gaussian-splatting/output/<scene> --custom_views --output_path <out>
```

Adds `--custom_views`, `--num_custom_views`, exports `rgb/`, `depth_raw/` (`expected_invdepth`), `camera_params/`.

## 2. Gaussian rasterizer patch (if using accel fork)

If `diff_gaussian_rasterization` exposes a `dc` argument (3dgs_accel branch), replace:

```text
gaussian-splatting/gaussian_renderer/__init__.py
```

with [gaussian_renderer__init__.py](gaussian_renderer__init__.py) in this folder.

Without this patch, accel rasterizer may CUDA-error when SH is passed only via `shs`.

## 3. Everything else

Train with upstream `train.py`, `convert.py`, submodules — unchanged. Do not commit full `gaussian-splatting/` or `SIBR_viewers/` to GSrefer3D Git.
