# Integration patches (3DGS-VLM)

Upstream trees are **not** vendored in full in this Git repository. Apply the changes below after cloning [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [RoboRefer](https://github.com/Zhoues/RoboRefer).

## Summary

| Area | Upstream change? | What we ship in Git |
|------|------------------|---------------------|
| **bridge/** | N/A (original) | Full directory |
| **3DGS `render.py`** | **Yes — new file** | `3DGS/render.py` (multi-view RGB + `depth_raw` + `camera_params`) |
| **3DGS `gaussian_renderer`** | **Yes — small patch** | `patches/3dgs/gaussian_renderer__init__.py` (accel `dc`/`shs` split) |
| **RoboRefer `datasets_mixture`** | **Yes** | `patches/roborefer/INTEGRATION.md` + rename fix + `data2_location` |
| **RoboRefer `llava_trainer`** | **Yes — 1 line** | transformers `log(..., start_time=)` compat |
| **RoboRefer `API/api.py`** | **Yes — defaults** | repo-relative weight paths |
| **RoboRefer rest (`llava/`, eval, …)** | No | clone upstream only |

## Apply

See [docs/UPSTREAM_SETUP.md](../docs/UPSTREAM_SETUP.md) for directory layout and commands.
