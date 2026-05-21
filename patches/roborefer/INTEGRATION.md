# RoboRefer integration changes

Clone upstream first, then apply **four small edits**. No need to fork the entire `llava/` tree into GSrefer3D Git.

## 1. `llava/data/datasets_mixture.py`

### 1a. Python identifier fix (required on Python 3.10+)

Rename illegal variables `2D_*` / `3D_*` to `ds_2d_*` / `ds_3d_*` (same `dataset_name` strings). See upstream issue: variable names cannot start with a digit.

### 1b. Register custom SFT mixture (this project)

Add inside `register_datasets_mixtures()`:

```python
    ### data2 desktop scene (custom Location SFT, GSrefer3D bridge export)
    data2_location = Dataset(
        dataset_name="data2_location",
        dataset_type="spatialdataset",
        data_path="../training_data/data2_sft/location_point.json",
        image_path="../training_data/data2_sft/image",
        depth_path="../training_data/data2_sft/depth",
    )
    add_dataset(data2_location)
```

Train with: `--data_mixture data2_location` (see `bridge/export_spatial_train.py` for how JSON is produced).

## 2. `llava/train/llava_trainer.py`

Compat with newer `transformers` `Trainer.log` signature:

```python
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
```

## 3. `API/api.py`

Default checkpoint paths relative to repo root (optional but convenient):

```python
from pathlib import Path
# ...
    _repo_root = Path(__file__).resolve().parents[2]
    _default_depth = str(_repo_root / "weights" / "depth_anything_v2_vitl.pth")
    _default_vlm = str(_repo_root / "RoboRefer-2B-SFT")
```

## 4. `scripts/setups/train.sh` (single GPU cloud)

When not on SLURM: `export SLURM_JOB_GPUS_PER_NODE=1`

## Not modified

Model architecture (`llava_arch.py`), encoders, projectors, eval scripts — use upstream as-is.

## Weights (download, not in Git)

| Asset | Suggested path |
|-------|----------------|
| RoboRefer-2B-SFT | `RoboRefer-2B-SFT/` (HF collection) |
| Depth Anything V2 ViT-L | `weights/depth_anything_v2_vitl.pth` |
| LoRA adapter (optional) | `RoboRefer-2B-SFT/data2_lora/` → merge for inference |

See [docs/UPSTREAM_SETUP.md](../../docs/UPSTREAM_SETUP.md).
