# Demo assets

| File | In Git? | Notes |
|------|---------|-------|
| `pipeline.png` | **Yes** | End-to-end diagram; root [README](../README.md) |
| `teaser_depth_ablation.png` | **Yes** | Depth source ablation bar chart |
| `teaser_train_data.png` | **Yes** | SFT label refine (proj → mask centroid) |
| `teaser_gaussian_seg.png` | **Yes** | Frustum-vote 3D instance vs hand OBB |
| `shaver.gif` | **Yes** | SIBR orbit — electric shaver (voted Gaussians) |
| `rabbit.gif` | **Yes** | SIBR orbit — brown rabbit |
| `golden_retriever.gif` | **Yes** | SIBR orbit — golden retriever |
| `umbrella.gif` | **Yes** | SIBR orbit — umbrella |
| `cake.gif` | **Yes** | SIBR orbit — toy cake |
| `hair_clip.gif` | **Yes** | SIBR orbit — hair clip |
| `double_sided_tape.gif` | **Yes** | SIBR orbit — hold-out tape |
| `teaser_3d_electric_shaver.gif` | no | Old fused-point / seed-marker orbit — **not** the 3D result |
| `teaser_3d_brown_rabbit.gif` | no | Same; do not cite as 3D eval |
| `teaser_base_lora_umbrella.png` | **Yes** | Base vs LoRA overlays — umbrella |
| `teaser_base_lora_golden_retriever.png` | **Yes** | Base vs LoRA overlays — golden retriever |
| `teaser_base_lora_rabbit.png` | **Yes** | Base vs LoRA overlays — brown rabbit |
| `teaser_base_lora_shaver.png` | **Yes** | Base vs LoRA overlays — electric shaver |
| `teaser_base_lora_tape.png` | **Yes** | Base vs LoRA overlays — hold-out tape |

### Export `teaser_base_lora_*.png`

From repo root (`envGS` or any env with **Pillow**):

```powershell
pip install pillow

python bridge/make_e2e_teaser.py --preset tape --output demo/teaser_base_lora_tape.png
python bridge/make_e2e_teaser.py --preset umbrella --output demo/teaser_base_lora_umbrella.png
python bridge/make_e2e_teaser.py --preset shaver --output demo/teaser_base_lora_shaver.png
python bridge/make_e2e_teaser.py --preset rabbit --output demo/teaser_base_lora_rabbit.png
python bridge/make_e2e_teaser.py --preset golden_retriever --output demo/teaser_base_lora_golden_retriever.png
```

Presets: `tape`, `shaver`, `rabbit`, `umbrella`, `golden_retriever` (run IDs from `docs/RESULTS.md`).  
Requires existing `runs/<run_id>/overlays_rgb/overlay_view_*.png` from `run_bridge_e2e.py`.

### Optional: smaller GIFs for slow networks

Official SIBR orbits are compressed in place to ~5–10 MB (full-res copies kept as `*.gif.orig`, not in Git):

```powershell
python bridge/compress_demo_gif.py --demo-dir demo
```

Do not compress `teaser_3d_*.gif` into the README; those are historical seed-marker orbits.

### Export `teaser_gaussian_seg.png`

```powershell
python bridge/plot_gaussian_seg_teaser.py
```

Official SIBR orbits (voted Gaussians): `demo/{shaver,rabbit,golden_retriever,umbrella,cake,hair_clip,double_sided_tape}.gif`. Do not use `teaser_3d_*.gif`.

| `pipeline_overview.mmd` | local | Mermaid source for `pipeline.png` |

Do not commit full `3DGS/test2/runs/*/overlays_rgb/` (72×N images). Copy representative PNGs only when building collages.
