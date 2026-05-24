# Demo assets

| File | In Git? | Notes |
|------|---------|-------|
| `pipeline.png` | **Yes** | End-to-end diagram; root [README](../README.md) |
| `teaser_depth_ablation.png` | **Yes** | Depth source ablation bar chart |
| `teaser_train_data.png` | **Yes** | SFT label refine (proj → mask centroid) |
| `teaser_3d_electric_shaver.gif` | **Yes** | SIBR orbit — electric shaver 3D anchor (LoRA) |
| `teaser_3d_brown_rabbit.gif` | **Yes** | SIBR orbit — brown rabbit 3D anchor (LoRA) |
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

| `pipeline_overview.mmd` | local | Mermaid source for `pipeline.png` |

Do not commit full `3DGS/test2/runs/*/overlays_rgb/` (72×N images). Copy representative PNGs only when building collages.
