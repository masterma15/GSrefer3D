# Changelog

本仓库版本以 Git tag `vMAJOR.MINOR.PATCH` 为准；`VERSION` 文件同步当前发布号。

## [0.2.0] — 2026-11-13

相对上次已推送提交 `32fd312`（`origin/master`）的增量。主线：softedge 视锥投票、门控 maskclose 后处理接入 e2e、data3/可见性工具。

### 新增可用模块（`bridge/`）

| 模块 | 用途 |
|---|---|
| `postprocess_seg.py` | 投票后处理；门控 `n_views≤30 ∧ n_selected<7000` → **mask-jury** |
| `batch_maskclose_gated.py` | 对 data3 softedge 核批量跑同上门控（命中为 mask-jury） |
| `frustum_segment.py`（大幅增强） | 两次视线加权 softedge（白集>0.03 / 终选>0.15）、遮挡与扇区等 |
| `run_bridge_e2e.py`（增强） | 选框默认同 `mask_grounding`；vote 后门控 mask-jury（少视角低选点）再 inject |
| `visible_mask.py` | 物体期望逆深度修剪 mask，按删比例标遮挡 |
| `render_object_invdepth.py` | 只光栅选中高斯 → 与 `depth_raw` 同定义的期望逆深度（envGS） |
| `render_object_weight.py` | 全图物体权重可视化（不读 SAM） |
| `overlay_visible_mask.py` | 可见 mask + 遮挡分类叠图 |
| `overlay_ray_occlusion.py` | 射线遮挡叠图（露出/挡住） |
| `export_ray_visible.py` | 由旧投票点云导出可见掩码与微调坐标 |
| `export_data3_sft.py` | data3 厨房 → 可见质心 SFT 包（对齐 data2_sft_ray 布局） |
| `select_nearby_views.py` | 邻视角选取（e2e 不调用；后续 SFT/扩充微调数据集做视角选取扩充） |
| `view_lists.py` | 显式删 kept / labels 视角 |
| `bbox_keys.py` | 训练目录名 → `bbox_data2.json` 键 |
| `repo_paths.py` | 相对仓库根写路径；读相对 / Win 绝对 / WSL `/mnt/...` |

### 测试

- `bridge/tests/test_postprocess_seg.py` — maskclose / 门控
- `bridge/tests/test_repo_paths.py`
- `bridge/tests/test_visible_mask.py`
- `bridge/tests/test_ablation_select.py`
- `bridge/tests/test_frustum_segment.py`（扩展）

### 工具与文档

- 展示脚本迁到 `tools/`（`compress_demo_gif`、`make_e2e_teaser`、`plot_*`）
- `tools/plot_text_only_ablation.py`、`plot_vote_compare.py`
- `docs/RESULTS.md`、`README.md`、`CLAUDE.md` 同步现行口径
- `docs/ablation_text_only.json`
- demo：`teaser_text_only_ablation.png`

### 移除 / 收敛

- 删除一次性脚本：`apply_filter_batch.py`、`patch_kept_rgb_paths.py`、`run_training_prepare.py`
- PCA / spur-only 试验路径从 `postprocess_seg` 收敛为门控 maskclose

### 未纳入本 tag 的本地产物（仍 gitignore / 勿推）

权重与模型目录、`*.tar*`、`training_data` 大体积、`3DGS/gaussian-splatting` 数据与 output、RoboRefer 权重、临时 `_tmp_*` / IDE 配置等。

## [0.1.0] — 既有 `origin/master` 尖端

以 `32fd312` 为界：官方 3D 口径改为视锥投票高斯、单命令 e2e、SIBR demo GIF、ray LoRA / OBB 评测等（见该段 commit 历史）。早期整库快照 tag：`backup/snapshot-f552b52-initial`。
