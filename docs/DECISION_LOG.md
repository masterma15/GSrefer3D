# 技术决策记录（Decision Log）

> 只记录 **为什么这样选**。实验数字与路径见 [`EXPERIMENT_DATA_INDEX.md`](EXPERIMENT_DATA_INDEX.md)。

---

| # | 决策 | 备选 | 为何放弃 / 弱化 | 最终方案 | 证据 |
|---|------|------|-----------------|----------|------|
| D1 | 2D→3D 深度 | DAV2 + COLMAP | 尺度不一致；NN 大、跨物体漂移 | 3DGS `expected_invdepth` + `camera_params` | §3.2：median 0.133 m vs 0.368 m |
| D2 | 相机外参 | JSON `rotation` 当 R_w2c | 3DGS 存 R_c2w，用错整体飞走 | `unproject.py` 转置 + pytest | `test_unproject_view000.py` |
| D3 | VLM 部署 | 本机 Windows API | 8G OOM；FlashAttention 需 Linux | WSL/云 4090 + HTTP 25547 | e2e 已通 |
| D4 | 跨环境 | 单进程双 conda | 依赖/CUDA 不兼容 | `run_bridge_e2e.py` 分阶段 | manifest 在 `runs/<id>/` |
| D5 | 多视角融合 | 单视角反投影 | 错指、遮挡 | RANSAC + 精炼 + snap | `fused.json` support |
| D6 | Qwen 可见性预筛 | `--visibility-check` | 与几何 outlier 重叠；瓶颈在指准 | 默认关 | D6 未系统 ablate |
| D7 | 视角数 | 36 视角 | 与 LoRA 不可比 | 统一 **72 视角** | median elev ≈ -19.8° |
| D8 | 主指标 | 只报 3D 融合点 | 俯拍 inlier 多 → 3D 偏轮廓 | **2D overlay 为主**；3D 为辅 | 见 limitation |
| D9 | 训练数据 | 人工 2D / 直接用 RefSpatial | 与 data2 域不符 | 3D 锚点 → 投影 → SAM2 → 469 条 | `training_data/data2_sft/` |
| D10 | 微调 | 全参 SFT | 小数据 + 算力 | **2B LoRA** · 1 epoch ~117 steps | loss 3.6→0.7 |
| D11 | Hold-out | 10 物体全训 | 需证非记忆 | **胶带**不在 SFT | §2.0 |
| D12 | 外部 benchmark | 只报 data2 | 防 leakage | Expand Base 复现；**LoRA 域外略降**（L −4.57 pp） | §9.1 |
| D13 | 3D 路线 | SceneSplat / Dr. Splat | 8G 训不了 encoder；与 RGB-D VLM 不对齐 | Inria 渲染 + 2D VLM + 几何融合 | [`ROUTE_POSITIONING.md`](ROUTE_POSITIONING.md) |
| D14 | 跨场景扩展 | 下载 splat + 伪造 COLMAP | 缺原始相机；合成轨道与街拍分布不一致，渲染 OOD | **已放弃**；改自拍 + 完整 COLMAP/SfM | — |

---

## Limitation（汇报结尾四句）

1. **域内为主**：data2 同场景；hold-out 胶带；Expand LoRA 域外 −4.57 pp，不替代域内结论。
2. **几何**：`expected_invdepth` 近似；3D 融合对俯拍分布敏感。
3. **规模**：469 条 / 1 epoch；单场景 SFT。
4. **路线**：未做 language-on-Gaussian；与 3DGS-native 工作互补非直接可比。
