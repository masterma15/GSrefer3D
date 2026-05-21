# 简历 & 面试速查

> 实验路径见 [`EXPERIMENT_DATA_INDEX.md`](EXPERIMENT_DATA_INDEX.md) · 决策见 [`DECISION_LOG.md`](DECISION_LOG.md)

---

## 30 秒 Pitch

> 在 3DGS 桌面场景里，把语言空间指代从 2D 点落到**世界系 3D 锚点**：验证 DAV2 反投影不可靠后改用 **3DGS 渲染深度 + 多视角融合**；搭建 `bridge/` 闭环；从 3D 锚点合成 **469 条** RGB-D SFT 并 **2B LoRA** 微调；Expand Base **50.21%** 复现，LoRA 域外 **45.64%**（−4.57 pp，域内为主）。

---

## 简历项目描述（可直接粘贴）

**3DGS-VLM：语言引导 3D 空间指代闭环**

- 对比并放弃 DAV2+COLMAP 反投影，采用 3DGS 渲染深度实现几何一致 2D→3D（20 组 NN median **0.133 m** vs DAV2 **0.368 m**）
- 设计 `bridge/` 跨环境管线（Windows 3DGS + WSL RoboRefer HTTP），多视角 RANSAC 融合至 3D 锚点并 SIBR 可视化
- 从 3D 锚点自动生成 **469 条 / 10 类** Location 数据（SAM2）；**2B LoRA** 微调（1 epoch，loss 3.6→0.7）
- data2 **72 视角** Base vs LoRA 域内对比；hold-out 胶带；Expand **Base 50.21% / LoRA 45.64%**（Location，域外 −4.57 pp）

---

## 军工（2025.01–11）

**排版**：接在 3DGS-VLM 四条 bullet **之后**；勿单独开「其他经历」标题。模板若全是 bullet，用「bullet 版」；若允许段后一句，用「一句版」。**不写**指标数字；对方追问时再说明（见下）。

**简历 · 一句版（可直接粘贴）**

```text
2025年1月至11月，参与海上移动目标遥感半实物仿真与检测系统研发，负责天基光学/SAR 成像仿真数据接入与校验、YOLOv8 舰船检测推理链路联调及多场景结项验收测试。
```

**简历 · bullet 版（与主项目格式一致时）**

```text
- 2025.01–11，参与海上移动目标遥感半实物仿真与检测项目，负责仿真数据集接入与校验、检测推理流水线联调及结项验收测试。
```

**面试 · 介绍本项目（约 40 秒，说清工作与闭环）**

> 这是海上移动目标的半实物仿真验证系统：成像侧按天基光学和 SAR 全链路生成仿真图像，并构建光学-雷达匹配数据集；检测侧用 YOLOv8 做多尺度舰船检测，再通过光学粗定位与 SAR 细提取做融合定位。我主要负责把仿真输出接入检测训练与推理流水线，核对数据格式和推理结果是否一致，并配合多目标、多分辨率场景的链路联调与结项验收测试。

**若被问「精度 / mAP / 指标」**（一句带过即可）：我侧重数据接入、链路联调和验收测试，检测模型由组内同学主研，简历未列具体数值。

技术细节与素材路径：[`MIL_CV_DATA_INDEX.md`](MIL_CV_DATA_INDEX.md)

**投国防 / 智驾 JD 时（勿投纯 VLM，可用 2 bullet）**

```text
- 参与海上移动目标半实物仿真系统建设，完成天基光学与 SAR 成像仿真数据生成，并负责光学-雷达匹配数据集的格式校验与检测侧接入。
- 协助基于 YOLOv8 的舰船多尺度检测与光学-SAR 融合定位链路联调，承担仿真数据进线、推理结果核对及多场景结项验收测试。
```

---

## 核心数字

| 类别 | 数值 |
|------|------|
| SFT 数据 | 469 / 10 物体 |
| LoRA | ~117 steps · 2.97% 参数 |
| 域内评测 | 72 视角 / test2 |
| RefSpatial-Expand Base | L **50.21%** (241) · P **48.50%** (200) |
| RefSpatial-Expand LoRA | L **45.64%** · P **47.00%**（Δ −4.57 / −1.50 pp） |
| 深度 ablation | 3DGS 15/20 组优于 DAV2 affine |

---

## 指标怎么说

- **域内**：2D overlay / success@mask 为主
- **3D**：support + snap_distance 为辅
- **域外**：RefSpatial（Expand 或原版 100 条）；hold-out 胶带
- **不要**用一个数概括全部

---

## 高频 Q&A

**Q：最大难点？**  
深度-几何一致 + 跨栈部署；不是只调 LoRA。

**Q：为什么不用 DAV2 做反投影？**  
VLM 的 depth 是语义辅助；反投影必须与相机+场景几何自洽（D1）。

**Q：微调有用吗？有 leakage 吗？**  
域内 data2 改善（overlay）；hold-out 胶带；Expand LoRA 域外略降（L −4.57 pp），符合小数据域适配，**主结论看域内**。

**Q：为什么不用 SceneSplat？**  
子问题不同：你是 2D RGB-D referring + 3D 锚点；他们是 3D 语义场。见 [`ROUTE_POSITIONING.md`](ROUTE_POSITIONING.md)。

**Q：RefSpatial 50% 算 SOTA 吗？**  
这是 **RoboRefer-2B-SFT 官方 Base** 在 Expand 上的论文数；你做的是**复现验证**。8B/RFT 更高；你的贡献在 3D 闭环 + 合成 SFT，不是刷榜。

**Q：最大缺口？**  
data2 域内 2D L2 自动表；可选原版 Bench 100 条。

---

## 多模态岗：还缺什么

| 优先级 | 缺口 | 怎么补 |
|--------|------|--------|
| P0 | 域内 2D L2 表 | predictions vs GT 脚本 |
| P1 | 原版 Bench 100 条 | 与论文 47% 直接可比 |
| P2 | Pipeline 图 + 6 张 overlay | 汇报素材 |

**岗位匹配**：VLM 应用 / 微调 / 3D×语言 ⭐⭐⭐⭐；预训练基座 ⭐⭐。

---

## 汇报 Checklist

- [ ] Pipeline 图（源：`docs/pipeline.md` · 附件：`demo/pipeline.png`）
- [x] DAV2 vs 3DGS 一行数（见 `results_table.md` §1）
- [ ] Base vs LoRA overlay 6–9 张
- [x] RefSpatial-Expand Base（L 50.21% / P 48.50%）
- [x] RefSpatial-Expand LoRA（L 45.64% / P 47.00%）
- [ ] SIBR 截图 1–2 张
