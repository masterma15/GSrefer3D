# 路线定位：本项目 vs 3DGS-native

> 面试被问「为什么不用 SceneSplat / ReferSplat」时用。

---

## 一张表

| | **本项目** | **SceneSplat / Dr. Splat** | **ReferSplat** |
|--|-----------|---------------------------|----------------|
| 输出 | 2D 点 + **P_world** | 3D open-vocab **mask/语义场** | 3D **指代分割** mask |
| 输入 | **RGB-D 图**（RoboRefer） | Gaussian 参数 / lang_feat | 文本 + 3DGS referring field |
| 训练 | 469 条 LoRA + 几何融合 | 大规模 3D 预训练 / per-scene 蒸馏 | per-scene ~45k iter |
| 评测 | RefSpatial / data2 overlay | ScanNet mIoU | Ref-LERF mIoU |

**一句话**：我做 **RGB-D 2D referring → 显式 3D 锚点**；他们做 **3DGS 上的语言场/分割**——同用 3DGS，**问题不同**。

---

## 亮点（面试主动说，5 条）

1. **几何可审计**：D1 定量 ablation，不是黑盒抬升。
2. **闭环可 demo**：render → RoboRefer → fuse → SIBR。
3. **数据飞轮**：3D 锚点 → 469 条 SFT → LoRA。
4. **对齐 RoboRefer / RefSpatial** 生态（真机 RGB-D 模态）。
5. **RefSpatial-Expand** Base 50.21% 复现；LoRA 45.64%（域外 −4.57 pp），评测链路通。

---

## 弱项（诚实承认）

- 无 3D mask / referring field；新视角纯遮挡不如 ReferSplat。
- 3D 点精度受融合影响；**主报 2D**。
- 未做 feed-forward 3D encoder（算力 + 问题定义）。

---

## 禁说

| ❌ | ✅ |
|----|-----|
| 比 Dr. Splat 3D 更准 | 2D referring + 3D 锚点闭环 |
| 只是显卡不够 | **算力** + **problem-first** 共同解释 |

---

## Related Work 一句版

| 工作 | 关系 |
|------|------|
| RoboRefer | 直接上游；你补 3D 闭环 + 合成数据 |
| ReferSplat | 最近邻居；mask vs point |
| SceneSplat | 3D encoder 预训练；互补 |
| Dr. Splat | language-on-Gaussian；互补 |
