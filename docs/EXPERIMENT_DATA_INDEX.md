# 实验数据路径索引

> 仓库根目录：`E:\3DGS-VLM\`  
> 最后更新：2026-05-21（移除下载 splat / data3–6 实验线）

**文档导航**：见 [`docs/README.md`](README.md) — 实验 / 决策 / 简历 / 军工索引已分文件。

---

## 1. 3DGS 场景与多视角渲染（共用输入）

| 用途 | 路径 |
|------|------|
| **data2 训练模型** | `3DGS/gaussian-splatting/output/data2/` |
| **snap 用点云** | `3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply` |
| **72 视角 RGB-D 包** | `3DGS/test2/` |
| ├─ RGB | `3DGS/test2/rgb/view_000.png` … `view_071.png` |
| ├─ 深度 raster | `3DGS/test2/depth_raw/view_*.npy` |
| ├─ 深度可视化 | `3DGS/test2/depth/ view_*.png` |
| └─ 相机 | `3DGS/test2/camera_params/view_*.json` |

---

## 2. E2E 指代实验（Base / LoRA 对比）

**每个 run 目录结构相同：**

```
3DGS/test2/runs/<run_id>/
  prompt.txt              # 完整 prompt
  run_manifest.json       # 场景路径、snap_ply、SIBR 提示
  predictions.json        # 72 视角 2D 预测 (nx, ny)
  fused.json              # 3D 融合 P_world、support、inlier
  marker.ply              # 融合点可视化
  overlays_rgb/           # 2D 指代叠加图 ★对比主看图
```

**SIBR 注入模型副本**（含红球）：  
`3DGS/gaussian-splatting/output/data2_runs/<run_id>/`

---

### 2.0 hold-out 胶带（已确认 Base / LoRA，72 视角）

Prompt：`Please point to the roll of clear double-sided adhesive tape on the desk.`  
（**不在** `training_data/data2_sft`，用于同场景泛化测试）

| 组别 | run_id | 完整路径 | support | P_world |
|------|--------|----------|---------|---------|
| **Base** | `20260519_000313_6c883d56` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_000313_6c883d56\` | 15 | [2.391, 0.881, 2.540] |
| **LoRA** | `20260519_132142_6c883d56` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_132142_6c883d56\` | 17 | [2.312, 0.798, 2.602] |

对比看图：`overlays_rgb\view_XXX.png`（两 run 同路径结构）。

---

### 2.1 LoRA（merged API）72 视角 — 训练物体 run_id

| 物体 | run_id | support | 备注 |
|------|--------|---------|------|
| 剃须刀 | `20260519_143457_4c3b9a32` | 20 | |
| 棕兔 | `20260519_144845_147bac82` | 52 | |
| 金毛 | `20260519_154013_f8dbfcc3` | 57 | |
| 金碗 | `20260519_154649_8d83a715` | 36 | |
| 雨伞 | `20260519_160219_d7bab60f` | 32 | |
| 玩具蛋糕 | `20260519_161222_7dd80c38` | 40 | |
| 饼干袋 | `20260519_162019_e51c780a` | 19 | |
| 药瓶 | `20260519_163004_04149b86` | 35 | |
| 手链 | `20260519_163546_cb2e562f` | 21 | |
| 发夹 | `20260519_164312_65bf02a5` | 25 | |

完整路径示例：  
`E:\3DGS-VLM\3DGS\test2\runs\20260519_144845_147bac82\overlays_rgb\view_000.png`

---

### 2.2 Base（RoboRefer-2B-SFT）72 视角 — 训练物体（已确认）

> API 为 `RoboRefer-2B-SFT`（未 merge），与 §2.1 同 prompt、同 `test2` 72 视角。  
> **10/10 训练物体 Base 72v 已齐**（手链 Base 见下表最后一行）。

| 物体 | run_id | 完整路径 | support | P_world |
|------|--------|----------|---------|---------|
| 剃须刀 | `20260519_170540_4c3b9a32` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_170540_4c3b9a32\` | 24 | [-0.734, 1.186, 2.088] |
| 棕兔 | `20260519_171359_147bac82` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_171359_147bac82\` | 42 | [0.306, 0.134, 1.509] |
| 金毛 | `20260519_172627_f8dbfcc3` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_172627_f8dbfcc3\` | 48 | [-0.424, 1.120, 0.411] |
| 金碗 | `20260519_173958_8d83a715` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_173958_8d83a715\` | 36 | [1.935, 1.447, 1.220] |
| 雨伞 | `20260519_174835_d7bab60f` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_174835_d7bab60f\` | 11 | [-1.004, 2.512, 0.362] |
| 玩具蛋糕 | `20260519_175733_7dd80c38` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_175733_7dd80c38\` | 28 | [-0.102, 0.597, 1.695] |
| 饼干袋 | `20260519_180800_e51c780a` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_180800_e51c780a\` | 17 | [-1.296, 1.965, 1.558] |
| 药瓶 | `20260519_182106_04149b86` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_182106_04149b86\` | 33 | [0.407, 1.986, 0.349] |
| 发夹 | `20260519_183039_65bf02a5` | `E:\3DGS-VLM\3DGS\test2\runs\20260519_183039_65bf02a5\` | 22 | [0.616, 1.839, 1.314] |
| 手链 | `20260520_001018_cb2e562f` | `E:\3DGS-VLM\3DGS\test2\runs\20260520_001018_cb2e562f\` | 22 | [-1.296, 1.965, 1.558] |

**旧 Base 36 视角（勿与 72v 直接比）**：`20260516_*`，如 `20260516_111848_4c3b9a32`（剃须刀）。

---

### 2.3 Base vs LoRA 对照总表（72 视角）

| 物体 | Base run_id | LoRA run_id | Base sup | LoRA sup |
|------|-------------|-------------|----------|----------|
| 剃须刀 | `170540_4c3b9a32` | `143457_4c3b9a32` | 24 | 20 |
| 棕兔 | `171359_147bac82` | `144845_147bac82` | 42 | 52 |
| 金毛 | `172627_f8dbfcc3` | `154013_f8dbfcc3` | 48 | 57 |
| 金碗 | `173958_8d83a715` | `154649_8d83a715` | 36 | 36 |
| 雨伞 | `174835_d7bab60f` | `160219_d7bab60f` | 11 | 32 |
| 玩具蛋糕 | `175733_7dd80c38` | `161222_7dd80c38` | 28 | 40 |
| 饼干袋 | `180800_e51c780a` | `162019_e51c780a` | 17 | 19 |
| 药瓶 | `182106_04149b86` | `163004_04149b86` | 33 | 35 |
| 手链 | `001018_cb2e562f` | `163546_cb2e562f` | 22 | 21 |
| 发夹 | `183039_65bf02a5` | `164312_65bf02a5` | 22 | 25 |
| 胶带 hold-out | `000313_6c883d56` | `132142_6c883d56` | 15 | 17 |

前缀多为 `20260519_`；手链 Base 为 `20260520_001018_cb2e562f`。完整路径 `E:\3DGS-VLM\3DGS\test2\runs\<run_id>\`。
---

## 3. DAV2 vs 3DGS 解耦实验 ★Decision Log D1

| 项目 | 路径 |
|------|------|
| **脚本** | `bridge/compare_depth_sources.py`；批量 `bridge/batch_compare_depth_sources.py` |
| **批量结果 JSON** | `docs/depth_compare_batch.json`（20 组，2026-05-20） |
| **输入视角** | `3DGS/test2/`（camera + rgb + depth_raw） |
| **NN 参考点云** | `3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply` |
| **DAV2 权重** | `weights/depth_anything_v2_vitl.pth` |

### 3.0 指标释义（速查）

**实验在做什么（D1）**  
固定同一视角、同一指代像素 `(nx,ny)`、同一相机外参，**只换深度来源**反投影得 `P_world`，再算到 `point_cloud.ply` 最近高斯中心的 **NN 距离（米）**。NN 越小，说明深度与 **3DGS 场景几何** 越一致。

**简历一行数（实验已完成）**  
> 20 组 NN median：**3DGS 0.133 m** vs **DAV2 affine 0.368 m**（15/20 组 3DGS 更优）

---

#### 什么是「逆深度」？

| 量 | 定义 | 在本项目里 |
|----|------|------------|
| **深度 z** | 相机坐标系下，沿光轴到点的距离（米） | 反投影公式里的 `z_cam` |
| **逆深度 inv** | **`inv = 1 / z`**（单位：1/米） | 近处 inv **大**，远处 inv **小**；对远景变化更平缓，优化/对齐更稳 |

**为何 3DGS 用逆深度**  
- `render.py` 存的是 raster 的 **`expected_invdepth`**（不是 z 本身）  
- `bridge/unproject.py`：`z_cam = 1 / max(inv, ε)` 再反投影  
- 官方深度正则化 `make_depth_scale.py` 也是在 **逆深度图** 上对 COLMAP 与单目深度做 **scale + offset**（median / MAD），不是在线性 z 上拟合

**DAV2 inv 列在测什么**  
仿 `make_depth_scale`：先把 DAV2 输出转成 `inv_dav2 = 1/DA`，在 valid 像素上拟合  
`inv_3dgs ≈ scale × inv_dav2 + offset`，再在该指代像素取对齐后的 inv → 转回 z → 反投影 → NN。  
这是「按 3DGS 训练惯例对齐单目深度」的 **对照组**；仍可能单点碰巧很小，但整图 **RMS** 仍大。

---

#### 表格各列含义

| 列 | 深度怎么来 | 含义 |
|----|------------|------|
| **3DGS NN** | `depth_raw/*.npy` 的 `expected_invdepth` → `z=1/inv` | **最终方案**；与 data2 辐射场同一几何 |
| **DAV2 raw** | DAV2 相对深度 `DA`，`z=1/DA`，**不对齐** | 单目深度缺米制尺度，NN 常很大 |
| **DAV2 affine** | 整图拟合 **`z_3dgs ≈ a·DA + b`**（线性 z） | 单视角「尽力对齐」；单点可能碰巧准 |
| **DAV2 inv** | 整图拟合 **`inv_3dgs ≈ a·inv_dav2 + b`**（仿 make_depth_scale） | 与 3DGS 深度监督同风格的对齐 |
| **RMS** | affine 后 **整图** valid 区：`sqrt(mean((z_3dgs−z_dav2)²))` | **不是 NN**；衡量深度图全局不一致，高则多视角仍会散 |

**共同前提**：相机用正确 `R_w2c`（`unproject.py` 对 JSON rotation 转置）；NN 参考 `iteration_30000/point_cloud.ply`。

**与 RoboRefer 的分工**：API 里 DAV2 给 VLM 当 **RGB-D 语义输入**；bridge 反投影必须用 **3DGS 渲染深度**（几何角色不同，见 D1）。

---

### 3.1 基准点（手工，view_000）

测试点：view_000, `(nx,ny)=(0.4958,0.9093)` pixel (507,695)

| 配置 | NN 距离 (m) |
|------|-------------|
| 3DGS expected_invdepth | **0.027** |
| DAV2 无对齐 | **1.555** |
| DAV2 + 全图 affine z | **0.281** |
| DAV2 + inv-depth scale/offset | **2.310** |
| 对齐后 depth map RMS | **5.44** |

### 3.2 多视角批量（10 物体 × 2 view，共 20 组）

**选点规则**：各物体 Base run 的 `predictions.json` 中 `parse_ok` 点；每物体取 **ny 最小**（偏水平）与 **ny 最大**（偏俯拍）各 1 视角。

| 物体 | view | (nx, ny) | 3DGS NN | DAV2 raw | DAV2 affine | DAV2 inv | RMS |
|------|------|----------|---------|----------|-------------|----------|-----|
| 剃须刀 | 027 | (0.300, 0.063) | 0.409 | 0.530 | 0.800 | 0.405 | 1.35 |
| 剃须刀 | 064 | (0.440, 0.948) | 0.045 | 1.253 | 0.045 | 4.934 | 3.81 |
| 棕兔 | 004 | (0.362, 0.133) | 0.147 | 0.195 | 0.011 | 0.155 | 1.48 |
| 棕兔 | 052 | (0.260, 0.756) | 0.010 | 0.423 | 0.312 | 0.045 | 1.82 |
| 金毛 | 004 | (0.368, 0.133) | 0.119 | 0.195 | 0.003 | 0.125 | 1.48 |
| 金毛 | 028 | (0.646, 0.854) | 0.017 | 0.619 | 0.374 | 0.111 | 1.55 |
| 金碗 | 040 | (0.577, 0.104) | 0.338 | 0.298 | 0.833 | 0.103 | 1.53 |
| 金碗 | 067 | (0.123, 0.978) | 0.095 | 0.572 | 0.424 | 1.266 | 1.53 |
| 雨伞 | 048 | (0.456, 0.107) | 0.173 | 0.596 | 0.174 | 0.084 | 1.30 |
| 雨伞 | 062 | (0.458, 0.911) | 0.031 | 0.920 | 0.110 | 0.858 | 1.65 |
| 玩具蛋糕 | 042 | (0.312, 0.107) | 0.019 | 0.337 | 0.312 | 0.119 | 2.21 |
| 玩具蛋糕 | 041 | (0.496, 0.904) | 0.054 | 0.204 | 0.361 | 0.167 | 1.19 |
| 饼干袋 | 047 | (0.350, 0.180) | 0.500 | 0.838 | 0.386 | 0.889 | 3.63 |
| 饼干袋 | 067 | (0.123, 0.978) | 0.095 | 0.572 | 0.424 | 1.266 | 1.53 |
| 药瓶 | 037 | (0.662, 0.072) | 0.362 | 1.316 | 0.300 | 0.086 | 1.91 |
| 药瓶 | 054 | (0.446, 0.969) | 0.191 | 0.686 | 0.242 | 0.199 | 0.66 |
| 手链 | 053 | (0.499, 0.169) | 0.322 | 0.759 | 0.392 | 0.118 | 1.52 |
| 手链 | 067 | (0.357, 0.963) | 0.478 | 0.572 | 0.512 | 0.848 | 1.53 |
| 发夹 | 064 | (0.536, 0.102) | 0.083 | 1.257 | 1.445 | 0.546 | 3.81 |
| 发夹 | 067 | (0.352, 0.957) | 0.436 | 0.572 | 0.465 | 0.887 | 1.53 |
| **汇总 median** | — | — | **0.133** | **0.572** | **0.368** | **0.183** | **1.53** |

**解读（20 组）**：3DGS NN **median 0.133 m** vs DAV2 affine **0.368 m**；**15/20** 组 3DGS NN &lt; DAV2 affine。DAV2 raw median **0.572 m** 仍系统性偏大。发夹 view_064、金碗 view_040 等少数组 affine 后仍差于 3DGS 或更差，与指代点落在弱纹理/遮挡边缘有关。

复现命令：

```powershell
# 单点（§3.1 基准）
D:\anaconda\envs\envGS\python.exe E:\3DGS-VLM\bridge\compare_depth_sources.py `
  --views-root E:\3DGS-VLM\3DGS\test2 --view 000 `
  --nx 0.4958 --ny 0.9093

# 批量 20 组（§3.2）
D:\anaconda\envs\envGS\python.exe E:\3DGS-VLM\bridge\batch_compare_depth_sources.py
```

相关验证脚本：`bridge/verify_unproject_vs_pointcloud.py`  
回归测试：`bridge/tests/test_unproject_view000.py`

---

## 9. RefSpatial-Expand-Bench 评测 ★D12

| 项目 | 内容 |
|------|------|
| **数据集** | `RefSpatial-Expand-Bench/`（软链至 `RoboRefer-main/Evaluation/RefSpatial-Bench`） |
| **API** | WSL · `RoboRefer-main/API` · port 25547 · RGB-D（`model_name` 含 `Depth`） |
| **权重** | Base：`RoboRefer-2B-SFT` · LoRA merge：`RoboRefer-2B-SFT-data2-merged/` |
| **脚本** | `RoboRefer-main/Evaluation/test_benchmark.py` + `summarize_acc.py` |
| **日期** | Base 2026-05-20 · LoRA 2026-05-20 |

### 9.1 结果对比（Base vs data2 LoRA）

| Task | n | **Base** `RoboRefer-2B-SFT-Depth` | **LoRA** `RoboRefer-2B-data2-LoRA-Depth` | Δ (LoRA−Base) | 论文 (Base) |
|------|---|-----------------------------------|----------------------------------------|---------------|-------------|
| **Location** | 241 | **50.21%** | **45.64%** | **−4.57 pp** | 50.21% ✅ |
| **Placement** | 200 | **48.50%** | **47.00%** | **−1.50 pp** | 48.50% ✅ |

- Base 输出：`RoboRefer-main/Evaluation/outputs/RoboRefer-2B-SFT-Depth/{Location,Placement}.jsonl`
- LoRA 输出：`RoboRefer-main/Evaluation/outputs/RoboRefer-2B-data2-LoRA-Depth/{Location,Placement}.jsonl`
- **解读**：469 条 data2 桌面 SFT 在 **域外** Expand 上略降，符合小数据域适配 trade-off；主验证仍看 data2 域内 overlay + hold-out 胶带。
- **说明**：Expand 与原版 RefSpatial-Bench（L=100，论文 47%）**不可直接比**；简历写清数据集名。

### 9.2 复现命令（WSL）

```bash
cd /mnt/e/3DGS-VLM/RoboRefer-main/Evaluation
ln -sfn ../../RefSpatial-Expand-Bench RefSpatial-Bench

# Base（API: RoboRefer-2B-SFT）
python test_benchmark.py --model_name RoboRefer-2B-SFT-Depth \
  --task_name Location --url http://127.0.0.1:25547
python summarize_acc.py --model_name RoboRefer-2B-SFT-Depth --task_name Location

# LoRA merge（API: RoboRefer-2B-SFT-data2-merged，model_name 仅用于输出目录命名）
python test_benchmark.py --model_name RoboRefer-2B-data2-LoRA-Depth \
  --task_name Location --url http://127.0.0.1:25547
python summarize_acc.py --model_name RoboRefer-2B-data2-LoRA-Depth --task_name Location
# Placement：--task_name Placement
```

### 9.3 待补

- [x] data2 **LoRA** merge · Expand Location / Placement（§9.1）
- [ ] [可选] 原版 RefSpatial-Bench（100+100+Unseen）· 与论文 47% 对齐

---

## 4. 训练数据（Location SFT）

| 项目 | 路径 |
|------|------|
| **导出 JSON + 图像** | `training_data/data2_sft/` |
| ├─ 标注 | `training_data/data2_sft/location_point.json`（469 条） |
| ├─ RGB | `training_data/data2_sft/image/` |
| └─ depth | `training_data/data2_sft/depth/` |
| **按物体中间产物** | `training_data/data2_<object>/`（projections、mask 等） |
| **mixture 注册名** | `data2_location`（`RoboRefer-main/llava/data/datasets_mixture.py`） |

---

## 5. 模型权重

| 模型 | 路径 |
|------|------|
| Base 2B | `RoboRefer-2B-SFT/` |
| LoRA adapter | `RoboRefer-2B-SFT/data2_lora/`（或 cloud `runs/train/data2_lora/`） |
| Merged（推理用） | `RoboRefer-2B-SFT-data2-merged/` |
| Depth-Anything ViT-L | `weights/depth_anything_v2_vitl.pth` |

---

## 6. LoRA 训练日志（云 AutoDL）

| 项目 | 典型路径 |
|------|----------|
| 训练输出 | `RoboRefer-main/runs/train/data2_lora/` |
| adapter | `adapter_model.safetensors`、`checkpoint-117/` |
| 超参 | 1 epoch，~117 steps，loss ≈ 3.6→0.7，train_loss ≈ 1.02 |

---

## 7. 文档与实习汇报素材

| 文档 | 路径 |
|------|------|
| **Decision Log** | `docs/DECISION_LOG.md` |
| **简历 / 面试** | `docs/RESUME_AND_INTERVIEW.md` |
| **路线对比** | `docs/ROUTE_POSITIONING.md` |
| **文档索引** | `docs/README.md` |
| **本索引** | `docs/EXPERIMENT_DATA_INDEX.md` |
| **结果总表** | `docs/results_table.md`（2D L2 · 深度 · Expand · 简历句） |
| 仓库说明 | `CLAUDE.md` |

**建议 figure 取材（组内汇报 / 实习总结 / 简历附件）：**

- overlay 对比：`runs/<run_id>/overlays_rgb/view_XXX.png`
- 3D 注入：`data2_runs/<run_id>/` → SIBR iteration_35000
- 解耦实验：终端 NN 表 → 实习报告「深度来源对比」小节

---

## 8. 仍缺 / 待补数据

- [x] hold-out **胶带** Base / LoRA 72v（§2.0）
- [x] Base 72v **10/10 训练物体**（§2.2；手链 Base `20260520_001018_cb2e562f`，support 22）
- [x] RefSpatial-Expand **Base** L 50.21% / P 48.50%（§9）
- [x] RefSpatial-Expand **LoRA** L 45.64% / P 47.00%（§9.1，Δ −4.57 / −1.50 pp）
- [x] 多视角 `compare_depth_sources.py` 汇总表（§3.2，20 组，`docs/depth_compare_batch.json`）
- [x] **域内 2D L2 总表**（`docs/results_table.md`，`bridge/eval_2d_vs_gt.py` → `docs/results_2d_eval.json`）

---

*NN = 反投影点 `P_world` 到 `point_cloud.ply` 最近高斯中心的欧氏距离。*
