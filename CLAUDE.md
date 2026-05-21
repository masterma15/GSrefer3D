# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

本仓库是一个**探索性研究工作区**，把三个本来独立的开源项目并列放在一起，用于研究 "3DGS + VLM 空间指代" 的结合：

- `gaussian-splatting/` — Inria 官方 3D Gaussian Splatting（图像 → 3D 场景重建 + 实时渲染）
- `RoboRefer-main/` — 基于 NVILA/VILA 的空间指代 VLM（RGB 或 RGB-D → 2D 点输出）
- `RefSpatial-Expand-Bench/` — 用于评测 RoboRefer 类模型的基准（Location / Placement 两个任务）

三个子项目来自不同上游、有**各自独立的 conda 环境、依赖、工作流**，在此仓库中并没有被统一构建。修改一个子项目时，不要假设其它子项目已经安装或可用。

## 三个子项目的各自工作流

### 1) gaussian-splatting（3DGS 训练/渲染）

**目录布局**：3DGS 代码位于 `3DGS/gaussian-splatting/`，这是一个扁平结构（`train.py`、`render.py`、`scene/`、`utils/`、`arguments/`、`gaussian_renderer/`、`submodules/` 都在同一层）。训练输出在 `3DGS/gaussian-splatting/output/<数据集名>/`，**模型目录名与训练数据集名一致**（`data`、`data2` 等），避免混用。多视角渲染输出在 `3DGS/test<N>/`，**test 目录与对应模型必须配套**（test2 → data2，test1 → data）。

**已有训练结果**：`data`（旧场景）、`data2`（电动剃须刀场景，当前主实验）。每个模型目录下有 `point_cloud/iteration_30000/`（最终权重）和 `iteration_35000/`（注入标记后的版本，用于 SIBR 可视化）。

**环境**：`conda activate envGS`（用户本机环境名，非 `gaussian_splatting`）。依赖清单见 `3DGS/environment-envGS.yml`（Python 3.9、PyTorch 2.4 + CUDA 11.8、**scipy** 等）。子模块需要手动 `pip install` 或 `python setup.py install`：`submodules/diff-gaussian-rasterization`、`submodules/simple-knn`、`submodules/fused-ssim`。

**典型命令**（在 `3DGS/` 下运行，即 `train.py`/`render.py` 所在目录）：
```powershell
# EXIF 修复（手机拍摄）
magick mogrify -strip gaussian-splatting/data2/input/*.jpg
# COLMAP 预处理
python gaussian-splatting/convert.py -s gaussian-splatting/data2
# 训练（-r 4 表示 1/4 分辨率）
python train.py -s gaussian-splatting/data2 -r 4
# 渲染多视角（输出到 test2）
python render.py -m gaussian-splatting/output/data2 --custom_views --output_path E:/GSrefer3D/3DGS/test2
# SIBR 查看器（必须从 bin/ 目录启动，否则 DLL 找不到）
Set-Location gaussian-splatting/viewers/bin
.\SIBR_gaussianViewer_app.exe -m "E:\GSrefer3D\3DGS\gaussian-splatting\output\data2"
# 加载注入标记的版本（iteration_35000）
.\SIBR_gaussianViewer_app.exe -m "E:\GSrefer3D\3DGS\gaussian-splatting\output\data2" --iteration 35000
```

**注意**：SIBR viewer 必须从 `viewers/bin/` 目录启动（`Set-Location` 后再运行），否则 `sibr_system.dll` 找不到。PowerShell 中 `--iteration` 不会报错，但 `--` 开头的参数在某些 PS 版本中需要用 `cmd /c` 包裹。

加速训练需切到 `diff-gaussian-rasterization` 的 `3dgs_accel` 分支并用 `--optimizer_type sparse_adam`（见 `gaussian-splatting/README.md` 中 "Training speed acceleration" 一节）。深度正则化需要 Depth-Anything-V2 生成深度图 + `utils/make_depth_scale.py` 生成 `depth_params.json`，然后训练时加 `-d <depth_dir>`。

### 2) RoboRefer-main（VLM 推理 / 训练 / 评测）

**这是一个包名为 `vila` 的 Python 包**（见 `pyproject.toml`），Python 源码目录叫 `llava/`（沿用 NVILA/VILA 命名，不要与 LLaVA 混淆）。

**环境**：`bash env_setup.sh roborefer` → `conda activate roborefer`（Python 3.10.14，PyTorch 2.5.1，CUDA 12.x，FlashAttention 2.5.8 的 Linux wheel — 该 setup 脚本默认假定 Linux）。安装过程还会把 `llava/train/deepspeed_replace/*` 复制覆盖到 site-packages 的 `deepspeed/` 下，这是已知的定制点，不要改 deepspeed 又忘了同步 `deepspeed_replace/`。

**三条主线**：

- **推理（API 服务 + 客户端）**：`cd API; python api.py --port 25547 --depth_model_path <DepthAnythingV2-vitl.pth> --vlm_model_path <RoboRefer weights>`，另一个终端 `python use_api.py --image_path ... --prompt ... --url http://127.0.0.1:25547`。`use_api.py` 中 `enable_depth=0/1` 切换纯 RGB / RGB-D。
- **训练**：脚本在 `scripts/RoboRefer/`（`depth_align_2B.sh`、`depth_sft_2B.sh`、`..._cluster.sh` 和 8B 变体）。训练是 **两阶段 SFT**：先做 depth alignment（深度编码器对齐），再做空间指代 SFT。自定义数据集须在 `llava/data/datasets_mixture.py` 的 `register_datasets_mixtures()` 中注册，`dataset_type="spatialdataset"` 同时支持 RGB 与 RGB-D（带 `depth_path` 就是 RGB-D，不带就是 RGB-only）。多个数据集在脚本的 `DATA_MIXTURE` 变量里用 `+` 连接。基础设施脚本 `scripts/setups/train.sh` 控制 NNODES / GPUS_PER_NODE / batch size，DeepSpeed 配置在 `scripts/zero*.json`。
- **基准评测**：需先下载 `RefSpatial-Bench` 到 `Evaluation/` 下：
  ```bash
  cd Evaluation
  git lfs install && git clone https://huggingface.co/datasets/BAAI/RefSpatial-Bench
  python test_benchmark.py --model_name RoboRefer-2B-SFT-Depth --task_name Location --url http://127.0.0.1:25547
  python summarize_acc.py --model_name RoboRefer-2B-SFT-Depth --task_name Location
  ```
  `--model_name` 名字里**含 `Depth` 就会启用深度输入**（`test_benchmark.py:35` 的 `enable_depth = int("Depth" in model_name)`），这个开关全靠命名约定。`--task_name` 可取 `Location`/`Placement`/`Unseen`/`all`。

**模型与权重存放**（仓库根，均在 `.gitignore`）：

| 路径 | 说明 |
|---|---|
| `RoboRefer-2B-SFT/` | 2B 基座（`llm/`、`vision_tower/`、`depth_tower/`、`mm_projector/`、`depth_projector/`） |
| `RoboRefer-2B-SFT/data2_lora/` | 本机已下载的 **2B LoRA adapter**（`adapter_model.safetensors` ~144MB） |
| `RoboRefer-2B-SFT-data2-merged/` | merge 后完整 2B 权重（`llava.load(lora, model_base=...)` → `save_pretrained`） |
| `RoboRefer-8B-SFT/` | 8B 基座（计划作对照组，云上待下载 ~18–20GB） |
| `weights/depth_anything_v2_vitl.pth` | Depth-Anything ViT-L |

**模型结构关键模块**：`llava/model/llava_arch.py`（多模态主干装配）、`llava/model/multimodal_encoder/`（视觉/深度 encoder，默认 `paligemma-siglip-so400m-patch14-448`）、`llava/model/multimodal_projector/`（2B 默认 `mlp_downsample_3x3_fix`；**8B 用 `mlp_downsample` + `dynamic_s2`**）。训练入口 `llava/train/train_mem.py`。

**本机 GPU 约束**：RTX **4060 Laptop 8GB** — 可跑 3DGS render/fuse/e2e；**不能**可靠跑 RoboRefer 2B/8B 训练或 API 推理（显存不足）。RoboRefer 推理/微调在 **AutoDL 4090D（24GB）**；本机通过 **SSH 隧道** 调云 API，`bridge/` **无需改代码**。

### 3) RefSpatial-Expand-Bench（基准数据集）

纯数据仓库，包含两种等价格式：HuggingFace `data/*.parquet` 格式（`location` + `placement` 两个 split）和原始 `Location/`、`Placement/` 目录（各含 `image/`、`mask/`、`question.json`）。`question.json` 每条样本有 `id`、`object`（目标描述）、`prompt`（完整指令）、`suffix`（回答格式要求，每个模型不同）、`rgb_path`、`mask_path`、`category`、`step`（推理步数/复杂度）、`scene`（indoor/outdoor）。

评测指标统一是**平均成功率** = 预测点落在 mask 内的比例。不同模型的 prompt 拼法不同（见 `README.md` 里 RoboRefer / Gemini / Molmo 三段示例），以及输出坐标归一化范围不同（RoboRefer 是 0–1，Gemini 是 0–1000，Molmo 是 0–100），解析后都要 scale 回原图尺寸。

## 跨项目的统一事项

- **平台**：Windows 11 + bash（msys/git-bash）。用 Unix 风格路径（正斜杠、`/dev/null`），但注意 3DGS 的用户命令文件里有 Windows 风格 `cd /d` 和反斜杠，属于历史遗留。
- **被 `.gitignore` 排除的大体积产物**：`3DGS/gaussian-splatting/data*/`、`output/`、`colmap-x64-windows-cuda/`、`viewers/bin/`、`submodules/**/build/`；`weights/*.pth`、`RoboRefer-2B-SFT/` 下各 tower/projector 与 `*.safetensors/*.pth/*.pt/*.bin`；`RefSpatial-Expand-Bench/data/` 与 `Location/image/`、`Placement/image/`；以及通配的 `**/*.db`、`**/images.bin`、`**/points3D.bin`、`*.parquet`。改这些目录里的东西前先确认是不是本机生成物。
- **Git 结构**：根仓库在 `master` 分支（commit `f552b52 chore: initial snapshot of GSrefer3D project`）。旧的 `gaussian-splatting/` 目录（含嵌套 `.git`）已废弃，3DGS 代码已迁移到 `3DGS/gaussian-splatting/`。

## 交互偏好

- 用中文回复用户（已在 `~/.claude/CLAUDE.md` 与本项目 memory 中记录）。
- 用户已有的操作记录/备忘在 `gaussian-splatting/安装命令.txt`、`gaussian-splatting/运行训练命令.txt`；涉及 3DGS 操作时优先与这些文件中的命令风格保持一致。

## 整合管线 (bridge/)

把 3DGS 渲染和 RoboRefer 空间指代连接成 "RGB-D 多视角 → 归一化 2D 点 → 世界 3D 点" 的闭环。所有桥接代码在仓库根目录的 `bridge/` 下，独立于两个子项目。依赖：`numpy`、`plyfile`（融合 + 可视化）、`requests`（RoboRefer HTTP 客户端）、**`scipy`**（`filter_views_3dgs.py` 射线透射 KD-tree，须在 **envGS** 内安装）。**不再依赖 `query_model.py` 或 `openai` 包**——客户端直接通过 HTTP POST 调用 RoboRefer API。

### 数据流

```
3DGS render.py --custom_views   →  <root>/{rgb, depth, depth_raw, camera_params}/
bridge/roborefer_client.py      →  <root>/predictions.json   （N 视角 (nx, ny)）
bridge/fuse_multiview.py        →  <root>/fused.json         （单一 P_world + RANSAC inliers）
bridge/visualize.py             →  <root>/marker.ply         （MeshLab/CloudCompare 叠看）
bridge/inject_gaussian_markers.py → output/<model>/point_cloud/iteration_35000/  （SIBR 可视化）
bridge/pipeline.py              →  上述步骤的编排器
```

### 关键文件

- `bridge/unproject.py` — `CameraView` + `Unprojector`。**约定（重要）**：`render.py` 保存的 `view.R` 是 3DGS 内部的 R_c2w（camera-to-world，glm 列主序约定的转置），`from_json()` 里会自动转置为 R_w2c 再使用，**不要直接把 JSON 里的 `rotation` 当 R_w2c 用**。`position` 是世界系下相机中心 C。反投影公式 `P_world = R_w2c.T @ P_cam + C`（即 `R_c2w @ P_cam + C`）。`depth_raw/*.npy` 是 raster 输出的 `expected_invdepth`，需要 `z_cam = 1 / max(inv, eps)`（已在 `Unprojector.sample_depth_raw` 处理）。
- `bridge/roborefer_client.py` — 单/批量模式。内置 HTTP 客户端，直接 POST `{image_url: [base64], depth_url: [base64], enable_depth, text}` 到 RoboRefer `/query` 端点，只需 `requests` 库。批量模式扫 `<root>/rgb/view_*.png` 的 view id，把回答解析为 `[{nx, ny}]` 写进 `predictions.json`，失败视角不中断（记录 `error`）。
- `bridge/fuse_multiview.py` — RANSAC（按 `--inlier-radius` 邻居计数）选最大簇 → **迭代精炼**（geometric median + k*median_dist 剔除远点，循环至收敛）→ 可选 `--ply` 做 snap-to-gaussian。参数：`--no-refine`（禁用迭代精炼）、`--refine-k`（精炼阈值倍数，默认 2.0）、`--exclude <view_id ...>`（手动剔除指向错误物体的视角）。
- `bridge/visualize.py` — 把 fused.json 渲染成带颜色的 ASCII PLY：红=融合点，绿=inlier，黄=outlier。用 MeshLab/CloudCompare 与 point_cloud.ply 叠看。
- `bridge/inject_gaussian_markers.py` — 把融合点注入为红色高斯球，写入新的 `iteration_35000/point_cloud.ply`，可直接在 SIBR viewer 中渲染。
- `bridge/pipeline.py` — 4 阶段编排器：`--stage render|query|fuse|all`。**query 阶段只需 `requests` 库**（检测 `import requests` 是否可用）；不可用时打印切环境指令并 `sys.exit(10)`。
- `bridge/filter_views_3dgs.py` — 训练数据视角过滤：`projections.json` + `fused.json` + 全场 `point_cloud.ply` → `projections_kept.json` / `projections_rejected.json`。拒帧规则：射线 `C→P_world`，簇深度带 `[z_lo,z_hi]`，`T(z_lo) < --ray-min-transmittance`（默认 0.55）→ `ray_foreground_occluded`。须在 envGS 且已装 `scipy`。
- `bridge/gen_training_data.py` — `--stage project` 生成投影；`--stage mask` 需 WSL roborefer + SAM2。
- `bridge/tests/` — pytest 回归。`test_unproject_view000.py` 硬编码金标准（view_000, nx=0.458, ny=0.298 → P_world=[-1.614, 0.703, -0.194]），任何改了渲染端字段、深度语义、外参约定的提交都会立刻报警。
- `bridge/verify_unproject_vs_pointcloud.py` — 验证脚本，检查反投影点到最近高斯点的 NN 距离。

### 跨环境运行（关键）

整条管线跨两个 conda 环境，**不能在一个 Python 进程里跑完**：

| 阶段 | 必须的 env |
|---|---|
| render | `envGS`（diff-gaussian-rasterization） |
| query  | 任何有 `requests` 的 env（推荐 `envGS`）；RoboRefer API server 需在 WSL Ubuntu 的 `roborefer` env 中运行 |
| fuse   | 任何有 numpy + plyfile 的 env（推荐 `envGS`） |

**RoboRefer API 部署**（二选一）：

1. **WSL `roborefer`**（需 ~10GB+ 显存，4060 8GB 易 OOM）：`http://127.0.0.1:25547`（WSL2 端口自动转发至 Windows）。
2. **AutoDL 云 GPU（推荐）**：在云上 `python api.py --port 25547 --host 0.0.0.0`；本机 **SSH 隧道** 后仍用 `--url http://127.0.0.1:25547`：
   ```powershell
   ssh -CNg -L 25547:127.0.0.1:25547 -p <AutoDL端口> root@<AutoDL主机>
   ```
   隧道占一个终端；**另开终端**跑 `run_bridge_e2e.py`。渲染/融合/overlay 产物仍写本机 `3DGS/test2/runs/`。

实际运行示例（data2 场景，电动剃须刀）：

```powershell
# 0) WSL Ubuntu: 起 RoboRefer API server（保持运行）
conda activate roborefer
cd /mnt/e/GSrefer3D/RoboRefer-main/API
python api.py --port 25547 \
     --depth_model_path "/mnt/e/GSrefer3D/weights/depth_anything_v2_vitl.pth" \
     --vlm_model_path "/mnt/e/GSrefer3D/RoboRefer-2B-SFT"

# 1) Windows envGS: 渲染多视角
conda activate envGS
Set-Location E:\GSrefer3D\3DGS
python render.py -m gaussian-splatting/output/data2 --custom_views --output_path E:/GSrefer3D/3DGS/test2

# 2) Windows envGS: 批量调 RoboRefer
Set-Location E:\GSrefer3D
python bridge/roborefer_client.py `
    --root E:/GSrefer3D/3DGS/test2 `
    --url http://127.0.0.1:25547 `
    --prompt "Please point to the electric shaver." `
    --output E:/GSrefer3D/3DGS/test2/predictions.json

# 3) Windows envGS: 融合 + snap
python bridge/fuse_multiview.py `
    --predictions E:/GSrefer3D/3DGS/test2/predictions.json `
    --inlier-radius 10.0 --min-inv 1e-3 --refine-k 1.75 `
    --ply E:/GSrefer3D/3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply `
    --output E:/GSrefer3D/3DGS/test2/fused.json

# 4) 导出 marker.ply（MeshLab/CloudCompare 叠看）
python bridge/visualize.py `
    --fused E:/GSrefer3D/3DGS/test2/fused.json `
    --output E:/GSrefer3D/3DGS/test2/marker.ply

# 5) 注入高斯标记（SIBR 可视化）
python bridge/inject_gaussian_markers.py `
    --ply E:\GSrefer3D\3DGS\gaussian-splatting\output\data2\point_cloud\iteration_30000\point_cloud.ply `
    --fused-json E:\GSrefer3D\3DGS\test2\fused.json `
    --out-iteration-dir E:\GSrefer3D\3DGS\gaussian-splatting\output\data2\point_cloud\iteration_35000

# 6) SIBR 查看（必须从 bin/ 目录启动）
Set-Location E:\GSrefer3D\3DGS\gaussian-splatting\viewers\bin
.\SIBR_gaussianViewer_app.exe -m "E:\GSrefer3D\3DGS\gaussian-splatting\output\data2" --iteration 35000
```

### 一体化端到端命令（`run_bridge_e2e.py`）

**推荐方式**：一条命令完成渲染 → RoboRefer → 融合 → overlay → 模型复制 + 注入的全流程。在仓库根 `E:\GSrefer3D` 下运行（`envGS` 环境，WSL RoboRefer API 需提前起好）：

```powershell
# 完整流程（含渲染）
Set-Location E:\GSrefer3D
python bridge/run_bridge_e2e.py `
    --model-path 3DGS/gaussian-splatting/output/data2 `
    --custom-views-out 3DGS/test2 `
    --prompt "Please point to the brown stuffed rabbit." `
    --snap --url http://127.0.0.1:25547

# 跳过渲染（已有多视角时）— 对比微调前后时用同一 test2 + --skip-render
python bridge/run_bridge_e2e.py `
    --model-path 3DGS/gaussian-splatting/output/data2 `
    --custom-views-out 3DGS/test2 `
    --prompt "Please point to the roll of clear double-sided adhesive tape on the desk." `
    --snap --skip-render --url http://127.0.0.1:25547
```

训前 baseline run：`3DGS/test2/runs/20260519_000313_6c883d56/`。训后换新 `run_id`；换 API 权重（基座 vs `RoboRefer-2B-SFT-data2-merged`）即可 A/B，**无需改 bridge 代码**。

**产物**：写入 `<custom-views-out>/runs/<run_id>/`（predictions.json、fused.json、marker.ply、overlays_rgb/、run_manifest.json、prompt.txt）；模型副本含注入标记写入 `<model-path>_runs/<run_id>/`。流程结束后终端打印可直接复制的 SIBR 启动命令。

**常用参数**：

| 参数 | 说明 |
|---|---|
| `--snap` | 自动选 `--model-path` 下最新迭代的 ply 做 snap（与 `--ply` 二选一） |
| `--ply <path>` | 显式指定 snap 用的 point_cloud.ply（推荐用 iteration_30000 避免与注入目录冲突） |
| `--skip-render` | 跳过 render.py，假定多视角已存在 |
| `--iteration N` | 指定渲染用的 checkpoint iteration |
| `--num-custom-views N` | 渲染视角数（默认 36） |
| `--inlier-radius R` | 融合 RANSAC 邻域半径（场景单位，推荐 5.0–10.0） |
| `--refine-k K` | 融合精炼阈值倍数（推荐 1.5–2.0） |
| `--exclude <id...>` | 融合时剔除指定 view_id |
| `--no-model-bundle` | 不复制模型、不注入，只输出 runs/ 下的预测与融合结果 |
| `--inject-surface-push M` | 标记从内部推出表面（毛绒/内嵌物体时试 0.03–0.1） |
| `--inject-log-scale V` | 标记大小（越不负越大，默认 -3.5） |
| `--visibility-check` | 启用 Qwen 可见性预筛（需 `QWEN_API_KEY`） |

**注入目录冲突处理**：若 `--snap` 自动选到 iteration_35000 且注入也默认写 iteration_35000，脚本自动改写到 iteration_36000 并记录在 run_manifest.json。建议显式用 `--ply .../iteration_30000/point_cloud.ply` 规避。

**仅重注入**（标记不可见时，不重跑推理）：
```powershell
python bridge/inject_gaussian_markers.py `
    --ply "E:\GSrefer3D\3DGS\gaussian-splatting\output\data2_runs\<run_id>\point_cloud\iteration_30000\point_cloud.ply" `
    --fused-json "E:\GSrefer3D\3DGS\test2\runs\<run_id>\fused.json" `
    --out-iteration-dir "E:\GSrefer3D\3DGS\gaussian-splatting\output\data2_runs\<run_id>\point_cloud\iteration_37000" `
    --surface-push 0.06 --log-scale -2.5 --marker-count 80
```

## 当前进度（2026-05）

### 已完成

| 项 | 状态 |
|---|---|
| data2 训练包 `training_data/data2_sft/` | **469 条** RGB-D，`location_point.json` + `image/` + `depth/` |
| 数据集注册 `data2_location` | `llava/data/datasets_mixture.py`（mixture 名；磁盘目录仍叫 `data2_sft`） |
| 导出脚本 | `bridge/export_spatial_train.py`（10 物体 → Bench 同款 tuple 格式） |
| AutoDL 环境 `roborefer` | PyTorch 2.5.1、`pip install -e ".[train,eval]"`、**`huggingface-hub==0.28.1`**（勿 `pip install -U huggingface_hub`） |
| 云上权重 | `/root/autodl-tmp/RoboRefer-2B-SFT`（已齐） |
| **2B LoRA 微调（1 epoch）** | 云上 `runs/train/data2_lora/`，~117 step、~6 min（4090D）；`train_loss≈1.02`，末段 loss ~0.7 |
| LoRA 本机备份 | `RoboRefer-2B-SFT/data2_lora/`（自 `data2_lora.tar.gz` 解压；merge 用根目录 adapter，**不必**用 `checkpoint-117/`） |
| e2e 训前 baseline（2B 基座，胶带） | `3DGS/test2/runs/20260519_000313_6c883d56/` |

### 云上必做补丁（RoboRefer-main，换实例或重传代码时需确认）

1. **`llava/data/datasets_mixture.py`**：`2D_*`/`3D_*` 变量名非法 → 改为 `ds_2d_*` / `ds_3d_*`（`dataset_name` 字符串不变）。
2. **`llava/train/llava_trainer.py`**：`log(self, logs, start_time=None)` — 兼容新版 `transformers`。
3. **`scripts/setups/train.sh` 单卡**：`export SLURM_JOB_GPUS_PER_NODE=1`（AutoDL 无 SLURM，否则默认 8 卡）。

### 进行中 / 下一步

1. **本机验证 2B LoRA**：WSL merge → `RoboRefer-2B-SFT-data2-merged/` → 云或 WSL API → `run_bridge_e2e.py --skip-render` 对比 baseline。
2. **对照实验（计划）**：8B 基座 e2e（不微调）→ 8B LoRA（同 `data2_location`）→ 与 2B 基座 / 2B LoRA 四方对比。
3. **Hold-out 验证**：**双面胶带**不在训练集；看 overlay / `fused.json`，**不必**单独 `val.json`。

### 验证 prompt（胶带，仅 e2e，不进训练集）

`Please point to the roll of clear double-sided adhesive tape on the desk.`

### 2B LoRA 微调命令摘要（AutoDL，cwd=`RoboRefer-main`）

```bash
export SLURM_JOB_GPUS_PER_NODE=1 GLOBAL_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=2
export WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
source scripts/setups/train.sh

torchrun --nproc_per_node=1 llava/train/train_mem.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path /root/autodl-tmp/RoboRefer-2B-SFT \
  --data_mixture data2_location \
  --lora_enable True --lora_llm True --lora_r 64 --lora_alpha 16 \
  --tune_vision_tower False --tune_mm_projector False \
  --tune_depth_tower False --tune_depth_projector False --tune_language_model False \
  --enable_depth True --use_depth_tower True \
  --mm_projector mlp_downsample_3x3_fix --depth_projector mlp_downsample_3x3_fix \
  --image_aspect_ratio dynamic --chat_template qwen2 \
  --output_dir runs/train/data2_lora \
  --num_train_epochs 1 --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 --model_max_length 4096 --save_strategy epoch \
  --report_to none
```

有效 batch = 2×2 = **4**；469 样本 ≈ **117** optimizer step。

### merge LoRA（WSL 或云上）

```python
import llava
model = llava.load("/path/to/data2_lora", model_base="/path/to/RoboRefer-2B-SFT")
model.save_pretrained("/path/to/RoboRefer-2B-SFT-data2-merged")
```

API：`--vlm_model_path` 指向 merged 目录。基座 `RoboRefer-2B-SFT/` **不会被覆盖**；每次 e2e 新建 `runs/<run_id>/`，baseline run **保留**。

### AutoDL 磁盘布局（50GB 数据盘 `autodl-tmp` 够用）

```
/root/autodl-tmp/
  RoboRefer-main/
  RoboRefer-2B-SFT/
  training_data/data2_sft/
  weights/depth_anything_v2_vitl.pth
  RoboRefer-8B-SFT/          # 计划下载 ~20GB
```

大文件放 **数据盘**；conda 在 **系统盘**。关机一般保留 `autodl-tmp`；**释放实例**会清空。

---

## 未来工作 / TODO

### 数据构造与微调

- [x] **3DGS 合成 data2 → `data2_sft` + `data2_location` 注册**
- [x] **2B LoRA 微调（AutoDL 4090D，469 条，1 epoch）**
- [ ] **2B LoRA merge + e2e 对比 baseline（胶带 + 1–2 训练物体）**
- [ ] **8B 基座 e2e 对照（云上 API，不微调）**
- [ ] **8B LoRA 微调（架构用 `dynamic_s2` + `mlp_downsample`，非 2B 参数）**
- [ ] 可选：RefSpatial-Expand-Bench Location 评测（非当前主验证路径）

- **跨场景扩展（方向 B）**：data2 验证可行后，用 data（体育馆）等扩充训练集

### Placement 任务（时间充裕后）

- Placement 类数据需人工在渲染图上标注落点坐标（空位无法用 SAM2 自动生成 mask）
- RoboRefer 本身偏向 Placement 指代，做完 Location 微调后扩展到 Placement 更有说服力

### 大场景测试

- data（体育馆大场景）：测试大场景下小目标空间指代精度，预期作为"挑战性测试集"展示系统局限性
- 改进方向：分层指代（先粗定位区域，再细定位物体）

### 视角过滤优化

- 当前 `--visibility-check`（Qwen 可见性预筛）效果未明显体现，需在遮挡较多的小物体场景下重点测试和调参

---

### 调参建议（基于实测）

- `--inlier-radius` 是**场景单位**，需根据场景尺度调整。**实测推荐 `5.0–10.0`**。
- `--refine-k` 控制迭代精炼剔除阈值（k * median_distance）。**实测推荐 `1.5–2.0`**，k 越小越激进。
- `--min-inv 1e-3` 过滤天空/远平面（`expected_invdepth` 太小 → z 爆炸）。
- **模型与数据必须配套**：test2 的 predictions.json 必须用 data2 的 point_cloud.ply 做 snap，混用会导致坐标偏移。
- **端到端验证结果（data2，电动剃须刀）**：`--inlier-radius 10.0 --refine-k 1.75`，坐标点落在剃须刀附近，相比 data/test1（屋顶，宽泛提示词）误差明显降低。
