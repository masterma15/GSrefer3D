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

把 3DGS 渲染和 RoboRefer 空间指代连接成闭环。`P_world` 只是生成 2D mask 的种子，3D 结果是视锥投票后的物体高斯。所有桥接代码在仓库根目录的 `bridge/` 下。依赖：`numpy`、`plyfile`、`requests`、**`scipy`**（`filter_views_3dgs.py` 射线透射，须在 **envGS**）。客户端 HTTP POST 调 RoboRefer API。

### 数据流

```
bridge/run_bridge_e2e.py 编排（subprocess / WSL）：
3DGS render.py --custom_views   →  <root>/{rgb, depth, depth_raw, camera_params}/
bridge/roborefer_client.py      →  runs/<id>/predictions.json
bridge/fuse_multiview.py        →  fused.json（P_world 种子）
bridge/filter_views_3dgs.py     →  projections_kept.json
bridge/gen_training_data.py     →  mask/（Grounding DINO + SAM2）
bridge/frustum_segment.py       →  gaussian_seg.json         （物体高斯）
bridge/inject_gaussian_seg.py   →  iteration_seg_<name>/     （SIBR）
bridge/eval_gaussian_seg.py     →  有手标 OBB 时：质心∈OBB、框内占比、2D mask 精度
```

### 关键文件

- `bridge/unproject.py` — `CameraView` + `Unprojector`。**约定（重要）**：`render.py` 保存的 `view.R` 是 3DGS 内部的 R_c2w（camera-to-world，glm 列主序约定的转置），`from_json()` 里会自动转置为 R_w2c 再使用，**不要直接把 JSON 里的 `rotation` 当 R_w2c 用**。`position` 是世界系下相机中心 C。反投影公式 `P_world = R_w2c.T @ P_cam + C`（即 `R_c2w @ P_cam + C`）。`depth_raw/*.npy` 是 raster 输出的 `expected_invdepth`，需要 `z_cam = 1 / max(inv, eps)`（已在 `Unprojector.sample_depth_raw` 处理）。
- `bridge/roborefer_client.py` — 单/批量模式。内置 HTTP 客户端，直接 POST `{image_url: [base64], depth_url: [base64], enable_depth, text}` 到 RoboRefer `/query` 端点，只需 `requests` 库。批量模式扫 `<root>/rgb/view_*.png` 的 view id，把回答解析为 `[{nx, ny}]` 写进 `predictions.json`，失败视角不中断（记录 `error`）。
- `bridge/fuse_multiview.py` — RANSAC + 迭代精炼得到 **mask 种子** `P_world`（不是 3D 评测结果）。`--ply` 供 `--depth-mode ray` 沿射线查高斯。`--no-refine`、`--refine-k`。
- `bridge/frustum_segment.py` — 多视 mask 视锥投票（默认 `--min-vote-frac 0.5`）→ `gaussian_seg.json`。
- `bridge/inject_gaussian_seg.py` — 选中高斯改色 + 手标 OBB 线框，写出独立 iteration（不写回 30000）。
- `bridge/eval_gaussian_seg.py` — **现行 3D 口径**：投票质心是否在 OBB 内、选中高斯框内占比、2D mask 重投影精度。
- `bridge/run_bridge_e2e.py` — 唯一用户入口：render → query → fuse 种子 → project/filter → WSL DINO+SAM → refine → 视锥投票 → SIBR 注入。API 不可用 `sys.exit(10)`；WSL/mask 环境缺 `sys.exit(11)`。
- `bridge/e2e_stages.py` — 渲染/融合/WSL mask 预检，供 e2e 调用，不是入口。
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

# 3) 融合种子（P_world 只给后续 mask 用）
python bridge/fuse_multiview.py `
    --predictions E:/GSrefer3D/3DGS/test2/predictions.json `
    --inlier-radius 10.0 --min-inv 1e-3 --refine-k 1.75 `
    --ply E:/GSrefer3D/3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply `
    --output E:/GSrefer3D/3DGS/test2/fused.json

# 4) 视锥投票 + SIBR 实例可视化 + 评测
python bridge/frustum_segment.py --obj-dir training_data/data2_shaver
python bridge/inject_gaussian_seg.py --seg training_data/data2_shaver/gaussian_seg.json
python bridge/eval_gaussian_seg.py --obj-glob training_data/data2_* --output docs/results_gaussian_seg.json

# 5) SIBR（必须从 viewers/bin 启动）
Set-Location E:\GSrefer3D\3DGS\gaussian-splatting\viewers\bin
.\SIBR_gaussianViewer_app.exe -m "E:\GSrefer3D\3DGS\gaussian-splatting\output\data2" --iteration seg_electric_shaver
```

### 一体化端到端命令（`run_bridge_e2e.py`）

**推荐方式**：一条命令从已有 3DGS 模型 + 提示词得到该物体的微调包和视锥投票点云（不含重建）。`--skip-render` = `--from query`，复用 `--custom-views-out` 下已有视角（默认 `3DGS/test2`，可改）。

```powershell
Set-Location E:\GSrefer3D
python bridge/run_bridge_e2e.py `
    --model-path 3DGS/gaussian-splatting/output/data2 `
    --custom-views-out 3DGS/test2 `
    --prompt "Please point to the brown stuffed rabbit." `
    --object "plush rabbit" `
    --name data2_rabbit `
    --url http://127.0.0.1:25547

python bridge/run_bridge_e2e.py `
    --model-path 3DGS/gaussian-splatting/output/data2 `
    --custom-views-out 3DGS/test2 `
    --prompt "Please point to the roll of clear double-sided adhesive tape on the desk." `
    --object "roll of clear double-sided tape" `
    --name data2_tape `
    --skip-render --url http://127.0.0.1:25547
```

训前 baseline run：`3DGS/test2/runs/20260519_000313_6c883d56/`。

**产物**：`training_data/<name>/`（`question.json`、`mask/`、`gaussian_seg.json`）；`<views>/runs/<run_id>/`（predictions、fused 种子）；`--model-path/point_cloud/iteration_seg_<name>/`。不写 `data2_sft`。

**常用参数**：`--from render|query|fuse|filter|mask|vote`、`--pause-after-mask`、`--bbox-key`、`--out-dir`、`--ply`、`--skip-render`、`--clean`。

## 当前进度（2026-05）

> 实验时间线与数字见 [README.md](README.md) · [docs/RESULTS.md](docs/RESULTS.md)。

### 已完成（主线闭环）

| 项 | 状态 |
|---|---|
| data2 训练包 `training_data/data2_sft/` | **469 条** RGB-D，`location_point.json` + `image/` + `depth/` |
| 数据集注册 `data2_location` | `llava/data/datasets_mixture.py`（mixture 名；磁盘目录仍叫 `data2_sft`） |
| 导出脚本 | `bridge/export_spatial_train.py`（10 物体 → Bench 同款 tuple 格式） |
| AutoDL 环境 `roborefer` | PyTorch 2.5.1、`pip install -e ".[train,eval]"`、**`huggingface-hub==0.28.1`**（勿 `pip install -U huggingface_hub`） |
| 云上权重 | `/root/autodl-tmp/RoboRefer-2B-SFT`（已齐） |
| **2B LoRA 微调（1 epoch）** | 云上 `runs/train/data2_lora/`，~117 step、~6 min（4090D）；`train_loss≈1.02`，末段 loss ~0.7 |
| LoRA 本机备份 | `RoboRefer-2B-SFT/data2_lora/`（自 `data2_lora.tar.gz` 解压；merge 用根目录 adapter，**不必**用 `checkpoint-117/`） |
| **2B LoRA merge + merged API 权重** | `RoboRefer-2B-SFT-data2-merged/` |
| **Base vs LoRA e2e** | 域内 10 物体 2D：LoRA median L2 ≤ Base（10/10）；见 `docs/results_2d_eval.json` |
| **Hold-out 双面胶带** | 不进 469 SFT；定性 2D overlay |
| **深度消融** | 3DGS `depth_raw` vs DAV2；`docs/depth_compare_batch.json` |
| **人工 OBB（11 物体）** | `docs/bbox_data2.json` + `docs/bbox_labels/` |
| **3D 实例评估** | 视锥投票质心 ∈ OBB + 框内高斯占比；`bridge/eval_gaussian_seg.py` |
| **SIBR 实例可视化** | `inject_gaussian_seg.py`（品红=框内选中，亮绿=框外，青=OBB） |
| **RefSpatial-Expand-Bench（OOD）** | Location Base **50.21%** / LoRA **45.64%**；作域适应 trade-off 对照，非主指标 |
| e2e 训前 baseline（2B 基座，胶带） | `3DGS/test2/runs/20260519_000313_6c883d56/`（保留对比） |

**口径说明：** `P_world` / `fused.json` 只作 mask 种子；**3D 主指标**是视锥投票高斯（`results_gaussian_seg.json`：质心 10/10 在手标 OBB 内，框内占比 0.993）。训练数据与 §5 2D 评估的种子来自历史 **invdepth + snap**；仓库 fuse CLI 默认已是 `depth_mode=ray`。不要把种子点 OBB 命中表当 3D 结果。

### 云上必做补丁（RoboRefer-main，换实例或重传代码时需确认）

1. **`llava/data/datasets_mixture.py`**：`2D_*`/`3D_*` 变量名非法 → 改为 `ds_2d_*` / `ds_3d_*`（`dataset_name` 字符串不变）。
2. **`llava/train/llava_trainer.py`**：`log(self, logs, start_time=None)` — 兼容新版 `transformers`。
3. **`scripts/setups/train.sh` 单卡**：`export SLURM_JOB_GPUS_PER_NODE=1`（AutoDL 无 SLURM，否则默认 8 卡）。

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

建议顺序：**近期收尾 → 8B 对照 → 训练管线升级 / 跨场景 → 系统鲁棒性**。本机 4060 跑 RoboRefer 推理/训练用 **AutoDL + SSH 隧道**；3DGS/bridge 在本机 `envGS`。

### 一、近期收尾（优先）

- [ ] **展示与材料**：简历/答辩与 README、RESULTS 对齐现行 `eval_gaussian_seg` 口径。

### 二、对照实验（下一档主实验）

- [x] **3DGS 合成 data2 → `data2_sft` + `data2_location` 注册**
- [x] **2B LoRA 微调（AutoDL 4090D，469 条，1 epoch）**
- [x] **2B LoRA merge + e2e 对比**（域内 2D + hold-out 胶带 + 视锥投票 3D）
- [ ] **8B 基座 e2e**（云上 API，`RoboRefer-8B-SFT`，不微调；同 test2 / prompt / 融合策略）
- [ ] **8B LoRA 微调**（同 `data2_location`；**`dynamic_s2` + `mlp_downsample`**，非 2B 的 `mlp_downsample_3x3_fix`）
- [ ] **四方对比表**：2B Base / 2B LoRA / 8B Base / 8B LoRA → 域内 2D + OBB + 胶带 hold-out

前置：云上拉齐 `RoboRefer-8B-SFT`（~18–20GB，见下方磁盘布局）。

### 三、数据与训练（方法升级，工作量大）

- [ ] **训练融合策略升级**：当前 469 条 SFT 锚点来自 **invdepth + snap**；可试 **ray 融合** 重导 `P_world` → `gen_training_data` project / SAM2 / export → 再 LoRA。
- [ ] **跨场景扩展（方向 B）**：data2 稳定后，用 **data（体育馆）** 扩训练集；单独报告大场景局限。
- [ ] **Placement 任务**：需人工标落点（空位无法 SAM2 自动生成 mask）；Location 闭环后再做更有说服力。

### 四、系统鲁棒性（中远期）

- [ ] **大场景挑战测试**：data 小目标指代；可探索 **分层指代**（先区域后物体）。
- [ ] **视角过滤**：`filter_views_3dgs.py` 射线透射拒帧（给 mask 用，不是进 VLM 前预筛）。
- [ ] **文档与默认行为一致**：新人勿混用历史 e2e 的 invdepth+snap run 与当前 fuse 默认 ray。

### 五、可选 / 非主路径

- [x] **RefSpatial-Expand-Bench**（已跑；LoRA 域内涨、OOD Location 略降，作讨论即可）
- [ ] RefSpatial / 其他 OOD 深挖（时间允许）

### 路线图（简）

```
近期收尾 (hair_clip / 可选 Base-ray / 材料)
    → 8B Base e2e → 8B LoRA → 四方对比
    → [可选] ray 重导 SFT + 再 LoRA
    → 跨场景 data / Placement
    → 大场景 + 可见性/拒帧优化
```

---

### 调参建议（基于实测）

- `--inlier-radius` 是**场景单位**，需根据场景尺度调整。**实测推荐 `5.0–10.0`**。
- `--refine-k` 控制迭代精炼剔除阈值（k * median_distance）。**实测推荐 `1.5–2.0`**，k 越小越激进。
- `--min-inv 1e-3` 过滤天空/远平面（`expected_invdepth` 太小 → z 爆炸）。
- **模型与数据必须配套**：test2 的 predictions.json 必须用 data2 的 point_cloud.ply 做 snap，混用会导致坐标偏移。
- **端到端验证结果（data2，电动剃须刀）**：`--inlier-radius 10.0 --refine-k 1.75`，坐标点落在剃须刀附近，相比 data/test1（屋顶，宽泛提示词）误差明显降低。
