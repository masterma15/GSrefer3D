# 上游依赖安装（本地克隆，不进 Git）

本仓库采用 **薄仓库（thin repo）** 策略：Git 只保留 `bridge/`、文档、补丁与 `3DGS/render.py`；**完整** 3DGS 与 RoboRefer 在本地克隆。

## 推荐目录布局

```text
3DGS-VLM/                    ← 本 Git 仓库
  bridge/
  docs/
  patches/
  3DGS/
    render.py                ← 本仓库提供
    environment-envGS.yml
    gaussian-splatting/      ← git clone Inria（本地，.gitignore）
  RoboRefer-main/             ← git clone Zhoues/RoboRefer（本地，.gitignore）
  weights/                     ← 下载权重（.gitignore）
  RoboRefer-2B-SFT/           ← HF 权重（.gitignore）
  training_data/data2_sft/   ← 合成训练集（.gitignore）
  RefSpatial-Expand-Bench/   ← 可选评测（图像 .gitignore）
```

## 1. 3D Gaussian Splatting

```bash
cd 3DGS
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
# 若用加速光栅，按官方 README 切 3dgs_accel 分支并安装对应 submodule
```

将本仓库 `3DGS/render.py` 放在 `3DGS/` 下（已包含则跳过）。

若使用 **accel** 光栅器，覆盖：

```text
gaussian-splatting/gaussian_renderer/__init__.py
  ← patches/3dgs/gaussian_renderer__init__.py
```

环境：

```bash
conda env create -f environment-envGS.yml   # 或 environment.yml
conda activate envGS
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
# 可选: submodules/fused-ssim
```

**不从 Git 下载：**

| 内容 | 获取方式 |
|------|----------|
| 训练数据 `data2/` | 自备 COLMAP 照片集 |
| `output/data2/point_cloud/` | 本地 `train.py` |
| SIBR `viewers/bin/` | [官方 binaries](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) |
| COLMAP CUDA 构建 | 本地编译或 Windows 预编译包 |

## 2. RoboRefer

```bash
git clone https://github.com/Zhoues/RoboRefer.git RoboRefer-main
cd RoboRefer-main
# 按上游 env_setup.sh / README 安装 roborefer 环境
```

应用补丁说明：[patches/roborefer/INTEGRATION.md](../patches/roborefer/INTEGRATION.md)。

**权重（Hugging Face，不进 Git）：**

| 文件 | 路径 |
|------|------|
| RoboRefer-2B-SFT | `RoboRefer-2B-SFT/` |
| Depth Anything V2 ViT-L | `weights/depth_anything_v2_vitl.pth` |
| data2 LoRA（可选） | `RoboRefer-2B-SFT/data2_lora/` → merge 为 `RoboRefer-2B-SFT-data2-merged/` |

API：

```bash
cd RoboRefer-main/API
python api.py --port 25547 \
  --depth_model_path ../../weights/depth_anything_v2_vitl.pth \
  --vlm_model_path ../../RoboRefer-2B-SFT-data2-merged
```

## 3. 端到端（本仓库 bridge）

```bash
# Windows envGS + RoboRefer API（WSL 或 SSH 隧道）
python bridge/run_bridge_e2e.py \
  --model-path 3DGS/gaussian-splatting/output/data2 \
  --custom-views-out 3DGS/test2 \
  --prompt "Please point to ..." \
  --snap --skip-render --url http://127.0.0.1:25547
```

## 4. 若你曾把整个 RoboRefer / 旧版 gaussian-splatting 提交进 Git

开源前建议从索引移除（保留本地文件）：

```bash
git rm -r --cached RoboRefer-main/ gaussian-splatting/ 2>/dev/null
git rm -r --cached 3DGS/gaussian-splatting/ 3DGS/test2/ 2>/dev/null
```

然后只 `git add bridge/ docs/ patches/ 3DGS/render.py 3DGS/environment-envGS.yml demo/ README.md LICENSE`

## 5. RefSpatial-Expand-Bench（可选）

```bash
git clone https://huggingface.co/datasets/BAAI/RefSpatial-Bench RefSpatial-Expand-Bench
```

评测脚本在 `RoboRefer-main/Evaluation/`；图像体积大，不 push 到本仓库。
