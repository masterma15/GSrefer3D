#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

# render.py 可能位于外层 gaussian-splatting/，而 scene / gaussian_renderer 仅在内层子目录。
_root = os.path.dirname(os.path.abspath(__file__))
_inner = os.path.join(_root, "gaussian-splatting")
if os.path.isfile(os.path.join(_inner, "scene", "__init__.py")):
    sys.path.insert(0, _inner)
elif os.path.isfile(os.path.join(_root, "scene", "__init__.py")):
    sys.path.insert(0, _root)

import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
import numpy as np
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def select_views_by_coverage(views, num_views=36):
    """FPS on direction vectors (camera→scene_center) for uniform angular coverage."""
    if len(views) <= num_views:
        return views

    positions = np.array([v.camera_center.cpu().numpy() for v in views])
    scene_center = positions.mean(axis=0)
    dirs = positions - scene_center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.where(norms > 1e-8, norms, 1.0)  # unit vectors

    selected = [0]
    min_dists = np.full(len(views), np.inf)
    for _ in range(num_views - 1):
        last = dirs[selected[-1]]
        # angular distance: 1 - cos(angle) in [0,2]
        d = 1.0 - dirs @ last
        min_dists = np.minimum(min_dists, d)
        min_dists[selected] = -np.inf
        selected.append(int(np.argmax(min_dists)))

    return [views[i] for i in selected]

def render_custom_views(dataset, gaussians, pipeline, background, output_path, num_views: int = 36):
    """渲染若干覆盖视角的 RGB + depth_raw + depth PNG + camera_params（view_000 …）。"""
    # 加载场景
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    
    # 获取所有相机
    all_views = scene.getTrainCameras() + scene.getTestCameras()
    
    selected_views = select_views_by_coverage(all_views, num_views)
    
    # 确保输出目录存在
    rgb_path = os.path.join(output_path, "rgb")
    depth_path = os.path.join(output_path, "depth")
    depth_raw_path = os.path.join(output_path, "depth_raw")
    camera_params_path = os.path.join(output_path, "camera_params")
    makedirs(rgb_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_raw_path, exist_ok=True)
    makedirs(camera_params_path, exist_ok=True)
    
    # 渲染每个视角
    for idx, view in enumerate(tqdm(selected_views, desc="Rendering custom views")):
        # 渲染RGB和深度
        rendering_result = render(view, gaussians, pipeline, background)
        rgb_image = rendering_result["render"]
        depth_image = rendering_result["depth"]
        
        # 保存RGB图像
        rgb_file = os.path.join(rgb_path, f"view_{idx:03d}.png")
        torchvision.utils.save_image(rgb_image, rgb_file)
        
        # 光栅器 depth 缓冲为 alpha 合成的 expected_invdepth（≈ 各高斯 1/z 的加权），不是线性 z。
        depth_np = depth_image.detach().squeeze().cpu().float().numpy()
        if depth_np.ndim != 2:
            depth_np = depth_np.reshape(depth_np.shape[-2], depth_np.shape[-1])
        raw_npy = os.path.join(depth_raw_path, f"view_{idx:03d}.npy")
        np.save(raw_npy, depth_np.astype(np.float32))

        # 归一化深度图并保存为PNG
        depth_min = depth_image.min()
        depth_max = depth_image.max()
        depth_normalized = (depth_image - depth_min) / (depth_max - depth_min + 1e-8)
        # 转换为正确的格式
        depth_save = depth_normalized.expand(3, -1, -1)  # 添加通道维度
        depth_file = os.path.join(depth_path, f"view_{idx:03d}.png")
        torchvision.utils.save_image(depth_save, depth_file)
        
        # 保存相机参数
        camera_params = {
            "position": view.camera_center.cpu().numpy().tolist(),
            "rotation": view.R.tolist(),
            "fov_x": float(view.FoVx),
            "fov_y": float(view.FoVy),
            "width": view.image_width,
            "height": view.image_height,
            "depth_raw_relative": os.path.relpath(raw_npy, start=os.path.dirname(camera_params_path)).replace("\\", "/"),
            "depth_raster_kind": "expected_invdepth",
            "depth_z_cam_note": (
                "Raster buffer is blended inverse depth (see diff-gaussian-rasterization forward). "
                "For pinhole unprojection use z_cam = 1.0 / max(inv, eps) as an approximation."
            ),
        }
        camera_file = os.path.join(camera_params_path, f"view_{idx:03d}.json")
        with open(camera_file, "w") as f:
            json.dump(camera_params, f, indent=2)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--custom_views", action="store_true", help="Render multi-view RGB-D pack under --output_path")
    parser.add_argument(
        "--num_custom_views",
        default=36,
        type=int,
        help="How many views to select (FPS on camera directions). Default 36.",
    )
    parser.add_argument("--output_path", default="test1", type=str, help="Output path for custom views")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.custom_views:
        # Render custom views with RGB-D
        with torch.no_grad():
            dataset = model.extract(args)
            gaussians = GaussianModel(dataset.sh_degree)
            pipeline_args = pipeline.extract(args)
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            render_custom_views(
                dataset, gaussians, pipeline_args, background, args.output_path,
                num_views=int(args.num_custom_views),
            )
    else:
        # Render standard sets
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)