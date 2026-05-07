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

import torch
from scene import Scene
import os
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

def select_views_by_coverage(views, num_views=12):
    """基于覆盖优先策略选择特定数量的视角"""
    if len(views) <= num_views:
        return views
    
    # 计算所有相机的位置
    camera_positions = []
    for view in views:
        camera_positions.append(view.camera_center.cpu().numpy())
    camera_positions = np.array(camera_positions)
    
    # 计算场景中心点
    scene_center = np.mean(camera_positions, axis=0)
    
    # 计算每个相机到中心点的距离
    distances = np.linalg.norm(camera_positions - scene_center, axis=1)
    
    # 按距离排序，选择不同距离的相机以确保覆盖
    sorted_indices = np.argsort(distances)
    
    # 均匀选择12个视角
    step = len(sorted_indices) // num_views
    selected_indices = [sorted_indices[i * step] for i in range(num_views)]
    
    # 如果数量不足，添加剩余的
    while len(selected_indices) < num_views:
        for i in range(len(sorted_indices)):
            if i not in selected_indices:
                selected_indices.append(i)
                if len(selected_indices) == num_views:
                    break
    
    return [views[i] for i in selected_indices]

def render_custom_views(dataset, gaussians, pipeline, background, output_path):
    """渲染12个特定视角的RGB和深度图"""
    # 加载场景
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    
    # 获取所有相机
    all_views = scene.getTrainCameras() + scene.getTestCameras()
    
    # 选择12个视角
    selected_views = select_views_by_coverage(all_views, 12)
    
    # 确保输出目录存在
    rgb_path = os.path.join(output_path, "rgb")
    depth_path = os.path.join(output_path, "depth")
    camera_params_path = os.path.join(output_path, "camera_params")
    makedirs(rgb_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
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
            "height": view.image_height
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
    parser.add_argument("--custom_views", action="store_true", help="Render 12 custom views with RGB-D output")
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
            render_custom_views(dataset, gaussians, pipeline_args, background, args.output_path)
    else:
        # Render standard sets
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)