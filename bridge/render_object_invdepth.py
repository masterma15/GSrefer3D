#!/usr/bin/env python3
"""只光栅投票选中的高斯，得到与 depth_raw 同定义的期望逆深度。

必须在装有 diff-gaussian-rasterization 的环境里跑（本机为 envGS）。
不改场景 ply，只在这一次渲染里把未选中高斯的不透明度暂置为 0。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_GS = _REPO / "3DGS" / "gaussian-splatting"
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))
if str(_GS) not in sys.path:
    sys.path.insert(0, str(_GS))

from visible_mask import (  # noqa: E402
    DEFAULT_HIGH_RATIO,
    DEFAULT_INV_EPS,
    DEFAULT_LOW_RATIO,
    DEFAULT_REL_TOL,
    trim_object_dir,
    world_to_view_matrix,
)


class _Pipe:
    """与官方 PipelineParams 默认值一致，避免为这一次渲染解析训练参数。"""

    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False


def minicam_from_camera_json(cam: dict):
    """用自定义视角 JSON 组装光栅器相机，外参与 render.py 写出时一致。"""
    import torch
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    rotation = np.asarray(cam["rotation"], dtype=np.float64)
    center = np.asarray(cam["position"], dtype=np.float64)
    w2c = world_to_view_matrix(rotation, center)
    world_view = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
    proj = getProjectionMatrix(
        znear=0.01,
        zfar=100.0,
        fovX=float(cam["fov_x"]),
        fovY=float(cam["fov_y"]),
    ).transpose(0, 1).cuda()
    full = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(
        int(cam["width"]),
        int(cam["height"]),
        float(cam["fov_y"]),
        float(cam["fov_x"]),
        0.01,
        100.0,
        world_view,
        full,
    )


def load_gaussian_model(ply_path: Path, sh_degree: int = 3):
    """从投票时用的 ply 加载高斯。编号与 gaussian_seg.json 的 indices 对齐。"""
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(sh_degree)
    model.load_ply(str(ply_path))
    return model


def render_selected_invdepth(model, cam: dict, indices: np.ndarray) -> np.ndarray:
    """返回物体期望逆深度，形状 (H, W)。未选中高斯在这次渲染中不透明度为 0。"""
    import torch
    from gaussian_renderer import render

    if indices.size == 0:
        h, w = int(cam["height"]), int(cam["width"])
        return np.zeros((h, w), dtype=np.float32)
    n = int(model.get_xyz.shape[0])
    if int(indices.min()) < 0 or int(indices.max()) >= n:
        raise IndexError(f"高斯编号超出 ply 范围 0..{n - 1}")

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    viewpoint = minicam_from_camera_json(cam)
    saved = model._opacity.data.detach().clone()
    try:
        model._opacity.data.fill_(-40.0)
        model._opacity.data[torch.as_tensor(indices, device="cuda", dtype=torch.long)] = saved[
            torch.as_tensor(indices, device="cuda", dtype=torch.long)
        ]
        with torch.no_grad():
            depth = render(viewpoint, model, _Pipe(), background)["depth"]
        arr = depth.detach().squeeze().float().cpu().numpy()
    finally:
        model._opacity.data.copy_(saved)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.shape != (int(cam["height"]), int(cam["width"])):
        raise RuntimeError(
            f"物体逆深度形状 {arr.shape} 与相机 {(cam['height'], cam['width'])} 不一致"
        )
    return arr.astype(np.float32)


def render_object_depths(
    obj_dir: Path,
    views_root: Path,
    *,
    ply: Path | None = None,
) -> dict[str, np.ndarray]:
    """为目录中每张掩码渲染物体逆深度。"""
    seg_path = obj_dir / "gaussian_seg.json"
    if not seg_path.is_file():
        raise FileNotFoundError(f"缺少 {seg_path}，请先投票")
    seg = json.loads(seg_path.read_text(encoding="utf-8"))
    indices = np.asarray(seg["indices"], dtype=np.int64)
    ply_path = ply or Path(seg["ply"])
    model = load_gaussian_model(ply_path)
    out: dict[str, np.ndarray] = {}
    for mask_path in sorted((obj_dir / "mask").glob("view_*.png")):
        view_id = mask_path.stem.split("_")[-1].zfill(3)
        cam_path = views_root / "camera_params" / f"view_{view_id}.json"
        if not cam_path.is_file():
            continue
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        out[view_id] = render_selected_invdepth(model, cam, indices)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="用物体逆深度修剪掩码并写出遮挡比例")
    ap.add_argument("--obj-dir", type=Path, required=True)
    ap.add_argument("--views-root", type=Path, required=True)
    ap.add_argument("--ply", type=Path, default=None)
    ap.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    ap.add_argument("--inv-eps", type=float, default=DEFAULT_INV_EPS)
    ap.add_argument("--low-ratio", type=float, default=DEFAULT_LOW_RATIO)
    ap.add_argument("--high-ratio", type=float, default=DEFAULT_HIGH_RATIO)
    args = ap.parse_args()
    obj_dir = args.obj_dir if args.obj_dir.is_absolute() else (_REPO / args.obj_dir).resolve()
    views_root = args.views_root if args.views_root.is_absolute() else (_REPO / args.views_root).resolve()
    depths = render_object_depths(obj_dir, views_root, ply=args.ply)
    records = trim_object_dir(
        obj_dir,
        views_root,
        depths,
        rel_tol=args.rel_tol,
        inv_eps=args.inv_eps,
        low=args.low_ratio,
        high=args.high_ratio,
    )
    kinds: dict[str, int] = {}
    for rec in records:
        kinds[rec["kind"]] = kinds.get(rec["kind"], 0) + 1
    print(f"[visible] {obj_dir.name} {len(records)} views {kinds}", flush=True)


if __name__ == "__main__":
    main()
