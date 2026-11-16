#!/usr/bin/env python3
"""用原射线的物体权重标出可见像素。全图都算，不读 SAM 掩码。

不透明度保持不动，遮挡物仍留在射线上。只改这一次渲染的颜色：
选中高斯为白，其余为黑，背景为 0。像素值是物体高斯的权重和 Σ αT，
T 仍由全部高斯连乘。

写出 ``<物体目录>/ray_weight_overlay/view_XXX.png``。
权重大于等于 0.5 盖绿色，0.15 到 0.5 盖橙色，其余不盖。
不改场景 ply，也不改 ``mask/`` 和 ``gaussian_seg.json``。
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

from render_object_invdepth import (  # noqa: E402
    _Pipe,
    load_gaussian_model,
    minicam_from_camera_json,
)

# 3DGS 直流颜色是 0.5 + C0 * dc。白、黑反解出直流系数。
_SH_C0 = 0.28209479177387814
_DC_WHITE = (1.0 - 0.5) / _SH_C0
_DC_BLACK = (0.0 - 0.5) / _SH_C0

VISIBLE_WEIGHT = 0.5
PARTIAL_WEIGHT = 0.15


def render_object_weight(model, cam: dict) -> np.ndarray:
    """返回物体权重图，形状 (H, W)，范围大约在 0 到 1。"""
    import torch
    from gaussian_renderer import render

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    viewpoint = minicam_from_camera_json(cam)
    with torch.no_grad():
        image = render(viewpoint, model, _Pipe(), background)["render"]
    arr = image.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    if arr.shape != (int(cam["height"]), int(cam["width"])):
        raise RuntimeError(
            f"权重图形状 {arr.shape} 与相机 {(cam['height'], cam['width'])} 不一致"
        )
    return arr.astype(np.float32)


def _prepare_white_set(model, indices: np.ndarray):
    """涂白选中高斯。返回 (idx_np, saved_dc, saved_rest, saved_opacity)。"""
    import torch

    n = int(model.get_xyz.shape[0])
    idx_np = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx_np.size and (int(idx_np.min()) < 0 or int(idx_np.max()) >= n):
        raise IndexError(f"高斯编号超出 ply 范围 0..{n - 1}")
    saved_dc = model._features_dc.data.detach().clone()
    saved_rest = model._features_rest.data.detach().clone()
    saved_opacity = model._opacity.data.detach().clone()
    model._features_rest.data.zero_()
    model._features_dc.data.fill_(_DC_BLACK)
    if idx_np.size:
        idx = torch.as_tensor(idx_np, device=model._features_dc.device, dtype=torch.long)
        model._features_dc.data[idx] = _DC_WHITE
    return idx_np, saved_dc, saved_rest, saved_opacity


def _restore_appearance(model, saved_dc, saved_rest, saved_opacity) -> None:
    model._features_dc.data.copy_(saved_dc)
    model._features_rest.data.copy_(saved_rest)
    model._opacity.data.copy_(saved_opacity)


def render_weight_maps(
    ply: Path,
    indices: np.ndarray,
    cams: list[dict],
) -> list[np.ndarray]:
    """选中高斯涂白、其余涂黑，遮挡物仍留在射线上。返回每张相机的权重图。"""
    model = load_gaussian_model(ply)
    saved = None
    maps: list[np.ndarray] = []
    try:
        _, saved_dc, saved_rest, saved_opacity = _prepare_white_set(model, indices)
        saved = (saved_dc, saved_rest, saved_opacity)
        for cam in cams:
            maps.append(render_object_weight(model, cam))
    finally:
        if saved is not None:
            _restore_appearance(model, *saved)
    return maps


def render_footprint_and_stain(
    ply: Path,
    indices: np.ndarray,
    cams: list[dict],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """足迹只保留白集的不透明度；染色保留全部不透明度。各返回与相机顺序相同的图。"""
    import torch

    idx_np = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx_np.size == 0:
        raise ValueError(
            "白集为空，无法渲染 footprint/stain（全部不透明度置空会触发 CUDA 非法访问）"
        )

    model = load_gaussian_model(ply)
    saved = None
    stains: list[np.ndarray] = []
    footprints: list[np.ndarray] = []
    try:
        idx_np, saved_dc, saved_rest, saved_opacity = _prepare_white_set(model, idx_np)
        saved = (saved_dc, saved_rest, saved_opacity)
        for cam in cams:
            stains.append(render_object_weight(model, cam))
        keep = torch.zeros(int(model.get_xyz.shape[0]), dtype=torch.bool, device=model._opacity.device)
        keep[torch.as_tensor(idx_np, device=model._opacity.device, dtype=torch.long)] = True
        model._opacity.data[~keep] = -50.0
        for cam in cams:
            footprints.append(render_object_weight(model, cam))
    finally:
        if saved is not None:
            _restore_appearance(model, *saved)
    return footprints, stains


def _font(size: int):
    from PIL import ImageFont

    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_weight_overlay(rgb: np.ndarray, weight: np.ndarray, view_id: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    visible = weight >= VISIBLE_WEIGHT
    partial = (weight >= PARTIAL_WEIGHT) & ~visible
    out = np.array(rgb, copy=True)
    out[visible] = np.array((46, 160, 67), dtype=np.uint8)
    out[partial] = np.array((214, 128, 18), dtype=np.uint8)
    image = Image.fromarray(out, mode="RGB")
    draw = ImageDraw.Draw(image)
    title = (
        f"view_{view_id}  可见(≥{VISIBLE_WEIGHT:g}) {int(visible.sum())}"
        f"  部分({PARTIAL_WEIGHT:g}–{VISIBLE_WEIGHT:g}) {int(partial.sum())}"
    )
    draw.rectangle([0, 0, image.width, 52], fill=(24, 24, 24))
    draw.text((12, 8), title, fill=(255, 255, 255), font=_font(32))
    return np.array(image)


def export_object(obj_dir: Path, views_root: Path, ply: Path) -> int:
    import torch
    from PIL import Image

    seg_path = obj_dir / "gaussian_seg.json"
    if not seg_path.is_file():
        raise FileNotFoundError(f"缺少 {seg_path}")
    seg = json.loads(seg_path.read_text(encoding="utf-8"))
    indices = np.asarray(seg["indices"], dtype=np.int64)
    model = load_gaussian_model(ply)
    n = int(model.get_xyz.shape[0])
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n):
        raise IndexError(f"高斯编号超出 ply 范围 0..{n - 1}")

    cam_paths = sorted((views_root / "camera_params").glob("view_*.json"))
    if not cam_paths:
        raise FileNotFoundError(f"{views_root / 'camera_params'} 下没有相机")

    out_dir = obj_dir / "ray_weight_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_dc = model._features_dc.data.detach().clone()
    saved_rest = model._features_rest.data.detach().clone()
    records = []
    try:
        model._features_rest.data.zero_()
        model._features_dc.data.fill_(_DC_BLACK)
        if indices.size:
            idx = torch.as_tensor(indices, device=model._features_dc.device, dtype=torch.long)
            model._features_dc.data[idx] = _DC_WHITE
        for cam_path in cam_paths:
            view_id = cam_path.stem.split("_")[-1].zfill(3)
            rgb_path = views_root / "rgb" / f"view_{view_id}.png"
            if not rgb_path.is_file():
                print(f"[skip] {obj_dir.name} view_{view_id}: 没有渲染图")
                continue
            cam = json.loads(cam_path.read_text(encoding="utf-8"))
            weight = render_object_weight(model, cam)
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            if rgb.shape[:2] != weight.shape:
                raise RuntimeError(
                    f"view_{view_id} 渲染图 {rgb.shape} 与权重 {weight.shape} 不一致"
                )
            Image.fromarray(draw_weight_overlay(rgb, weight, view_id)).save(
                out_dir / f"view_{view_id}.png"
            )
            visible = int((weight >= VISIBLE_WEIGHT).sum())
            partial = int(((weight >= PARTIAL_WEIGHT) & (weight < VISIBLE_WEIGHT)).sum())
            records.append(
                {
                    "view_id": view_id,
                    "visible_px": visible,
                    "partial_px": partial,
                    "max_weight": float(weight.max()),
                }
            )
    finally:
        model._features_dc.data.copy_(saved_dc)
        model._features_rest.data.copy_(saved_rest)

    (obj_dir / "ray_weight.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    n_vis = sum(1 for rec in records if rec["visible_px"] >= 500)
    print(
        f"[weight] {obj_dir.name} {len(records)} views, "
        f"visible_px>=500 on {n_vis} -> {out_dir}",
        flush=True,
    )
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description="全图渲染物体射线权重并叠加到原图")
    ap.add_argument("--obj-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--views-root", type=Path, default=Path("3DGS/test2"))
    ap.add_argument(
        "--ply",
        type=Path,
        default=Path("3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply"),
    )
    args = ap.parse_args()
    views_root = args.views_root if args.views_root.is_absolute() else (_REPO / args.views_root).resolve()
    ply = args.ply if args.ply.is_absolute() else (_REPO / args.ply).resolve()
    total = 0
    for raw in args.obj_dir:
        obj_dir = raw if raw.is_absolute() else (_REPO / raw).resolve()
        total += export_object(obj_dir, views_root, ply)
    if total == 0:
        sys.exit("[error] 没有写出权重图")
    print(f"[done] {total}")


if __name__ == "__main__":
    main()
