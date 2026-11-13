#!/usr/bin/env python3
"""把射线遮挡盖到原渲染图上。

绿色是露出的部分（权重不低于 0.5）。黄色是物体轮廓里没进绿色的部分，也就是被挡住的区域。
轮廓来自只渲染旧投票高斯的那一次。不改掩码和 ray_occlusion.json。

默认出剃须刀、蛋糕、饼干。写出 ``published_vote/data2_<物体>/occlusion_overlay/view_XXX.png``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from export_ray_visible import (  # noqa: E402
    PLY,
    PUBLISHED,
    VIEWS,
    _out_dir,
    _paint,
    _restore,
    _with_selected,
    load_indices,
)
from overlay_visible_mask import KIND_LABEL, _font  # noqa: E402
from render_object_invdepth import load_gaussian_model  # noqa: E402
from render_object_weight import VISIBLE_WEIGHT, render_object_weight  # noqa: E402

GREEN = np.array((46, 160, 67), dtype=np.uint8)
YELLOW = np.array((242, 196, 24), dtype=np.uint8)
DEFAULT_KEYS = ("shaver", "toy_cake", "cookie")


def _draw(rgb: np.ndarray, green: np.ndarray, yellow: np.ndarray, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    out = np.array(rgb, copy=True)
    out[yellow] = YELLOW
    out[green] = GREEN
    image = Image.fromarray(out, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    bar_h = 48
    draw.rectangle([0, 0, image.width, bar_h], fill=(24, 24, 24))
    draw.text((12, 8), title, fill=(255, 255, 255), font=font)
    return np.array(image)


def overlay_object(key: str, iteration: str) -> Path:
    from PIL import Image

    out = _out_dir(key)
    rows = json.loads((out / "ray_occlusion.json").read_text(encoding="utf-8"))
    by_view = {row["view_id"]: row for row in rows}
    indices = load_indices(key, iteration)
    cam_paths = sorted((VIEWS / "camera_params").glob("view_*.json"))
    model = load_gaussian_model(PLY)
    saved_dc, saved_rest = _paint(model, indices)
    selected = _with_selected(model, indices)
    dest = out / "occlusion_overlay"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        for cam_path in cam_paths:
            view_id = cam_path.stem.split("_")[-1].zfill(3)
            row = by_view[view_id]
            cam = json.loads(cam_path.read_text(encoding="utf-8"))
            rgb = np.array(Image.open(VIEWS / "rgb" / f"view_{view_id}.png").convert("RGB"))
            green = np.array(Image.open(out / "mask_ray" / f"view_{view_id}.png")) >= 128
            if row["kind"] == "silhouette_failed":
                yellow = np.zeros(green.shape, dtype=bool)
                label = "轮廓光栅失败"
            else:
                alone = render_object_weight(model, cam)
                yellow = (alone >= VISIBLE_WEIGHT) & ~green
                label = KIND_LABEL.get(row["kind"], row["kind"])
            ratio = row.get("ratio")
            ratio_text = "—" if ratio is None else f"遮挡 {ratio * 100:.0f}%"
            title = f"view_{view_id}  {label}  {ratio_text}   绿=露出  黄=遮挡"
            Image.fromarray(_draw(rgb, green, yellow, title)).save(dest / f"view_{view_id}.png")
            print(f"[overlay] {key} view_{view_id} {label}", flush=True)
    finally:
        _restore(model, selected)
        model._features_dc.data.copy_(saved_dc)
        model._features_rest.data.copy_(saved_rest)
        del model
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    print(f"[overlay] {key} -> {dest}", flush=True)
    return dest


def main() -> None:
    wanted = sys.argv[1:] or list(DEFAULT_KEYS)
    by_key = {key: iteration for key, iteration, _prompt_dir in PUBLISHED}
    for key in wanted:
        if key not in by_key:
            raise SystemExit(f"未知物体 {key}")
        overlay_object(key, by_key[key])


if __name__ == "__main__":
    main()
