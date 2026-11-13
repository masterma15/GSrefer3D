#!/usr/bin/env python3
"""用物体高斯的期望逆深度修剪掩码，并按删掉的比例标记遮挡。

射线滤帧模块保留给现有端到端和已导出的 469 条。这一步不调用它。
第一次投票已经筛过视角一致性。这里只比较两条逆深度：

- 整图渲染逆深度：``depth_raw/view_XXX.npy``，当时全场景光栅的结果。
- 物体逆深度：同一套光栅器、同一相机，只让投票选中的高斯有不透明度。

物体逆深度接近 0，视为这条线上没有选中高斯，像素删掉。
两边都有效且相对差小于门槛，像素留下。否则删掉。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

# 与融合里过滤天空的 min-inv 一致：逆深度低于此值视为没有表面。
DEFAULT_INV_EPS = 1e-3
# 可见表面的相对差中位数大约 0.04：整图逆深度还会叠上物体后面的高斯，
# 物体单独渲染会略小。0.05 卡在这条偏差的上沿，边缘被整片删掉。
# 0.10 盖过这条偏差，相对差到 0.3 的遮挡仍然删掉。
DEFAULT_REL_TOL = 0.10
DEFAULT_LOW_RATIO = 0.2
DEFAULT_HIGH_RATIO = 0.8
FG_THRESH = 128


def keep_visible_pixels(
    scene_inv: np.ndarray,
    object_inv: np.ndarray,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    inv_eps: float = DEFAULT_INV_EPS,
) -> np.ndarray:
    """整图逆深度与物体逆深度接近的像素。返回与输入同形状的布尔图。

    物体逆深度不超过 ``inv_eps`` 的像素删掉（空射线）。
    相对差是 ``|整图 - 物体| / max(|整图|, inv_eps)``，小于 ``rel_tol`` 才留下。
    """
    scene = np.asarray(scene_inv, dtype=np.float64)
    obj = np.asarray(object_inv, dtype=np.float64)
    if scene.ndim != 2 or obj.ndim != 2:
        raise ValueError(f"逆深度须为二维，得到 scene {scene.shape} object {obj.shape}")
    if scene.shape != obj.shape:
        raise ValueError(f"逆深度形状不一致：scene {scene.shape} object {obj.shape}")
    if rel_tol < 0 or inv_eps <= 0:
        raise ValueError("rel_tol 须 >= 0，inv_eps 须 > 0")

    finite = np.isfinite(scene) & np.isfinite(obj)
    present = finite & (obj > inv_eps) & (scene > inv_eps)
    denom = np.maximum(np.abs(scene), inv_eps)
    rel = np.abs(scene - obj) / denom
    return present & (rel < rel_tol)


def occlusion_ratio(mask: np.ndarray, kept: np.ndarray, *, fg_thresh: int = FG_THRESH) -> float:
    """删掉的前景像素占原掩码前景的比例。原掩码为空时返回 nan。"""
    if mask.shape != kept.shape:
        raise ValueError(f"掩码与保留图形状不一致：{mask.shape} vs {kept.shape}")
    fg = np.asarray(mask) >= fg_thresh
    n = int(fg.sum())
    if n == 0:
        return float("nan")
    removed = int(np.count_nonzero(fg & ~np.asarray(kept, dtype=bool)))
    return removed / n


def classify_occlusion(
    ratio: float,
    *,
    low: float = DEFAULT_LOW_RATIO,
    high: float = DEFAULT_HIGH_RATIO,
) -> str:
    """按删掉比例分类。恰好等于 low 或 high 时归入部分遮挡。"""
    if not (low < high):
        raise ValueError(f"需要 low < high，得到 {low}, {high}")
    if ratio != ratio:
        return "empty_mask"
    if ratio < low:
        return "almost_unoccluded"
    if ratio > high:
        return "fully_occluded"
    return "partial"


def mask_centroid(kept: np.ndarray) -> tuple[float, float] | None:
    """保留像素的质心（列、行）。没有前景时返回 None。"""
    ys, xs = np.nonzero(np.asarray(kept, dtype=bool))
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def trim_mask(
    mask: np.ndarray,
    scene_inv: np.ndarray,
    object_inv: np.ndarray,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    inv_eps: float = DEFAULT_INV_EPS,
    low: float = DEFAULT_LOW_RATIO,
    high: float = DEFAULT_HIGH_RATIO,
    fg_thresh: int = FG_THRESH,
) -> dict[str, Any]:
    """修剪一张掩码。完全遮挡和空掩码不给质心。"""
    if mask.shape[:2] != scene_inv.shape:
        raise ValueError(
            f"掩码与整图逆深度形状不一致：mask {mask.shape} scene {scene_inv.shape}"
        )
    fg = np.asarray(mask) >= fg_thresh
    visible = keep_visible_pixels(scene_inv, object_inv, rel_tol=rel_tol, inv_eps=inv_eps)
    kept = fg & visible
    ratio = occlusion_ratio(mask, kept, fg_thresh=fg_thresh)
    kind = classify_occlusion(ratio, low=low, high=high)
    centroid = mask_centroid(kept) if kind in ("almost_unoccluded", "partial") else None
    return {
        "kept": kept,
        "ratio": ratio,
        "kind": kind,
        "centroid_uv": centroid,
    }


def world_to_view_matrix(rotation_c2w: np.ndarray, camera_center: np.ndarray) -> np.ndarray:
    """由 JSON 里的 R_c2w 和相机中心得到与 ``getWorld2View2`` 相同的世界到相机矩阵。

    返回值尚未转置。光栅器要的是这个矩阵的转置，与 ``Camera.world_view_transform`` 一致。
    """
    rotation = np.asarray(rotation_c2w, dtype=np.float64).reshape(3, 3)
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    r_w2c = rotation.T
    translation = -r_w2c @ center
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = r_w2c
    view[:3, 3] = translation
    return view


def _depth_raw_path(views_root: Path, cam: dict, view_id: str) -> Path:
    """``depth_raw_relative`` 相对视角根目录，与 render.py 写出时的起点一致。"""
    rel = cam.get("depth_raw_relative")
    if rel:
        return (views_root / rel).resolve()
    return views_root / "depth_raw" / f"view_{view_id}.npy"


def _write_label(width: int, height: int, centroid: tuple[float, float] | None) -> dict[str, float] | None:
    if centroid is None or width <= 0 or height <= 0:
        return None
    u, v = centroid
    return {"u": u, "v": v, "nx": u / width, "ny": v / height}


def trim_object_dir(
    obj_dir: Path,
    views_root: Path,
    object_inv_by_view: dict[str, np.ndarray],
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    inv_eps: float = DEFAULT_INV_EPS,
    low: float = DEFAULT_LOW_RATIO,
    high: float = DEFAULT_HIGH_RATIO,
) -> list[dict[str, Any]]:
    """修剪目录里每一张已有掩码。``object_inv_by_view`` 的键是三位视角编号。"""
    from PIL import Image

    mask_dir = obj_dir / "mask"
    out_dir = obj_dir / "mask_visible"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    mask_paths = sorted(mask_dir.glob("view_*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"{mask_dir} 下没有 view_*.png")

    for mask_path in mask_paths:
        view_id = mask_path.stem.split("_")[-1].zfill(3)
        cam_path = views_root / "camera_params" / f"view_{view_id}.json"
        if not cam_path.is_file():
            records.append({"view_id": view_id, "kind": "missing_camera", "ratio": None})
            continue
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        scene_path = _depth_raw_path(views_root, cam, view_id)
        if not scene_path.is_file():
            records.append({"view_id": view_id, "kind": "missing_scene_depth", "ratio": None})
            continue
        if view_id not in object_inv_by_view:
            records.append({"view_id": view_id, "kind": "missing_object_depth", "ratio": None})
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        scene_inv = np.load(scene_path)
        if scene_inv.ndim != 2:
            scene_inv = np.squeeze(scene_inv)
        result = trim_mask(
            mask,
            scene_inv,
            object_inv_by_view[view_id],
            rel_tol=rel_tol,
            inv_eps=inv_eps,
            low=low,
            high=high,
        )
        Image.fromarray(result["kept"].astype(np.uint8) * 255, mode="L").save(
            out_dir / f"view_{view_id}.png"
        )
        width = int(cam.get("width", mask.shape[1]))
        height = int(cam.get("height", mask.shape[0]))
        ratio = result["ratio"]
        records.append(
            {
                "view_id": view_id,
                "kind": result["kind"],
                "ratio": None if ratio != ratio else float(ratio),
                "label": _write_label(width, height, result["centroid_uv"]),
            }
        )
    (obj_dir / "occlusion.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records
