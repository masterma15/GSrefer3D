"""手标 OBB 几何：命中判定、线框采样、SH 颜色。供投票评测与 SIBR 注入复用。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

C0 = 0.28209479177387814


def load_bbox(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("objects", data)


def half_extent(obb: dict[str, Any]) -> np.ndarray:
    if "half_extent" in obb:
        return np.asarray(obb["half_extent"], dtype=np.float64)
    return np.asarray(obb["width"], dtype=np.float64) / 2.0


# 旧名，避免大面积改调用
_half_extent = half_extent


def obb_local(p_world: np.ndarray, obb: dict[str, Any]) -> np.ndarray:
    center = np.asarray(obb["center"], dtype=np.float64)
    rot = np.asarray(obb["rotation_columns"], dtype=np.float64)
    return rot.T @ (p_world - center)


def obb_hit(p_world: np.ndarray, obb: dict[str, Any], *, margin: float = 0.0) -> bool:
    local = obb_local(p_world, obb)
    half = half_extent(obb) + margin
    return bool(np.all(np.abs(local) <= half))


def obb_hits(xyz: np.ndarray, obb: dict[str, Any], *, margin: float = 0.0) -> np.ndarray:
    """批量命中，语义与 ``obb_hit`` 相同。"""
    if xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    center = np.asarray(obb["center"], dtype=np.float64).reshape(1, 3)
    rot = np.asarray(obb["rotation_columns"], dtype=np.float64)
    half = (half_extent(obb) + float(margin)).reshape(1, 3)
    local = (xyz - center) @ rot
    return np.all(np.abs(local) <= half, axis=1)


def obb_corners_world(obb: dict[str, Any]) -> np.ndarray:
    center = np.asarray(obb["center"], dtype=np.float64)
    rot = np.asarray(obb["rotation_columns"], dtype=np.float64)
    half = half_extent(obb)
    corners: list[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
                corners.append(center + rot @ local)
    return np.stack(corners, axis=0)


def _corner_index(sx: float, sy: float, sz: float) -> int:
    return (4 if sx > 0 else 0) + (2 if sy > 0 else 0) + (1 if sz > 0 else 0)


OBB_WIREFRAME_EDGES: list[tuple[int, int]] = [
    (_corner_index(-1, -1, -1), _corner_index(-1, -1, 1)),
    (_corner_index(-1, -1, -1), _corner_index(-1, 1, -1)),
    (_corner_index(-1, -1, -1), _corner_index(1, -1, -1)),
    (_corner_index(-1, -1, 1), _corner_index(-1, 1, 1)),
    (_corner_index(-1, -1, 1), _corner_index(1, -1, 1)),
    (_corner_index(-1, 1, -1), _corner_index(-1, 1, 1)),
    (_corner_index(-1, 1, -1), _corner_index(1, 1, -1)),
    (_corner_index(-1, 1, 1), _corner_index(1, 1, 1)),
    (_corner_index(1, -1, -1), _corner_index(1, -1, 1)),
    (_corner_index(1, -1, -1), _corner_index(1, 1, -1)),
    (_corner_index(1, -1, 1), _corner_index(1, 1, 1)),
    (_corner_index(1, 1, -1), _corner_index(1, 1, 1)),
]


def obb_wireframe_samples(obb: dict[str, Any], *, step_m: float = 0.025) -> np.ndarray:
    corners = obb_corners_world(obb)
    pts: list[np.ndarray] = []
    for i, j in OBB_WIREFRAME_EDGES:
        a, b = corners[i], corners[j]
        seg_len = float(np.linalg.norm(b - a))
        n = max(2, int(math.ceil(seg_len / step_m)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            pts.append(a + t * (b - a))
    return np.stack(pts, axis=0)


def rgb_to_f_dc(rgb: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    rgb_a = np.asarray(rgb, dtype=np.float64).reshape(3)
    sh = (rgb_a - 0.5) / C0
    return float(sh[0]), float(sh[1]), float(sh[2])


_rgb_to_f_dc = rgb_to_f_dc
