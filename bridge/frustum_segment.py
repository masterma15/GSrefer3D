#!/usr/bin/env python3
"""多视 mask 视锥过滤 + T(z) 加权投票 + 簇深度切片 → 目标高斯集合（P0/P1）。

输入（物体目录，如 training_data/data2_shaver）：
  projections_kept.json（没有则用 projections.json）、mask/view_*.png、
  filter_report.json（含 P_world 与 T_at_z_lo；可缺）。
相机与 ply 默认读 filter_report，可用 CLI 覆盖。
只有 projections.json 时，各视角权重为 1，不使用射线透射率。

流程：
  1. 只用 kept 视角（T(z) 已拒帧）。
  2. 将全体高斯中心 μ 投到该视 mask：落在前景即视锥命中。
  3. 投票：命中则加上该视权重（P1：权重=T_at_z_lo，缺省 1）。
  4. 深度切片（默认关；``--depth-slice`` 开启）：至少在一个命中视角上，
     z_cam 落在簇深度带 [z_lo,z_hi]（``--z-slack`` 可放宽）。
  5. 写出 gaussian_seg.json 与选中点的 xyzrgb ply（不是语言场，也不上 BEV/虚相机）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from bbox_keys import OBJ_DIR_TO_BBOX_KEY  # noqa: E402
from ray_visibility import cluster_depth_band, load_ply_xyz  # noqa: E402
from repo_paths import resolve_data_path  # noqa: E402
from unproject import CameraView  # noqa: E402


def world_to_cam_uvz(xyz: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量：世界点 → (u, v, z_cam)。z<=0 的 u/v 为 nan。

    JSON rotation 为 R_c2w，与 ``CameraView.project_world`` 同一外参。
    """
    view = CameraView.from_camera_dict(cam)
    p_cam = view.world_to_cam_points(xyz)
    z = p_cam[:, 2]
    fx, fy, cx, cy = view.intrinsics
    z_ok = z > 1e-8
    z_safe = np.where(z_ok, z, np.nan)
    u = fx * p_cam[:, 0] / z_safe + cx
    v = fy * p_cam[:, 1] / z_safe + cy
    return u, v, z


def frustum_hit_mask(
    xyz: np.ndarray,
    cam: dict,
    mask: np.ndarray,
    *,
    fg_thresh: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """视锥命中：μ 投到像平面且落在 mask 前景。

    返回 (hit, z_cam)，hit 为 (N,) bool。
    """
    u, v, z = world_to_cam_uvz(xyz, cam)
    h, w = int(mask.shape[0]), int(mask.shape[1])
    valid = np.isfinite(u) & np.isfinite(v) & (z > 1e-8)
    hit = np.zeros(xyz.shape[0], dtype=bool)
    if np.any(valid):
        ui = np.floor(u[valid]).astype(np.int64)
        vi = np.floor(v[valid]).astype(np.int64)
        in_img = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        sel = np.nonzero(valid)[0][in_img]
        hit[sel] = mask[vi[in_img], ui[in_img]] >= fg_thresh
    return hit, z


def select_cluster_xyz(
    xyz: np.ndarray,
    p_world: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """P_world 邻域内的高斯中心，用于算各视簇深度带。"""
    d = np.linalg.norm(xyz - p_world.reshape(1, 3), axis=1)
    idx = np.nonzero(d <= radius)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(d))], dtype=np.int64)
    return xyz[idx]


def view_weight_from_report(entry: dict[str, Any] | None, *, use_t_weight: bool) -> float:
    """P1：kept 视角权重用 T(z)；无 T 或关闭加权则为 1。拒帧权重为 0。"""
    if entry is None:
        return 1.0
    if entry.get("status") == "rejected":
        return 0.0
    if not use_t_weight:
        return 1.0
    t = entry.get("T_at_z_lo")
    if t is None:
        return 1.0
    return float(np.clip(float(t), 0.0, 1.0))


def camera_azimuth_about(p_world: np.ndarray, cam: dict[str, Any]) -> float:
    """水平方位角 [0, 2π)：从 P_world 指向相机中心，atan2(dx, dz)。"""
    c = np.asarray(cam["position"], dtype=np.float64).reshape(3)
    p = np.asarray(p_world, dtype=np.float64).reshape(3)
    d = c - p
    return float(np.arctan2(d[0], d[2]) % (2.0 * np.pi))


def principal_coverage_arc(az_rad: np.ndarray) -> tuple[float, float]:
    """对方位角集合取主覆盖弧：去掉最大空隙后的互补弧 (start, length)，单位弧度。"""
    a = np.unique(np.sort(np.asarray(az_rad, dtype=np.float64) % (2.0 * np.pi)))
    if a.size == 0:
        return 0.0, 0.0
    if a.size == 1:
        return float(a[0]), 0.0
    gaps = np.diff(a)
    wrap = float(a[0] + 2.0 * np.pi - a[-1])
    all_gaps = np.append(gaps, wrap)
    i = int(np.argmax(all_gaps))
    length = float(2.0 * np.pi - all_gaps[i])
    start = float(a[0] if i == len(gaps) else a[i + 1])
    return start, max(0.0, length)


def apply_sector_vote_weights(
    views: list[dict[str, Any]],
    p_world: np.ndarray,
    *,
    sector_deg: float = 30.0,
    cap_mult: float = 3.0,
) -> dict[str, Any]:
    """按 P_world 水平覆盖弧做等角扇区软加权（原地改 views[*].weight）。

    步骤：主覆盖弧 → 约 ``sector_deg`` 细扇区 → ``1/count`` → 稀扇区封顶（≤ cap_mult×中位）
    → 乘现有权重（通常含 T）→ 归一到均值 1，使 ``min_vote_frac × N`` 门槛尺度不变。
    """
    if not views:
        return {"n_sectors": 0, "arc_deg": 0.0, "sector_counts": []}

    az = np.array([camera_azimuth_about(p_world, v["cam"]) for v in views], dtype=np.float64)
    start, arc_len = principal_coverage_arc(az)
    sector_rad = max(1e-6, float(sector_deg) * np.pi / 180.0)
    if arc_len <= 1e-9:
        n_sec = 1
        bin_size = 1.0
        ids = np.zeros(len(views), dtype=np.int64)
    else:
        n_sec = max(1, int(np.ceil(arc_len / sector_rad)))
        bin_size = arc_len / float(n_sec)
        rel = (az - start) % (2.0 * np.pi)
        # 数值落在弧外时钳到弧内（浮点）
        rel = np.minimum(rel, arc_len - 1e-12)
        ids = np.minimum(n_sec - 1, np.floor(rel / bin_size).astype(np.int64))

    counts = np.bincount(ids, minlength=n_sec).astype(np.float64)
    inv = np.zeros(n_sec, dtype=np.float64)
    nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    pos = inv[nz]
    if pos.size:
        med = float(np.median(pos))
        if med > 0 and cap_mult > 0:
            inv = np.minimum(inv, float(cap_mult) * med)

    raw = np.array([float(views[i]["weight"]) * float(inv[ids[i]]) for i in range(len(views))], dtype=np.float64)
    mean = float(raw.mean()) if raw.size else 1.0
    if mean <= 1e-12:
        mean = 1.0
    scaled = raw / mean
    for i, v in enumerate(views):
        v["weight"] = float(scaled[i])
        v["sector_id"] = int(ids[i])

    meta = {
        "n_sectors": int(n_sec),
        "arc_start_deg": float(np.degrees(start) % 360.0),
        "arc_deg": float(np.degrees(arc_len)),
        "sector_deg_target": float(sector_deg),
        "sector_deg_effective": float(np.degrees(bin_size)) if arc_len > 1e-9 else 0.0,
        "cap_mult": float(cap_mult),
        "sector_counts": [int(c) for c in counts.tolist()],
        "weight_mean_before_norm": mean,
    }
    return meta


_AXIS_NAMES = ("x", "y", "z")
PAIR_TAU_DEG = 20.0
PAIR_FRAME_MARGIN = 8
PAIR_WHITE_MIN_SCORE = 0.03
PAIR_FINAL_MIN_SCORE = 0.15
# 掩码内侧仍是整票赞成。外侧距离 d 的反对强度是 sigmoid((d-2)/0.5)。
SOFT_EDGE_D0 = 2.0
SOFT_EDGE_TAU = 0.5


def assign_view_axis(
    p_world: np.ndarray,
    cam: dict[str, Any],
    *,
    max_deg: float = 35.0,
) -> int | None:
    """相机相对 P_world 的方向归到最近的世界轴。

    与该轴夹角大于 ``max_deg`` 时不归入任何轴（斜视不充当第二根轴）。
    返回 0/1/2 = X/Y/Z，正负方向合成同一根轴。
    """
    c = np.asarray(cam["position"], dtype=np.float64).reshape(3)
    d = c - np.asarray(p_world, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-8:
        return None
    abs_u = np.abs(d / nrm)
    axis = int(np.argmax(abs_u))
    cos_min = float(np.cos(np.deg2rad(float(max_deg))))
    if float(abs_u[axis]) + 1e-12 < cos_min:
        return None
    return axis


def camera_az_el(p_world: np.ndarray, cam: dict[str, Any]) -> tuple[float, float]:
    """锚点指向相机的水平方位角 [0, 2π) 和仰角（弧度）。半径不返回。"""
    c = np.asarray(cam["position"], dtype=np.float64).reshape(3)
    d = c - np.asarray(p_world, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-8:
        return 0.0, 0.0
    az = float(np.arctan2(d[0], d[2]) % (2.0 * np.pi))
    el = float(np.arcsin(np.clip(d[1] / nrm, -1.0, 1.0)))
    return az, el


def patch_grid_shape(n_bins: int) -> tuple[int, int]:
    """64 格是相机角度范围上的 8×8。格数须为完全平方。"""
    n = int(n_bins)
    side = int(round(n ** 0.5))
    if side < 1 or side * side != n:
        raise ValueError(f"方位格数须为完全平方，得到 {n_bins}")
    return side, side


def assign_view_patch_bins(
    p_world: np.ndarray,
    cams: list[dict[str, Any]],
    *,
    n_bins: int = 64,
) -> tuple[list[int], dict[str, Any]]:
    """在相机实际占据的方位角弧 × 仰角范围上切 ``n_az × n_el`` 格。

    不铺满整圈。半径不参与。空格由调用方跳过。
    """
    n_az, n_el = patch_grid_shape(n_bins)
    if not cams:
        return [], {
            "n_az": n_az,
            "n_el": n_el,
            "arc_start_deg": 0.0,
            "arc_deg": 0.0,
            "el_min_deg": 0.0,
            "el_max_deg": 0.0,
        }
    angles = [camera_az_el(p_world, cam) for cam in cams]
    az = np.array([a for a, _ in angles], dtype=np.float64)
    el = np.array([e for _, e in angles], dtype=np.float64)
    start, arc_len = principal_coverage_arc(az)
    el_min = float(el.min())
    el_max = float(el.max())
    el_span = el_max - el_min
    ids: list[int] = []
    for a, e in angles:
        if arc_len <= 1e-9:
            ia = 0
        else:
            rel = float((a - start) % (2.0 * np.pi))
            if rel > arc_len:
                rel = arc_len
            ia = int(rel / arc_len * n_az)
            if ia >= n_az:
                ia = n_az - 1
        if el_span <= 1e-9:
            ie = 0
        else:
            ie = int((e - el_min) / el_span * n_el)
            if ie >= n_el:
                ie = n_el - 1
        ids.append(int(ie * n_az + ia))
    meta = {
        "n_az": int(n_az),
        "n_el": int(n_el),
        "arc_start_deg": float(np.degrees(start) % 360.0),
        "arc_deg": float(np.degrees(arc_len)),
        "el_min_deg": float(np.degrees(el_min)),
        "el_max_deg": float(np.degrees(el_max)),
    }
    return ids, meta


def mean_azimuth_bin_rates(
    hit_mass: dict[int, np.ndarray],
    sizes: dict[int, int],
) -> np.ndarray:
    """每格 ``sum(T × 命中) / 张数`` 的平均。空格不在 ``hit_mass`` 里，不进分母。

    分母是格内相机张数，与全局票的 ``比例 × 视角数`` 相同。T 为 1 时退回命中比例。
    """
    if not hit_mass:
        raise ValueError("没有有相机的方位格")
    acc: np.ndarray | None = None
    for bin_id, mass in hit_mass.items():
        rate = mass.astype(np.float64) / float(sizes[bin_id])
        acc = rate if acc is None else acc + rate
    assert acc is not None
    return acc / float(len(hit_mass))


def active_axis_ids(axis_ids: list[int | None], *, min_views: int) -> list[int]:
    """张数达到 ``min_views`` 的轴才存在。"""
    counts = [sum(1 for a in axis_ids if a == i) for i in range(3)]
    return [i for i, n in enumerate(counts) if n >= int(min_views)]


def segment_gaussians_axis(
    xyz: np.ndarray,
    *,
    p_world: np.ndarray,
    views: list[dict[str, Any]],
    active_axes: list[int],
    group_frac: float,
    cluster_radius: float,
    z_slack: float,
    apply_depth_slice: bool,
    min_axes: int = 2,
    axis_names: tuple[str, str, str] = _AXIS_NAMES,
) -> dict[str, Any]:
    """跨轴硬条件：至少 ``min_axes`` 根活跃轴上，组内命中比例 ≥ ``group_frac``。

    不套全局 ``min_vote_frac``。斜视（无 axis_id）不进任何组的分母。
    """
    n = xyz.shape[0]
    cluster_xyz = select_cluster_xyz(xyz, p_world, radius=cluster_radius)
    sizes = {ax: sum(1 for v in views if v.get("axis_id") == ax) for ax in active_axes}
    hit_counts = {ax: np.zeros(n, dtype=np.int32) for ax in active_axes}
    per_view_stats: list[dict[str, Any]] = []

    for item in views:
        cam = item["cam"]
        mask = item["mask"]
        ax = item.get("axis_id")
        hit, selected, z_lo, z_hi = view_frustum_masks(
            xyz,
            cam,
            mask,
            cluster_xyz,
            z_slack=z_slack,
            apply_depth_slice=apply_depth_slice,
        )
        used = hit if not apply_depth_slice else selected
        if ax in hit_counts:
            hit_counts[ax] += used.astype(np.int32)
        per_view_stats.append(
            {
                "view_id": item["view_id"],
                "weight": float(item.get("weight") or 0.0),
                "axis_id": None if ax is None else int(ax),
                "axis": None if ax is None else axis_names[int(ax)],
                "frustum_hits": int(used.sum()),
                "z_lo": z_lo,
                "z_hi": z_hi,
            }
        )

    support = np.zeros(n, dtype=np.int32)
    for ax, hc in hit_counts.items():
        need = float(group_frac) * float(sizes[ax])
        support += (hc.astype(np.float64) + 1e-9 >= need).astype(np.int32)
    selected = support >= int(min_axes)
    idx = np.nonzero(selected)[0]
    centroid = xyz[idx].mean(axis=0) if idx.size else p_world.copy()
    return {
        "indices": idx,
        "votes": support.astype(np.float64),
        "selected": selected,
        "centroid": centroid,
        "per_view": per_view_stats,
        "n_selected": int(idx.size),
        "cluster_size": int(cluster_xyz.shape[0]),
        "axis_sizes": {axis_names[i]: int(sizes[i]) for i in active_axes},
    }


def pair_attention(
    p_world: np.ndarray,
    cams: list[dict[str, Any]],
    *,
    tau_deg: float = 20.0,
) -> dict[str, Any]:
    """视线夹角上的固定配对权重。同一方向的图分量变小，远处的一对权重大。"""
    tau = float(tau_deg)
    if tau <= 0.0:
        raise ValueError(f"tau_deg 必须 > 0，得到 {tau_deg}")
    dirs = []
    p = np.asarray(p_world, dtype=np.float64).reshape(3)
    for cam in cams:
        d = np.asarray(cam["position"], dtype=np.float64).reshape(3) - p
        nrm = float(np.linalg.norm(d))
        dirs.append(d / nrm if nrm >= 1e-8 else np.array([0.0, 0.0, 1.0]))
    u = np.stack(dirs, axis=0)
    gram = np.clip(u @ u.T, -1.0, 1.0)
    theta = np.arccos(gram)
    closeness = np.exp(-theta / np.deg2rad(tau))
    a = 1.0 / np.maximum(closeness.sum(axis=1), 1e-12)
    alpha = a[:, None] * a[None, :] * (theta / np.pi)
    np.fill_diagonal(alpha, 0.0)
    alpha = 0.5 * (alpha + alpha.T)
    np.fill_diagonal(alpha, 0.0)
    drow = alpha.sum(axis=1)
    denom = float(a.sum() + 0.5 * alpha.sum())
    return {
        "a": a,
        "alpha": alpha,
        "drow": drow,
        "denom": denom,
        "tau_deg": tau,
    }


def pair_scores(
    positive: np.ndarray,
    attention: dict[str, Any],
    *,
    negative: np.ndarray | None = None,
    chunk: int = 200_000,
) -> np.ndarray:
    """赞成矩阵 (高斯, 视角) → 分数。全空为 0。保留条件用 score > 阈值。

    ``negative`` 缺省时，没赞成的都算反对，和第一次投票相同。
    传入反对矩阵后，赞成和反对都是 0 的位置弃权，不减分。
    """
    alpha = np.asarray(attention["alpha"], dtype=np.float64)
    a = np.asarray(attention["a"], dtype=np.float64)
    drow = np.asarray(attention["drow"], dtype=np.float64)
    denom = float(attention["denom"])
    if denom <= 0.0:
        raise ValueError("配对权重分母为 0")
    n = int(positive.shape[0])
    out = np.empty(n, dtype=np.float64)
    step = max(1, int(chunk))
    for start in range(0, n, step):
        p = np.asarray(positive[start : start + step], dtype=np.float64)
        if negative is None:
            quad = np.sum((p @ alpha) * p, axis=1)
            out[start : start + p.shape[0]] = (p @ a - p @ drow + 1.5 * quad) / denom
        else:
            neg = np.asarray(negative[start : start + step], dtype=np.float64)
            quad = np.sum((p @ alpha) * p, axis=1)
            oppose = np.sum((p @ alpha) * neg, axis=1)
            out[start : start + p.shape[0]] = (p @ a + 0.5 * quad - oppose) / denom
    return out


def _interior_pixels(
    xyz: np.ndarray,
    cam: dict[str, Any],
    mask: np.ndarray,
    *,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """落在画面内部、离开边缘 ``margin`` 像素的高斯。返回 (行号, u, v)。"""
    u, v, z = world_to_cam_uvz(xyz, cam)
    h, w = int(mask.shape[0]), int(mask.shape[1])
    edge = max(0, int(margin))
    valid = np.isfinite(u) & np.isfinite(v) & (z > 1e-8)
    if not np.any(valid):
        return None
    ui = np.floor(u[valid]).astype(np.int64)
    vi = np.floor(v[valid]).astype(np.int64)
    inside = (
        (ui >= edge)
        & (ui < w - edge)
        & (vi >= edge)
        & (vi < h - edge)
    )
    if not np.any(inside):
        return None
    sel = np.nonzero(valid)[0][inside]
    return sel, ui[inside], vi[inside]


def outside_oppose_strength(dist: np.ndarray) -> np.ndarray:
    """掩码外侧的反对强度。d=1 约 0.12，d≥4 接近 1。"""
    x = (np.asarray(dist, dtype=np.float64) - SOFT_EDGE_D0) / SOFT_EDGE_TAU
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def _mask_outside_distance(mask: np.ndarray, fg_thresh: int) -> np.ndarray:
    """每个背景像素到最近掩码像素的距离。掩码上为 0。"""
    from scipy.ndimage import distance_transform_edt

    fg = mask >= int(fg_thresh)
    if not np.any(fg):
        return np.full(fg.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(~fg).astype(np.float32)


def observe_frame(
    xyz: np.ndarray,
    cam: dict[str, Any],
    mask: np.ndarray,
    *,
    margin: int = 8,
    fg_thresh: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """画面内部：mask 里整票赞成。mask 外按软边缘减分。画面外和边缘弃权。"""
    positive = np.zeros(xyz.shape[0], dtype=np.float32)
    negative = np.zeros(xyz.shape[0], dtype=np.float32)
    picked = _interior_pixels(xyz, cam, mask, margin=margin)
    if picked is None:
        return positive, negative
    sel, ui, vi = picked
    dist_map = _mask_outside_distance(mask, fg_thresh)
    fg = dist_map[vi, ui] == 0
    positive[sel[fg]] = 1.0
    out = ~fg
    if np.any(out):
        negative[sel[out]] = outside_oppose_strength(dist_map[vi[out], ui[out]])
    return positive, negative


def observe_footprint(
    xyz: np.ndarray,
    cam: dict[str, Any],
    mask: np.ndarray,
    footprint: np.ndarray,
    stain: np.ndarray,
    *,
    visible_weight: float = 0.5,
    margin: int = 8,
    fg_thresh: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """赞成仍是画面内的整票 mask。外侧按软边缘减分。足迹高且染色低时弃权。"""
    if footprint.shape[:2] != mask.shape[:2] or stain.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"足迹 {footprint.shape}、染色 {stain.shape} 与 mask {mask.shape} 不一致"
        )
    positive = np.zeros(xyz.shape[0], dtype=np.float32)
    negative = np.zeros(xyz.shape[0], dtype=np.float32)
    picked = _interior_pixels(xyz, cam, mask, margin=margin)
    if picked is None:
        return positive, negative
    sel, ui, vi = picked
    dist_map = _mask_outside_distance(mask, fg_thresh)
    fg = dist_map[vi, ui] == 0
    on_object = footprint[vi, ui] >= float(visible_weight)
    seen = stain[vi, ui] >= float(visible_weight)
    occluded = on_object & ~seen
    positive[sel[fg]] = 1.0
    out = ~fg & ~occluded
    if np.any(out):
        negative[sel[out]] = outside_oppose_strength(dist_map[vi[out], ui[out]])
    return positive, negative


def observe_weight_map(
    xyz: np.ndarray,
    cam: dict[str, Any],
    mask: np.ndarray,
    weight: np.ndarray,
    *,
    visible_weight: float = 0.5,
    fg_thresh: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """染色权重图上的赞成和反对。权重低于阈值或投出画面的高斯弃权。"""
    if weight.shape[:2] != mask.shape[:2]:
        raise ValueError(f"权重图 {weight.shape} 与 mask {mask.shape} 不一致")
    u, v, z = world_to_cam_uvz(xyz, cam)
    h, w = int(mask.shape[0]), int(mask.shape[1])
    positive = np.zeros(xyz.shape[0], dtype=bool)
    negative = np.zeros(xyz.shape[0], dtype=bool)
    valid = np.isfinite(u) & np.isfinite(v) & (z > 1e-8)
    if not np.any(valid):
        return positive, negative
    ui = np.floor(u[valid]).astype(np.int64)
    vi = np.floor(v[valid]).astype(np.int64)
    in_img = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    sel = np.nonzero(valid)[0][in_img]
    ui = ui[in_img]
    vi = vi[in_img]
    seen = weight[vi, ui] >= float(visible_weight)
    fg = mask[vi, ui] >= int(fg_thresh)
    positive[sel] = seen & fg
    negative[sel] = seen & ~fg
    return positive, negative


def segment_gaussians_pair(
    xyz: np.ndarray,
    *,
    p_world: np.ndarray,
    views: list[dict[str, Any]],
    tau_deg: float,
    min_score: float,
    cluster_radius: float,
    z_slack: float,
    apply_depth_slice: bool,
    weight_maps: list[np.ndarray] | None = None,
    footprint_maps: list[np.ndarray] | None = None,
    visible_weight: float = 0.5,
    frame_margin: int = 8,
) -> dict[str, Any]:
    """视角对加减：同向降权，远处的不同意多减。score > min_score 才保留。

    画面外和边缘弃权。传入足迹和染色图时，足迹高且染色低也弃权；
    赞成仍是画面内部的 mask，不看染色权重。不使用透射率。
    """
    n = xyz.shape[0]
    attention = pair_attention(p_world, [item["cam"] for item in views], tau_deg=tau_deg)
    positive = np.zeros((n, len(views)), dtype=np.float32)
    negative = np.zeros((n, len(views)), dtype=np.float32)
    per_view_stats: list[dict[str, Any]] = []
    use_maps = weight_maps is not None or footprint_maps is not None
    if use_maps and (weight_maps is None or footprint_maps is None):
        raise ValueError("足迹和染色图要一起传入")
    if use_maps and (len(weight_maps) != len(views) or len(footprint_maps) != len(views)):
        raise ValueError("足迹、染色图的张数必须和视角一致")
    for col, item in enumerate(views):
        if use_maps:
            pos, neg = observe_footprint(
                xyz,
                item["cam"],
                item["mask"],
                footprint_maps[col],
                weight_maps[col],
                visible_weight=visible_weight,
                margin=frame_margin,
            )
        else:
            pos, neg = observe_frame(
                xyz,
                item["cam"],
                item["mask"],
                margin=frame_margin,
            )
        positive[:, col] = pos
        negative[:, col] = neg
        per_view_stats.append(
            {
                "view_id": item["view_id"],
                "weight": float(item.get("weight") or 0.0),
                "frustum_hits": int(np.count_nonzero(pos > 0)),
                "opposed": float(neg.sum()),
            }
        )
    scores = pair_scores(positive, attention, negative=negative)
    cluster_xyz = select_cluster_xyz(xyz, p_world, radius=cluster_radius)
    selected_mask = scores > float(min_score)
    idx = np.nonzero(selected_mask)[0]
    centroid = xyz[idx].mean(axis=0) if idx.size else p_world.copy()
    return {
        "indices": idx,
        "votes": scores,
        "selected": selected_mask,
        "centroid": centroid,
        "per_view": per_view_stats,
        "n_selected": int(idx.size),
        "cluster_size": int(cluster_xyz.shape[0]),
        "pair": {
            "tau_deg": float(attention["tau_deg"]),
            "min_score": float(min_score),
            "denom": float(attention["denom"]),
            "n_views": int(len(views)),
            "keep": "score > min_score",
            "occlusion": bool(use_maps),
            "frame_margin": int(frame_margin),
            "visible_weight": None if not use_maps else float(visible_weight),
            "soft_edge_d0": SOFT_EDGE_D0,
            "soft_edge_tau": SOFT_EDGE_TAU,
        },
    }


def apply_weight_observations(
    xyz: np.ndarray,
    views: list[dict[str, Any]],
    weight_maps: list[np.ndarray],
    *,
    visible_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """按染色权重图填赞成和反对。低权重的像素弃权。"""
    if len(weight_maps) != len(views):
        raise ValueError(f"权重图 {len(weight_maps)} 张，视角 {len(views)} 张")
    n = xyz.shape[0]
    positive = np.zeros((n, len(views)), dtype=bool)
    negative = np.zeros((n, len(views)), dtype=bool)
    for col, (item, weight) in enumerate(zip(views, weight_maps)):
        pos, neg = observe_weight_map(
            xyz,
            item["cam"],
            item["mask"],
            weight,
            visible_weight=visible_weight,
        )
        positive[:, col] = pos
        negative[:, col] = neg
    return positive, negative


def segment_gaussians_azimuth(
    xyz: np.ndarray,
    *,
    p_world: np.ndarray,
    views: list[dict[str, Any]],
    n_bins: int,
    min_frac: float,
    cluster_radius: float,
    z_slack: float,
    apply_depth_slice: bool,
) -> dict[str, Any]:
    """相机占据的方位角弧 × 仰角范围切成正方形网格。

    每格先算 ``sum(T × 命中) / 张数``，再对有相机的格取平均，平均 ≥ ``min_frac`` 才保留。
    空格跳过。相机远近不进格子。T 来自该视角权重，与全局票相同。
    """
    n = xyz.shape[0]
    cluster_xyz = select_cluster_xyz(xyz, p_world, radius=cluster_radius)
    bin_ids, patch = assign_view_patch_bins(
        p_world, [item["cam"] for item in views], n_bins=n_bins
    )
    counts = np.bincount(np.asarray(bin_ids, dtype=np.int64), minlength=int(n_bins))
    occupied = [i for i, c in enumerate(counts.tolist()) if c > 0]
    sizes = {i: int(counts[i]) for i in occupied}
    hit_mass = {i: np.zeros(n, dtype=np.float64) for i in occupied}
    per_view_stats: list[dict[str, Any]] = []

    for item, bin_id in zip(views, bin_ids):
        cam = item["cam"]
        mask = item["mask"]
        hit, selected, z_lo, z_hi = view_frustum_masks(
            xyz,
            cam,
            mask,
            cluster_xyz,
            z_slack=z_slack,
            apply_depth_slice=apply_depth_slice,
        )
        used = hit if not apply_depth_slice else selected
        hit_mass[bin_id] += float(item["weight"]) * used.astype(np.float64)
        per_view_stats.append(
            {
                "view_id": item["view_id"],
                "weight": float(item.get("weight") or 0.0),
                "az_bin": int(bin_id),
                "frustum_hits": int(used.sum()),
                "z_lo": z_lo,
                "z_hi": z_hi,
            }
        )

    mean_rate = mean_azimuth_bin_rates(hit_mass, sizes)
    selected = mean_rate + 1e-9 >= float(min_frac)
    idx = np.nonzero(selected)[0]
    centroid = xyz[idx].mean(axis=0) if idx.size else p_world.copy()
    return {
        "indices": idx,
        "votes": mean_rate,
        "selected": selected,
        "centroid": centroid,
        "per_view": per_view_stats,
        "n_selected": int(idx.size),
        "cluster_size": int(cluster_xyz.shape[0]),
        "bin_counts": [int(c) for c in counts.tolist()],
        "n_occupied": int(len(occupied)),
        "patch": patch,
    }


def load_kept_view_ids(obj_dir: Path) -> list[str]:
    kept_path = obj_dir / "projections_kept.json"
    if not kept_path.is_file():
        raise FileNotFoundError(f"缺少 {kept_path}，请先跑 filter_views_3dgs")
    recs = json.loads(kept_path.read_text(encoding="utf-8"))
    return [str(r["view_id"]).zfill(3) for r in recs]


def resolve_vote_view_ids(obj_dir: Path, *, ignore_kept: bool = False) -> tuple[list[str], bool]:
    """投票用的视角编号，以及这些编号是否来自射线滤帧后的保留列表。

    有 ``projections_kept.json`` 时沿用它，现有物体目录的行为不变。
    ``ignore_kept`` 时改读全部画面内投影，滤帧报告仍留在目录里。
    新目录只有画面内投影、没有滤帧列表时，用 ``projections.json``，权重稍后强制为 1。
    """
    kept_path = obj_dir / "projections_kept.json"
    if kept_path.is_file() and not ignore_kept:
        recs = json.loads(kept_path.read_text(encoding="utf-8"))
        return [str(r["view_id"]).zfill(3) for r in recs], True
    proj_path = obj_dir / "projections.json"
    if proj_path.is_file():
        recs = json.loads(proj_path.read_text(encoding="utf-8"))
        return [str(r["view_id"]).zfill(3) for r in recs], False
    raise FileNotFoundError(
        f"{obj_dir} 缺少 projections_kept.json 和 projections.json"
    )


def load_filter_view_index(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for v in report.get("views") or []:
        out[str(v["view_id"]).zfill(3)] = v
    return out


def resolve_mask_path(obj_dir: Path, view_id: str, *, mask_dirname: str = "mask") -> Path | None:
    p = obj_dir / mask_dirname / f"view_{view_id}.png"
    return p if p.is_file() else None


def view_frustum_masks(
    xyz: np.ndarray,
    cam: dict,
    mask: np.ndarray,
    cluster_xyz: np.ndarray,
    *,
    z_slack: float,
    apply_depth_slice: bool,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """一帧的 mask 命中，以及这一帧自己的选中集合。

    返回 (mask_hit, selected, z_lo, z_hi)。
    有簇深度带时，selected 是 mask 命中且 z 落在放宽带内的高斯。
    没有深度带时 selected 等于 mask 命中，z_lo、z_hi 为 None。
    """
    hit, z_cam = frustum_hit_mask(xyz, cam, mask)
    band = None
    if apply_depth_slice:
        band = cluster_depth_band(
            cluster_xyz,
            cam,
            frame_margin=0,
            w=int(cam["width"]),
            h=int(cam["height"]),
        )
    if band is None:
        return hit, hit, None, None
    z_lo, z_hi = band
    lo = z_lo * (1.0 - z_slack)
    hi = z_hi * (1.0 + z_slack)
    selected = hit & (z_cam >= lo) & (z_cam <= hi)
    return hit, selected, float(z_lo), float(z_hi)


def select_view_frustum(
    xyz: np.ndarray,
    cam: dict,
    mask: np.ndarray,
    cluster_xyz: np.ndarray,
    *,
    z_slack: float,
    apply_depth_slice: bool,
) -> np.ndarray:
    """这一帧 mask 视锥（含该帧深度带）选中的高斯行号。"""
    _, selected, _, _ = view_frustum_masks(
        xyz,
        cam,
        mask,
        cluster_xyz,
        z_slack=z_slack,
        apply_depth_slice=apply_depth_slice,
    )
    return np.nonzero(selected)[0].astype(np.int64)


def segment_gaussians(
    xyz: np.ndarray,
    *,
    p_world: np.ndarray,
    views: list[dict[str, Any]],
    min_vote: float,
    cluster_radius: float,
    z_slack: float,
    apply_depth_slice: bool,
) -> dict[str, Any]:
    """对已装好相机/mask/权重的视角做投票与深度切片。

    views 每项：view_id, cam, mask, weight, width, height。
    """
    n = xyz.shape[0]
    votes = np.zeros(n, dtype=np.float64)
    # 深度：任一命中视角上 z 落在该视簇带内
    depth_ok = np.zeros(n, dtype=bool)
    per_view_stats: list[dict[str, Any]] = []

    cluster_xyz = select_cluster_xyz(xyz, p_world, radius=cluster_radius)

    for item in views:
        cam = item["cam"]
        mask = item["mask"]
        w = float(item["weight"])
        view_id = item["view_id"]
        hit, selected, z_lo, z_hi = view_frustum_masks(
            xyz,
            cam,
            mask,
            cluster_xyz,
            z_slack=z_slack,
            apply_depth_slice=apply_depth_slice,
        )
        votes += w * hit.astype(np.float64)
        n_hit = int(hit.sum())
        if z_lo is not None:
            depth_ok |= selected
        per_view_stats.append(
            {
                "view_id": view_id,
                "weight": w,
                "frustum_hits": n_hit,
                "z_lo": z_lo,
                "z_hi": z_hi,
            }
        )

    voted = votes >= float(min_vote)
    if apply_depth_slice:
        # 没有任何视角算出深度带时，不因 depth_ok 全假而清空
        if np.any(depth_ok) or any(s["z_lo"] is not None for s in per_view_stats):
            selected = voted & depth_ok
        else:
            selected = voted
    else:
        selected = voted

    idx = np.nonzero(selected)[0]
    if idx.size:
        centroid = xyz[idx].mean(axis=0)
    else:
        centroid = p_world.copy()

    return {
        "indices": idx,
        "votes": votes,
        "selected": selected,
        "centroid": centroid,
        "per_view": per_view_stats,
        "n_selected": int(idx.size),
        "cluster_size": int(cluster_xyz.shape[0]),
    }


def write_xyzrgb_ply(path: Path, xyz: np.ndarray, rgb: tuple[int, int, int]) -> None:
    """选中高斯中心写成彩色点云，供 MeshLab / CloudCompare 叠看。"""
    from plyfile import PlyData, PlyElement

    n = xyz.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=dtype)
    arr["x"] = xyz[:, 0].astype(np.float32)
    arr["y"] = xyz[:, 1].astype(np.float32)
    arr["z"] = xyz[:, 2].astype(np.float32)
    arr["red"] = np.uint8(rgb[0])
    arr["green"] = np.uint8(rgb[1])
    arr["blue"] = np.uint8(rgb[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def load_views_for_object(
    obj_dir: Path,
    views_root: Path,
    report_index: dict[str, dict[str, Any]],
    *,
    use_t_weight: bool,
    mask_dirname: str = "mask",
    ignore_kept: bool = False,
) -> list[dict[str, Any]]:
    """组装有掩码和相机的视角。滤帧列表存在时才使用透射率权重。"""
    from PIL import Image

    view_ids, from_kept = resolve_vote_view_ids(obj_dir, ignore_kept=ignore_kept)
    out: list[dict[str, Any]] = []
    for vid in view_ids:
        mask_path = resolve_mask_path(obj_dir, vid, mask_dirname=mask_dirname)
        cam_path = views_root / "camera_params" / f"view_{vid}.json"
        if mask_path is None or not cam_path.is_file():
            continue
        entry = report_index.get(vid)
        # 没有滤帧列表时不读透射率，被挡视角也能进入第一次投票。
        weight = (
            view_weight_from_report(entry, use_t_weight=use_t_weight)
            if from_kept
            else 1.0
        )
        if weight <= 0.0:
            continue
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        mask = np.array(Image.open(mask_path).convert("L"))
        out.append(
            {
                "view_id": vid,
                "cam": cam,
                "mask": mask,
                "weight": weight,
                "mask_path": str(mask_path.as_posix()),
            }
        )
    return out


def _read_p_world(path: Path) -> np.ndarray:
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))["P_world"], dtype=np.float64)


def resolve_p_world(
    obj_dir: Path,
    report: dict[str, Any],
    fused: Path | None,
) -> tuple[np.ndarray, str | None]:
    """P_world：--fused，其次 report 里的 fused 路径，其次物体目录里的 fused.json，最后 report 内嵌的点。"""
    if fused is not None:
        path = resolve_data_path(fused)
        if path.is_file():
            return _read_p_world(path), path.as_posix()
    fp = report.get("fused_path")
    if fp:
        path = resolve_data_path(fp)
        if path.is_file():
            return _read_p_world(path), path.as_posix()
    local = obj_dir / "fused.json"
    if local.is_file():
        return _read_p_world(local), local.as_posix()
    if report.get("P_world"):
        return np.asarray(report["P_world"], dtype=np.float64), report.get("fused_path")
    raise FileNotFoundError(f"{obj_dir.name} 无 P_world（无 fused.json / filter_report.P_world）")


def run_object(
    obj_dir: Path,
    *,
    views_root: Path | None,
    ply: Path | None,
    fused: Path | None,
    min_vote: float,
    min_vote_frac: float | None,
    use_t_weight: bool,
    cluster_radius: float | None,
    z_slack: float,
    apply_depth_slice: bool,
    out_dir: Path | None,
    xyz: np.ndarray | None = None,
    mask_dirname: str = "mask",
    seg_filename: str = "gaussian_seg.json",
    ignore_kept: bool = False,
    sector_weight: bool = False,
    sector_deg: float = 30.0,
    sector_cap_mult: float = 3.0,
    axis_vote: bool = False,
    axis_max_deg: float = 35.0,
    axis_min_views: int = 4,
    axis_group_frac: float = 0.3,
    az_bins: bool = False,
    az_bin_count: int = 64,
    pair_vote: bool = False,
    pair_tau: float = 20.0,
    pair_min_score: float = 0.0,
    pair_occlusion_from: Path | None = None,
) -> dict[str, Any]:
    obj_dir = obj_dir.resolve()
    report_path = obj_dir / "filter_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report_index = load_filter_view_index(report)

    if views_root is None:
        vr = report.get("views_root")
        views_root = Path(vr) if vr else _REPO / "3DGS" / "test2"
    views_root = views_root.resolve()

    if ply is None:
        pp = report.get("ply_path")
        ply = Path(pp) if pp else _REPO / "3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply"
    ply = ply.resolve()

    p_world, fused_used = resolve_p_world(obj_dir, report, fused)

    if cluster_radius is None:
        cluster_radius = float(report.get("cluster_radius_effective") or 0.2)

    views = load_views_for_object(
        obj_dir,
        views_root,
        report_index,
        use_t_weight=use_t_weight,
        mask_dirname=mask_dirname,
        ignore_kept=ignore_kept,
    )
    if not views:
        raise SystemExit(f"{obj_dir.name}: 没有同时具备投影列表、mask 和相机的视角")
    if sum(bool(flag) for flag in (sector_weight, axis_vote, az_bins, pair_vote)) > 1:
        raise SystemExit("不要同时开 --sector-weight、--axis-vote、--az-bins、--pair-vote")

    sector_meta: dict[str, Any] | None = None
    axis_meta: dict[str, Any] | None = None
    az_meta: dict[str, Any] | None = None
    pair_meta: dict[str, Any] | None = None
    if sector_weight:
        sector_meta = apply_sector_vote_weights(
            views,
            p_world,
            sector_deg=sector_deg,
            cap_mult=sector_cap_mult,
        )
        print(
            f"[seg] sector-weight on: arc={sector_meta['arc_deg']:.1f}deg "
            f"n_sec={sector_meta['n_sectors']} counts={sector_meta['sector_counts']}",
            flush=True,
        )

    use_axis = False
    if axis_vote:
        for v in views:
            v["axis_id"] = assign_view_axis(p_world, v["cam"], max_deg=axis_max_deg)
        ids = [v["axis_id"] for v in views]
        counts = [sum(1 for a in ids if a == i) for i in range(3)]
        active = active_axis_ids(ids, min_views=axis_min_views)
        axis_meta = {
            "max_deg": float(axis_max_deg),
            "min_views": int(axis_min_views),
            "group_frac": float(axis_group_frac),
            "counts": {_AXIS_NAMES[i]: int(counts[i]) for i in range(3)},
            "active": [_AXIS_NAMES[i] for i in active],
            "unassigned": int(sum(1 for a in ids if a is None)),
        }
        if len(active) >= 2:
            use_axis = True
            axis_meta["mode"] = "axis"
            print(
                f"[seg] axis-vote: active={axis_meta['active']} counts={axis_meta['counts']} "
                f"unassigned={axis_meta['unassigned']} group_frac={axis_group_frac}",
                flush=True,
            )
        else:
            axis_meta["mode"] = "fallback_frac"
            min_vote_frac = 0.35
            print(
                f"[seg] axis-vote fallback: only {len(active)} axis "
                f"(need 2, min_views={axis_min_views}); plain frac=0.35",
                flush=True,
            )

    use_pair = bool(pair_vote) or pair_occlusion_from is not None
    use_az = bool(az_bins)
    if use_pair:
        min_vote = float(pair_min_score)
        min_vote_frac = None
        print(
            f"[seg] pair-vote: tau={float(pair_tau):.1f}deg keep score>{min_vote:.3f}",
            flush=True,
        )
    elif use_az:
        if min_vote_frac is None or float(min_vote_frac) < 0.0:
            raise SystemExit("--az-bins 需要 --min-vote-frac ≥ 0")
        min_vote = float(min_vote_frac)
        print(
            f"[seg] az-bins: n={int(az_bin_count)} mean_frac={min_vote:.2f}",
            flush=True,
        )
    elif not use_axis and min_vote_frac is not None and float(min_vote_frac) >= 0.0:
        min_vote = float(min_vote_frac) * len(views)
    print(
        f"[seg] {obj_dir.name} views={len(views)} "
        f"{'pair' if use_pair else 'az' if use_az else 'axis' if use_axis else f'min_vote={min_vote:.2f}'} "
        f"masks={mask_dirname} ply={ply}",
        flush=True,
    )
    if xyz is None:
        xyz = load_ply_xyz(ply)
    weight_maps = None
    footprint_maps = None
    if use_pair and pair_occlusion_from is not None:
        src = Path(pair_occlusion_from)
        if not src.is_file():
            raise FileNotFoundError(f"缺少白集 {src}")
        white = np.asarray(json.loads(src.read_text(encoding="utf-8"))["indices"], dtype=np.int64)
        from render_object_weight import render_footprint_and_stain

        print(f"[seg] footprint+stain from {src.name} white={white.size}", flush=True)
        if white.size == 0:
            raise SystemExit(
                f"[seg] 白集为空（{src.name}），终止：无法做 softedge footprint/stain"
            )
        footprint_maps, weight_maps = render_footprint_and_stain(
            ply, white, [item["cam"] for item in views]
        )
    if use_pair:
        result = segment_gaussians_pair(
            xyz,
            p_world=p_world,
            views=views,
            tau_deg=float(pair_tau),
            min_score=float(pair_min_score),
            cluster_radius=cluster_radius,
            z_slack=z_slack,
            apply_depth_slice=apply_depth_slice,
            weight_maps=weight_maps,
            footprint_maps=footprint_maps,
        )
        pair_meta = result["pair"]
        print(
            f"[seg] pair-vote views={pair_meta['n_views']} denom={pair_meta['denom']:.3f}",
            flush=True,
        )
    elif use_az:
        result = segment_gaussians_azimuth(
            xyz,
            p_world=p_world,
            views=views,
            n_bins=int(az_bin_count),
            min_frac=float(min_vote_frac),
            cluster_radius=cluster_radius,
            z_slack=z_slack,
            apply_depth_slice=apply_depth_slice,
        )
        patch = result["patch"]
        az_meta = {
            "bin_count": int(az_bin_count),
            "n_az": int(patch["n_az"]),
            "n_el": int(patch["n_el"]),
            "arc_start_deg": float(patch["arc_start_deg"]),
            "arc_deg": float(patch["arc_deg"]),
            "el_min_deg": float(patch["el_min_deg"]),
            "el_max_deg": float(patch["el_max_deg"]),
            "n_occupied": int(result["n_occupied"]),
            "bin_counts": result["bin_counts"],
            "score": "mean of per-bin sum(T*hit)/count on the camera az-el patch; empty bins skipped",
        }
        print(
            f"[seg] az-bins {patch['n_az']}x{patch['n_el']} "
            f"arc={patch['arc_deg']:.1f}deg el={patch['el_min_deg']:.1f}..{patch['el_max_deg']:.1f} "
            f"occupied={az_meta['n_occupied']}/{int(az_bin_count)}",
            flush=True,
        )
    elif use_axis:
        result = segment_gaussians_axis(
            xyz,
            p_world=p_world,
            views=views,
            active_axes=active,
            group_frac=axis_group_frac,
            cluster_radius=cluster_radius,
            z_slack=z_slack,
            apply_depth_slice=apply_depth_slice,
        )
        min_vote = 0.0
        min_vote_frac = None
    else:
        result = segment_gaussians(
            xyz,
            p_world=p_world,
            views=views,
            min_vote=min_vote,
            cluster_radius=cluster_radius,
            z_slack=z_slack,
            apply_depth_slice=apply_depth_slice,
        )

    out_dir = (out_dir or obj_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = result["indices"]
    payload = {
        "object_dir": obj_dir.name,
        "bbox_key": OBJ_DIR_TO_BBOX_KEY.get(obj_dir.name),
        "ply": str(ply.as_posix()),
        "views_root": str(views_root.as_posix()),
        "fused": fused_used,
        "P_world": p_world.tolist(),
        "centroid": result["centroid"].tolist(),
        "n_gaussians": int(xyz.shape[0]),
        "n_selected": result["n_selected"],
        "n_views_used": len(views),
        "min_vote": min_vote,
        "min_vote_frac": min_vote_frac,
        "use_t_weight": use_t_weight,
        "sector_weight": bool(sector_weight),
        "sector_weight_meta": sector_meta,
        "axis_vote": bool(axis_vote),
        "axis_vote_meta": axis_meta,
        "az_bins": bool(az_bins),
        "az_bins_meta": az_meta,
        "pair_vote": bool(pair_vote),
        "pair_vote_meta": pair_meta,
        "cluster_radius": cluster_radius,
        "cluster_size": result["cluster_size"],
        "z_slack": z_slack,
        "apply_depth_slice": apply_depth_slice,
        "indices": idx.astype(int).tolist(),
        "per_view": result["per_view"],
    }
    json_path = out_dir / seg_filename
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    ply_out = out_dir / f"{Path(seg_filename).stem}.ply"
    if idx.size:
        write_xyzrgb_ply(ply_out, xyz[idx], (220, 40, 40))
    else:
        write_xyzrgb_ply(ply_out, p_world.reshape(1, 3), (220, 40, 40))

    print(
        f"[seg] selected={result['n_selected']}/{xyz.shape[0]} "
        f"centroid={result['centroid'].tolist()} -> {json_path}",
        flush=True,
    )
    return payload


def run_default_pair_vote(
    obj_dir: Path,
    *,
    white_json: Path,
    final_out_dir: Path | None,
    views_root: Path | None,
    ply: Path | None,
    fused: Path | None,
    cluster_radius: float | None,
    z_slack: float,
    ignore_kept: bool,
    xyz: np.ndarray | None = None,
) -> dict[str, Any]:
    """两次视线加权投票。白集写到 ``white_json``，最终 ``gaussian_seg.json`` 写到物体目录或 ``final_out_dir``。"""
    common: dict[str, Any] = dict(
        views_root=views_root,
        ply=ply,
        fused=fused,
        min_vote=0.0,
        min_vote_frac=None,
        use_t_weight=False,
        cluster_radius=cluster_radius,
        z_slack=z_slack,
        apply_depth_slice=False,
        xyz=xyz,
        ignore_kept=ignore_kept,
        pair_vote=True,
        pair_tau=PAIR_TAU_DEG,
    )
    run_object(
        obj_dir,
        out_dir=white_json.parent,
        seg_filename=white_json.name,
        pair_min_score=PAIR_WHITE_MIN_SCORE,
        **common,
    )
    return run_object(
        obj_dir,
        out_dir=final_out_dir,
        seg_filename="gaussian_seg.json",
        pair_min_score=PAIR_FINAL_MIN_SCORE,
        pair_occlusion_from=white_json,
        **common,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Frustum-filter 3DGS Gaussians from multi-view masks")
    ap.add_argument("--obj-dir", type=Path, action="append", dest="obj_dirs", help="training_data/data2_* (repeatable)")
    ap.add_argument("--obj-glob", type=str, default=None, help="e.g. training_data/data2_* (excludes data2_sft)")
    ap.add_argument("--views-root", type=Path, default=None)
    ap.add_argument("--ply", type=Path, default=None)
    ap.add_argument("--fused", type=Path, default=None)
    ap.add_argument(
        "--min-vote-frac",
        type=float,
        default=None,
        help="显式使用旧的全局比例票。不传则走两次视线加权投票（白集 0.03，最终 0.15）。",
    )
    ap.add_argument("--no-t-weight", action="store_true", help="Each kept view votes 1 (ignore T(z))")
    ap.add_argument("--cluster-radius", type=float, default=None, help="Override filter_report cluster radius")
    ap.add_argument("--z-slack", type=float, default=0.15, help="With --depth-slice: expand [z_lo,z_hi] by this fraction")
    ap.add_argument(
        "--depth-slice",
        action="store_true",
        help="Restrict votes to the seed-cluster depth band (default: off)",
    )
    ap.add_argument(
        "--no-depth-slice",
        action="store_true",
        help=argparse.SUPPRESS,
    )  # legacy alias; depth slice is already off by default
    ap.add_argument(
        "--sector-weight",
        action="store_true",
        help="Soft-balance vote weights by azimuth sectors about P_world (default: off)",
    )
    ap.add_argument(
        "--sector-deg",
        type=float,
        default=30.0,
        help="Target sector width in degrees along the coverage arc (default 30)",
    )
    ap.add_argument(
        "--sector-cap-mult",
        type=float,
        default=3.0,
        help="Cap 1/count at this multiple of the median before ×T and mean-1 norm",
    )
    ap.add_argument(
        "--axis-vote",
        action="store_true",
        help="Keep a Gaussian only if it is supported on two world axes (default: off)",
    )
    ap.add_argument("--axis-max-deg", type=float, default=35.0, help="Max angle from an axis to join that group")
    ap.add_argument("--axis-min-views", type=int, default=4, help="Min views for an axis to count")
    ap.add_argument("--axis-group-frac", type=float, default=0.3, help="Within-axis hit fraction to count as support")
    ap.add_argument(
        "--az-bins",
        action="store_true",
        help="8x8 bins on the cameras' azimuth-arc x elevation patch about P_world (default: off)",
    )
    ap.add_argument(
        "--az-bin-count",
        type=int,
        default=64,
        help="Square grid over the occupied azimuth arc and elevation range (default 64 = 8x8)",
    )
    ap.add_argument(
        "--pair-vote",
        action="store_true",
        help="Signed view-pair score; keep gaussians with score > --pair-min-score (default: off)",
    )
    ap.add_argument(
        "--pair-tau",
        type=float,
        default=20.0,
        help="Angular width in degrees for same-direction downweight (default 20)",
    )
    ap.add_argument(
        "--pair-min-score",
        type=float,
        default=0.0,
        help="Keep gaussians with pair score strictly greater than this (default 0)",
    )
    ap.add_argument(
        "--pair-occlusion-from",
        type=Path,
        default=None,
        help="Second pair vote: abstain where stained object weight is below 0.5. Path is a gaussian_seg.json white set.",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Only valid with a single --obj-dir")
    args = ap.parse_args()

    dirs: list[Path] = list(args.obj_dirs or [])
    if args.obj_glob:
        dirs.extend(sorted(p for p in Path().glob(args.obj_glob) if p.is_dir() and p.name != "data2_sft"))
    if not dirs:
        ap.error("provide --obj-dir and/or --obj-glob")
    if args.out_dir is not None and len(dirs) != 1:
        ap.error("--out-dir requires exactly one --obj-dir")

    legacy = args.min_vote_frac is not None
    other_mode = any(
        (
            args.sector_weight,
            args.axis_vote,
            args.az_bins,
            args.pair_vote,
            args.pair_occlusion_from is not None,
        )
    )
    if legacy and other_mode:
        ap.error("--min-vote-frac 是旧的全局比例票，不要和其它投票开关一起用")
    failed = 0
    for d in dirs:
        try:
            if not legacy and not other_mode:
                fd, name = tempfile.mkstemp(prefix="gaussian_seg_white_", suffix=".json")
                os.close(fd)
                white = Path(name)
                try:
                    run_default_pair_vote(
                        d,
                        white_json=white,
                        final_out_dir=args.out_dir,
                        views_root=args.views_root,
                        ply=args.ply,
                        fused=args.fused,
                        cluster_radius=args.cluster_radius,
                        z_slack=args.z_slack,
                        ignore_kept=False,
                    )
                finally:
                    white.unlink(missing_ok=True)
                    white.with_suffix(".ply").unlink(missing_ok=True)
                continue
            run_object(
                d,
                views_root=args.views_root,
                ply=args.ply,
                fused=args.fused,
                min_vote=0.0,
                min_vote_frac=args.min_vote_frac,
                use_t_weight=not args.no_t_weight,
                cluster_radius=args.cluster_radius,
                z_slack=args.z_slack,
                apply_depth_slice=bool(args.depth_slice),
                out_dir=args.out_dir,
                sector_weight=bool(args.sector_weight),
                sector_deg=float(args.sector_deg),
                sector_cap_mult=float(args.sector_cap_mult),
                axis_vote=bool(args.axis_vote),
                axis_max_deg=float(args.axis_max_deg),
                axis_min_views=int(args.axis_min_views),
                axis_group_frac=float(args.axis_group_frac),
                az_bins=bool(args.az_bins),
                az_bin_count=int(args.az_bin_count),
                pair_vote=bool(args.pair_vote) or args.pair_occlusion_from is not None,
                pair_tau=float(args.pair_tau),
                pair_min_score=float(args.pair_min_score),
                pair_occlusion_from=args.pair_occlusion_from,
            )
        except (FileNotFoundError, SystemExit) as e:
            failed += 1
            print(f"[skip] {d}: {e}", flush=True)
    if failed:
        print(f"[seg] done with {failed} skipped", flush=True)


if __name__ == "__main__":
    main()
