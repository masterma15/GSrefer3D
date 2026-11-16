#!/usr/bin/env python3
"""投票结果的后处理。

默认/批量/e2e 走门控：``n_views ≤ 30`` 且 ``n_selected < 7000`` 才后处理，否则保留 softedge。
（少视角且选点仍偏少，覆盖中间剪刀；同档左/右剪、药瓶选点过万则跳过。）
门控命中时跑 **mask-jury**（多帧高置信 mask 删杂散 + 密度主门控补点）。

``--mask-direct`` / e2e ``--force-maskclose``：绕过门控强制后处理（force 在 e2e 也走 mask-jury；
CLI ``--mask-direct`` 仍为旧 maskclose 补点对照）。

``--mask-jury``：显式跑 mask-jury，不看门控。

另有一套按弱连接块挑选的接口，供对照，不是默认路径。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from repo_paths import resolve_data_path  # noqa: E402
from unproject import CameraView  # noqa: E402

# 场景尺度：0.015 个场景单位只相当于现实 0.15 厘米。1 现实厘米 = 0.1 个场景单位。
# 物体按现实大约 1.5 厘米仍算连在一起，所以生长半径用 0.15 个场景单位。
REAL_CM_TO_SCENE = 0.1
MASK_GROW_RADIUS = 1.5 * REAL_CM_TO_SCENE
LINK_RADIUS_M = 0.015
FILL_RADIUS_M = 0.015
# 一侧算「大」：不少于最大块的 20%。最大块自己总是大的。两侧都大的不送去给掩码挑。
LARGE_FRAC = 0.20
# 大块上掩码外突支：相对该大块点数 < α 且细颈（触主体比例低）才整支删。
SPUR_FRAC = 0.15
SPUR_NECK_TOUCH_MAX = 0.35
# 落在掩码里的比例差小于这个值，视为分不出胜负，弱连接块都不删。
MASK_RATIO_TIE = 0.10
# 薄片：λ2/λ1 ≥ 0.45 且 λ3/λ2 < 0.35。细条：λ2/λ1 < 0.35。
SHEET_LAMBDA21 = 0.45
SHEET_LAMBDA32 = 0.35
STICK_LAMBDA21 = 0.35
# 外缘一次最多剥掉这块的 20%。再多，或者会拆成两大块，就整步取消。
PEEL_MAX_FRAC = 0.20
MASK_FG = 128
# 后处理门控：少视角 ∧ 选点仍偏少 → mask-jury。
MASKCLOSE_FEW_VIEWS_MAX = 30
MASKCLOSE_FEW_VIEWS_N_SELECTED_MAX = 7000
# 旧「极端欠选」阈值，现行门控不再单独使用；保留给历史引用。
MASKCLOSE_N_SELECTED_MAX = 1500
# 已废弃的名字门控子串（勿再用于判定）。
POSTPROCESS_GATE_NAME_SUBSTR = "middle_scissors"
# mask-jury：高置信视角集 + 删点多数决 + 密度主门控补点
MASK_JURY_SCORE_FLOOR = 0.50
MASK_JURY_MAX_VIEWS = 5
MASK_JURY_KEEP_FRAC = 0.5
MASK_JURY_DIST_MAX = 20.0
MASK_JURY_IOU_MIN = 0.35
MASK_JURY_DENSE_RADIUS = 2.0 * MASK_GROW_RADIUS
MASK_JURY_MIN_DENSE = 8
MASK_JURY_LINK_RADIUS = MASK_GROW_RADIUS


@dataclass(frozen=True)
class PostprocessConfig:
    """后处理用的距离和形状门槛。默认就是上面这组。"""

    link_radius_m: float = LINK_RADIUS_M
    fill_radius_m: float = FILL_RADIUS_M
    large_frac: float = LARGE_FRAC
    mask_ratio_tie: float = MASK_RATIO_TIE
    sheet_lambda21: float = SHEET_LAMBDA21
    sheet_lambda32: float = SHEET_LAMBDA32
    stick_lambda21: float = STICK_LAMBDA21
    peel_max_frac: float = PEEL_MAX_FRAC


def connected_labels(xyz: np.ndarray, radius: float) -> np.ndarray:
    """距离小于 ``radius`` 的点连成一块。返回每个点的块编号，从 0 起。"""
    n = int(xyz.shape[0])
    labels = np.arange(n, dtype=np.int64)
    if n <= 1:
        return labels
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(float(radius), output_type="ndarray")
    if len(pairs) == 0:
        return labels
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    graph = csr_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return comp.astype(np.int64)


def large_component_ids(labels: np.ndarray, *, large_frac: float) -> set[int]:
    """最大块，以及不少于它 20% 的块，都算大块。"""
    if labels.size == 0:
        return set()
    ids, counts = np.unique(labels, return_counts=True)
    n_max = int(counts.max())
    cutoff = max(1, int(np.ceil(float(large_frac) * n_max)))
    return {int(i) for i, n in zip(ids, counts) if int(n) >= cutoff}


def fill_same_component(
    core_xyz: np.ndarray,
    core_labels: np.ndarray,
    white_xyz: np.ndarray,
    *,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """白集点只补进同一块。同时挨到两块的是桥，不加。

    返回 (白集行号, 这些点所属的块号)。
    """
    empty = np.zeros(0, dtype=np.int64)
    if core_xyz.size == 0 or white_xyz.size == 0:
        return empty, empty
    tree = cKDTree(core_xyz)
    neigh = tree.query_ball_point(white_xyz, r=float(radius))
    chosen: list[int] = []
    owners: list[int] = []
    for i, ids in enumerate(neigh):
        if not ids:
            continue
        labs = core_labels[np.asarray(ids, dtype=np.int64)]
        if int(labs.min()) != int(labs.max()):
            continue
        chosen.append(i)
        owners.append(int(labs[0]))
    if not chosen:
        return empty, empty
    return np.asarray(chosen, dtype=np.int64), np.asarray(owners, dtype=np.int64)


def _lambda_ratios(xyz: np.ndarray) -> tuple[float, float] | None:
    """返回 (λ2/λ1, λ3/λ2)。点太少或没有伸展时没有形状。"""
    if xyz.shape[0] < 6:
        return None
    centered = xyz - xyz.mean(axis=0)
    cov = (centered.T @ centered) / float(xyz.shape[0])
    lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    if float(lam[-1]) <= 1e-12:
        return None
    r21 = float(lam[-2] / lam[-1])
    r32 = float(lam[-3] / lam[-2]) if float(lam[-2]) > 1e-12 else 0.0
    return r21, r32


def shape_name(xyz: np.ndarray, cfg: PostprocessConfig) -> str:
    """整块或一个邻域的形状：sheet / stick / volume / small。"""
    ratios = _lambda_ratios(xyz)
    if ratios is None:
        return "small"
    r21, r32 = ratios
    if r21 >= cfg.sheet_lambda21 and r32 < cfg.sheet_lambda32:
        return "sheet"
    if r21 < cfg.stick_lambda21:
        return "stick"
    return "volume"


def peel_peripheral_sheets(
    xyz: np.ndarray,
    cfg: PostprocessConfig,
) -> np.ndarray:
    """外缘的薄层行号。整块是薄片或细条时不剥。剥太多或拆成两大块则取消。"""
    n = int(xyz.shape[0])
    empty = np.zeros(0, dtype=np.int64)
    if n < 12 or shape_name(xyz, cfg) != "volume":
        return empty
    tree = cKDTree(xyz)
    neigh = tree.query_ball_point(xyz, r=cfg.link_radius_m)
    drop: list[int] = []
    for i, ids in enumerate(neigh):
        if len(ids) < 6:
            continue
        pts = xyz[np.asarray(ids, dtype=np.int64)]
        if shape_name(pts, cfg) != "sheet":
            continue
        offset = float(np.linalg.norm(pts.mean(axis=0) - xyz[i]))
        if offset < 0.3 * cfg.link_radius_m:
            continue
        drop.append(i)
    if not drop:
        return empty
    if len(drop) > cfg.peel_max_frac * n:
        return empty
    keep = np.ones(n, dtype=bool)
    keep[np.asarray(drop, dtype=np.int64)] = False
    kept_labels = connected_labels(xyz[keep], cfg.link_radius_m)
    _, counts = np.unique(kept_labels, return_counts=True)
    large = [int(c) for c in counts if int(c) >= max(1, int(np.ceil(cfg.large_frac * n)))]
    if len(large) >= 2:
        return empty
    return np.asarray(drop, dtype=np.int64)


def frame_and_mask(
    xyz: np.ndarray,
    cam: CameraView,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """每个点是否投进画面、是否落在掩码前景。"""
    n = int(xyz.shape[0])
    in_frame = np.zeros(n, dtype=bool)
    inside = np.zeros(n, dtype=bool)
    if n == 0:
        return in_frame, inside
    p_cam = cam.world_to_cam_points(xyz)
    z = p_cam[:, 2]
    fx, fy, cx, cy = cam.intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * p_cam[:, 0] / z + cx
        v = fy * p_cam[:, 1] / z + cy
    h, w = mask.shape[:2]
    in_frame = (z > 1e-8) & np.isfinite(u) & np.isfinite(v)
    in_frame &= (u >= 0.0) & (v >= 0.0) & (u < w) & (v < h)
    ui = np.clip(np.nan_to_num(u, nan=0.0).astype(np.int64), 0, w - 1)
    vi = np.clip(np.nan_to_num(v, nan=0.0).astype(np.int64), 0, h - 1)
    inside[in_frame] = mask[vi[in_frame], ui[in_frame]] >= MASK_FG
    return in_frame, inside


def mask_grow_fill(
    base_xyz: np.ndarray,
    base_index: np.ndarray,
    pool_xyz: np.ndarray,
    pool_index: np.ndarray,
    pool_in_frame: np.ndarray,
    pool_inside: np.ndarray,
    *,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """掩码门控生长：只补「投进画面且在掩码内、距当前点集 ≤ R」的白集点。"""
    sel_xyz = np.asarray(base_xyz, dtype=np.float64).copy()
    sel_index = np.asarray(base_index, dtype=np.int64).copy()
    cand = np.asarray(pool_in_frame, dtype=bool) & np.asarray(pool_inside, dtype=bool)
    cand_index = np.asarray(pool_index[cand], dtype=np.int64)
    cand_xyz = np.asarray(pool_xyz[cand], dtype=np.float64)
    if sel_index.size and cand_index.size:
        have = {int(i) for i in sel_index.tolist()}
        fresh = np.array([int(i) not in have for i in cand_index.tolist()], dtype=bool)
        cand_index = cand_index[fresh]
        cand_xyz = cand_xyz[fresh]
    added = 0
    while sel_xyz.shape[0] > 0 and cand_index.size > 0:
        dist, _ = cKDTree(sel_xyz).query(cand_xyz, k=1)
        take = dist <= float(radius)
        if not np.any(take):
            break
        sel_xyz = np.concatenate([sel_xyz, cand_xyz[take]])
        sel_index = np.concatenate([sel_index, cand_index[take]])
        added += int(np.count_nonzero(take))
        cand_xyz = cand_xyz[~take]
        cand_index = cand_index[~take]
    return sel_xyz, sel_index, added


def morph_close_fill(
    base_xyz: np.ndarray,
    base_index: np.ndarray,
    pool_xyz: np.ndarray,
    pool_index: np.ndarray,
    *,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """形态学闭包补点：膨胀再腐蚀，不看掩码。

    宇宙 = 基点 ∪ 候选池（可为白集或全场邻域）。先把距基点 ≤ ``radius`` 的
    池内点胀进来，再丢掉胀开后仍贴着「未胀入点」的外缘壳；
    缝宽 ≤ 约 ``2*radius`` 的缺口会留下。基点一律保留。
    """
    base_xyz = np.asarray(base_xyz, dtype=np.float64)
    base_index = np.asarray(base_index, dtype=np.int64)
    pool_xyz = np.asarray(pool_xyz, dtype=np.float64)
    pool_index = np.asarray(pool_index, dtype=np.int64)
    if base_index.size == 0:
        return base_xyz.copy(), base_index.copy(), 0
    if pool_index.size == 0:
        return base_xyz.copy(), base_index.copy(), 0

    base_set = {int(i) for i in base_index.tolist()}
    fresh = np.array([int(i) not in base_set for i in pool_index.tolist()], dtype=bool)
    pool_xyz = pool_xyz[fresh]
    pool_index = pool_index[fresh]
    if pool_index.size == 0:
        return base_xyz.copy(), base_index.copy(), 0

    dist_to_base, _ = cKDTree(base_xyz).query(pool_xyz, k=1)
    dil_pool = dist_to_base <= float(radius)
    dil_xyz = np.concatenate([base_xyz, pool_xyz[dil_pool]])
    dil_index = np.concatenate([base_index, pool_index[dil_pool]])
    dil_set = {int(i) for i in dil_index.tolist()}

    uni_xyz = np.concatenate([base_xyz, pool_xyz])
    uni_index = np.concatenate([base_index, pool_index])
    outside = np.array([int(i) not in dil_set for i in uni_index.tolist()], dtype=bool)
    if not np.any(outside):
        kept_xyz, kept_index = dil_xyz, dil_index
    else:
        dist_out, _ = cKDTree(uni_xyz[outside]).query(dil_xyz, k=1)
        # 距未胀入点 > R → 落在结构元内部，腐蚀后仍在
        is_base = np.array([int(i) in base_set for i in dil_index.tolist()], dtype=bool)
        survive = is_base | (dist_out > float(radius))
        kept_xyz = dil_xyz[survive]
        kept_index = dil_index[survive]

    n_added = int(sum(1 for i in kept_index.tolist() if int(i) not in base_set))
    return kept_xyz, kept_index, n_added


def drop_exterior_spurs(
    xyz: np.ndarray,
    *,
    in_frame: np.ndarray,
    inside: np.ndarray,
    labels: np.ndarray,
    large_ids: set[int],
    radius: float,
    spur_frac: float = SPUR_FRAC,
    neck_touch_max: float = SPUR_NECK_TOUCH_MAX,
) -> tuple[np.ndarray, int]:
    """大块上掩码外、细颈连出的外突支行号掩码，以及删掉的支数。

    投不进画面的点当种子（不判外突）。掩码内空洞若触主体比例高则保留，避免掏空。
    """
    n = int(xyz.shape[0])
    drop = np.zeros(n, dtype=bool)
    n_spurs = 0
    if n == 0 or not large_ids:
        return drop, 0
    seed = (~in_frame) | inside
    for lab in large_ids:
        sel = labels == int(lab)
        n_main = int(np.count_nonzero(sel))
        if n_main == 0:
            continue
        out = sel & in_frame & ~inside
        if not np.any(out):
            continue
        seed_local = sel & seed
        if not np.any(seed_local):
            continue
        out_idx = np.flatnonzero(out)
        spur_labs = connected_labels(xyz[out_idx], radius)
        tree = cKDTree(xyz[seed_local])
        for s in np.unique(spur_labs):
            members = out_idx[spur_labs == int(s)]
            n_spur = int(members.size)
            if n_spur >= float(spur_frac) * n_main:
                continue
            dist, _ = tree.query(xyz[members], k=1)
            n_neck = int(np.count_nonzero(dist <= float(radius)))
            if n_spur > 0 and (n_neck / n_spur) <= float(neck_touch_max):
                drop[members] = True
                n_spurs += 1
    return drop, n_spurs


def _drop_blocks_and_spurs(
    sel_xyz: np.ndarray,
    sel_index: np.ndarray,
    *,
    cam: CameraView,
    mask: np.ndarray,
    radius: float,
    large_frac: float,
    outside_keep: float,
    spur_frac: float,
    neck_touch_max: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """离群小块 + 外突支删除。统计里带两类删点的 ply 行号。"""
    labels = connected_labels(sel_xyz, radius)
    large_ids = large_component_ids(labels, large_frac=large_frac)
    in_frame, inside = frame_and_mask(sel_xyz, cam, mask)
    drop_lab: set[int] = set()
    for lab in np.unique(labels):
        lab = int(lab)
        if lab in large_ids:
            continue
        sel = labels == lab
        framed = in_frame[sel]
        if not np.any(framed):
            continue
        ratio = float(np.mean(inside[sel][framed]))
        if ratio < float(outside_keep):
            drop_lab.add(lab)

    block_drop = np.array([int(lab) in drop_lab for lab in labels], dtype=bool)
    keep = ~block_drop
    keep_xyz = sel_xyz[keep]
    keep_index = sel_index[keep]
    block_drop_index = np.unique(sel_index[block_drop])

    labels2 = connected_labels(keep_xyz, radius)
    large2 = large_component_ids(labels2, large_frac=large_frac)
    in_frame2, inside2 = frame_and_mask(keep_xyz, cam, mask)
    spur_drop, n_spurs = drop_exterior_spurs(
        keep_xyz,
        in_frame=in_frame2,
        inside=inside2,
        labels=labels2,
        large_ids=large2,
        radius=radius,
        spur_frac=spur_frac,
        neck_touch_max=neck_touch_max,
    )
    spur_drop_index = np.unique(keep_index[spur_drop])
    out_xyz = keep_xyz[~spur_drop]
    out_index = keep_index[~spur_drop]
    return out_xyz, out_index, {
        "n_blocks_dropped": int(len(drop_lab)),
        "n_spurs_dropped": int(n_spurs),
        "n_deleted": int(block_drop_index.size) + int(spur_drop_index.size),
        "n_large_blocks": int(len(large_ids)),
        "block_drop_index": block_drop_index.astype(np.int64),
        "spur_drop_index": spur_drop_index.astype(np.int64),
    }


def scene_close_pool(
    sel_xyz: np.ndarray,
    sel_index: np.ndarray,
    scene_xyz: np.ndarray,
    *,
    band: float,
) -> tuple[np.ndarray, np.ndarray]:
    """当前点集 ``band`` 邻域内的全场点（不含已选），作闭包候选池。"""
    sel_set = {int(i) for i in np.asarray(sel_index, dtype=np.int64).tolist()}
    if sel_xyz.shape[0] == 0 or scene_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int64)
    dist, _ = cKDTree(sel_xyz).query(scene_xyz, k=1)
    near = dist <= float(band)
    idx = np.flatnonzero(near)
    keep = np.array([int(i) not in sel_set for i in idx.tolist()], dtype=bool)
    idx = idx[keep]
    return np.asarray(scene_xyz[idx], dtype=np.float64), idx.astype(np.int64)


def mask_direct_select(
    base_xyz: np.ndarray,
    base_index: np.ndarray,
    pool_xyz: np.ndarray,
    pool_index: np.ndarray,
    pool_in_frame: np.ndarray,
    pool_inside: np.ndarray,
    *,
    cam: CameraView,
    mask: np.ndarray,
    radius: float,
    large_frac: float = LARGE_FRAC,
    outside_keep: float = 0.5,
    spur_frac: float = SPUR_FRAC,
    neck_touch_max: float = SPUR_NECK_TOUCH_MAX,
    scene_xyz: np.ndarray | None = None,
    close_band: float | None = None,
) -> dict[str, Any]:
    """掩码补点清支 →（可选）全场邻域闭包补结构 → 再清外突支。

    1. 基点全留；白集掩码内按 ``radius`` 生长补点。
    2. 离群小块 + 细颈外突支删除。
    3. 若给 ``scene_xyz``：在选中点 ``close_band``（默认 ``2*radius``）邻域做闭包。
    4. 闭包后再清一次外突支。
    """
    sel_xyz, sel_index, added_mask = mask_grow_fill(
        base_xyz,
        base_index,
        pool_xyz,
        pool_index,
        pool_in_frame,
        pool_inside,
        radius=radius,
    )

    if sel_index.size == 0:
        return {
            "indices": np.zeros(0, dtype=np.int64),
            "n_base": int(base_index.size),
            "n_deleted": 0,
            "n_added": 0,
            "n_added_mask": 0,
            "n_added_close": 0,
            "n_selected": 0,
            "delete_mode": "empty",
        }

    sel_xyz, sel_index, st1 = _drop_blocks_and_spurs(
        sel_xyz,
        sel_index,
        cam=cam,
        mask=mask,
        radius=radius,
        large_frac=large_frac,
        outside_keep=outside_keep,
        spur_frac=spur_frac,
        neck_touch_max=neck_touch_max,
    )
    n_deleted = int(st1["n_deleted"])
    n_blocks = int(st1["n_blocks_dropped"])
    n_spurs = int(st1["n_spurs_dropped"])
    n_large = int(st1["n_large_blocks"])
    added_close = 0

    if scene_xyz is not None and sel_index.size > 0:
        band = float(2.0 * radius if close_band is None else close_band)
        pool_xyz_s, pool_index_s = scene_close_pool(
            sel_xyz, sel_index, np.asarray(scene_xyz, dtype=np.float64), band=band
        )
        sel_xyz, sel_index, added_close = morph_close_fill(
            sel_xyz, sel_index, pool_xyz_s, pool_index_s, radius=radius
        )
        sel_xyz, sel_index, st2 = _drop_blocks_and_spurs(
            sel_xyz,
            sel_index,
            cam=cam,
            mask=mask,
            radius=radius,
            large_frac=large_frac,
            outside_keep=outside_keep,
            spur_frac=spur_frac,
            neck_touch_max=neck_touch_max,
        )
        n_deleted += int(st2["n_deleted"])
        n_blocks += int(st2["n_blocks_dropped"])
        n_spurs += int(st2["n_spurs_dropped"])

    final_index = np.unique(sel_index)
    modes: list[str] = []
    if n_blocks:
        modes.append("discrete_blocks")
    if n_spurs:
        modes.append("exterior_spurs")
    if added_close:
        modes.append("scene_morph_close")
    return {
        "indices": final_index,
        "n_base": int(base_index.size),
        "n_deleted": n_deleted,
        "n_added": int(added_mask + added_close),
        "n_added_mask": int(added_mask),
        "n_added_close": int(added_close),
        "n_selected": int(final_index.size),
        "delete_mode": "+".join(modes) if modes else "none",
        "n_blocks_dropped": n_blocks,
        "n_spurs_dropped": n_spurs,
        "n_large_blocks": n_large,
    }


def mask_inside_ratio(
    xyz: np.ndarray,
    cam: CameraView,
    mask: np.ndarray,
) -> float:
    """投进画面的点里，落在掩码前景的比例。没有点投进画面时记 0。"""
    in_frame, inside = frame_and_mask(xyz, cam, mask)
    if not np.any(in_frame):
        return 0.0
    return float(np.mean(inside[in_frame]))


def weak_keep_ids(
    labels: np.ndarray,
    *,
    large_ids: set[int],
    inside_ratio: dict[int, float],
    tie: float,
) -> tuple[set[int], str]:
    """弱连接块里留一块。比例接近时一块都不删。返回留下的块号和原因。"""
    weak = [int(i) for i in np.unique(labels) if int(i) not in large_ids]
    if len(weak) <= 1:
        return large_ids | set(weak), "no_contest"
    ratios = sorted((float(inside_ratio.get(i, 0.0)), i) for i in weak)
    best_ratio, best_id = ratios[-1]
    second_ratio = ratios[-2][0]
    if best_ratio - second_ratio < float(tie):
        return large_ids | set(weak), "tie"
    return large_ids | {int(best_id)}, "best"


def best_select_view(rows: list[dict[str, Any]]) -> tuple[str, float]:
    """``question.json`` 里综合分最高的视角。没有掩码记录时报错。"""
    best_id: str | None = None
    best_score = -1e9
    for row in rows:
        boxes = (row.get("mask_meta") or {}).get("ranked_boxes") or []
        if not boxes or "view_id" not in row:
            continue
        score = float(boxes[0].get("select_score") or -1e9)
        if score > best_score:
            best_score = score
            best_id = str(row["view_id"])
    if best_id is None:
        raise FileNotFoundError("question.json 里没有带 select_score 的掩码")
    return best_id, best_score


def _view_mask_quality_ok(row: dict[str, Any]) -> bool:
    meta = row.get("mask_meta") or {}
    if bool(meta.get("point_in_mask")):
        return True
    dist = meta.get("dist_to_mask")
    if dist is not None and float(dist) <= MASK_JURY_DIST_MAX:
        return True
    iou = meta.get("box_mask_iou")
    if iou is not None and float(iou) >= MASK_JURY_IOU_MIN:
        return True
    return False


def select_jury_views(
    rows: list[dict[str, Any]],
    *,
    obj_dir: Path,
    score_floor: float = MASK_JURY_SCORE_FLOOR,
    max_views: int = MASK_JURY_MAX_VIEWS,
) -> tuple[list[tuple[str, float]], str]:
    """高置信 mask 视角：score 下限 + 质量过滤，最多 ``max_views``。

    返回 ``([(view_id, score), ...], fallback_reason)``。
    ``fallback_reason`` 为空表示正常；``best_only`` 表示过滤后为空、回退最高分一帧。
    """
    scored: list[tuple[str, float]] = []
    for row in rows:
        boxes = (row.get("mask_meta") or {}).get("ranked_boxes") or []
        if not boxes or "view_id" not in row:
            continue
        score = float(boxes[0].get("select_score") or -1e9)
        if score < float(score_floor):
            continue
        if not _view_mask_quality_ok(row):
            continue
        vid = str(row["view_id"]).zfill(3)
        if not (obj_dir / "mask" / f"view_{vid}.png").is_file():
            continue
        scored.append((vid, score))
    scored.sort(key=lambda x: -x[1])
    # 同 view 去重，保留最高分
    seen: set[str] = set()
    uniq: list[tuple[str, float]] = []
    for vid, score in scored:
        if vid in seen:
            continue
        seen.add(vid)
        uniq.append((vid, score))
        if len(uniq) >= int(max_views):
            break
    if uniq:
        return uniq, ""
    best_id, best_score = best_select_view(rows)
    vid = str(best_id).zfill(3)
    if not (obj_dir / "mask" / f"view_{vid}.png").is_file():
        raise FileNotFoundError(f"回退视角 mask 不存在: view_{vid}.png")
    return [(vid, float(best_score))], "best_only"


def load_view_cams_masks(
    view_ids: list[str],
    *,
    obj_dir: Path,
    views_root: Path,
) -> tuple[list[CameraView], list[np.ndarray], list[str]]:
    from PIL import Image

    cams: list[CameraView] = []
    masks: list[np.ndarray] = []
    kept: list[str] = []
    for vid in view_ids:
        cam_path = views_root / "camera_params" / f"view_{vid}.json"
        mask_path = obj_dir / "mask" / f"view_{vid}.png"
        if not cam_path.is_file() or not mask_path.is_file():
            continue
        cams.append(CameraView.from_json(cam_path))
        masks.append(np.asarray(Image.open(mask_path).convert("L")))
        kept.append(vid)
    if not cams:
        raise FileNotFoundError("没有可用的 camera/mask 视角")
    return cams, masks, kept


def mask_hit_vis_counts(
    xyz: np.ndarray,
    cams: list[CameraView],
    masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """每个点在多视上的 mask 命中数 / 可投影数。"""
    n = int(xyz.shape[0])
    n_hit = np.zeros(n, dtype=np.int32)
    n_vis = np.zeros(n, dtype=np.int32)
    if n == 0 or not cams:
        return n_hit, n_vis
    for cam, mask in zip(cams, masks):
        in_frame, inside = frame_and_mask(xyz, cam, mask)
        n_vis += in_frame.astype(np.int32)
        n_hit += (in_frame & inside).astype(np.int32)
    return n_hit, n_vis


def jury_delete_keep_mask(
    xyz: np.ndarray,
    cams: list[CameraView],
    masks: list[np.ndarray],
    *,
    keep_frac: float = MASK_JURY_KEEP_FRAC,
    min_vis: int = 2,
) -> np.ndarray:
    """返回与 ``xyz`` 同长的 bool：True 保留。"""
    n_hit, n_vis = mask_hit_vis_counts(xyz, cams, masks)
    keep = np.ones(int(xyz.shape[0]), dtype=bool)
    enough = n_vis >= int(min_vis)
    ratio = np.zeros(int(xyz.shape[0]), dtype=np.float64)
    ratio[enough] = n_hit[enough] / np.maximum(n_vis[enough], 1)
    keep[enough & (ratio < float(keep_frac))] = False
    return keep


def density_mask_fill(
    core_xyz: np.ndarray,
    core_index: np.ndarray,
    pool_xyz: np.ndarray,
    pool_index: np.ndarray,
    cams: list[CameraView],
    masks: list[np.ndarray],
    *,
    dense_radius: float = MASK_JURY_DENSE_RADIUS,
    min_dense: int = MASK_JURY_MIN_DENSE,
    link_radius: float = MASK_JURY_LINK_RADIUS,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """密度主门控补点：核邻域够密，且至少一视 mask 命中。

    返回 ``(sel_xyz, sel_index, n_added, n_rejected)``。
    """
    core_xyz = np.asarray(core_xyz, dtype=np.float64)
    core_index = np.asarray(core_index, dtype=np.int64)
    pool_xyz = np.asarray(pool_xyz, dtype=np.float64)
    pool_index = np.asarray(pool_index, dtype=np.int64)
    if core_index.size == 0 or pool_index.size == 0:
        return core_xyz.copy(), core_index.copy(), 0, int(pool_index.size)

    have = {int(i) for i in core_index.tolist()}
    fresh = np.array([int(i) not in have for i in pool_index.tolist()], dtype=bool)
    pool_xyz = pool_xyz[fresh]
    pool_index = pool_index[fresh]
    if pool_index.size == 0:
        return core_xyz.copy(), core_index.copy(), 0, 0

    n_hit, n_vis = mask_hit_vis_counts(pool_xyz, cams, masks)
    mask_ok = (n_vis >= 1) & (n_hit >= 1)

    tree = cKDTree(core_xyz)
    # 邻居数（不含自身；候选不在树上）
    neigh = tree.query_ball_point(pool_xyz, r=float(dense_radius))
    dense_ok = np.array([len(ids) >= int(min_dense) for ids in neigh], dtype=bool)
    link_ok = np.array([len(ids) >= 1 for ids in tree.query_ball_point(pool_xyz, r=float(link_radius))], dtype=bool)

    take = mask_ok & dense_ok & link_ok
    n_added = int(np.count_nonzero(take))
    n_rejected = int(pool_index.size - n_added)
    if n_added == 0:
        return core_xyz.copy(), core_index.copy(), 0, n_rejected
    sel_xyz = np.concatenate([core_xyz, pool_xyz[take]])
    sel_index = np.concatenate([core_index, pool_index[take]])
    return sel_xyz, sel_index, n_added, n_rejected


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def postprocess_indices(
    core_xyz: np.ndarray,
    core_index: np.ndarray,
    white_xyz: np.ndarray,
    white_index: np.ndarray,
    *,
    mask_xyz_ratio,
    cfg: PostprocessConfig | None = None,
) -> dict[str, Any]:
    """按四步得到新的 ply 行号。

    ``mask_xyz_ratio(labels, xyz)`` 给出每个块落在最高分掩码里的比例。
    这样单测不必读图片。
    """
    cfg = cfg or PostprocessConfig()
    labels = connected_labels(core_xyz, cfg.link_radius_m)
    large_ids = large_component_ids(labels, large_frac=cfg.large_frac)
    fill_rows, filled_labels = fill_same_component(
        core_xyz, labels, white_xyz, radius=cfg.fill_radius_m
    )
    filled_index = white_index[fill_rows]
    filled_xyz = white_xyz[fill_rows]

    # 补洞点跟它所在的块走。薄层按块剥，掩码选择之前做。
    ply_ids = [core_index]
    piece_labels = [labels]
    if filled_index.size:
        ply_ids.append(filled_index)
        piece_labels.append(filled_labels)
    all_index = np.concatenate(ply_ids)
    all_labels = np.concatenate(piece_labels)
    all_xyz = np.concatenate([core_xyz, filled_xyz]) if filled_xyz.size else core_xyz

    drop_mask = np.zeros(all_index.shape[0], dtype=bool)
    for lab in np.unique(all_labels):
        sel = np.flatnonzero(all_labels == int(lab))
        local_drop = peel_peripheral_sheets(all_xyz[sel], cfg)
        if local_drop.size:
            drop_mask[sel[local_drop]] = True
    keep_geom = ~drop_mask
    ratios = mask_xyz_ratio(all_labels[keep_geom], all_xyz[keep_geom])
    kept_ids, reason = weak_keep_ids(
        all_labels[keep_geom],
        large_ids=large_ids,
        inside_ratio=ratios,
        tie=cfg.mask_ratio_tie,
    )
    choose = np.array([int(lab) in kept_ids for lab in all_labels], dtype=bool)
    final_sel = keep_geom & choose
    final_index = np.unique(all_index[final_sel])

    comp_rows = []
    for lab in sorted(int(i) for i in np.unique(labels)):
        comp_rows.append(
            {
                "id": lab,
                "n_core": int(np.sum(labels == lab)),
                "large": lab in large_ids,
                "inside_ratio": None if lab not in ratios else float(ratios[lab]),
                "kept": lab in kept_ids,
            }
        )
    return {
        "indices": final_index,
        "n_core": int(core_index.size),
        "n_filled": int(filled_index.size),
        "n_peeled": int(drop_mask.sum()),
        "n_selected": int(final_index.size),
        "mask_choice": reason,
        "components": comp_rows,
    }


def _ratios_from_mask(cam: CameraView, mask: np.ndarray):
    """返回闭包：按块把点投进这张掩码。"""

    def ratios(labels: np.ndarray, xyz: np.ndarray) -> dict[int, float]:
        out: dict[int, float] = {}
        for lab in np.unique(labels):
            sel = labels == int(lab)
            out[int(lab)] = mask_inside_ratio(xyz[sel], cam, mask)
        return out

    return ratios


def postprocess_vote(
    core_json: Path,
    white_json: Path,
    *,
    out_json: Path,
    cfg: PostprocessConfig | None = None,
) -> dict[str, Any]:
    """读核、白集、最高分掩码，写出新的 ``gaussian_seg.json``。不注入。"""
    from ray_visibility import load_ply_xyz

    cfg = cfg or PostprocessConfig()
    core_payload = _load_json(core_json)
    white_payload = _load_json(white_json)
    if not isinstance(core_payload, dict) or not isinstance(white_payload, dict):
        raise TypeError("gaussian_seg.json 应为对象")
    core_index = np.asarray(core_payload["indices"], dtype=np.int64)
    white_index = np.asarray(white_payload["indices"], dtype=np.int64)
    ply_path = resolve_data_path(core_payload["ply"])
    xyz = load_ply_xyz(ply_path)
    obj_dir = _object_dir(core_json)
    question = _load_json(obj_dir / "question.json")
    if not isinstance(question, list):
        raise TypeError("question.json 应为列表")
    view_id, select_score = best_select_view(question)
    cam = CameraView.from_json(_views_root(core_payload) / "camera_params" / f"view_{view_id}.json")
    from PIL import Image

    mask = np.asarray(Image.open(obj_dir / "mask" / f"view_{view_id}.png").convert("L"))
    white_only = np.setdiff1d(white_index, core_index, assume_unique=False)
    result = postprocess_indices(
        xyz[core_index],
        core_index,
        xyz[white_only],
        white_only,
        mask_xyz_ratio=_ratios_from_mask(cam, mask),
        cfg=cfg,
    )
    out = dict(core_payload)
    idx = result["indices"]
    out["indices"] = idx.astype(int).tolist()
    out["n_selected"] = int(idx.size)
    out["centroid"] = xyz[idx].mean(axis=0).tolist() if idx.size else core_payload.get("centroid")
    out["postprocess"] = {
        "link_radius_m": cfg.link_radius_m,
        "fill_radius_m": cfg.fill_radius_m,
        "large_frac": cfg.large_frac,
        "mask_ratio_tie": cfg.mask_ratio_tie,
        "mask_view": view_id,
        "select_score": select_score,
        "n_core": result["n_core"],
        "n_filled": result["n_filled"],
        "n_peeled": result["n_peeled"],
        "n_selected": result["n_selected"],
        "mask_choice": result["mask_choice"],
        "components": result["components"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[post] view_{view_id} score={select_score:.3f} "
        f"core={result['n_core']} fill={result['n_filled']} "
        f"peel={result['n_peeled']} selected={result['n_selected']} "
        f"choice={result['mask_choice']} -> {out_json}",
        flush=True,
    )
    return out


def mask_direct_vote(
    base_json: Path,
    white_json: Path,
    *,
    obj_dir: Path,
    out_json: Path,
    radius: float = MASK_GROW_RADIUS,
) -> dict[str, Any]:
    """maskclose：掩码白集补点 → 清支 → 全场邻域闭包 → 再清支。"""
    from PIL import Image

    from ray_visibility import load_ply_xyz

    base_payload = _load_json(base_json)
    white_payload = _load_json(white_json)
    if not isinstance(base_payload, dict) or not isinstance(white_payload, dict):
        raise TypeError("gaussian_seg.json 应为对象")
    question = _load_json(obj_dir / "question.json")
    if not isinstance(question, list):
        raise TypeError("question.json 应为列表")
    view_id, select_score = best_select_view(question)
    ply_path = resolve_data_path(base_payload["ply"])
    xyz = load_ply_xyz(ply_path)
    base_index = np.asarray(base_payload["indices"], dtype=np.int64)
    white_index = np.setdiff1d(
        np.asarray(white_payload["indices"], dtype=np.int64),
        base_index,
        assume_unique=False,
    )
    cam = CameraView.from_json(_views_root(base_payload) / "camera_params" / f"view_{view_id}.json")
    mask = np.asarray(Image.open(obj_dir / "mask" / f"view_{view_id}.png").convert("L"))
    pool_in_frame, pool_inside = frame_and_mask(xyz[white_index], cam, mask)
    result = mask_direct_select(
        xyz[base_index],
        base_index,
        xyz[white_index],
        white_index,
        pool_in_frame,
        pool_inside,
        cam=cam,
        mask=mask,
        radius=radius,
        scene_xyz=xyz,
        close_band=2.0 * float(radius),
    )
    out = dict(base_payload)
    idx = result["indices"]
    out["indices"] = idx.astype(int).tolist()
    out["n_selected"] = int(idx.size)
    out["centroid"] = xyz[idx].mean(axis=0).tolist() if idx.size else base_payload.get("centroid")
    out["postprocess"] = {
        "mode": "mask_then_scene_morph_close",
        "grow_radius_scene": float(radius),
        "grow_radius_real_cm": float(radius) / REAL_CM_TO_SCENE,
        "close_band_scene": 2.0 * float(radius),
        "spur_frac": SPUR_FRAC,
        "spur_neck_touch_max": SPUR_NECK_TOUCH_MAX,
        "mask_view": view_id,
        "select_score": select_score,
        "n_base": result["n_base"],
        "n_deleted": result["n_deleted"],
        "n_added": result["n_added"],
        "n_added_mask": result.get("n_added_mask"),
        "n_added_close": result.get("n_added_close"),
        "n_selected": result["n_selected"],
        "delete_mode": result.get("delete_mode"),
        "n_blocks_dropped": result.get("n_blocks_dropped"),
        "n_spurs_dropped": result.get("n_spurs_dropped"),
        "n_large_blocks": result.get("n_large_blocks"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[mask] view_{view_id} score={select_score:.3f} radius={radius:.3f} "
        f"base={result['n_base']} del={result['n_deleted']} "
        f"add_mask={result.get('n_added_mask')} add_close={result.get('n_added_close')} "
        f"blocks_drop={result.get('n_blocks_dropped')} "
        f"spurs_drop={result.get('n_spurs_dropped')} selected={result['n_selected']} "
        f"-> {out_json}",
        flush=True,
    )
    return out


def mask_jury_vote(
    base_json: Path,
    white_json: Path | None,
    *,
    obj_dir: Path,
    out_json: Path,
    score_floor: float = MASK_JURY_SCORE_FLOOR,
    max_views: int = MASK_JURY_MAX_VIEWS,
    keep_frac: float = MASK_JURY_KEEP_FRAC,
    dense_radius: float = MASK_JURY_DENSE_RADIUS,
    min_dense: int = MASK_JURY_MIN_DENSE,
) -> dict[str, Any]:
    """多帧高置信 mask 删点 + 密度主门控补点（白集可选）。"""
    from ray_visibility import load_ply_xyz

    base_payload = _load_json(base_json)
    if not isinstance(base_payload, dict):
        raise TypeError("gaussian_seg.json 应为对象")
    question = _load_json(obj_dir / "question.json")
    if not isinstance(question, list):
        raise TypeError("question.json 应为列表")

    jury, fallback = select_jury_views(
        question, obj_dir=obj_dir, score_floor=score_floor, max_views=max_views
    )
    views_root = _views_root(base_payload)
    cams, masks, jury_ids = load_view_cams_masks(
        [v for v, _ in jury], obj_dir=obj_dir, views_root=views_root
    )
    jury_scores = {vid: sc for vid, sc in jury}

    ply_path = resolve_data_path(base_payload["ply"])
    xyz = load_ply_xyz(ply_path)
    base_index = np.asarray(base_payload["indices"], dtype=np.int64)
    base_xyz = xyz[base_index]

    keep = jury_delete_keep_mask(base_xyz, cams, masks, keep_frac=keep_frac)
    kept_index = base_index[keep]
    kept_xyz = base_xyz[keep]
    n_deleted = int((~keep).sum())

    n_added = 0
    n_rejected_fill = 0
    if white_json is not None and Path(white_json).is_file() and kept_index.size > 0:
        white_payload = _load_json(white_json)
        if not isinstance(white_payload, dict):
            raise TypeError("gaussian_seg_white.json 应为对象")
        white_index = np.setdiff1d(
            np.asarray(white_payload["indices"], dtype=np.int64),
            kept_index,
            assume_unique=False,
        )
        # 补点 mask：优先 Jury；若某点在 Jury 上全不可见，用同一套 Jury 统计即可（n_vis=0 则不补）
        kept_xyz, kept_index, n_added, n_rejected_fill = density_mask_fill(
            kept_xyz,
            kept_index,
            xyz[white_index],
            white_index,
            cams,
            masks,
            dense_radius=dense_radius,
            min_dense=min_dense,
        )

    final_index = np.unique(kept_index)
    out = dict(base_payload)
    out["indices"] = final_index.astype(int).tolist()
    out["n_selected"] = int(final_index.size)
    out["centroid"] = (
        xyz[final_index].mean(axis=0).tolist() if final_index.size else base_payload.get("centroid")
    )
    out["postprocess"] = {
        "mode": "mask_jury",
        "jury_views": [
            {"view_id": vid, "select_score": float(jury_scores.get(vid, 0.0))} for vid in jury_ids
        ],
        "jury_fallback": fallback or None,
        "mask_score_floor": float(score_floor),
        "mask_keep_frac": float(keep_frac),
        "dense_radius_scene": float(dense_radius),
        "min_dense_count": int(min_dense),
        "n_base": int(base_index.size),
        "n_deleted_mask": n_deleted,
        "n_added_gated": n_added,
        "n_rejected_fill": n_rejected_fill,
        "n_selected": int(final_index.size),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[jury] views={jury_ids} fallback={fallback or '-'} "
        f"base={base_index.size} del={n_deleted} add={n_added} "
        f"reject_fill={n_rejected_fill} selected={final_index.size} -> {out_json}",
        flush=True,
    )
    return out


def should_run_maskclose(
    n_selected: int | None = None,
    n_views_used: int | None = None,
    *,
    object_dir: str | Path | None = None,
    n_selected_max: int = MASKCLOSE_N_SELECTED_MAX,
    few_views_max: int = MASKCLOSE_FEW_VIEWS_MAX,
    few_views_n_selected_max: int = MASKCLOSE_FEW_VIEWS_N_SELECTED_MAX,
) -> tuple[bool, str]:
    """是否对 softedge 核跑后处理（e2e/门控默认：mask-jury）。

    现行：``n_views_used ≤ few_views_max`` 且 ``n_selected < few_views_n_selected_max``。
    ``object_dir`` / ``n_selected_max`` 保留签名兼容，不参与判定。
    """
    _ = (object_dir, n_selected_max)
    if n_selected is None:
        return False, "missing_n_selected"
    if n_views_used is None:
        return False, "missing_n_views"
    n_sel = int(n_selected)
    n_views = int(n_views_used)
    if n_views <= int(few_views_max) and n_sel < int(few_views_n_selected_max):
        return True, f"few_views<={int(few_views_max)}_and_n_selected<{int(few_views_n_selected_max)}"
    return False, "softedge_ok"


def maskclose_gate_from_core(
    core_payload: dict[str, Any],
    *,
    object_dir: str | Path | None = None,
    **kwargs: Any,
) -> tuple[bool, str, int | None, int | None]:
    """从 softedge ``gaussian_seg.json`` 读选点数/视角数并门控。"""
    n_sel = core_payload.get("n_selected")
    if n_sel is None and "indices" in core_payload:
        n_sel = len(core_payload["indices"])
    n_views = core_payload.get("n_views_used")
    enable, reason = should_run_maskclose(
        None if n_sel is None else int(n_sel),
        None if n_views is None else int(n_views),
        object_dir=object_dir if object_dir is not None else core_payload.get("object_dir"),
        **kwargs,
    )
    return enable, reason, None if n_sel is None else int(n_sel), None if n_views is None else int(n_views)


def write_gated_skip(
    core_json: Path,
    *,
    out_json: Path,
    reason: str,
    n_selected: int | None,
    n_views_used: int | None,
) -> dict[str, Any]:
    """门控未命中：把 softedge 核原样写出，并记下跳过原因。"""
    payload = _load_json(core_json)
    if not isinstance(payload, dict):
        raise TypeError("gaussian_seg.json 应为对象")
    out = dict(payload)
    out["postprocess"] = {
        "mode": "gated_skip",
        "gate_reason": reason,
        "n_selected_core": n_selected,
        "n_views_used": n_views_used,
        "thresholds": {
            "few_views_max": MASKCLOSE_FEW_VIEWS_MAX,
            "few_views_n_selected_max": MASKCLOSE_FEW_VIEWS_N_SELECTED_MAX,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[gate] skip maskclose reason={reason} n_selected={n_selected} "
        f"n_views={n_views_used} -> {out_json}",
        flush=True,
    )
    return out


def _object_dir(core_json: Path) -> Path:
    """``试验_pair_softedge/gaussian_seg.json`` 的上两级是物体目录。"""
    parent = core_json.resolve().parent
    if parent.name.startswith("试验_"):
        return parent.parent
    return parent


def _views_root(payload: dict[str, Any]) -> Path:
    raw = payload.get("views_root")
    if not raw:
        raise FileNotFoundError("gaussian_seg.json 没有 views_root")
    return resolve_data_path(raw)


def inject_post(seg_json: Path, *, iteration_name: str) -> Path:
    """用现有注入把后处理索引写成 SIBR 迭代。不画 OBB。"""
    from inject_gaussian_seg import inject_gaussian_seg

    seg = _load_json(seg_json)
    if not isinstance(seg, dict):
        raise TypeError("gaussian_seg.json 应为对象")
    ply = resolve_data_path(seg["ply"])
    model = ply.parent.parent.parent
    out_dir = model / "point_cloud" / f"iteration_{iteration_name}"
    inject_gaussian_seg(
        ply=ply,
        seg=seg,
        out_iteration_dir=out_dir,
        obb=None,
        object_key=None,
        add_obb=False,
        add_seed=True,
    )
    print(f"[post] inject {iteration_name}", flush=True)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="投票点云后处理（maskclose / 门控 maskclose / mask-jury）")
    ap.add_argument("--core", type=Path, required=True, help="分数 > 0.15 的 gaussian_seg.json")
    ap.add_argument(
        "--white",
        type=Path,
        default=None,
        help="分数 > 0.03 的 gaussian_seg_white.json；--mask-jury 时可缺省（只删不补）",
    )
    ap.add_argument("--out", type=Path, required=True, help="写出的 gaussian_seg.json")
    ap.add_argument("--inject", action="store_true", help="写完后注入 SIBR 迭代")
    ap.add_argument(
        "--inject-on-skip",
        action="store_true",
        help="与 --gated-maskclose 联用：门控跳过时也按 --iteration 注入（内容仍是 softedge）",
    )
    ap.add_argument("--iteration", type=str, default=None, help="--inject 时的 iteration 名，不含 iteration_ 前缀")
    ap.add_argument(
        "--mask-direct",
        action="store_true",
        help="强制 maskclose（忽略门控）：掩码白集补点 → 清支 → 全场邻域闭包 → 再清支",
    )
    ap.add_argument(
        "--gated-maskclose",
        action="store_true",
        help=(
            "门控后处理：n_views≤30 且 n_selected<7000 时跑 mask-jury；"
            "否则写出 softedge 并跳过"
        ),
    )
    ap.add_argument(
        "--mask-jury",
        action="store_true",
        help="显式跑 mask-jury（多帧高置信删点 + 密度补点），不看门控",
    )
    ap.add_argument("--mask-score-floor", type=float, default=MASK_JURY_SCORE_FLOOR)
    ap.add_argument("--mask-jury-max", type=int, default=MASK_JURY_MAX_VIEWS)
    ap.add_argument("--mask-keep-frac", type=float, default=MASK_JURY_KEEP_FRAC)
    ap.add_argument("--dense-radius", type=float, default=MASK_JURY_DENSE_RADIUS)
    ap.add_argument("--min-dense-count", type=int, default=MASK_JURY_MIN_DENSE)
    ap.add_argument("--obj-dir", type=Path, default=None, help="物体目录，里面有 question.json 和 mask/")
    ap.add_argument("--radius", type=float, default=MASK_GROW_RADIUS, help="补点的场景单位半径，默认约 1.5 现实厘米")
    args = ap.parse_args()
    modes = [args.mask_direct, args.gated_maskclose, args.mask_jury]
    if sum(bool(x) for x in modes) > 1:
        ap.error("不要同时指定 --mask-direct / --gated-maskclose / --mask-jury")
    if args.mask_direct or args.gated_maskclose:
        if args.white is None:
            ap.error("--mask-direct / --gated-maskclose 需要 --white")
    obj_dir = args.obj_dir if args.obj_dir is not None else _object_dir(args.core)
    ran_maskclose = False
    skipped = False
    if args.mask_jury:
        mask_jury_vote(
            args.core,
            args.white,
            obj_dir=obj_dir,
            out_json=args.out,
            score_floor=args.mask_score_floor,
            max_views=args.mask_jury_max,
            keep_frac=args.mask_keep_frac,
            dense_radius=args.dense_radius,
            min_dense=args.min_dense_count,
        )
        ran_maskclose = True
    elif args.gated_maskclose:
        core_payload = _load_json(args.core)
        if not isinstance(core_payload, dict):
            raise TypeError("gaussian_seg.json 应为对象")
        enable, reason, n_sel, n_views = maskclose_gate_from_core(
            core_payload, object_dir=obj_dir
        )
        print(
            f"[gate] enable={enable} reason={reason} n_selected={n_sel} n_views={n_views}",
            flush=True,
        )
        if enable:
            if args.white is None:
                raise SystemExit("门控命中需要 --white（mask-jury 密度补点）")
            out = mask_jury_vote(
                args.core,
                args.white,
                obj_dir=obj_dir,
                out_json=args.out,
                score_floor=args.mask_score_floor,
                max_views=args.mask_jury_max,
                keep_frac=args.mask_keep_frac,
                dense_radius=args.dense_radius,
                min_dense=args.min_dense_count,
            )
            pp = out.setdefault("postprocess", {})
            if isinstance(pp, dict):
                pp["gate_reason"] = reason
                pp["gate_n_selected_core"] = n_sel
                pp["gate_n_views_used"] = n_views
                args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            ran_maskclose = True
        else:
            write_gated_skip(
                args.core,
                out_json=args.out,
                reason=reason,
                n_selected=n_sel,
                n_views_used=n_views,
            )
            skipped = True
    elif args.mask_direct:
        mask_direct_vote(args.core, args.white, obj_dir=obj_dir, out_json=args.out, radius=args.radius)
        ran_maskclose = True
    else:
        if args.white is None:
            ap.error("默认后处理需要 --white")
        postprocess_vote(args.core, args.white, out_json=args.out)

    do_inject = bool(args.inject) and (
        ran_maskclose or (skipped and args.inject_on_skip) or not args.gated_maskclose
    )
    if args.inject and args.gated_maskclose and skipped and not args.inject_on_skip:
        print("[gate] skip inject (softedge kept; pass --inject-on-skip to inject anyway)", flush=True)
    if do_inject:
        if not args.iteration:
            ap.error("--inject 需要 --iteration")
        inject_post(args.out, iteration_name=args.iteration)


if __name__ == "__main__":
    main()
