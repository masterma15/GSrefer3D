#!/usr/bin/env python3
"""从一个种子视角选出仍看着同一块表面的邻视角。

端到端不调用。带方位的指代只送一张图。这个脚本留到以后做微调数据选视角。

用种子图已有的 depth_raw 抽点反投影，再投进其他相机。
不新渲染。召回率是种子表面点仍落在候选画面内的比例。
幅宽比是候选相机到这些点的中位距离除以种子的中位距离（同一视场角下幅宽正比于距离）。
视线绕房间竖直方向取有符号夹角。种子留在选出集合的中位，两侧各取同样张数，单侧不超过 45°。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from unproject import CameraView, in_frame  # noqa: E402

MIN_INV = 1e-3
BORDER = 8
RECALL_MIN = 0.7
FOOTPRINT_MAX = 1.45
DUP_GAZE_DEG = 8.0
GAZE_MAX_DEG = 45.0
GRID_TARGET = 1000


def _load_view(root: Path, view_id: int) -> tuple[CameraView, dict]:
    path = root / "camera_params" / f"view_{view_id:03d}.json"
    if not path.is_file():
        raise SystemExit(f"缺少相机 {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CameraView.from_camera_dict(raw), raw


def _forward(raw: dict) -> np.ndarray:
    rotation = np.asarray(raw["rotation"], dtype=np.float64)
    forward = rotation[:, 2]
    return forward / np.linalg.norm(forward)


def _gaze_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _room_up(raws: list[dict]) -> np.ndarray:
    downs = [np.asarray(raw["rotation"], dtype=np.float64)[:, 1] for raw in raws]
    up = -np.mean(downs, axis=0)
    return up / np.linalg.norm(up)


def _signed_deg(forward: np.ndarray, seed_horizontal: np.ndarray, up: np.ndarray) -> float:
    horizontal = forward - up * float(np.dot(forward, up))
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-8:
        return 0.0
    horizontal = horizontal / norm
    sine = float(np.dot(np.cross(seed_horizontal, horizontal), up))
    cosine = float(np.clip(np.dot(seed_horizontal, horizontal), -1.0, 1.0))
    return math.degrees(math.atan2(sine, cosine))


def _discover(root: Path) -> list[int]:
    found = []
    for path in sorted((root / "camera_params").glob("view_*.json")):
        found.append(int(path.stem.split("_")[1]))
    if not found:
        raise SystemExit(f"{root}/camera_params 没有 view_*.json")
    return found


def _seed_points(root: Path, view_id: int, view: CameraView) -> tuple[np.ndarray, np.ndarray]:
    depth_path = root / "depth_raw" / f"view_{view_id:03d}.npy"
    if not depth_path.is_file():
        raise SystemExit(f"缺少深度 {depth_path}")
    depth = np.load(depth_path)
    if depth.ndim != 2:
        depth = np.squeeze(depth)
    height, width = depth.shape
    if (width, height) != (view.width, view.height):
        raise SystemExit(
            f"view_{view_id:03d} 深度 {width}x{height} 与相机 {view.width}x{view.height} 不一致"
        )
    step = max(1, int(round(math.sqrt((width * height) / GRID_TARGET))))
    ys, xs = np.mgrid[0:height:step, 0:width:step]
    inv = depth[ys, xs]
    keep = np.isfinite(inv) & (inv >= MIN_INV)
    if int(keep.sum()) < 10:
        raise SystemExit(f"view_{view_id:03d} 有效深度点太少")
    z = 1.0 / inv[keep]
    u = xs[keep].astype(np.float64)
    v = ys[keep].astype(np.float64)
    fx, fy, cx, cy = view.intrinsics
    p_cam = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    return p_cam @ view.R_w2c + view.camera_center, u


def _project(points: np.ndarray, view: CameraView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - view.camera_center
    p_cam = centered @ view.R_w2c.T
    z = p_cam[:, 2]
    fx, fy, cx, cy = view.intrinsics
    z_ok = z > 1e-8
    z_safe = np.where(z_ok, z, np.nan)
    u = fx * p_cam[:, 0] / z_safe + cx
    v = fy * p_cam[:, 1] / z_safe + cy
    return u, v, z


def _inside(u: np.ndarray, v: np.ndarray, z: np.ndarray, view: CameraView) -> np.ndarray:
    finite = np.isfinite(u) & np.isfinite(v)
    framed = np.array(
        [
            finite[i] and z[i] > 1e-8 and in_frame(float(u[i]), float(v[i]), view.width, view.height, BORDER)
            for i in range(len(z))
        ],
        dtype=bool,
    )
    return framed


def select_nearby(
    root: Path,
    seed: int,
    *,
    count: int,
    recall_min: float,
    footprint_max: float,
) -> dict:
    if count < 1:
        raise SystemExit("--count 至少为 1")
    view_ids = _discover(root)
    if seed not in view_ids:
        raise SystemExit(f"种子视角 {seed} 不在 {root}")
    loaded = {view_id: _load_view(root, view_id) for view_id in view_ids}
    seed_view, seed_raw = loaded[seed]
    points, seed_u = _seed_points(root, seed, seed_view)
    left_band = seed_u < seed_view.width / 3.0
    right_band = seed_u >= seed_view.width * 2.0 / 3.0
    seed_forward = _forward(seed_raw)
    up = _room_up([raw for _, raw in loaded.values()])
    seed_horizontal = seed_forward - up * float(np.dot(seed_forward, up))
    seed_horizontal = seed_horizontal / np.linalg.norm(seed_horizontal)
    seed_dist = float(np.median(np.linalg.norm(points - seed_view.camera_center, axis=1)))
    rows = []
    for view_id in view_ids:
        if view_id == seed:
            continue
        view, raw = loaded[view_id]
        u, v, z = _project(points, view)
        inside = _inside(u, v, z, view)
        recall = float(inside.mean())
        recall_left = float(inside[left_band].mean()) if left_band.any() else 0.0
        recall_right = float(inside[right_band].mean()) if right_band.any() else 0.0
        dist = float(np.median(np.linalg.norm(points - view.camera_center, axis=1)))
        footprint = dist / seed_dist
        forward = _forward(raw)
        gaze = _gaze_deg(seed_forward, forward)
        signed = _signed_deg(forward, seed_horizontal, up)
        if abs(signed) < DUP_GAZE_DEG:
            reason = "duplicate"
        elif abs(signed) > GAZE_MAX_DEG or gaze > GAZE_MAX_DEG:
            reason = "angle"
        elif footprint > footprint_max:
            reason = "footprint"
        elif min(recall_left, recall_right) < recall_min:
            reason = "edge"
        elif recall < recall_min:
            reason = "recall"
        else:
            reason = "eligible"
        rows.append(
            {
                "view_id": view_id,
                "recall": round(recall, 4),
                "recall_left": round(recall_left, 4),
                "recall_right": round(recall_right, 4),
                "footprint_ratio": round(footprint, 4),
                "gaze_deg": round(gaze, 2),
                "signed_deg": round(signed, 2),
                "kept": False,
                "reason": reason,
            }
        )
    n_side = max(0, (count - 1) // 2)
    left = _pick_side(rows, -1, n_side, recall_min, footprint_max)
    right = _pick_side(rows, 1, n_side, recall_min, footprint_max)
    n_pair = min(len(left), len(right), n_side)
    chosen = _trim_near(left, n_pair) + _trim_near(right, n_pair)
    chosen_ids = {row["view_id"] for row in chosen}
    for row in rows:
        if row["view_id"] in chosen_ids:
            row["kept"] = True
            row["reason"] = "near" if abs(row["signed_deg"]) < DUP_GAZE_DEG else "eligible"
        elif row["reason"] == "eligible":
            row["reason"] = "rank"
    view_out = sorted([seed, *(row["view_id"] for row in chosen)])
    return {
        "seed": seed,
        "n_surface_points": int(points.shape[0]),
        "seed_surface_distance": round(seed_dist, 4),
        "recall_min": recall_min,
        "footprint_max": footprint_max,
        "duplicate_gaze_deg": DUP_GAZE_DEG,
        "gaze_max_deg": GAZE_MAX_DEG,
        "border_px": BORDER,
        "count_requested": count,
        "per_side": n_pair,
        "view_ids": view_out,
        "candidates": rows,
    }


def _trim_near(rows: list[dict], n_keep: int) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (abs(row["signed_deg"]), row["view_id"]))
    return ordered[:n_keep]


def _edge_recall(row: dict) -> float:
    return min(row["recall_left"], row["recall_right"])


def _take_spread(pool: list[dict], n_side: int) -> list[dict]:
    """沿夹角从近到远均匀留下，避免只挤在 30°–40°。"""
    ordered = sorted(pool, key=lambda row: (abs(row["signed_deg"]), row["view_id"]))
    if len(ordered) <= n_side:
        return ordered
    if n_side == 1:
        return [ordered[0]]
    indexes = [round(i * (len(ordered) - 1) / (n_side - 1)) for i in range(n_side)]
    chosen = []
    seen: set[int] = set()
    for index in indexes:
        row = ordered[index]
        if row["view_id"] in seen:
            continue
        seen.add(row["view_id"])
        chosen.append(row)
    return chosen


def _passes_edge(row: dict, recall_min: float, footprint_max: float) -> bool:
    return (
        _edge_recall(row) >= recall_min
        and row["footprint_ratio"] <= footprint_max
        and row["gaze_deg"] <= GAZE_MAX_DEG
    )


def _on_side(signed: float, sign: int, *, near: bool) -> bool:
    if sign < 0:
        return -DUP_GAZE_DEG < signed <= 0.0 if near else -GAZE_MAX_DEG <= signed <= -DUP_GAZE_DEG
    return 0.0 <= signed < DUP_GAZE_DEG if near else DUP_GAZE_DEG <= signed <= GAZE_MAX_DEG


def _pick_side(
    rows: list[dict],
    sign: int,
    n_side: int,
    recall_min: float,
    footprint_max: float,
) -> list[dict]:
    """一侧先取 8°–45° 且左右边都达标的视角。不够时从小于 8°、召回率仍达标的近邻补。"""
    if n_side < 1:
        return []
    strict = [
        row
        for row in rows
        if _on_side(row["signed_deg"], sign, near=False) and _passes_edge(row, recall_min, footprint_max)
    ]
    near = [
        row
        for row in rows
        if _on_side(row["signed_deg"], sign, near=True) and _passes_edge(row, recall_min, footprint_max)
    ]
    chosen = _take_spread(strict, n_side)
    used = {row["view_id"] for row in chosen}
    if len(chosen) < n_side:
        for row in _take_spread(near, n_side - len(chosen)):
            if row["view_id"] in used:
                continue
            chosen.append(row)
            used.add(row["view_id"])
    chosen.sort(key=lambda row: row["signed_deg"])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="从种子视角选出仍看着同一块表面的邻视角")
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4, help="种子视角编号，默认 4（view_004）")
    parser.add_argument("--count", type=int, default=11, help="含种子在内的张数，两侧均分，默认 11")
    parser.add_argument("--recall-min", type=float, default=RECALL_MIN)
    parser.add_argument("--footprint-max", type=float, default=FOOTPRINT_MAX)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = select_nearby(
        args.views_root,
        args.seed,
        count=args.count,
        recall_min=args.recall_min,
        footprint_max=args.footprint_max,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[nearby] {len(payload['view_ids'])} -> {args.output}")
    else:
        print(text)
    ids = " ".join(f"{view_id:03d}" for view_id in payload["view_ids"])
    print(f"[nearby] views {ids}")


if __name__ == "__main__":
    main()
