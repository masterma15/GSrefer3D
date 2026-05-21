#!/usr/bin/env python3
"""Append marker Gaussians to a 3DGS-exported point_cloud.ply (same vertex schema).

Reads ``fused.json`` for world position (``P_world`` or optional ``P_world_before_snap``).
By default re-snaps that point to the nearest vertex on ``--ply`` so markers sit on the
reconstruction you are viewing (see project notes on fuse-time vs view-time ply).

Example:
  python bridge/inject_gaussian_markers.py \\
    --ply E:/GSrefer3D/3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply \\
    --fused-json E:/GSrefer3D/3DGS/test2/fused.json \\
    --out-iteration-dir E:/GSrefer3D/3DGS/gaussian-splatting/output/data2/point_cloud/iteration_35000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def _inverse_sigmoid(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def _rgb_to_f_dc(rgb: np.ndarray) -> tuple[float, float, float]:
    rgb = np.asarray(rgb, dtype=np.float64).reshape(3)
    sh = (rgb - 0.5) / C0
    return float(sh[0]), float(sh[1]), float(sh[2])


def _nearest_vertex(p: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.linalg.norm(xyz - p.reshape(1, 3), axis=1)
    i = int(np.argmin(d))
    return xyz[i].copy(), float(d[i])


def _surface_push(p: np.ndarray, xyz: np.ndarray, push: float, k: int) -> np.ndarray:
    """Nudge ``p`` along (p - mean(nearest k points)) to move markers slightly out of dense interiors."""
    if push <= 0.0:
        return p
    kk = min(max(k, 4), len(xyz))
    d = np.linalg.norm(xyz - p.reshape(1, 3), axis=1)
    ii = np.argpartition(d, kk - 1)[:kk]
    m = xyz[ii].mean(axis=0)
    v = p - m
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return p
    return p + (push / norm) * v


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject marker Gaussians into 3DGS point_cloud.ply")
    ap.add_argument("--ply", type=Path, required=True, help="Source point_cloud.ply (base + markers)")
    ap.add_argument("--fused-json", type=Path, default=None)
    ap.add_argument(
        "--world-coords",
        choices=("snap", "unsnapped"),
        default="snap",
        help="snap: fused['P_world']. unsnapped: fused['P_world_before_snap'] if present.",
    )
    ap.add_argument(
        "--no-resnap-to-input-ply",
        action="store_true",
        help="Do not move marker center to nearest vertex on --ply (use json coords as-is).",
    )
    ap.add_argument("--x", type=float, default=None)
    ap.add_argument("--y", type=float, default=None)
    ap.add_argument("--z", type=float, default=None)
    ap.add_argument("--marker-count", type=int, default=50,
                    help="Number of marker Gaussians per fused point (more = denser blob).")
    ap.add_argument(
        "--log-scale",
        type=float,
        default=-3.5,
        help="Log of Gaussian scale (3DGS convention); LESS negative => LARGER splats (e.g. -2.2 vs -3.5).",
    )
    ap.add_argument("--opacity-sigmoid", type=float, default=0.99,
                    help="Target opacity after sigmoid; higher = more opaque markers.")
    ap.add_argument("--jitter", type=float, default=0.012,
                    help="Std-dev of jitter around center in scene units (slightly bigger cloud).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for marker jitter.")
    ap.add_argument(
        "--max-resnap-dist",
        type=float,
        default=None,
        metavar="D",
        help=(
            "If set, only snap marker center to nearest PLY vertex when distance <= D (scene units). "
            "Otherwise keep fused coordinates — avoids snapping to unrelated distant geometry on large clouds."
        ),
    )
    ap.add_argument("--out-iteration-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--all-candidates", action="store_true",
                    help="Inject ALL candidate points from fused.json (green=inlier, yellow=outlier, red=fused)")
    ap.add_argument(
        "--surface-push",
        type=float,
        default=0.0,
        metavar="M",
        help=(
            "After resnap, shift center by M meters along (p - mean(nearest-k vertices)) "
            "to pull markers out of the interior of thick objects (try 0.02–0.08)."
        ),
    )
    ap.add_argument(
        "--surface-push-k",
        type=int,
        default=48,
        help="Neighbour count for --surface-push (default 48).",
    )
    ap.add_argument(
        "--marker-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Extra world-space translation added to marker center after resnap/push (scene units).",
    )
    args = ap.parse_args()

    if args.fused_json is not None:
        with args.fused_json.open("r", encoding="utf-8") as f:
            fused = json.load(f)
        if args.world_coords == "unsnapped":
            raw = fused.get("P_world_before_snap")
            if raw is None:
                raise SystemExit(
                    "fused.json has no P_world_before_snap; use --world-coords snap or re-fuse with updated bridge."
                )
            px, py, pz = float(raw[0]), float(raw[1]), float(raw[2])
        else:
            px = float(fused["P_world"][0])
            py = float(fused["P_world"][1])
            pz = float(fused["P_world"][2])
    elif args.x is not None and args.y is not None and args.z is not None:
        px, py, pz = args.x, args.y, args.z
    else:
        ap.error("provide --fused-json or --x --y --z")

    src = args.ply.resolve()
    if not src.is_file():
        raise SystemExit(f"missing source ply: {src}")

    rng = np.random.default_rng(args.seed)
    ply = PlyData.read(str(src))
    el0 = ply.elements[0]
    if el0.name != "vertex":
        raise SystemExit(f"expected first element 'vertex', got {el0.name!r}")

    data = el0.data.copy()
    n = len(data)
    if n == 0:
        raise SystemExit("empty vertex buffer")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)

    props = list(data.dtype.names)
    if "opacity" not in props or not all(f"scale_{i}" in props for i in range(3)):
        raise SystemExit("unexpected ply schema (need opacity, scale_0..2)")

    op = _inverse_sigmoid(args.opacity_sigmoid)
    ls = float(args.log_scale)

    def _make_marker_rows(center, rgb, count, jitter, rng):
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        d2 = np.sum((xyz - np.array([cx, cy, cz])) ** 2, axis=1)
        ti = int(np.argmin(d2))
        r, g, b = _rgb_to_f_dc(np.asarray(rgb, dtype=np.float64))
        rows = []
        for _ in range(count):
            row = data[ti].copy()
            j = rng.normal(scale=jitter, size=3)
            row["x"] = cx + float(j[0])
            row["y"] = cy + float(j[1])
            row["z"] = cz + float(j[2])
            row["nx"] = 0.0
            row["ny"] = 0.0
            row["nz"] = 0.0
            if "f_dc_0" in props:
                row["f_dc_0"] = r
                row["f_dc_1"] = g
                row["f_dc_2"] = b
            for name in props:
                if name.startswith("f_rest_"):
                    row[name] = 0.0
            row["opacity"] = op
            row["scale_0"] = ls
            row["scale_1"] = ls
            row["scale_2"] = ls
            if all(f"rot_{i}" in props for i in range(4)):
                row["rot_0"] = 1.0
                row["rot_1"] = 0.0
                row["rot_2"] = 0.0
                row["rot_3"] = 0.0
            rows.append(tuple(row))
        return rows

    rng = np.random.default_rng(args.seed)
    new_rows = []

    if args.all_candidates and args.fused_json is not None:
        with args.fused_json.open("r", encoding="utf-8") as f:
            fused = json.load(f)
        inlier_set = set(fused.get("inlier_indices", []))
        candidates = fused.get("candidates", [])
        for i, cand in enumerate(candidates):
            pw = np.array(cand["P_world"], dtype=np.float64)
            if i in inlier_set:
                rgb = [0.1, 0.9, 0.1]  # green
            else:
                rgb = [0.95, 0.85, 0.1]  # yellow
            new_rows.extend(_make_marker_rows(pw, rgb, 20, args.jitter * 0.7, rng))
        # Fused point in red (larger cluster)
        fused_p = np.array(fused["P_world"], dtype=np.float64)
        fused_p = _surface_push(fused_p, xyz, float(args.surface_push), int(args.surface_push_k))
        fused_p = fused_p + np.asarray(args.marker_offset, dtype=np.float64).reshape(3)
        new_rows.extend(_make_marker_rows(fused_p, [0.98, 0.08, 0.06], args.marker_count, args.jitter, rng))
        print(f"[info] injected {len(candidates)} candidates + 1 fused point "
              f"({len(inlier_set)} inliers green, {len(candidates)-len(inlier_set)} outliers yellow, fused red)")
    else:
        p0 = np.array([px, py, pz], dtype=np.float64)
        if not args.no_resnap_to_input_ply:
            p_snapped, rdist = _nearest_vertex(p0, xyz)
            print(f"[info] json center {p0.tolist()}  -> resnap to input ply {p_snapped.tolist()}  (dist={rdist:.6f})")
            maxd = args.max_resnap_dist
            if maxd is not None and rdist > float(maxd):
                print(f"[info] resnap distance {rdist:.6f} > --max-resnap-dist {maxd:g}; keeping json center (no snap).")
            else:
                p0 = p_snapped
        p0 = _surface_push(p0, xyz, float(args.surface_push), int(args.surface_push_k))
        off = np.asarray(args.marker_offset, dtype=np.float64).reshape(3)
        p0 = p0 + off
        if float(args.surface_push) > 0.0 or np.any(off != 0.0):
            print(f"[info] marker center after push/offset {p0.tolist()}")
        new_rows.extend(_make_marker_rows(p0, [0.98, 0.08, 0.06], args.marker_count, args.jitter, rng))

    extra = np.array(new_rows, dtype=data.dtype)
    merged = np.empty(n + len(extra), dtype=data.dtype)
    merged[:n] = data
    merged[n:] = extra

    out_el = PlyElement.describe(merged, "vertex")
    out_ply = PlyData([out_el])

    if args.out_iteration_dir is not None:
        out_dir = args.out_iteration_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "point_cloud.ply"
        out_ply.write(str(out_path))
        print(f"[ok] wrote {out_path}  ({len(merged)} vertices = {n} + {len(extra)})")
    if args.output is not None:
        outp = args.output.resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_ply.write(str(outp))
        print(f"[ok] wrote {outp}  ({len(merged)} vertices)")
    if args.out_iteration_dir is None and args.output is None:
        ap.error("set --out-iteration-dir and/or --output")


if __name__ == "__main__":
    main()
