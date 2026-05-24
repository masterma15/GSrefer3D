#!/usr/bin/env python3
"""Export a marker .ply for the fused 3D point.

Reads bridge/fuse_multiview.py output (fused.json) and writes a small ASCII
PLY with one big sphere-ish vertex cluster at the fused P_world plus inlier
candidates (green). Outlier candidates are omitted.

Drop the .ply into MeshLab / CloudCompare next to point_cloud.ply, or load
it in the SIBR viewer's external-pointcloud overlay if you build it in.

No external deps beyond numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# RGB colors used in the PLY (uchar 0-255).
COLOR_FUSED = (255, 64, 64)       # bright red, large
COLOR_INLIER = (64, 200, 64)      # green
COLOR_OUTLIER = (200, 200, 64)    # yellow


def _sphere_cluster(center: np.ndarray, radius: float, n: int) -> np.ndarray:
    """Generate ``n`` jittered points inside a sphere around ``center``.

    Used to make the marker visible without writing real mesh faces.
    """
    rng = np.random.default_rng(seed=0)
    # uniform inside-sphere via cubic root
    u = rng.uniform(0.0, 1.0, size=n) ** (1.0 / 3.0)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return center[None, :] + dirs * (u * radius)[:, None]


def write_marker_ply(
    fused: dict,
    output: Path,
    *,
    marker_radius: float = 0.08,
    marker_count: int = 250,
    candidate_radius: float = 0.028,
    candidate_count: int = 55,
) -> None:
    P = np.asarray(fused["P_world"], dtype=np.float64)

    inlier_set = set(fused.get("inlier_indices", []))
    cands = fused.get("candidates", [])

    chunks: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    # 1) Big red marker at fused point
    chunks.append((_sphere_cluster(P, marker_radius, marker_count), COLOR_FUSED))

    # 2) Inlier candidates only (green); outliers omitted
    for i, c in enumerate(cands):
        if i not in inlier_set:
            continue
        p = np.asarray(c["P_world"], dtype=np.float64)
        chunks.append((_sphere_cluster(p, candidate_radius, candidate_count), COLOR_INLIER))

    points = np.concatenate([c[0] for c in chunks], axis=0)
    colors = np.concatenate([
        np.tile(np.asarray(c[1], dtype=np.uint8), (c[0].shape[0], 1))
        for c in chunks
    ], axis=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a colored marker .ply for fused.json")
    ap.add_argument("--fused", type=Path, required=True, help="bridge/fuse_multiview.py output (fused.json)")
    ap.add_argument("--output", type=Path, default=None, help="Output marker .ply (default: fused.parent / marker.ply)")
    ap.add_argument("--marker-radius", type=float, default=0.08)
    ap.add_argument("--marker-count", type=int, default=250)
    ap.add_argument("--candidate-radius", type=float, default=0.028)
    ap.add_argument("--candidate-count", type=int, default=55)
    args = ap.parse_args()

    with args.fused.open("r", encoding="utf-8") as f:
        fused = json.load(f)
    out = args.output or (args.fused.parent / "marker.ply")
    write_marker_ply(
        fused,
        out,
        marker_radius=args.marker_radius,
        marker_count=args.marker_count,
        candidate_radius=args.candidate_radius,
        candidate_count=args.candidate_count,
    )
    print(f"[summary] wrote {out}")
    print(f"          fused P_world: {fused['P_world']}")
    cands = fused.get("candidates", [])
    inlier_set = set(fused.get("inlier_indices", []))
    n_out = len(cands) - len(inlier_set)
    print(f"          {len(inlier_set)} inliers shown, {n_out} outliers omitted")


if __name__ == "__main__":
    main()
