#!/usr/bin/env python3
"""
Unproject (nx, ny) with depth_raw/view_XXX.npy + camera_params JSON,
then report Euclidean distance to the nearest Gaussian in point_cloud.ply.

Run from repo root inside the gaussian_splatting conda env (needs plyfile; scipy optional for speed).

Example:
  python bridge/verify_unproject_vs_pointcloud.py \\
    --camera gaussian-splatting/gaussian-splatting/test1/camera_params/view_000.json \\
    --depth-npy gaussian-splatting/gaussian-splatting/test1/depth_raw/view_000.npy \\
    --ply gaussian-splatting/gaussian-splatting/output/<run>/point_cloud/iteration_30000/point_cloud.ply \\
    --nx 0.458 --ny 0.298
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "bridge"))

from unproject import load_camera, linear_z_from_raster_npy, unproject  # noqa: E402


def load_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as e:
        raise SystemExit("Install plyfile (gaussian_splatting env: conda install plyfile -c conda-forge)") from e
    ply = PlyData.read(str(path))
    v = ply.elements[0]
    x = np.asarray(v["x"], dtype=np.float64)
    y = np.asarray(v["y"], dtype=np.float64)
    z = np.asarray(v["z"], dtype=np.float64)
    return np.stack([x, y, z], axis=1)


def nn_distance(query: np.ndarray, points: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        d, _ = tree.query(query, k=1)
        return float(d)
    except Exception:
        return float(np.linalg.norm(points - query, axis=1).min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=Path, required=True)
    ap.add_argument("--depth-npy", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--nx", type=float, required=True)
    ap.add_argument("--ny", type=float, required=True)
    ap.add_argument("--inv-eps", type=float, default=1e-6)
    ap.add_argument(
        "--linear-z",
        action="store_true",
        help="If set, .npy stores linear camera z instead of expected_invdepth.",
    )
    args = ap.parse_args()

    cam = load_camera(args.camera)
    w, h = int(cam["width"]), int(cam["height"])
    u = int(np.clip(args.nx * w, 0, w - 1))
    v = int(np.clip(args.ny * h, 0, h - 1))
    stored, z = linear_z_from_raster_npy(
        args.depth_npy, u, v, inv_to_z=not args.linear_z, eps=args.inv_eps
    )
    _pcam, p_world = unproject(args.nx, args.ny, cam, z)
    if stored is not None and not args.linear_z:
        print("expected_invdepth at pixel:", stored)
    print("z_cam (linear, used):", z)
    print("P_world:", p_world)

    if not args.ply.is_file():
        print("PLY not found:", args.ply.resolve(), file=sys.stderr)
        sys.exit(1)
    xyz = load_ply_xyz(args.ply)
    d = nn_distance(p_world.astype(np.float64), xyz)
    print("num_gaussians:", xyz.shape[0])
    print("nearest_neighbor_distance:", d)


if __name__ == "__main__":
    main()
