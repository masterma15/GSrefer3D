#!/usr/bin/env python3
"""Align DAv2 relative depth to absolute scale using COLMAP sparse points.

For each rendered view in test2:
1. Project COLMAP 3D sparse points into the view using camera_params
2. Sample the DAv2 uint16 depth at those pixel locations
3. Fit scale/shift: metric_invdepth = scale * dav2_value + shift
4. Apply to full DAv2 depth map, save as float32 .npy (same format as depth_raw/)

Convention (matching bridge/unproject.py):
- rotation = R_w2c (world-to-camera)
- position = C (camera center in world)
- P_cam = R_w2c @ (P_world - C)
- invdepth = 1 / z_cam

Usage:
    python bridge/align_dav2_depth.py \
        --colmap-dir 3DGS/gaussian-splatting/data2/sparse/0 \
        --camera-dir 3DGS/test2/camera_params \
        --dav2-dir 3DGS/test2/depth_dav2 \
        --out-dir 3DGS/test2/depth_raw_dav2
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np


def read_points3d_bin(path: Path) -> np.ndarray:
    """Read COLMAP points3D.bin, return (N, 3) float64 array of XYZ."""
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        points = []
        for _ in range(num):
            _pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            _rgb = struct.unpack("<BBB", f.read(3))
            _err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)  # skip track
            points.append(xyz)
    return np.array(points, dtype=np.float64)


def load_camera(cam_json: Path) -> dict:
    with open(cam_json, "r", encoding="utf-8") as f:
        return json.load(f)


def project_points_to_view(
    points_world: np.ndarray,  # (N, 3)
    cam: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a camera view.

    Returns:
        uv: (M, 2) pixel coordinates of valid projections
        invdepth_gt: (M,) ground-truth inverse depth (1/z_cam)
        mask: (M,) boolean mask of points that project into frame
    """
    R_w2c = np.array(cam["rotation"], dtype=np.float64)
    C = np.array(cam["position"], dtype=np.float64)
    W, H = cam["width"], cam["height"]
    fov_x, fov_y = cam["fov_x"], cam["fov_y"]

    fx = W / (2.0 * np.tan(fov_x / 2.0))
    fy = H / (2.0 * np.tan(fov_y / 2.0))
    cx, cy = W / 2.0, H / 2.0

    # World to camera
    P_cam = (R_w2c @ (points_world - C).T).T  # (N, 3)
    z = P_cam[:, 2]

    # Only keep points in front of camera
    valid = z > 0.01
    P_valid = P_cam[valid]
    z_valid = P_valid[:, 2]

    # Project to pixel
    u = fx * P_valid[:, 0] / z_valid + cx
    v = fy * P_valid[:, 1] / z_valid + cy

    # Only keep points inside image bounds (with margin)
    margin = 2
    in_frame = (u >= margin) & (u < W - margin) & (v >= margin) & (v < H - margin)

    uv = np.stack([u[in_frame], v[in_frame]], axis=1)
    invdepth_gt = 1.0 / z_valid[in_frame]

    return uv, invdepth_gt, in_frame


def fit_scale_shift(
    dav2_values: np.ndarray,
    gt_invdepth: np.ndarray,
    *,
    ransac_iters: int = 200,
    inlier_thresh: float = 0.1,
) -> tuple[float, float, int]:
    """Robust linear fit: gt_invdepth = scale * dav2_value + shift.

    Uses RANSAC for robustness to outliers.
    Returns (scale, shift, num_inliers).
    """
    n = len(dav2_values)
    if n < 5:
        # Too few points, fallback to least squares
        A = np.stack([dav2_values, np.ones(n)], axis=1)
        result = np.linalg.lstsq(A, gt_invdepth, rcond=None)
        scale, shift = result[0]
        return float(scale), float(shift), n

    best_inliers = 0
    best_scale, best_shift = 1.0, 0.0
    rng = np.random.default_rng(42)

    for _ in range(ransac_iters):
        idx = rng.choice(n, size=2, replace=False)
        x = dav2_values[idx]
        y = gt_invdepth[idx]
        dx = x[1] - x[0]
        if abs(dx) < 1e-10:
            continue
        s = (y[1] - y[0]) / dx
        b = y[0] - s * x[0]

        residual = np.abs(gt_invdepth - (s * dav2_values + b))
        inliers = residual < inlier_thresh
        count = inliers.sum()
        if count > best_inliers:
            best_inliers = count
            # Refit on inliers
            A = np.stack([dav2_values[inliers], np.ones(count)], axis=1)
            result = np.linalg.lstsq(A, gt_invdepth[inliers], rcond=None)
            best_scale, best_shift = result[0]

    return float(best_scale), float(best_shift), int(best_inliers)


def main() -> None:
    ap = argparse.ArgumentParser(description="Align DAv2 depth to absolute scale via COLMAP sparse points")
    ap.add_argument("--colmap-dir", type=Path, required=True, help="COLMAP sparse/0/ directory")
    ap.add_argument("--camera-dir", type=Path, required=True, help="test2/camera_params/ directory")
    ap.add_argument("--dav2-dir", type=Path, required=True, help="test2/depth_dav2/ (uint16 PNGs)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dir for aligned .npy files")
    ap.add_argument("--inlier-thresh", type=float, default=0.1,
                    help="RANSAC inlier threshold for scale/shift fitting")
    args = ap.parse_args()

    # Load COLMAP sparse points
    p3d_path = args.colmap_dir / "points3D.bin"
    if not p3d_path.is_file():
        sys.exit(f"points3D.bin not found: {p3d_path}")
    print(f"[info] loading COLMAP points from {p3d_path}")
    points_world = read_points3d_bin(p3d_path)
    print(f"[info] {len(points_world)} sparse 3D points")

    # Find all view camera params
    cam_files = sorted(args.camera_dir.glob("view_*.json"))
    if not cam_files:
        sys.exit(f"No view_*.json in {args.camera_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cam_file in cam_files:
        view_name = cam_file.stem  # e.g. view_000
        dav2_png = args.dav2_dir / f"{view_name}.png"
        if not dav2_png.is_file():
            print(f"[skip] {view_name}: no DAv2 depth at {dav2_png}")
            continue

        cam = load_camera(cam_file)

        # Project COLMAP points into this view
        uv, invdepth_gt, _ = project_points_to_view(points_world, cam)
        if len(uv) < 10:
            print(f"[skip] {view_name}: only {len(uv)} valid projections")
            continue

        # Load DAv2 uint16 depth
        dav2_raw = cv2.imread(str(dav2_png), cv2.IMREAD_UNCHANGED)
        if dav2_raw is None or dav2_raw.dtype != np.uint16:
            print(f"[skip] {view_name}: cannot read uint16 PNG")
            continue

        # Resize to render resolution if needed (DAv2 on real photos may be
        # much larger than the rendered camera_params resolution)
        render_W, render_H = cam["width"], cam["height"]
        if dav2_raw.shape[0] != render_H or dav2_raw.shape[1] != render_W:
            print(f"  resize DAv2 {dav2_raw.shape[1]}x{dav2_raw.shape[0]} -> {render_W}x{render_H}")
            dav2_raw = cv2.resize(dav2_raw, (render_W, render_H), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        dav2_float = dav2_raw.astype(np.float64) / 65535.0

        # Sample DAv2 at projected pixel locations
        u_int = np.round(uv[:, 0]).astype(int)
        v_int = np.round(uv[:, 1]).astype(int)
        H, W = dav2_float.shape
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_int, v_int = u_int[valid], v_int[valid]
        invdepth_gt_valid = invdepth_gt[valid]

        dav2_samples = dav2_float[v_int, u_int]

        # Fit scale/shift
        scale, shift, n_inliers = fit_scale_shift(
            dav2_samples, invdepth_gt_valid, inlier_thresh=args.inlier_thresh
        )

        # Apply to full depth map — output at render resolution
        aligned_invdepth = scale * dav2_float + shift
        aligned_invdepth = np.clip(aligned_invdepth, 1e-6, None)

        # Save as float32 .npy (same format/resolution as depth_raw/)
        out_npy = args.out_dir / f"{view_name}.npy"
        np.save(str(out_npy), aligned_invdepth.astype(np.float32))

        # Compute fit quality
        fitted = scale * dav2_samples + shift
        residual = np.abs(invdepth_gt_valid - fitted)
        med_err = float(np.median(residual))

        print(f"[done] {view_name}: {len(invdepth_gt_valid)} pts, "
              f"inliers={n_inliers}, scale={scale:.4f}, shift={shift:.6f}, "
              f"median_err={med_err:.5f}")
        results.append({
            "view": view_name,
            "n_points": len(invdepth_gt_valid),
            "n_inliers": n_inliers,
            "scale": scale,
            "shift": shift,
            "median_error": med_err,
        })

    # Summary
    if results:
        avg_err = np.mean([r["median_error"] for r in results])
        avg_inliers = np.mean([r["n_inliers"] for r in results])
        print(f"\n[summary] {len(results)} views aligned")
        print(f"          avg median_error={avg_err:.5f}, avg inliers={avg_inliers:.0f}")
        print(f"          output: {args.out_dir}")
    else:
        print("[error] no views were aligned")


if __name__ == "__main__":
    main()
