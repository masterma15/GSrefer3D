#!/usr/bin/env python3
"""Decouple rotation vs depth: 3DGS raster vs DAV2 unprojection NN distance.

Same view, same (nx, ny), same camera_params; only depth source changes.
Uses correct R_w2c (transpose of JSON rotation). Optional wrong-R ablation.

Example (envGS or any env with torch+cv2+plyfile):

  python bridge/compare_depth_sources.py \\
    --views-root 3DGS/test2 --view 000 \\
    --nx 0.4958 --ny 0.9093 \\
    --ply 3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "bridge"))
sys.path.insert(0, str(_REPO))

from unproject import CameraView, Unprojector  # noqa: E402


def load_ply_xyz(path: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply.elements[0]
    return np.stack(
        [np.asarray(v["x"], float), np.asarray(v["y"], float), np.asarray(v["z"], float)],
        axis=1,
    )


def nn_distance(query: np.ndarray, points: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree

        return float(cKDTree(points).query(query, k=1)[0])
    except Exception:
        return float(np.linalg.norm(points - query, axis=1).min())


def load_dav2_model(ckpt: Path, device: str):
    dav2_root = _REPO / "RoboRefer-main" / "API" / "Depth_Anything_V2"
    sys.path.insert(0, str(dav2_root))
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    }
    model = DepthAnythingV2(**cfg)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model = model.to(device).eval()
    return model, torch


def dav2_depth_map(model, torch_mod, rgb_path: Path, device: str, input_size: int = 518) -> np.ndarray:
    raw = cv2.imread(str(rgb_path))
    if raw is None:
        raise FileNotFoundError(rgb_path)
    with torch_mod.no_grad():
        depth = model.infer_image(raw, input_size=input_size, device=device)
    return np.asarray(depth, dtype=np.float64)


def fit_affine_z(z_ref: np.ndarray, z_src: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """z_ref ≈ scale * z_src + offset on masked pixels."""
    yr = z_ref[mask].astype(np.float64)
    xs = z_src[mask].astype(np.float64)
    if yr.size < 50:
        raise ValueError(f"too few pixels for affine fit: {yr.size}")
    A = np.stack([xs, np.ones_like(xs)], axis=1)
    scale, offset = np.linalg.lstsq(A, yr, rcond=None)[0]
    return float(scale), float(offset)


def fit_invdepth_scale_offset(inv_ref: np.ndarray, inv_src: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """inv_ref ≈ scale * inv_src + offset (make_depth_scale style)."""
    yr = inv_ref[mask].astype(np.float64)
    xs = inv_src[mask].astype(np.float64)
    if yr.size < 50:
        raise ValueError(f"too few pixels for inv fit: {yr.size}")
    t_ref = np.median(yr)
    s_ref = np.mean(np.abs(yr - t_ref)) + 1e-9
    t_src = np.median(xs)
    s_src = np.mean(np.abs(xs - t_src)) + 1e-9
    scale = s_ref / s_src
    offset = t_ref - t_src * scale
    return float(scale), float(offset)


def unproject_with_rotation(
    view: CameraView,
    nx: float,
    ny: float,
    z_cam: float,
    *,
    use_wrong_r: bool = False,
) -> np.ndarray:
    r_w2c = np.asarray(view.R_w2c, dtype=np.float64)
    if use_wrong_r:
        # Bug: treat JSON rotation (R_c2w) as R_w2c without transpose
        r_w2c = r_w2c.T
    fx, fy, cx, cy = view.intrinsics
    fu = nx * view.width
    fv = ny * view.height
    x = (fu - cx) * z_cam / fx
    y = (fv - cy) * z_cam / fy
    p_cam = np.array([x, y, z_cam], dtype=np.float64)
    return r_w2c.T @ p_cam + view.camera_center


def main() -> None:
    ap = argparse.ArgumentParser(description="3DGS vs DAV2 depth unprojection decoupling")
    ap.add_argument("--views-root", type=Path, default=_REPO / "3DGS" / "test2")
    ap.add_argument("--view", type=str, default="000")
    ap.add_argument("--nx", type=float, default=0.4958)
    ap.add_argument("--ny", type=float, default=0.9093)
    ap.add_argument(
        "--ply",
        type=Path,
        default=_REPO / "3DGS" / "gaussian-splatting" / "output" / "data2" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
    )
    ap.add_argument("--dav2-ckpt", type=Path, default=_REPO / "weights" / "depth_anything_v2_vitl.pth")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--inv-eps", type=float, default=1e-3)
    ap.add_argument("--skip-dav2", action="store_true", help="Only report 3DGS (+ wrong-R if --wrong-r)")
    ap.add_argument("--wrong-r", action="store_true", help="Also run wrong-R ablation with 3DGS depth")
    args = ap.parse_args()

    root = args.views_root
    cam_path = root / "camera_params" / f"view_{args.view}.json"
    depth_npy = root / "depth_raw" / f"view_{args.view}.npy"
    rgb_path = root / "rgb" / f"view_{args.view}.png"

    for p in (cam_path, depth_npy, rgb_path, args.ply):
        if not p.is_file():
            sys.exit(f"missing: {p}")

    view = CameraView.from_json(cam_path)
    unp = Unprojector(view, inv_eps=args.inv_eps)
    u, v = unp.normalized_to_pixel(args.nx, args.ny)

    gs_inv = np.load(depth_npy).astype(np.float64)
    gs_stored, z_gs = unp.sample_depth_raw(depth_npy, u, v, kind="expected_invdepth")
    _, p_gs, _ = unp.normalized_to_world(args.nx, args.ny, z_gs)

    xyz = load_ply_xyz(args.ply)
    nn_gs = nn_distance(p_gs, xyz)

    print("=" * 60)
    print(f"view_{args.view}  (nx,ny)=({args.nx},{args.ny})  pixel=({u},{v})")
    print(f"camera: {cam_path}")
    print(f"ply: {args.ply}  ({xyz.shape[0]} gaussians)")
    print("-" * 60)
    print("[3DGS expected_invdepth]")
    print(f"  inv @ pixel     = {gs_stored:.6f}")
    print(f"  z_cam           = {z_gs:.6f}")
    print(f"  P_world         = [{p_gs[0]:.4f}, {p_gs[1]:.4f}, {p_gs[2]:.4f}]")
    print(f"  NN distance     = {nn_gs:.4f}")

    if args.wrong_r:
        p_wrong = unproject_with_rotation(view, args.nx, args.ny, z_gs, use_wrong_r=True)
        nn_wrong = nn_distance(p_wrong, xyz)
        print("-" * 60)
        print("[3DGS + WRONG R (R_c2w used as R_w2c)]")
        print(f"  P_world         = [{p_wrong[0]:.4f}, {p_wrong[1]:.4f}, {p_wrong[2]:.4f}]")
        print(f"  NN distance     = {nn_wrong:.4f}")

    if args.skip_dav2:
        return

    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("-" * 60)
    print(f"[DAV2] loading vitl on {device} ...")
    model, torch_mod = load_dav2_model(args.dav2_ckpt, device)
    da = dav2_depth_map(model, torch_mod, rgb_path, device)
    if da.shape != gs_inv.shape:
        da = cv2.resize(da, (gs_inv.shape[1], gs_inv.shape[0]), interpolation=cv2.INTER_LINEAR)

    z_da_raw = float(da[v, u])
    inv_da = 1.0 / np.maximum(da, 1e-6)

    # Valid mask: central scene invdepth band (exclude sky / far junk)
    z_gs_map = 1.0 / np.maximum(gs_inv, args.inv_eps)
    mask = (
        (gs_inv > args.inv_eps)
        & (gs_inv < 2.0)
        & (da > 1e-6)
        & np.isfinite(da)
        & np.isfinite(gs_inv)
    )
    n_valid = int(mask.sum())
    print(f"  DAV2 z @ pixel   = {z_da_raw:.6f}  (relative, pre-align)")
    print(f"  affine fit pixels= {n_valid}")

    # --- DAV2 raw (no scale): use inverse of relative depth as arbitrary z ---
    z_da_unit = 1.0 / max(z_da_raw, 1e-6)
    p_da_raw = unproject_with_rotation(view, args.nx, args.ny, z_da_unit, use_wrong_r=False)
    nn_da_raw = nn_distance(p_da_raw, xyz)
    print("-" * 60)
    print("[DAV2 raw, z=1/DA_depth @ pixel, NO alignment]")
    print(f"  z_cam used      = {z_da_unit:.6f}")
    print(f"  NN distance     = {nn_da_raw:.4f}")

    if n_valid >= 50:
        # Per-view affine on linear z
        sc_z, off_z = fit_affine_z(z_gs_map, da, mask)
        z_da_affine = sc_z * z_da_raw + off_z
        p_da_affine = unproject_with_rotation(view, args.nx, args.ny, z_da_affine, use_wrong_r=False)
        nn_da_affine = nn_distance(p_da_affine, xyz)

        # Per-view scale+offset on inverse depth (make_depth_scale style)
        sc_i, off_i = fit_invdepth_scale_offset(gs_inv, inv_da, mask)
        inv_aligned = sc_i * (1.0 / max(z_da_raw, 1e-6)) + off_i
        # also build full-map inv aligned for consistency at pixel
        inv_da_aligned_map = sc_i * inv_da + off_i
        z_da_invfit = 1.0 / max(float(inv_da_aligned_map[v, u]), args.inv_eps)
        p_da_inv = unproject_with_rotation(view, args.nx, args.ny, z_da_invfit, use_wrong_r=False)
        nn_da_inv = nn_distance(p_da_inv, xyz)

        # Oracle: scale z at query pixel to match 3DGS exactly
        z_oracle = z_gs
        p_oracle = p_gs.copy()
        nn_oracle = nn_gs

        print("-" * 60)
        print(f"[DAV2 + per-view affine z]  z = {sc_z:.4f}*DA + {off_z:.4f}")
        print(f"  z_cam @ pixel   = {z_da_affine:.6f}  (gs {z_gs:.6f})")
        print(f"  NN distance     = {nn_da_affine:.4f}")
        print("-" * 60)
        print(f"[DAV2 + inv-depth scale/offset (make_depth_scale style)]")
        print(f"  scale, offset   = {sc_i:.4f}, {off_i:.4f}")
        print(f"  z_cam @ pixel   = {z_da_invfit:.6f}")
        print(f"  NN distance     = {nn_da_inv:.4f}")
        print("-" * 60)
        print("[oracle: force z_cam = 3DGS z at pixel]")
        print(f"  NN distance     = {nn_oracle:.4f}  (same as 3DGS by construction)")

        # Multi-view proxy: spread if we used only DAV2 affine at many pixels - report depth RMS on mask
        z_da_affine_map = sc_z * da + off_z
        rms = float(np.sqrt(np.mean((z_gs_map[mask] - z_da_affine_map[mask]) ** 2)))
        print("-" * 60)
        print(f"[depth map RMS on valid mask after affine z]  = {rms:.4f} m")
        print("  (high RMS → multi-view 3D points still scatter even if R is correct)")
    else:
        print("[skip] too few valid pixels for DAV2 alignment fit")

    print("=" * 60)
    print("Summary:")
    print(f"  3DGS NN          {nn_gs:.4f}")
    if args.wrong_r:
        print(f"  3DGS wrong-R NN  {nn_wrong:.4f}")
    print(f"  DAV2 no align NN {nn_da_raw:.4f}")
    if n_valid >= 50:
        print(f"  DAV2 affine z NN {nn_da_affine:.4f}")
        print(f"  DAV2 inv-fit NN  {nn_da_inv:.4f}")


if __name__ == "__main__":
    main()
