"""Refine unproject depth along the camera ray using local 3DGS Gaussians.

Given (nx, ny) and raster expected_invdepth, build an initial camera depth z0, then
select Gaussians near the ray and near the click in image space; set depth from their
camera-space z using p75 in the band, then ``z_ref = max(z0, z_pick)`` (never shallower
than raster z0).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gen_training_data import world_to_image
from ray_visibility import RayOcclusionModel, _quat_to_rot
from unproject import CameraView, Unprojector


@dataclass(frozen=True)
class RayUnprojectConfig:
    perp_radius: float = 0.06
    pixel_radius: float = 6.0
    z_band_frac: float = 2.0
    min_alpha_on_ray: float = 0.45
    k_max: int = 64
    sample_step: float = 0.025


def camera_dict_from_view(view: CameraView) -> dict:
    return {
        "rotation": view.R_w2c.T,
        "position": view.camera_center,
        "width": view.width,
        "height": view.height,
        "fov_x": view.fov_x,
        "fov_y": view.fov_y,
    }


def _ray_dir_world(view: CameraView, fu: float, fv: float) -> np.ndarray:
    fx, fy, cx, cy = view.intrinsics
    d_cam = np.array([(fu - cx) / fx, (fv - cy) / fy, 1.0], dtype=np.float64)
    d_cam /= np.linalg.norm(d_cam) + 1e-12
    return view.R_w2c.T @ d_cam


def _gaussian_alpha_on_ray(
    model: RayOcclusionModel,
    idx: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Per-Gaussian alpha weight (opacity * Gaussian falloff on ray)."""
    mu = model.xyz[idx]
    opacity = model.opacity[idx]
    scales = model.scales[idx]
    rot = model.rot[idx]
    v = mu - C
    t_ray = v @ d
    closest = C + t_ray[:, None] * d
    delta = mu - closest
    r = _quat_to_rot(rot)
    s = np.zeros((mu.shape[0], 3, 3), dtype=np.float64)
    s[:, 0, 0] = scales[:, 0]
    s[:, 1, 1] = scales[:, 1]
    s[:, 2, 2] = scales[:, 2]
    l_mat = np.matmul(r, s)
    cov = np.matmul(l_mat, np.transpose(l_mat, (0, 2, 1)))
    power = np.empty(mu.shape[0], dtype=np.float64)
    for i in range(mu.shape[0]):
        try:
            inv = np.linalg.inv(cov[i])
            power[i] = float(delta[i] @ inv @ delta[i])
        except np.linalg.LinAlgError:
            power[i] = 1e6
    g = np.exp(-0.5 * np.clip(power, 0.0, 50.0))
    return np.clip(opacity * g, 0.0, 0.99)


def refine_z_on_ray(
    view: CameraView,
    unp: Unprojector,
    nx: float,
    ny: float,
    z0: float,
    model: RayOcclusionModel,
    cam: dict,
    *,
    cfg: RayUnprojectConfig,
) -> tuple[float, dict]:
    """Return (z_refined, debug dict). Falls back to z0 if no valid Gaussians."""
    u0, v0 = unp.normalized_to_pixel(nx, ny)
    fu = nx * view.width
    fv = ny * view.height
    C = view.camera_center
    d = _ray_dir_world(view, fu, fv)
    _, p0, _ = unp.normalized_to_world(nx, ny, z0)
    dist0 = float(np.linalg.norm(p0 - C))
    z_lo = z0 * (1.0 - cfg.z_band_frac)
    z_hi = z0 * (1.0 + cfg.z_band_frac)
    t_end = max(dist0 * (1.0 + cfg.z_band_frac), 0.15)
    idx = model._candidate_indices(C, d, t_end)
    meta: dict = {
        "z_cam_raw": float(z0),
        "n_ray_candidates": int(idx.size),
        "n_local": 0,
        "min_alpha_on_ray": float(cfg.min_alpha_on_ray),
        "method": "fallback_z0",
    }
    if idx.size == 0:
        return z0, meta

    p_cam_all = (model.xyz[idx] - C) @ view.R_w2c
    z_cam_all = p_cam_all[:, 2]
    alphas = _gaussian_alpha_on_ray(model, idx, C, d)

    keep = []
    for j, gi in enumerate(idx):
        if z_cam_all[j] < z_lo or z_cam_all[j] > z_hi:
            continue
        u, v, _ = world_to_image(model.xyz[gi], cam)
        if abs(u - fu) > cfg.pixel_radius or abs(v - fv) > cfg.pixel_radius:
            continue
        if alphas[j] < cfg.min_alpha_on_ray:
            continue
        keep.append(j)

    meta["n_local"] = len(keep)
    if not keep:
        return z0, meta

    z_sel = z_cam_all[np.asarray(keep, dtype=np.int64)]
    z_pick = float(np.percentile(z_sel, 75))
    z_ref = float(max(z0, z_pick))
    meta["method"] = "p75_max_z0"
    meta["z_pick"] = z_pick
    meta["z_cam_refined"] = z_ref
    meta["delta_z"] = z_ref - z0
    return z_ref, meta


def unproject_with_ray_depth(
    view: CameraView,
    unp: Unprojector,
    nx: float,
    ny: float,
    depth_path: Path,
    model: RayOcclusionModel,
    *,
    cfg: RayUnprojectConfig | None = None,
    kind: str = "expected_invdepth",
) -> dict:
    """Like normalized_with_depth_raw but z from ray-local Gaussians when possible."""
    cfg = cfg or RayUnprojectConfig()
    u, v = unp.normalized_to_pixel(nx, ny)
    stored, z0 = unp.sample_depth_raw(depth_path, u, v, kind=kind)
    cam = camera_dict_from_view(view)
    z_ref, ray_meta = refine_z_on_ray(view, unp, nx, ny, z0, model, cam, cfg=cfg)
    p_cam, p_world, pix = unp.normalized_to_world(nx, ny, z_ref)
    return {
        "pixel": pix,
        "stored_value": stored,
        "depth_kind": kind,
        "z_cam": z_ref,
        "z_cam_raw": z0,
        "P_cam": p_cam,
        "P_world": p_world,
        "depth_mode": "ray",
        "ray": ray_meta,
    }
