"""Ray transmittance visibility for 3DGS (reject-mode B: cluster depth band).

Requires ``scipy`` (``cKDTree``); listed in ``3DGS/environment-envGS.yml``.

Per view: one ray from camera center C through fused P_world. Cluster Gaussians
define camera-depth band [z_lo, z_hi]. Foreground occlusion is
``T(z_lo) = prod(1 - alpha_i)`` over scene Gaussians near the ray with z_cam < z_lo,
using 3D Gaussian falloff (opacity * exp(-0.5 * mahalanobis^2) at closest point on ray).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gen_training_data import world_to_image


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) per row -> (N,3,3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    q = q / n
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def load_ply_gaussians(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load xyz (N,3), opacity (N,), scales (N,3), rot (N,4) wxyz from 3DGS ply."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply.elements[0]
    xyz = np.column_stack(
        [np.asarray(v["x"], dtype=np.float64), np.asarray(v["y"], dtype=np.float64), np.asarray(v["z"], dtype=np.float64)]
    )
    opacity = _sigmoid(np.asarray(v["opacity"], dtype=np.float64).reshape(-1))
    scales = np.exp(
        np.column_stack(
            [
                np.asarray(v["scale_0"], dtype=np.float64),
                np.asarray(v["scale_1"], dtype=np.float64),
                np.asarray(v["scale_2"], dtype=np.float64),
            ]
        )
    )
    rot = np.column_stack(
        [
            np.asarray(v["rot_0"], dtype=np.float64),
            np.asarray(v["rot_1"], dtype=np.float64),
            np.asarray(v["rot_2"], dtype=np.float64),
            np.asarray(v["rot_3"], dtype=np.float64),
        ]
    )
    return xyz, opacity, scales, rot


def cluster_depth_band(cluster_xyz: np.ndarray, cam: dict, *, frame_margin: int, w: int, h: int) -> tuple[float, float] | None:
    """Camera z range [z_lo, z_hi] for cluster points in frame."""
    from gen_training_data import in_frame

    zs: list[float] = []
    for p in cluster_xyz:
        u, v, z = world_to_image(p, cam)
        if z > 0 and in_frame(u, v, w, h, margin=frame_margin):
            zs.append(float(z))
    if not zs:
        return None
    return float(min(zs)), float(max(zs))


def _world_to_cam_batch(p: np.ndarray, cam: dict) -> np.ndarray:
    r_c2w = np.asarray(cam["rotation"], dtype=np.float64)
    c = np.asarray(cam["position"], dtype=np.float64)
    return (p - c) @ r_c2w


@dataclass
class RayOcclusionModel:
    """Scene Gaussians + KD-tree for ray-cylinder candidate queries."""

    xyz: np.ndarray
    opacity: np.ndarray
    scales: np.ndarray
    rot: np.ndarray
    perp_radius: float
    sample_step: float
    _tree: Any = None

    @classmethod
    def from_ply(cls, path: Path, *, perp_radius: float = 0.08, sample_step: float = 0.025) -> RayOcclusionModel:
        xyz, opacity, scales, rot = load_ply_gaussians(path)
        try:
            from scipy.spatial import cKDTree
        except ImportError as e:
            raise ImportError("ray reject mode requires scipy (cKDTree)") from e
        tree = cKDTree(xyz)
        return cls(
            xyz=xyz,
            opacity=opacity,
            scales=scales,
            rot=rot,
            perp_radius=perp_radius,
            sample_step=sample_step,
            _tree=tree,
        )

    def _candidate_indices(self, C: np.ndarray, d: np.ndarray, t_end: float) -> np.ndarray:
        assert self._tree is not None
        idx: set[int] = set()
        t = max(self.sample_step, 0.01)
        while t <= t_end + 1e-6:
            pt = C + d * t
            for j in self._tree.query_ball_point(pt, self.perp_radius):
                idx.add(int(j))
            t += self.sample_step
        if not idx:
            return np.zeros(0, dtype=np.int64)
        return np.fromiter(idx, dtype=np.int64)

    def transmittance_before(
        self,
        cam: dict,
        p_world: np.ndarray,
        z_cut: float,
        *,
        z_hi: float | None = None,
    ) -> dict[str, float]:
        """Integrate alpha along C->P_world; return T and stats at camera depth z_cut."""
        c = np.asarray(cam["position"], dtype=np.float64)
        d = p_world - c
        dist = float(np.linalg.norm(d))
        if dist < 1e-8:
            return {"T_at_z_cut": 0.0, "n_candidates": 0, "n_foreground": 0}
        d = d / dist
        t_end = dist * 1.05 if z_hi is None else dist * 1.2
        idx = self._candidate_indices(c, d, t_end)
        if idx.size == 0:
            return {"T_at_z_cut": 1.0, "n_candidates": 0, "n_foreground": 0}

        mu = self.xyz[idx]
        opacity = self.opacity[idx]
        scales = self.scales[idx]
        rot = self.rot[idx]

        v = mu - c
        t_ray = v @ d
        closest = c + t_ray[:, None] * d
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
        alpha = np.clip(opacity * g, 0.0, 0.99)

        p_cam = _world_to_cam_batch(mu, cam)
        z_cam = p_cam[:, 2]

        order = np.argsort(z_cam)
        t_val = 1.0
        n_fg = 0
        for i in order:
            if z_cam[i] >= z_cut:
                break
            if t_ray[i] <= 0:
                continue
            a = float(alpha[i])
            if a < 1e-4:
                continue
            n_fg += 1
            t_val *= 1.0 - a
            if t_val < 1e-4:
                t_val = 0.0
                break
        return {
            "T_at_z_cut": float(t_val),
            "n_candidates": int(idx.size),
            "n_foreground": int(n_fg),
        }

    def transmittance_beam_min(
        self,
        cam: dict,
        p_world: np.ndarray,
        z_cut: float,
        *,
        z_hi: float | None = None,
        beam_offset_m: float = 0.05,
    ) -> dict[str, float]:
        """Min T over center ray + 4 offsets in camera plane (catches plush beside the line)."""
        r_c2w = np.asarray(cam["rotation"], dtype=np.float64)
        right = r_c2w[:, 0]
        up = r_c2w[:, 1]
        targets = [np.asarray(p_world, dtype=np.float64)]
        for du, dv in ((beam_offset_m, 0), (-beam_offset_m, 0), (0, beam_offset_m), (0, -beam_offset_m)):
            targets.append(targets[0] + right * du + up * dv)
        t_min = 1.0
        n_cand = 0
        n_fg = 0
        for t_world in targets:
            st = self.transmittance_before(cam, t_world, z_cut, z_hi=z_hi)
            t_min = min(t_min, st["T_at_z_cut"])
            n_cand = max(n_cand, st["n_candidates"])
            n_fg = max(n_fg, st["n_foreground"])
        return {
            "T_at_z_cut": float(t_min),
            "n_candidates": n_cand,
            "n_foreground": n_fg,
            "beam_offset_m": beam_offset_m,
        }


def ray_reject_reason(
  T_at_z_lo: float | None,
  *,
  min_transmittance: float,
) -> str | None:
    if T_at_z_lo is None:
        return "no_cluster_depth_band"
    if T_at_z_lo < min_transmittance:
        return "ray_foreground_occluded"
    return None
