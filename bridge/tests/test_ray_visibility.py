"""Tests for ray transmittance visibility (reject-mode B)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from ray_visibility import RayOcclusionModel, cluster_depth_band, ray_reject_reason  # noqa: E402


def test_ray_reject_low_transmittance() -> None:
    assert ray_reject_reason(0.1, min_transmittance=0.45) == "ray_foreground_occluded"
    assert ray_reject_reason(0.6, min_transmittance=0.45) is None
    assert ray_reject_reason(0.40, min_transmittance=0.45) == "ray_foreground_occluded"


def test_transmittance_foreground_blocks() -> None:
    """Opaque Gaussian in front of anchor should drop T(z_lo)."""
    xyz = np.array(
        [
            [0.0, 0.0, 2.0],  # foreground occluder
            [0.0, 0.0, 4.0],  # target cluster
        ],
        dtype=np.float64,
    )
    opacity = np.array([0.95, 0.9], dtype=np.float64)
    scales = np.array([[0.15, 0.15, 0.15], [0.1, 0.1, 0.1]], dtype=np.float64)
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
    model = RayOcclusionModel(
        xyz=xyz,
        opacity=opacity,
        scales=scales,
        rot=rot,
        perp_radius=0.2,
        sample_step=0.05,
    )
    from scipy.spatial import cKDTree

    model._tree = cKDTree(xyz)
    cam = {
        "position": [0.0, 0.0, 0.0],
        "rotation": np.eye(3).tolist(),
        "fov_x": 0.8,
        "fov_y": 0.6,
        "width": 800,
        "height": 600,
    }
    p_world = np.array([0.0, 0.0, 4.0])
    stats = model.transmittance_before(cam, p_world, z_cut=3.5, z_hi=4.5)
    assert stats["T_at_z_cut"] < 0.2
    stats_clear = model.transmittance_before(cam, p_world, z_cut=1.5, z_hi=4.5)
    assert stats_clear["T_at_z_cut"] > 0.5


def test_cluster_depth_band() -> None:
    cam = {
        "position": [0.0, 0.0, 0.0],
        "rotation": np.eye(3).tolist(),
        "fov_x": 0.8,
        "fov_y": 0.6,
        "width": 800,
        "height": 600,
    }
    cluster = np.array([[0.0, 0.0, 3.0], [0.1, 0.0, 3.5]], dtype=np.float64)
    band = cluster_depth_band(cluster, cam, frame_margin=0, w=800, h=600)
    assert band is not None
    z_lo, z_hi = band
    assert z_lo == pytest.approx(3.0)
    assert z_hi == pytest.approx(3.5)
