"""Tests for bridge/ray_unproject.py depth pick logic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from ray_unproject import refine_z_on_ray, RayUnprojectConfig  # noqa: E402
from unproject import CameraView, Unprojector  # noqa: E402


class _StubModel:
  def __init__(self, xyz: np.ndarray):
    self.xyz = xyz

  def _candidate_indices(self, C, d, t_end):  # noqa: ANN001
    return np.arange(len(self.xyz))


def test_p75_max_z0_pushes_deeper_not_shallower(monkeypatch) -> None:
    view = CameraView(
        width=1000,
        height=800,
        fov_x=1.0,
        fov_y=0.8,
        R_w2c=np.eye(3),
        camera_center=np.zeros(3),
    )
    unp = Unprojector(view)
    z0 = 2.0
  # Gaussians mostly behind z0 in camera z (deeper = larger z_cam)
    xyz = np.array([
        [0.0, 0.0, 2.5],
        [0.0, 0.0, 2.8],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 3.2],
    ], dtype=np.float64)
    model = _StubModel(xyz)
    cam = {"rotation": np.eye(3), "position": np.zeros(3), "width": 1000, "height": 800, "fov_x": 1.0, "fov_y": 0.8}

    import ray_unproject as ru

    monkeypatch.setattr(ru, "world_to_image", lambda p, c: (500.0, 400.0, float(p[2])))
    monkeypatch.setattr(ru, "_gaussian_alpha_on_ray", lambda *a, **k: np.ones(len(a[1])))

    z_ref, meta = refine_z_on_ray(view, unp, 0.5, 0.5, z0, model, cam, cfg=RayUnprojectConfig())
    assert meta["method"] == "p75_max_z0"
    assert z_ref >= z0
    assert meta["delta_z"] >= 0


def test_p75_max_z0_keeps_z0_when_pick_shallower(monkeypatch) -> None:
    view = CameraView(
        width=1000,
        height=800,
        fov_x=1.0,
        fov_y=0.8,
        R_w2c=np.eye(3),
        camera_center=np.zeros(3),
    )
    unp = Unprojector(view)
    z0 = 3.0
    xyz = np.array([
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 1.8],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.2],
    ], dtype=np.float64)
    model = _StubModel(xyz)
    cam = {"rotation": np.eye(3), "position": np.zeros(3), "width": 1000, "height": 800, "fov_x": 1.0, "fov_y": 0.8}

    import ray_unproject as ru

    monkeypatch.setattr(ru, "world_to_image", lambda p, c: (500.0, 400.0, float(p[2])))
    monkeypatch.setattr(ru, "_gaussian_alpha_on_ray", lambda *a, **k: np.ones(len(a[1])))

    z_ref, meta = refine_z_on_ray(view, unp, 0.5, 0.5, z0, model, cam, cfg=RayUnprojectConfig())
    assert z_ref == z0
    assert meta["delta_z"] == 0.0
