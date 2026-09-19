"""视锥命中、T 加权投票与深度切片的单元测试（不读真实 ply）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from frustum_segment import (  # noqa: E402
    frustum_hit_mask,
    segment_gaussians,
    view_weight_from_report,
    world_to_cam_uvz,
)


def _cam() -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "rotation": np.eye(3).tolist(),
        "fov_x": 0.8,
        "fov_y": 0.6,
        "width": 100,
        "height": 80,
    }


def test_world_to_cam_center_pixel() -> None:
    cam = _cam()
    xyz = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    u, v, z = world_to_cam_uvz(xyz, cam)
    assert z[0] == np.float64(3.0) or abs(float(z[0]) - 3.0) < 1e-9
    assert abs(float(u[0]) - 50.0) < 1e-6
    assert abs(float(v[0]) - 40.0) < 1e-6


def test_frustum_hit_only_center_blob() -> None:
    cam = _cam()
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[35:46, 45:56] = 255
    xyz = np.array(
        [
            [0.0, 0.0, 3.0],  # 应命中（主点）
            [0.8, 0.0, 3.0],  # 偏出 mask
            [0.0, 0.0, 6.0],  # 仍在主点射线，也会命中（视锥无限深）
        ],
        dtype=np.float64,
    )
    hit, _ = frustum_hit_mask(xyz, cam, mask)
    assert bool(hit[0]) is True
    assert bool(hit[1]) is False
    assert bool(hit[2]) is True


def test_t_weight_rejected_is_zero() -> None:
    assert view_weight_from_report({"status": "rejected", "T_at_z_lo": 0.9}, use_t_weight=True) == 0.0
    assert view_weight_from_report({"status": "kept", "T_at_z_lo": 0.2}, use_t_weight=True) == 0.2
    assert view_weight_from_report({"status": "kept", "T_at_z_lo": 0.2}, use_t_weight=False) == 1.0


def test_vote_and_depth_slice_drops_far_along_ray() -> None:
    """两视角投票选中物体；深度带切掉射线上更远的桌面点。"""
    cam = _cam()
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[30:51, 40:61] = 255
    xyz = np.array(
        [
            [0.0, 0.0, 3.0],  # 物体
            [0.0, 0.0, 8.0],  # 同射线更远
            [1.2, 0.0, 3.0],  # 画外/mask 外
        ],
        dtype=np.float64,
    )
    views = [
        {"view_id": "000", "cam": cam, "mask": mask, "weight": 1.0},
        {"view_id": "001", "cam": cam, "mask": mask, "weight": 1.0},
    ]
    out = segment_gaussians(
        xyz,
        p_world=np.array([0.0, 0.0, 3.0]),
        views=views,
        min_vote=2.0,
        cluster_radius=0.3,
        z_slack=0.15,
        apply_depth_slice=True,
    )
    sel = set(int(i) for i in out["indices"])
    assert 0 in sel
    assert 1 not in sel
    assert 2 not in sel
    assert out["n_selected"] == 1
