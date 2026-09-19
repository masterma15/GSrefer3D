"""针对 bridge/unproject.py 的回归测试，使用 3DGS/test1/view_000。

锁定 2026-05-08 端到端验证过的数值：

  RoboRefer（RGB-D 模式）在 view_000 上返回 (nx, ny) = (0.458, 0.298)。
  -> 像素 (350, 169)，expected_invdepth = 0.4641861617565155
  -> z_cam = 2.154308082377821
  -> P_world = [-1.61395605, 0.70258973, -0.19355955]

  到 point_cloud.ply 的最近邻距离：0.2545（场景对角线 189.3）

若渲染约定、相机 JSON 结构或深度语义发生变化而本测试未同步更新，
回归会立刻失败。不要悄悄放宽容差；应修正上游改动。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from unproject import CameraView, Unprojector  # noqa: E402

TEST1 = REPO / "3DGS" / "test1"
CAMERA_JSON = TEST1 / "camera_params" / "view_000.json"
DEPTH_NPY = TEST1 / "depth_raw" / "view_000.npy"

NX, NY = 0.458, 0.298
EXPECTED_PIXEL = (350, 169)
EXPECTED_INVDEPTH = 0.4641861617565155
EXPECTED_Z_CAM = 2.154308082377821
EXPECTED_P_WORLD = np.array([0.61385237, 0.76297073, -0.04477087])

# 最近邻距离的回归守卫（point_cloud.ply，137 万个高斯）。
# 场景对角线约 189 时观测值为 0.2545。留出较宽松余量，
# 避免微小渲染调整误报，但约定翻转仍会触发。
NN_DISTANCE_MAX = 0.6


pytestmark = pytest.mark.skipif(
    not (CAMERA_JSON.is_file() and DEPTH_NPY.is_file()),
    reason="3DGS/test1/view_000 artifacts missing; re-render with render.py --custom_views",
)


@pytest.fixture(scope="module")
def unp() -> Unprojector:
    return Unprojector(CameraView.from_json(CAMERA_JSON))


def test_intrinsics_are_derived_from_fov(unp: Unprojector) -> None:
    fx, fy, cx, cy = unp.view.intrinsics
    assert cx == unp.view.width / 2.0
    assert cy == unp.view.height / 2.0
    # 合理性边界：FoV 约 72deg/57deg → fx, fy 应落在 [W/4, W] 内
    assert unp.view.width / 4 < fx < unp.view.width
    assert unp.view.height / 4 < fy < unp.view.height


def test_normalized_to_pixel(unp: Unprojector) -> None:
    assert unp.normalized_to_pixel(NX, NY) == EXPECTED_PIXEL


def test_depth_raw_sampling(unp: Unprojector) -> None:
    u, v = EXPECTED_PIXEL
    stored, z_cam = unp.sample_depth_raw(DEPTH_NPY, u, v, kind="expected_invdepth")
    assert stored == pytest.approx(EXPECTED_INVDEPTH, rel=0, abs=1e-9)
    assert z_cam == pytest.approx(EXPECTED_Z_CAM, rel=0, abs=1e-9)


def test_end_to_end_world_point_matches_roborefer_sample(unp: Unprojector) -> None:
    result = unp.normalized_with_depth_raw(NX, NY, DEPTH_NPY)
    assert result["pixel"] == EXPECTED_PIXEL
    assert result["z_cam"] == pytest.approx(EXPECTED_Z_CAM, rel=0, abs=1e-9)
    np.testing.assert_allclose(result["P_world"], EXPECTED_P_WORLD, atol=1e-6)


def test_world_point_lies_in_scene_bbox(unp: Unprojector) -> None:
    """防止 R_w2c.T @ P_cam + C 发生符号翻转。"""
    # 来自 point_cloud.ply（iteration_30000）的观测场景包围盒：
    bbox_min = np.array([-60.57, -30.57, -65.03])
    bbox_max = np.array([80.20, 26.52, 47.92])
    _, p_world, _ = unp.normalized_to_world(NX, NY, EXPECTED_Z_CAM)
    assert np.all(p_world >= bbox_min)
    assert np.all(p_world <= bbox_max)


# --- 可选：点云最近邻距离检查（仅当 plyfile 与 PLY 存在时） ---

PLY_CANDIDATES = sorted(
    (REPO / "3DGS" / "gaussian-splatting" / "output").glob(
        "*/point_cloud/iteration_30000/point_cloud.ply"
    )
)


@pytest.mark.skipif(not PLY_CANDIDATES, reason="no trained point_cloud.ply under 3DGS/gaussian-splatting/output")
def test_nearest_gaussian_distance_is_small(unp: Unprojector) -> None:
    try:
        from plyfile import PlyData
    except ImportError:
        pytest.skip("plyfile not installed in this env")

    ply = PlyData.read(str(PLY_CANDIDATES[0]))
    v = ply.elements[0]
    xyz = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1
    ).astype(np.float64)

    result = unp.normalized_with_depth_raw(NX, NY, DEPTH_NPY)
    p = result["P_world"]
    # 一次性回归用暴力搜索即可（约 140 万点，数秒）
    d = float(np.linalg.norm(xyz - p, axis=1).min())
    assert d < NN_DISTANCE_MAX, f"NN distance {d:.4f} exceeds regression ceiling {NN_DISTANCE_MAX}"
