"""Regression test for bridge/unproject.py using 3DGS/test1/view_000.

Pins the numbers that were end-to-end verified on 2026-05-08:

  RoboRefer (RGB-D mode) on view_000 returned (nx, ny) = (0.458, 0.298).
  -> pixel (350, 169), expected_invdepth = 0.4641861617565155
  -> z_cam = 2.154308082377821
  -> P_world = [-1.61395605, 0.70258973, -0.19355955]

  nearest-neighbor distance to point_cloud.ply: 0.2545 (scene diag 189.3)

If the rendering convention, camera JSON schema, or depth semantics ever
change without updating this test, the regression fails loudly. Do NOT
silently bump tolerances; fix the upstream change instead.
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

# Regression guard on NN distance (point_cloud.ply, 1.37M gaussians).
# Observed 0.2545 in a scene with diagonal ~189. Keep a generous headroom so
# a minor rendering tweak doesn't false-alarm, but a convention flip will.
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
    # sanity bounds: FoV ~72deg/57deg -> fx,fy well inside [W/4, W]
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
    """Guards against sign flips in R_w2c.T @ P_cam + C."""
    # Observed scene bbox from point_cloud.ply (iteration_30000):
    bbox_min = np.array([-60.57, -30.57, -65.03])
    bbox_max = np.array([80.20, 26.52, 47.92])
    _, p_world, _ = unp.normalized_to_world(NX, NY, EXPECTED_Z_CAM)
    assert np.all(p_world >= bbox_min)
    assert np.all(p_world <= bbox_max)


# --- Optional: point-cloud NN distance check (only if plyfile + PLY present) ---

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
    # brute force is fine for a one-off regression (~1.4M points, a few seconds)
    d = float(np.linalg.norm(xyz - p, axis=1).min())
    assert d < NN_DISTANCE_MAX, f"NN distance {d:.4f} exceeds regression ceiling {NN_DISTANCE_MAX}"
