"""Tests for bridge/fuse_multiview.py.

Focuses on the math-only functions (RANSAC, geometric median) plus a smoke
test that ``gather_candidates`` + ``fuse`` roundtrip a synthetic predictions
JSON pointing at the real test1/ rasters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from fuse_multiview import (  # noqa: E402
    fuse,
    gather_candidates,
    geometric_median,
    ransac_select_inliers,
)

TEST1 = REPO / "3DGS" / "test1"


# --- RANSAC ----------------------------------------------------------------


def test_ransac_picks_dense_cluster_over_outliers() -> None:
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.02, -0.01],
        [0.1, -0.05, 0.0],
        [10.0, 0.0, 0.0],   # outlier
        [-9.0, 5.0, 1.0],   # outlier
    ])
    inliers, support = ransac_select_inliers(pts, radius=0.5)
    assert support == 3
    assert sorted(inliers) == [0, 1, 2]


def test_ransac_single_point() -> None:
    pts = np.array([[1.0, 2.0, 3.0]])
    inliers, support = ransac_select_inliers(pts, radius=0.1)
    assert inliers == [0]
    assert support == 1


def test_ransac_empty() -> None:
    inliers, support = ransac_select_inliers(np.zeros((0, 3)), radius=1.0)
    assert inliers == []
    assert support == 0


# --- Geometric median -------------------------------------------------------


def test_geometric_median_robust_to_outlier() -> None:
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [100.0, 100.0, 100.0],   # heavy outlier
    ])
    gm = geometric_median(pts)
    # Mean would be pulled toward (25, 25, 25); median should stay near origin
    assert np.linalg.norm(gm) < 1.0


def test_geometric_median_single() -> None:
    p = np.array([[3.0, -1.0, 4.0]])
    np.testing.assert_allclose(geometric_median(p), [3.0, -1.0, 4.0])


# --- gather_candidates + fuse smoke test against real test1 rasters --------


@pytest.mark.skipif(
    not (TEST1 / "camera_params" / "view_000.json").is_file()
    or not (TEST1 / "depth_raw" / "view_000.npy").is_file(),
    reason="3DGS/test1 artifacts missing",
)
def test_fuse_smoke_using_synthetic_predictions(tmp_path: Path) -> None:
    """Build a predictions.json that aims all parsed views at the same pixel
    region, run fuse(), and check it returns a 3D point inside the scene bbox.
    """
    # Pick 4 views and the ground-truth pixel from view_000 testing baseline.
    nx, ny = 0.458, 0.298
    view_ids = [0, 1, 2, 3]
    views = []
    for vid in view_ids:
        name = f"view_{vid:03d}"
        views.append({
            "view_id": vid,
            "rgb_path": f"rgb/{name}.png",
            "depth_path": f"depth/{name}.png",
            "depth_raw_path": f"depth_raw/{name}.npy",
            "camera_path": f"camera_params/{name}.json",
            "raw_answer": f"[({nx}, {ny})]",
            "points": [{"nx": nx, "ny": ny}],
            "parse_ok": True,
            "error": None,
        })
    # Add an "outlier" view with an extreme nx (likely yields a far world point)
    views.append({
        "view_id": 5,
        "rgb_path": "rgb/view_005.png",
        "depth_path": "depth/view_005.png",
        "depth_raw_path": "depth_raw/view_005.npy",
        "camera_path": "camera_params/view_005.json",
        "raw_answer": "[(0.99, 0.01)]",
        "points": [{"nx": 0.99, "ny": 0.01}],
        "parse_ok": True,
        "error": None,
    })
    # And one failed view that should be skipped silently.
    views.append({
        "view_id": 6,
        "rgb_path": "rgb/view_006.png",
        "depth_path": "depth/view_006.png",
        "depth_raw_path": "depth_raw/view_006.npy",
        "camera_path": "camera_params/view_006.json",
        "raw_answer": None,
        "points": [],
        "parse_ok": False,
        "error": "server down",
    })

    predictions = {
        "prompt": "test",
        "suffix": "",
        "url": "fake",
        "enable_depth": 1,
        "root": str(TEST1).replace("\\", "/"),
        "views": views,
    }

    cands = gather_candidates(predictions, min_inv=1e-3)
    assert len(cands) >= 4  # at least 4 from view_ids; outlier may also pass min_inv

    result = fuse(predictions, inlier_radius=2.0, min_inv=1e-3)
    # All real candidates from the same (nx, ny) shouldn't be miles apart
    # (their world points come from independent cameras pointing at the same
    # nominal pixel, so they may not coincide exactly, but support should be >= 1).
    assert result.support >= 1
    p = result.P_world
    bbox_min = np.array([-100.0, -100.0, -100.0])
    bbox_max = np.array([100.0, 100.0, 100.0])
    assert np.all(p >= bbox_min) and np.all(p <= bbox_max)


def test_view_000_single_candidate_matches_unproject_baseline(tmp_path: Path) -> None:
    """Single view, single point -> fuse must return the unprojection itself."""
    if not (TEST1 / "camera_params" / "view_000.json").is_file():
        pytest.skip("test1 artifacts missing")

    predictions = {
        "prompt": "test", "suffix": "", "url": "fake", "enable_depth": 1,
        "root": str(TEST1).replace("\\", "/"),
        "views": [{
            "view_id": 0,
            "rgb_path": "rgb/view_000.png",
            "depth_path": "depth/view_000.png",
            "depth_raw_path": "depth_raw/view_000.npy",
            "camera_path": "camera_params/view_000.json",
            "raw_answer": "[(0.458, 0.298)]",
            "points": [{"nx": 0.458, "ny": 0.298}],
            "parse_ok": True,
            "error": None,
        }],
    }
    result = fuse(predictions, inlier_radius=0.5, min_inv=1e-3)
    expected = np.array([-1.61395605, 0.70258973, -0.19355955])
    np.testing.assert_allclose(result.P_world, expected, atol=1e-6)
    assert result.support == 1
