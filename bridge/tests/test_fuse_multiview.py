"""bridge/fuse_multiview.py 的测试。

覆盖纯数学函数（RANSAC、几何中位数），以及用合成 predictions
JSON 指向真实 test1/ 光栅、对 ``gather_candidates`` + ``fuse`` 做往返的冒烟测试。
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
        [10.0, 0.0, 0.0],   # 离群点
        [-9.0, 5.0, 1.0],   # 离群点
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


# --- 几何中位数 -------------------------------------------------------------


def test_geometric_median_robust_to_outlier() -> None:
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [100.0, 100.0, 100.0],   # 强离群点
    ])
    gm = geometric_median(pts)
    # 均值会被拉向 (25, 25, 25)；中位数应仍靠近原点
    assert np.linalg.norm(gm) < 1.0


def test_geometric_median_single() -> None:
    p = np.array([[3.0, -1.0, 4.0]])
    np.testing.assert_allclose(geometric_median(p), [3.0, -1.0, 4.0])


# --- 用真实 test1 光栅对 gather_candidates + fuse 做冒烟测试 ----------------


@pytest.mark.skipif(
    not (TEST1 / "camera_params" / "view_000.json").is_file()
    or not (TEST1 / "depth_raw" / "view_000.npy").is_file(),
    reason="3DGS/test1 artifacts missing",
)
def test_fuse_smoke_using_synthetic_predictions(tmp_path: Path) -> None:
    """构造一份 predictions.json，让所有解析成功的视角指向同一像素区域，
    运行 fuse()，并检查返回的 3D 点落在场景包围盒内。
    """
    # 选取 4 个视角，以及 view_000 测试基线的真值像素。
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
    # 再加一个极端 nx 的「离群」视角（很可能得到很远的世界点）
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
    # 以及一个失败视角，应被静默跳过。
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
    assert len(cands) >= 4  # 至少来自 view_ids 的 4 个；离群点也可能通过 min_inv

    result = fuse(predictions, inlier_radius=2.0, min_inv=1e-3)
    # 同一 (nx, ny) 的真实候选不应相距过远
    # （世界点来自指向同一标称像素的独立相机，未必完全重合，但 support 应 >= 1）。
    assert result.support >= 1
    p = result.P_world
    bbox_min = np.array([-100.0, -100.0, -100.0])
    bbox_max = np.array([100.0, 100.0, 100.0])
    assert np.all(p >= bbox_min) and np.all(p <= bbox_max)


def test_view_000_single_candidate_matches_unproject_baseline(tmp_path: Path) -> None:
    """单视角、单点 → fuse 必须返回该反投影本身。"""
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
