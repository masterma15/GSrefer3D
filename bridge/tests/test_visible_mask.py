"""物体逆深度修剪：空射线、相对差和遮挡分类。不启动光栅器。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from frustum_segment import resolve_vote_view_ids  # noqa: E402
from visible_mask import (  # noqa: E402
    _depth_raw_path,
    classify_occlusion,
    keep_visible_pixels,
    occlusion_ratio,
    trim_mask,
    world_to_view_matrix,
)


def test_empty_ray_is_removed() -> None:
    scene = np.array([[0.4, 0.0]], dtype=np.float64)
    obj = np.array([[0.0, 0.0]], dtype=np.float64)
    kept = keep_visible_pixels(scene, obj)
    assert kept.tolist() == [[False, False]]


def test_relative_agreement_keeps_only_close_pixels() -> None:
    scene = np.array([[0.50, 0.80, 0.50]], dtype=np.float64)
    obj = np.array([[0.51, 0.50, 0.50]], dtype=np.float64)
    kept = keep_visible_pixels(scene, obj, rel_tol=0.05)
    # 0.01/0.50 = 0.02 留下；0.30/0.80 = 0.375 是更近的整图，删掉；相等留下。
    assert kept.tolist() == [[True, False, True]]


def test_classification_boundaries() -> None:
    assert classify_occlusion(0.199) == "almost_unoccluded"
    assert classify_occlusion(0.2) == "partial"
    assert classify_occlusion(0.8) == "partial"
    assert classify_occlusion(0.801) == "fully_occluded"
    assert classify_occlusion(float("nan")) == "empty_mask"


def test_full_occlusion_has_no_centroid() -> None:
    mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    scene = np.array([[0.8, 0.8], [0.2, 0.2]])
    obj = np.array([[0.2, 0.2], [0.2, 0.2]])
    result = trim_mask(mask, scene, obj)
    assert result["kind"] == "fully_occluded"
    assert result["centroid_uv"] is None
    assert result["ratio"] == 1.0


def test_partial_centroid_uses_kept_pixels_only() -> None:
    mask = np.full((2, 4), 255, dtype=np.uint8)
    scene = np.array([[0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.9, 0.9]])
    obj = np.array([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    result = trim_mask(mask, scene, obj)
    assert result["kind"] == "partial"
    assert abs(result["ratio"] - 0.5) < 1e-9
    u, v = result["centroid_uv"]
    assert abs(u - 0.5) < 1e-9
    assert abs(v - 0.5) < 1e-9


def test_empty_mask_ratio_is_nan() -> None:
    mask = np.zeros((2, 2), dtype=np.uint8)
    kept = np.zeros((2, 2), dtype=bool)
    ratio = occlusion_ratio(mask, kept)
    assert ratio != ratio


def test_depth_raw_relative_is_under_views_root(tmp_path: Path) -> None:
    cam = {"depth_raw_relative": "depth_raw/view_000.npy"}
    got = _depth_raw_path(tmp_path, cam, "000")
    assert got == (tmp_path / "depth_raw" / "view_000.npy").resolve()


def test_world_to_view_recovers_camera_center() -> None:
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    center = np.array([1.5, -2.0, 0.25], dtype=np.float64)
    view = world_to_view_matrix(rotation, center)
    c2w = np.linalg.inv(view)
    assert np.allclose(c2w[:3, 3], center)


def test_vote_ids_prefer_kept_list(tmp_path: Path) -> None:
    (tmp_path / "projections_kept.json").write_text(
        json.dumps([{"view_id": "2"}, {"view_id": "10"}]),
        encoding="utf-8",
    )
    (tmp_path / "projections.json").write_text(
        json.dumps([{"view_id": "1"}]),
        encoding="utf-8",
    )
    ids, from_kept = resolve_vote_view_ids(tmp_path)
    assert from_kept is True
    assert ids == ["002", "010"]
    ids, from_kept = resolve_vote_view_ids(tmp_path, ignore_kept=True)
    assert from_kept is False
    assert ids == ["001"]


def test_vote_ids_fall_back_to_all_projections(tmp_path: Path) -> None:
    (tmp_path / "projections.json").write_text(
        json.dumps([{"view_id": "7"}]),
        encoding="utf-8",
    )
    ids, from_kept = resolve_vote_view_ids(tmp_path)
    assert from_kept is False
    assert ids == ["007"]
