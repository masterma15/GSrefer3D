"""消融选择：文本只按 DINO 分数选框。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from mask_grounding import GroundingSamConfig, _rank_boxes  # noqa: E402


def _boxes() -> tuple[np.ndarray, np.ndarray]:
    # 近处低分框含点 (10, 10)；远处高分框。100x100 上默认距离阈值约 21 像素。
    boxes = np.array(
        [
            [0.0, 0.0, 20.0, 20.0],
            [80.0, 80.0, 95.0, 95.0],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.2, 0.9], dtype=np.float64)
    return boxes, scores


def test_point_rank_drops_far_high_score_box() -> None:
    boxes, scores = _boxes()
    ranked = _rank_boxes(boxes, scores, 10.0, 10.0, 100, 100, GroundingSamConfig())
    assert [r[0] for r in ranked] == [0]


def test_text_only_keeps_far_box_by_dino_score() -> None:
    boxes, scores = _boxes()
    cfg = GroundingSamConfig(text_only=True)
    ranked = _rank_boxes(boxes, scores, 10.0, 10.0, 100, 100, cfg)
    assert [r[0] for r in ranked] == [1, 0]
    assert ranked[0][1] == 0.9


def test_box_larger_than_image_fraction_is_dropped() -> None:
    """面积超过画面 0.35 的框直接丢掉，即使检测分更高。"""
    boxes = np.array(
        [
            [0.0, 0.0, 90.0, 90.0],
            [8.0, 8.0, 18.0, 18.0],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.70, 0.50], dtype=np.float64)
    ranked = _rank_boxes(boxes, scores, 10.0, 10.0, 100, 100, GroundingSamConfig())
    assert [r[0] for r in ranked] == [1]


def test_higher_score_within_gate_beats_closer_lower_score() -> None:
    """距离门内，置信度差过距离加权时，高分框优先。"""
    boxes = np.array(
        [
            [0.0, 0.0, 20.0, 20.0],
            [25.0, 10.0, 40.0, 25.0],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.40, 0.55], dtype=np.float64)
    ranked = _rank_boxes(boxes, scores, 10.0, 10.0, 100, 100, GroundingSamConfig())
    assert ranked[0][0] == 1


def test_text_only_keeps_large_high_score_box() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 90.0, 90.0],
            [80.0, 80.0, 95.0, 95.0],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.99, 0.4], dtype=np.float64)
    cfg = GroundingSamConfig(text_only=True, max_box_area_ratio=0.35)
    ranked = _rank_boxes(boxes, scores, 10.0, 10.0, 100, 100, cfg)
    assert [r[0] for r in ranked] == [0, 1]
