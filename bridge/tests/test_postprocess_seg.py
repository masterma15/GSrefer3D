"""后处理：桥点不补，大块不进掩码挑选，弱块比例接近时不删；maskclose 相关。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from postprocess_seg import (  # noqa: E402
    connected_labels,
    fill_same_component,
    large_component_ids,
    mask_direct_select,
    morph_close_fill,
    postprocess_indices,
    should_run_maskclose,
    weak_keep_ids,
)


def test_bridge_point_is_not_filled() -> None:
    core = np.array(
        [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.028, 0.0, 0.0], [0.033, 0.0, 0.0]],
        dtype=np.float64,
    )
    labels = connected_labels(core, 0.015)
    assert int(labels[0]) == int(labels[1])
    assert int(labels[2]) == int(labels[3])
    assert int(labels[0]) != int(labels[2])
    white = np.array([[0.0165, 0.0, 0.0], [-0.010, 0.0, 0.0]], dtype=np.float64)
    rows, owners = fill_same_component(core, labels, white, radius=0.015)
    assert rows.tolist() == [1]
    assert int(owners[0]) == int(labels[0])


def test_two_large_parts_are_not_a_mask_contest() -> None:
    labels = np.array([0] * 100 + [1] * 40, dtype=np.int64)
    large = large_component_ids(labels, large_frac=0.20)
    assert large == {0, 1}
    kept, reason = weak_keep_ids(
        labels, large_ids=large, inside_ratio={0: 0.2, 1: 0.9}, tie=0.10
    )
    assert kept == {0, 1}
    assert reason == "no_contest"


def test_clear_mask_ratio_drops_the_other_weak_piece() -> None:
    core = np.zeros((12, 3), dtype=np.float64)
    core[:10, 0] = np.linspace(0.0, 0.09, 10)
    core[10, 0] = 0.30
    core[11, 0] = 0.50

    def ratios(labels: np.ndarray, xyz: np.ndarray) -> dict[int, float]:
        out = {}
        for lab in np.unique(labels):
            center = float(xyz[labels == lab][:, 0].mean())
            out[int(lab)] = 0.95 if center < 0.4 else 0.05
        return out

    result = postprocess_indices(
        core,
        np.arange(12, dtype=np.int64),
        np.zeros((0, 3), dtype=np.float64),
        np.zeros(0, dtype=np.int64),
        mask_xyz_ratio=ratios,
    )
    assert result["mask_choice"] == "best"
    assert set(result["indices"].tolist()) == set(range(11))


def test_close_ratios_drop_nothing() -> None:
    labels = np.array([0] * 100 + [1] * 5 + [2] * 5, dtype=np.int64)
    large = large_component_ids(labels, large_frac=0.20)
    kept, reason = weak_keep_ids(
        labels,
        large_ids=large,
        inside_ratio={0: 0.5, 1: 0.90, 2: 0.84},
        tie=0.10,
    )
    assert reason == "tie"
    assert kept == {0, 1, 2}


def test_morph_close_fills_gap_without_mask() -> None:
    """缺口两侧基点 + 缝中白集：闭包能补上；远离的白集外缘被腐蚀掉。"""
    base = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float64)
    base_index = np.array([0, 1], dtype=np.int64)
    pool = np.array([[0.2, 0.0, 0.0], [0.55, 0.0, 0.0], [0.70, 0.0, 0.0]], dtype=np.float64)
    pool_index = np.array([10, 11, 12], dtype=np.int64)
    xyz, idx, n_added = morph_close_fill(base, base_index, pool, pool_index, radius=0.21)
    kept = set(idx.tolist())
    assert {0, 1, 10}.issubset(kept)
    assert 12 not in kept
    assert n_added >= 1


def test_mask_direct_keeps_main_and_drops_discrete_outside() -> None:
    """主体保留；掩码内近邻白集可补；远处小块整块删（无全场闭包）。"""

    class _Cam:
        intrinsics = (1.0, 1.0, 0.0, 0.0)

        def world_to_cam_points(self, xyz: np.ndarray) -> np.ndarray:
            out = np.zeros_like(xyz, dtype=np.float64)
            out[:, 0] = xyz[:, 0]
            out[:, 1] = xyz[:, 1]
            out[:, 2] = 1.0
            return out

    base_xyz = np.zeros((11, 3), dtype=np.float64)
    base_xyz[:10, 0] = np.linspace(0.0, 0.9, 10)
    base_xyz[10] = [15.0, 0.0, 0.0]
    base_index = np.arange(11, dtype=np.int64)
    pool_xyz = np.array([[1.0, 0.0, 0.0], [15.05, 0.0, 0.0]], dtype=np.float64)
    pool_index = np.array([20, 21], dtype=np.int64)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 1:10] = 255

    result = mask_direct_select(
        base_xyz,
        base_index,
        pool_xyz,
        pool_index,
        np.array([True, False]),
        np.array([True, False]),
        cam=_Cam(),
        mask=mask,
        radius=0.15,
    )
    kept = set(result["indices"].tolist())
    assert set(range(10)).issubset(kept)
    assert 10 not in kept
    assert 20 in kept
    assert 21 not in kept
    assert result["n_added"] >= 1
    assert "discrete_blocks" in str(result["delete_mode"])


def test_mask_direct_drops_thin_exterior_spur() -> None:
    """主体旁细颈外突支整支删；主体上掩码空洞点保留。"""

    class _Cam:
        intrinsics = (1.0, 1.0, 0.0, 0.0)

        def world_to_cam_points(self, xyz: np.ndarray) -> np.ndarray:
            out = np.zeros_like(xyz, dtype=np.float64)
            out[:, 0] = xyz[:, 0]
            out[:, 1] = xyz[:, 1]
            out[:, 2] = 1.0
            return out

    n_body = 40
    spur = np.array(
        [[10.15, 0.0, 0.0], [10.40, 0.0, 0.0], [10.65, 0.0, 0.0], [10.90, 0.0, 0.0], [11.15, 0.0, 0.0]],
        dtype=np.float64,
    )
    base_xyz = np.zeros((n_body + spur.shape[0], 3), dtype=np.float64)
    base_xyz[:n_body, 0] = np.linspace(0.0, 9.9, n_body)
    base_xyz[n_body:] = spur
    base_index = np.arange(base_xyz.shape[0], dtype=np.int64)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 0:10] = 255
    mask[:, 4] = 0
    radius = 0.31

    result = mask_direct_select(
        base_xyz,
        base_index,
        np.zeros((0, 3), dtype=np.float64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=bool),
        np.zeros(0, dtype=bool),
        cam=_Cam(),
        mask=mask,
        radius=radius,
        spur_frac=0.15,
        neck_touch_max=0.35,
    )
    kept = set(result["indices"].tolist())
    assert set(range(n_body)).issubset(kept)
    assert not kept.intersection(set(range(n_body, n_body + spur.shape[0])))
    assert int(result["n_spurs_dropped"]) >= 1
    assert "exterior_spurs" in str(result["delete_mode"])


def test_maskclose_gate_chopsticks_and_scissors() -> None:
    """右筷极端欠选、中间剪刀少视角欠完整应开；低选点但视角多、或少视角但选点过万不开。"""
    ok, reason = should_run_maskclose(642, 77)
    assert ok and "n_selected<" in reason
    ok, reason = should_run_maskclose(3413, 27)
    assert ok and "few_views" in reason
    ok, reason = should_run_maskclose(2476, 101)
    assert not ok and reason == "softedge_ok"
    ok, reason = should_run_maskclose(10369, 29)
    assert not ok and reason == "softedge_ok"
    ok, reason = should_run_maskclose(8779, 90)
    assert not ok and reason == "softedge_ok"
