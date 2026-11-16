"""后处理：桥点不补，大块不进掩码挑选，弱块比例接近时不删；maskclose 相关。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from postprocess_seg import (  # noqa: E402
    connected_labels,
    density_mask_fill,
    fill_same_component,
    jury_delete_keep_mask,
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


def test_maskclose_gate_few_views_low_selected() -> None:
    """n_views≤30 且 n_selected<7000 开门控；左/右剪、药瓶、大视角右筷跳过。"""
    ok, reason = should_run_maskclose(5569, 27)
    assert ok and "few_views" in reason and "7000" in reason
    ok, reason = should_run_maskclose(6999, 30)
    assert ok
    ok, reason = should_run_maskclose(7000, 27)
    assert not ok and reason == "softedge_ok"
    ok, reason = should_run_maskclose(11013, 29)
    assert not ok and reason == "softedge_ok"
    ok, reason = should_run_maskclose(12307, 27)
    assert not ok and reason == "softedge_ok"
    ok, reason = should_run_maskclose(8623, 77)
    assert not ok and reason == "softedge_ok"
    ok, reason = should_run_maskclose(100, None)
    assert not ok and reason == "missing_n_views"
    ok, reason = should_run_maskclose(None, 10)
    assert not ok and reason == "missing_n_selected"


class _OrthoCam:
    """世界 (x,y) → 像素 (x,y)，z=1；便于假 mask 单测。"""

    def __init__(self, w: int = 20, h: int = 20) -> None:
        self.width = w
        self.height = h

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        return (1.0, 1.0, 0.0, 0.0)

    def world_to_cam_points(self, xyz: np.ndarray) -> np.ndarray:
        out = np.zeros_like(xyz, dtype=np.float64)
        out[:, 0] = xyz[:, 0]
        out[:, 1] = xyz[:, 1]
        out[:, 2] = 1.0
        return out


def test_jury_delete_drops_points_outside_majority_masks() -> None:
    """两点：A 两视 mask 内保留；B 两视可见但都不在 mask → 删。"""
    cam = _OrthoCam()
    m0 = np.zeros((20, 20), dtype=np.uint8)
    m0[0:5, 0:5] = 255
    m1 = np.zeros((20, 20), dtype=np.uint8)
    m1[0:5, 0:5] = 255
    xyz = np.array([[2.0, 2.0, 0.0], [12.0, 12.0, 0.0]], dtype=np.float64)
    # A hit/vis=2/2；B hit/vis=0/2 < 0.5 → 删 B
    keep = jury_delete_keep_mask(xyz, [cam, cam], [m0, m1], keep_frac=0.5, min_vis=2)
    assert bool(keep[0]) is True
    assert bool(keep[1]) is False


def test_density_fill_requires_neighbors_and_one_mask_hit() -> None:
    """洞内近邻够密且 mask 命中可补；远处或无 mask 不补。"""
    cam = _OrthoCam()
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:8, 0:8] = 255
    core = np.zeros((9, 3), dtype=np.float64)
    xs = np.linspace(1.0, 3.0, 3)
    i = 0
    for x in xs:
        for y in xs:
            core[i] = [x, y, 0.0]
            i += 1
    core_index = np.arange(9, dtype=np.int64)
    pool = np.array(
        [
            [2.0, 2.0, 0.0],  # 洞心，密度够，mask 内
            [15.0, 15.0, 0.0],  # 远且 mask 外
            [2.0, 2.0, 0.0],  # duplicate index handled by caller; use distinct
        ],
        dtype=np.float64,
    )
    pool[2] = [6.5, 6.5, 0.0]  # mask 内但离核远、邻居不够
    pool_index = np.array([100, 101, 102], dtype=np.int64)
    xyz, idx, n_added, n_rej = density_mask_fill(
        core,
        core_index,
        pool,
        pool_index,
        [cam],
        [mask],
        dense_radius=1.5,
        min_dense=4,
        link_radius=1.5,
    )
    kept = set(idx.tolist())
    assert 100 in kept
    assert 101 not in kept
    assert 102 not in kept
    assert n_added == 1
    assert n_rej >= 2
