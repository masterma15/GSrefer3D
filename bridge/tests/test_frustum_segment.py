"""视锥命中、T 加权投票与深度切片的单元测试（不读真实 ply）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from frustum_segment import (  # noqa: E402
    PAIR_FINAL_MIN_SCORE,
    PAIR_WHITE_MIN_SCORE,
    active_axis_ids,
    apply_sector_vote_weights,
    assign_view_patch_bins,
    assign_view_axis,
    frustum_hit_mask,
    mean_azimuth_bin_rates,
    observe_footprint,
    observe_frame,
    outside_oppose_strength,
    observe_weight_map,
    pair_attention,
    pair_scores,
    principal_coverage_arc,
    segment_gaussians,
    select_view_frustum,
    view_weight_from_report,
    world_to_cam_uvz,
)
from obb_geom import obb_axis_coverage  # noqa: E402

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
    assert view_weight_from_report({"status": "kept", "T_at_z_lo": 0.9}, use_t_weight=True) == 0.9
    assert view_weight_from_report({"status": "kept", "T_at_z_lo": 0.2}, use_t_weight=True) == 0.2
    assert view_weight_from_report({"status": "kept", "T_at_z_lo": 0.2}, use_t_weight=False) == 1.0


def test_principal_coverage_arc_drops_largest_gap() -> None:
    # 三视挤在 0°附近 + 一视在 180° → 主弧应盖住有点的一侧而非硬 360
    az = np.deg2rad(np.array([0.0, 10.0, 20.0, 180.0], dtype=np.float64))
    start, length = principal_coverage_arc(az)
    assert length < np.deg2rad(200.0)
    assert length > np.deg2rad(150.0)


def test_assign_view_axis_rejects_diagonal() -> None:
    p = np.zeros(3, dtype=np.float64)
    assert assign_view_axis(p, {"position": [3.0, 0.0, 0.0]}, max_deg=35.0) == 0
    assert assign_view_axis(p, {"position": [3.0, 0.4, 0.0]}, max_deg=35.0) == 0
    assert assign_view_axis(p, {"position": [1.0, 1.0, 0.0]}, max_deg=35.0) is None
    assert active_axis_ids([0, 0, 0, 0, 1, 1, 1, None], min_views=4) == [0]


def test_patch_bins_use_azimuth_and_elevation_on_the_camera_span() -> None:
    """同一视线远近同格；仰角不同则不同格。格子盖住相机弧，不把对面空弧算进去。"""
    p = np.zeros(3, dtype=np.float64)
    cams = [
        {"position": [0.0, 0.0, 5.0]},
        {"position": [0.0, 0.0, 12.0]},
        {"position": [0.0, 5.0, 5.0]},
        {"position": [5.0, 0.0, 0.0]},
    ]
    ids, meta = assign_view_patch_bins(p, cams, n_bins=64)
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]
    assert ids[0] != ids[3]
    assert meta["n_az"] == 8 and meta["n_el"] == 8
    assert meta["arc_deg"] < 180.0


def test_pair_attention_shrinks_a_crowd_to_one_share() -> None:
    p = np.zeros(3, dtype=np.float64)
    cams = [{"position": [0.0, 0.0, 1.0]} for _ in range(10)]
    cams.append({"position": [0.0, 0.0, -1.0]})
    att = pair_attention(p, cams, tau_deg=20.0)
    crowd = float(att["a"][:10].sum())
    assert abs(crowd - 1.0) < 1e-4
    assert abs(float(att["a"][10]) - 1.0) < 2e-3


def test_pair_score_empty_is_zero_and_above_a_single_hit() -> None:
    p = np.zeros(3, dtype=np.float64)
    cams = [
        {"position": [0.0, 0.0, 1.0]},
        {"position": [0.0, 0.0, -1.0]},
        {"position": [1.0, 0.0, 0.0]},
    ]
    att = pair_attention(p, cams, tau_deg=20.0)
    hits = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )
    scores = pair_scores(hits, att)
    assert abs(float(scores[0])) < 1e-12
    assert float(scores[1]) < float(scores[0])
    assert float(scores[2]) > 0.0


def test_soft_edge_is_steep() -> None:
    near = float(outside_oppose_strength(np.array([1.0]))[0])
    mid = float(outside_oppose_strength(np.array([2.0]))[0])
    far = float(outside_oppose_strength(np.array([5.0]))[0])
    assert 0.10 < near < 0.15
    assert abs(mid - 0.5) < 1e-5
    assert far > 0.99


def test_default_pair_thresholds() -> None:
    assert PAIR_WHITE_MIN_SCORE == 0.03
    assert PAIR_FINAL_MIN_SCORE == 0.15


def test_half_frame_and_occluded_surface_do_not_oppose() -> None:
    cam = _cam()
    center = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    outside = np.array([[30.0, 0.0, 3.0]], dtype=np.float64)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[35:46, 45:56] = 255
    pos_c, neg_c = observe_frame(center, cam, mask, margin=8)
    pos_o, neg_o = observe_frame(outside, cam, mask, margin=8)
    assert pos_c[0] and not neg_c[0]
    assert not pos_o[0] and not neg_o[0]
    blank = np.zeros((80, 100), dtype=np.uint8)
    low = np.zeros((80, 100), dtype=np.float32)
    high = np.ones((80, 100), dtype=np.float32)
    # 画面内、不在 mask：足迹高染色低是挡住，弃权；足迹低才反对
    _, neg_block = observe_footprint(center, cam, blank, high, low, margin=8)
    _, neg_side = observe_footprint(center, cam, blank, low, low, margin=8)
    pos_hole, _ = observe_footprint(center, cam, mask, low, low, margin=8)
    assert not neg_block[0]
    assert neg_side[0]
    assert pos_hole[0]


def test_low_object_weight_abstains_instead_of_opposing() -> None:
    cam = _cam()
    xyz = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    mask = np.zeros((80, 100), dtype=np.uint8)
    hidden = np.full((80, 100), 0.1, dtype=np.float32)
    visible = np.full((80, 100), 0.8, dtype=np.float32)
    pos_h, neg_h = observe_weight_map(xyz, cam, mask, hidden)
    pos_v, neg_v = observe_weight_map(xyz, cam, mask, visible)
    assert not pos_h[0] and not neg_h[0]
    assert not pos_v[0] and neg_v[0]
    p = np.zeros(3, dtype=np.float64)
    cams = [{"position": [0.0, 0.0, 1.0]}, {"position": [0.0, 0.0, -1.0]}]
    att = pair_attention(p, cams, tau_deg=20.0)
    abstain = pair_scores(np.array([[1.0, 0.0]]), att, negative=np.array([[0.0, 0.0]]))
    oppose = pair_scores(np.array([[1.0, 0.0]]), att, negative=np.array([[0.0, 1.0]]))
    assert float(abstain[0]) > float(oppose[0])


def test_azimuth_bin_mean_multiplies_hits_by_transmittance() -> None:
    """格内分母仍是张数。两张都命中但 T=0.5 时，这一格是 0.5，不是 1。"""
    mass = np.array([1.0, 0.0], dtype=np.float64)
    mean = mean_azimuth_bin_rates({0: mass}, {0: 2})
    assert abs(float(mean[0]) - 0.5) < 1e-9
    assert abs(float(mean[1]) - 0.0) < 1e-9


def test_azimuth_bin_mean_gives_a_single_view_the_same_weight_as_a_full_bin() -> None:
    """8 张挤在一格、1 张单独一格时，两格各贡献 0 或 1，张数不再加成。"""
    dense = np.array([8, 4, 0], dtype=np.int32)
    sparse = np.array([0, 1, 0], dtype=np.int32)
    mean = mean_azimuth_bin_rates({0: dense, 1: sparse}, {0: 8, 1: 1})
    assert abs(float(mean[0]) - 0.5) < 1e-9
    assert abs(float(mean[1]) - 0.75) < 1e-9
    assert abs(float(mean[2]) - 0.0) < 1e-9


def test_sector_weights_downweight_dense_side_mean_one() -> None:
    """密扇区单视变轻；稀扇区变重；归一后均值≈1。"""
    p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    views: list[dict] = []
    # 0° 附近 6 张（密），90° 1 张（稀）；T 已编进 weight
    for i, ang_deg in enumerate([0, 5, 10, 15, 20, 25, 90]):
        rad = np.deg2rad(ang_deg)
        views.append(
            {
                "view_id": f"{i:03d}",
                "cam": {"position": [float(np.sin(rad)), 0.0, float(np.cos(rad))]},
                "mask": None,
                "weight": 1.0,
            }
        )
    meta = apply_sector_vote_weights(views, p, sector_deg=30.0, cap_mult=3.0)
    ws = np.array([v["weight"] for v in views], dtype=np.float64)
    assert abs(float(ws.mean()) - 1.0) < 1e-6
    # 稀侧（最后一张）应重于密侧平均
    assert float(ws[-1]) > float(ws[:-1].mean())
    assert meta["n_sectors"] >= 2


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


def test_one_view_keeps_only_depth_band() -> None:
    """同一帧 mask 里两个点，只有落在该帧深度带里的留下。"""
    cam = _cam()
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[30:51, 40:61] = 255
    xyz = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 8.0],
        ],
        dtype=np.float64,
    )
    cluster = np.array([[0.0, 0.0, 3.0]], dtype=np.float64)
    idx = select_view_frustum(
        xyz, cam, mask, cluster, z_slack=0.15, apply_depth_slice=True
    )
    assert idx.tolist() == [0]


def test_one_view_without_band_keeps_mask_hits() -> None:
    cam = _cam()
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[30:51, 40:61] = 255
    xyz = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 8.0]], dtype=np.float64)
    cluster = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    idx = select_view_frustum(
        xyz, cam, mask, cluster, z_slack=0.15, apply_depth_slice=True
    )
    assert idx.tolist() == [0, 1]


def _unit_obb() -> dict:
    return {
        "center": [0.0, 0.0, 0.0],
        "half_extent": [1.0, 1.0, 1.0],
        "rotation_columns": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def test_empty_selection_has_no_axis_coverage() -> None:
    assert obb_axis_coverage(np.zeros((0, 3), dtype=np.float64), _unit_obb()) is None


def test_corner_spread_is_tighter_than_full_box() -> None:
    obb = _unit_obb()
    corner = np.array([[0.8, 0.8, 0.8], [0.9, 0.8, 0.8]], dtype=np.float64)
    full = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    corner_cov = obb_axis_coverage(corner, obb)
    full_cov = obb_axis_coverage(full, obb)
    assert corner_cov is not None and full_cov is not None
    assert corner_cov["min"] < full_cov["min"]
    assert obb_axis_coverage(np.zeros((0, 3)), obb) is None
