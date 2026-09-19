"""注入脚本：OBB 分色、拒绝写回 30000、SIBR iteration 名（不读真实大 ply）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from inject_gaussian_seg import (  # noqa: E402
    COLOR_IN_OBB,
    COLOR_OUT_OBB,
    inject_gaussian_seg,
    sibr_iteration_arg,
)
from obb_geom import obb_hits, rgb_to_f_dc  # noqa: E402


def _aabb_obb(*, center, half) -> dict:
    return {
        "center": list(center),
        "half_extent": list(half),
        "width": [2 * float(h) for h in half],
        "rotation_columns": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def _tiny_ply(path: Path, xyz: np.ndarray) -> None:
    n = xyz.shape[0]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("f_rest_0", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["f_dc_0"] = 0.1
    data["opacity"] = 0.0
    data["scale_0"] = data["scale_1"] = data["scale_2"] = -4.0
    data["rot_0"] = 1.0
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def test_sibr_iteration_arg() -> None:
    assert sibr_iteration_arg(Path("point_cloud/iteration_38100")) == "38100"
    assert sibr_iteration_arg(Path("iteration_seg_electric_shaver")) == "seg_electric_shaver"
    assert sibr_iteration_arg(Path("custom")) == "custom"


def test_obb_hits_axis_aligned() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    obb = _aabb_obb(center=(0.0, 0.0, 0.0), half=(0.5, 0.5, 0.5))
    hits = obb_hits(xyz, obb)
    assert bool(hits[0]) and not bool(hits[1])


def test_recolor_in_vs_out_and_refuse_30000(tmp_path: Path) -> None:
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    src = tmp_path / "src.ply"
    _tiny_ply(src, xyz)
    obb = _aabb_obb(center=(0.0, 0.0, 0.0), half=(0.5, 0.5, 0.5))
    seg = {"indices": [0, 1], "P_world": [0.0, 0.0, 0.0], "min_vote_frac": 0.5}
    out_dir = tmp_path / "iteration_seg_toy"
    man = inject_gaussian_seg(
        ply=src,
        seg=seg,
        out_iteration_dir=out_dir,
        obb=obb,
        object_key="toy",
        add_obb=True,
        add_seed=True,
        wireframe_step_m=0.5,
    )
    assert man["n_recolored_in_obb"] == 1
    assert man["n_recolored_out_obb"] == 1
    assert man["n_wireframe"] > 0
    assert (out_dir / "point_cloud.ply").is_file()
    assert (out_dir / "inject_manifest.json").is_file()

    written = PlyData.read(str(out_dir / "point_cloud.ply")).elements[0].data
    r_in, g_in, b_in = rgb_to_f_dc(COLOR_IN_OBB)
    r_out, g_out, b_out = rgb_to_f_dc(COLOR_OUT_OBB)
    assert abs(float(written["f_dc_0"][0]) - r_in) < 1e-5
    assert abs(float(written["f_dc_1"][0]) - g_in) < 1e-5
    assert abs(float(written["f_dc_2"][0]) - b_in) < 1e-5
    assert abs(float(written["f_dc_0"][1]) - r_out) < 1e-5
    assert abs(float(written["f_dc_1"][1]) - g_out) < 1e-5
    assert abs(float(written["f_dc_2"][1]) - b_out) < 1e-5
    # 未选中的第 2 个高斯保持原色
    assert abs(float(written["f_dc_0"][2]) - 0.1) < 1e-5
    assert json.loads((out_dir / "inject_manifest.json").read_text(encoding="utf-8"))["sibr_iteration"] == "seg_toy"

    bad = tmp_path / "iteration_30000"
    try:
        inject_gaussian_seg(
            ply=src, seg=seg, out_iteration_dir=bad, obb=obb, object_key="toy", add_obb=False, add_seed=False,
        )
        raise AssertionError("should refuse iteration_30000")
    except ValueError as e:
        assert "30000" in str(e)
