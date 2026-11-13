from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from e2e_stages import STAGES, guess_ply  # noqa: E402
from gen_training_data import _load_projections  # noqa: E402
from run_bridge_e2e import (  # noqa: E402
    RAY_FILTER_MIN_KEPT,
    _should_run,
    mask_uses_kept_views,
    parse_args,
    resolve_mask_use_kept,
)


def test_guess_ply_skips_seg_iteration(tmp_path: Path) -> None:
    root = tmp_path / "model"
    (root / "point_cloud" / "iteration_30000").mkdir(parents=True)
    (root / "point_cloud" / "iteration_35000").mkdir()
    (root / "point_cloud" / "iteration_seg_data2_shaver").mkdir()
    for name in ("iteration_30000", "iteration_35000", "iteration_seg_data2_shaver"):
        (root / "point_cloud" / name / "point_cloud.ply").write_bytes(b"ply\n")
    got = guess_ply(root)
    assert got is not None
    assert got.parent.name == "iteration_35000"


def test_skip_render_maps_to_query() -> None:
    args = parse_args(
        [
            "--model-path",
            "3DGS/gaussian-splatting/output/data2",
            "--prompt",
            "Please point to the electric shaver on the desk.",
            "--object",
            "electric shaver",
            "--name",
            "data2_shaver",
            "--skip-render",
        ]
    )
    from_stage = args.from_stage
    if args.skip_render and from_stage == "render":
        from_stage = "query"
    assert from_stage == "query"
    assert _should_run("query", from_stage)
    assert not _should_run("render", from_stage)
    assert _should_run("vote", from_stage)


def test_from_vote_skips_earlier() -> None:
    assert not _should_run("mask", "vote")
    assert _should_run("vote", "vote")
    assert STAGES[0] == "render"


def test_mask_list_ignores_ray_filter_kept(tmp_path: Path) -> None:
    (tmp_path / "projections_kept.json").write_text(
        json.dumps([{"view_id": "001"}]),
        encoding="utf-8",
    )
    (tmp_path / "projections.json").write_text(
        json.dumps([{"view_id": "001"}, {"view_id": "002"}]),
        encoding="utf-8",
    )
    records = _load_projections(tmp_path)
    assert [r["view_id"] for r in records] == ["001", "002"]
    kept_only = _load_projections(tmp_path, use_kept=True)
    assert [r["view_id"] for r in kept_only] == ["001"]


def test_ray_filter_fallback_uses_all_views_when_kept_is_low(tmp_path: Path) -> None:
    report = {"counts": {"input": 56, "kept": RAY_FILTER_MIN_KEPT - 1}}
    (tmp_path / "filter_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert mask_uses_kept_views(36, fallback=True) is False
    assert mask_uses_kept_views(57, fallback=True) is True
    assert mask_uses_kept_views(36, fallback=False) is True
    assert resolve_mask_use_kept(tmp_path, ray_filter=True, fallback=True) is False


def test_parse_args_accepts_ray_filter_fallback() -> None:
    args = parse_args(
        [
            "--model-path",
            "m",
            "--prompt",
            "Please point to x.",
            "--object",
            "x",
            "--name",
            "n",
            "--ray-filter",
            "--ray-filter-fallback",
        ]
    )
    assert args.ray_filter_fallback is True


def test_parse_args_accepts_ray_filter() -> None:
    args = parse_args(
        [
            "--model-path",
            "m",
            "--prompt",
            "Please point to x.",
            "--object",
            "x",
            "--name",
            "n",
            "--ray-filter",
        ]
    )
    assert args.ray_filter is True


def test_parse_args_rejects_removed_ray_filter_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-path",
                "m",
                "--prompt",
                "Please point to x.",
                "--object",
                "x",
                "--name",
                "n",
                "--ray-min-transmittance",
                "0.2",
            ]
        )


def test_parse_args_requires_object_and_name() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-path",
                "m",
                "--prompt",
                "Please point to x.",
            ]
        )
