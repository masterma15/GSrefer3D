from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from e2e_stages import STAGES, guess_ply  # noqa: E402
from run_bridge_e2e import _should_run, parse_args  # noqa: E402


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
