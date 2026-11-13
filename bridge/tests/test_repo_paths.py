"""路径解析：相对仓库根、Windows 盘符、/mnt 三种写法。"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from repo_paths import resolve_data_path, to_repo_relative  # noqa: E402


def test_repo_relative_roundtrip() -> None:
    sample = REPO / "bridge" / "repo_paths.py"
    rel = to_repo_relative(sample)
    assert rel == "bridge/repo_paths.py"
    assert resolve_data_path(rel).resolve() == sample.resolve()


def test_windows_drive_maps_when_missing() -> None:
    missing = "Z:/no/such/file.png"
    got = resolve_data_path(missing)
    assert got.as_posix().endswith("no/such/file.png")
    assert str(got).startswith("/mnt/z") or str(got).startswith("\\mnt\\z") or "mnt" in got.as_posix()
