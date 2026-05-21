"""Tests for yellow-overlay depth ratio helper (filter_views_3dgs)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bridge"))

from filter_views_3dgs import ratio_depth_visible  # noqa: E402


def test_ratio_yellow_uses_relative_only() -> None:
    assert ratio_depth_visible(0.36, 0.40, rel_slack=0.15)
    assert not ratio_depth_visible(0.30, 0.40, rel_slack=0.15)
