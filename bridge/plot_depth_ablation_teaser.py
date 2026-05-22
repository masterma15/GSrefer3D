#!/usr/bin/env python3
"""Plot demo/teaser_depth_ablation.png from docs/depth_compare_batch.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    data = json.loads((_REPO / "docs" / "depth_compare_batch.json").read_text(encoding="utf-8"))
    rows = [r for r in data["results"] if r.get("ok")]

    def med(key: str) -> float:
        return float(np.median([r[key] for r in rows]))

    labels = ["3DGS render depth", "DAV2 affine", "DAV2 raw", "DAV2 inv"]
    keys = ["nn_3dgs", "nn_dav2_affine", "nn_dav2_raw", "nn_dav2_inv"]
    values = [med(k) for k in keys]
    wins = sum(1 for r in rows if r["nn_3dgs"] < r["nn_dav2_affine"])

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    colors = ["#27ae60", "#c0392b", "#7f8c8d", "#bdc3c7"]
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="#2c3e50", linewidth=0.9, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Median NN distance to point cloud (m)", fontsize=11)
    ax.set_title("Depth source ablation (20 groups; fixed pixel + camera)", fontsize=12, fontweight="bold")
    ymax = max(values) * 1.25
    ax.set_ylim(0, ymax)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + ymax * 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.annotate(
        f"3DGS better than DAV2 affine: {wins}/20 groups",
        xy=(0.5, 0.94),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#f1c40f"),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out = _REPO / "demo" / "teaser_depth_ablation.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[ok] {out}  medians={dict(zip(keys, values))}  wins={wins}/20")


if __name__ == "__main__":
    main()
