#!/usr/bin/env python3
"""Referring-point ablation chart from docs/ablation_text_only.json.

In-box count is round(n_selected * frac_selected_in_obb), which recovers
eval_obb's n_selected_in_obb. Out-of-box count is the remainder.
Bars are those counts, not the in-box fraction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "data2_bowl": "Golden bowl",
    "data2_bracelet": "Bracelet",
    "data2_cookie": "Cookie bag",
    "data2_golden_retriever": "Golden retriever",
    "data2_hair_clip": "Hair clip",
    "data2_medicine_bottle": "Medicine bottle",
    "data2_rabbit": "Brown rabbit",
    "data2_shaver": "Electric shaver",
    "data2_toy_cake": "Toy cake",
    "data2_umbrella": "Umbrella",
    "data2_tape": "Tape (hold-out)",
}
WITH_COLOR = "#2ca02c"
TEXT_COLOR = "#d62728"


def counts(row: dict) -> tuple[int, int]:
    """Return (in-box, out-of-box) Gaussian counts."""
    n = int(row["n_selected"])
    frac = float(row["frac_selected_in_obb"])
    if n == 0:
        if frac != 0.0:
            raise SystemExit(f"n_selected=0 but frac={frac}")
        return 0, 0
    raw = n * frac
    n_in = int(round(raw))
    if abs(raw - n_in) > 1e-4:
        raise SystemExit(f"in-box count not an integer recovery: n={n} frac={frac} raw={raw}")
    if n_in < 0 or n_in > n:
        raise SystemExit(f"in-box count out of range: {n_in} / {n}")
    return n_in, n - n_in


def _draw_pair(ax, y: np.ndarray, with_n: np.ndarray, text_n: np.ndarray, xlabel: str) -> None:
    height = 0.36
    with_plot = np.where(with_n > 0, with_n, np.nan)
    text_plot = np.where(text_n > 0, text_n, np.nan)
    ax.barh(y - height / 2, with_plot, height=height, color=WITH_COLOR, label="With referring point", zorder=2)
    ax.barh(y + height / 2, text_plot, height=height, color=TEXT_COLOR, label="Text only", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_xlim(1, 4e5)
    ax.grid(axis="x", which="major", ls=":", color="#cccccc", zorder=0)
    for yi, a, b in zip(y, with_n, text_n):
        _label(ax, yi - height / 2, int(a), WITH_COLOR)
        _label(ax, yi + height / 2, int(b), TEXT_COLOR)


def _label(ax, yi: float, value: int, color: str) -> None:
    if value <= 0:
        ax.text(1.15, yi, "0", va="center", ha="left", fontsize=7.5, color=color)
        return
    ax.text(value * 1.12, yi, f"{value:,}", va="center", ha="left", fontsize=7.5, color="#222222")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "docs" / "ablation_text_only.json")
    ap.add_argument("--output", type=Path, default=ROOT / "demo" / "teaser_text_only_ablation.png")
    args = ap.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    names = [LABELS.get(r["object_dir"], r["object_dir"]) for r in rows]
    with_in, with_out, text_in, text_out = [], [], [], []
    for r in rows:
        wi, wo = counts(r["baseline"])
        ti, to = counts(r["text_only"])
        with_in.append(wi)
        with_out.append(wo)
        text_in.append(ti)
        text_out.append(to)
        print(
            f"{r['object_dir']}\t{wi}\t{ti}\t{ti - wi}\t{wo}\t{to}\t{to - wo}"
        )

    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4), sharey=True)
    fig.suptitle(
        "Referring-point ablation — voted Gaussian counts vs hand OBB",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    _draw_pair(
        axes[0],
        y,
        np.asarray(with_in),
        np.asarray(text_in),
        "Gaussians inside the hand OBB  (log)",
    )
    _draw_pair(
        axes[1],
        y,
        np.asarray(with_out),
        np.asarray(text_out),
        "Gaussians outside the hand OBB  (log)",
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 0.955))
    fig.text(
        0.5,
        0.015,
        "Same kept views and frustum vote (min-vote-frac 0.5). "
        "Text only drops the 2D point; the object name still goes to Grounding DINO. "
        "Count = round(n_selected × fraction in OBB). "
        "Medicine bottle text-only selects 0 Gaussians.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.90))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
