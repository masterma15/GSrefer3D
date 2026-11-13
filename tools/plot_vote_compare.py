"""data2 mask-precision change: fraction 0.5 vs view-angle vote."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "golden_bowl": "Golden bowl",
    "bracelet": "Bracelet",
    "cookie_bag": "Cookie bag",
    "golden_retriever": "Golden retriever",
    "hair_clip": "Hair clip",
    "medicine_bottle": "Medicine bottle",
    "brown_rabbit": "Brown rabbit",
    "electric_shaver": "Electric shaver",
    "toy_cake": "Toy cake",
    "umbrella": "Umbrella",
    "double_sided_tape": "Double-sided tape",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "docs" / "results_vote_compare_data2.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "demo" / "teaser_vote_compare_data2.png",
    )
    args = ap.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(
        payload["results"],
        key=lambda r: (
            r["view_angle"]["mean_view_mask_precision"]
            - r["fraction_0.5"]["mean_view_mask_precision"]
        ),
        reverse=True,
    )
    names = [LABELS.get(r["fraction_0.5"]["object_key"], r["name"]) for r in rows]
    old = np.array(
        [r["fraction_0.5"]["mean_view_mask_precision"] for r in rows], dtype=np.float64
    )
    new = np.array(
        [r["view_angle"]["mean_view_mask_precision"] for r in rows], dtype=np.float64
    )
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    h = 0.36
    ax.barh(y + h / 2, old, height=h, color="#b7b7b7", label="Fraction 0.5")
    ax.barh(y - h / 2, new, height=h, color="#4c78a8", label="View-angle")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0.55, 1.05)
    ax.set_xlabel("2D mask reprojection precision")
    ax.set_title(
        "data2 vote change    ·    11 objects, same masks and hand OBBs",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    for yi, a, b in zip(y, old, new):
        ax.text(max(a, b) + 0.008, yi, f"{b - a:+.3f}", va="center", fontsize=9, color="#1a376c")
    ax.legend(loc="lower right", frameon=False)
    ax.axvline(float(np.mean(old)), color="#888888", ls="--", lw=0.8)
    ax.axvline(float(np.mean(new)), color="#4c78a8", ls="--", lw=0.8)
    fig.text(
        0.5,
        0.01,
        (
            f"centroid ∈ OBB  11/11 both"
            f"    ·    mean in-box fraction  0.992 both"
            f"    ·    mean precision  {np.mean(old):.3f} → {np.mean(new):.3f}"
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"mean {np.mean(old):.4f} -> {np.mean(new):.4f}")


if __name__ == "__main__":
    main()
