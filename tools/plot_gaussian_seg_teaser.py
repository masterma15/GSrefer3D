"""Bar chart for frustum-vote 3D instance metrics (docs/results_gaussian_seg.json)."""

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
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "docs" / "results_gaussian_seg.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "demo" / "teaser_gaussian_seg.png",
    )
    args = ap.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["results"]
    names = [LABELS.get(r["object_key"], r["object_key"]) for r in rows]
    frac = np.array([r["frac_selected_in_obb"] for r in rows], dtype=np.float64)
    prec = np.array(
        [r["mask2d"]["mean_view_mask_precision"] for r in rows], dtype=np.float64
    )
    summary = payload["summary"]

    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharey=True)
    fig.suptitle(
        "Frustum-vote 3D instance vs hand OBB   (10 objects;  P_world is seed only)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    axes[0].barh(y, frac, color="#2ca02c", height=0.62)
    axes[0].set_xlabel("Fraction of selected Gaussians inside OBB")
    axes[0].set_xlim(0.80, 1.005)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].axvline(summary["mean_frac_selected_in_obb"], color="#1a376c", ls="--", lw=1.0)
    for yi, v in zip(y, frac):
        axes[0].text(min(v + 0.004, 0.992), yi, f"{v:.3f}", va="center", fontsize=9)
    axes[0].invert_yaxis()

    axes[1].barh(y, prec, color="#4c78a8", height=0.62)
    axes[1].set_xlabel("2D mask reprojection precision")
    axes[1].set_xlim(0.80, 1.005)
    axes[1].axvline(summary["mean_view_mask_precision"], color="#1a376c", ls="--", lw=1.0)
    for yi, v in zip(y, prec):
        axes[1].text(min(v + 0.004, 0.992), yi, f"{v:.3f}", va="center", fontsize=9)

    fig.text(
        0.5,
        0.02,
        (
            f"centroid ∈ OBB  {int(summary['centroid_obb_hit_rate'])}%  ({summary['n_ok']}/{summary['n_ok']})"
            f"   ·   mean frac in OBB  {summary['mean_frac_selected_in_obb']:.3f}"
            f"   ·   mean view-mask precision  {summary['mean_view_mask_precision']:.3f}"
        ),
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#f0c14b"),
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
