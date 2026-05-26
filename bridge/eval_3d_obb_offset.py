#!/usr/bin/env python3
"""OBB outside offset + Base vs LoRA comparison + example plots.

For fused P_world vs docs/bbox_data2.json OBB:
  - inside  → outside offset = 0
  - outside → Euclidean distance to nearest box surface (local frame clamp)

Also plots one LoRA in-box and one LoRA out-of-box example (wireframe + point).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_BRIDGE = Path(__file__).resolve().parent
if str(_BRIDGE) not in sys.path:
    sys.path.insert(0, str(_BRIDGE))

from eval_3d_obb_hit import (  # noqa: E402
    DEFAULT_FUSED_BY_GROUP,
    DEFAULT_ROWS,
    _half_extent,
    eval_run,
    load_bbox,
    load_rows,
    obb_hit,
    obb_local,
    summarize,
)

_REPO = _BRIDGE.parent


def obb_corners_world(obb: dict[str, Any]) -> np.ndarray:
    center = np.asarray(obb["center"], dtype=np.float64)
    rot = np.asarray(obb["rotation_columns"], dtype=np.float64)
    half = _half_extent(obb)
    corners: list[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
                corners.append(center + rot @ local)
    return np.stack(corners, axis=0)


def _corner_index(sx: float, sy: float, sz: float) -> int:
    return (4 if sx > 0 else 0) + (2 if sy > 0 else 0) + (1 if sz > 0 else 0)


OBB_WIREFRAME_EDGES: list[tuple[int, int]] = [
    (_corner_index(-1, -1, -1), _corner_index(-1, -1, 1)),
    (_corner_index(-1, -1, -1), _corner_index(-1, 1, -1)),
    (_corner_index(-1, -1, -1), _corner_index(1, -1, -1)),
    (_corner_index(-1, -1, 1), _corner_index(-1, 1, 1)),
    (_corner_index(-1, -1, 1), _corner_index(1, -1, 1)),
    (_corner_index(-1, 1, -1), _corner_index(-1, 1, 1)),
    (_corner_index(-1, 1, -1), _corner_index(1, 1, -1)),
    (_corner_index(-1, 1, 1), _corner_index(1, 1, 1)),
    (_corner_index(1, -1, -1), _corner_index(1, -1, 1)),
    (_corner_index(1, -1, -1), _corner_index(1, 1, -1)),
    (_corner_index(1, -1, 1), _corner_index(1, 1, 1)),
    (_corner_index(1, 1, -1), _corner_index(1, 1, 1)),
]


def obb_nearest_on_box_world(
    p_world: np.ndarray, obb: dict[str, Any], *, margin: float = 0.0
) -> np.ndarray:
    local = obb_local(p_world, obb)
    half = _half_extent(obb) + margin
    clamped = np.clip(local, -half, half)
    rot = np.asarray(obb["rotation_columns"], dtype=np.float64)
    center = np.asarray(obb["center"], dtype=np.float64)
    return center + rot @ clamped


def obb_outside_metrics(
    p_world: np.ndarray, obb: dict[str, Any], *, margin: float = 0.0
) -> dict[str, float]:
    local = obb_local(p_world, obb)
    half = _half_extent(obb) + margin
    signed_margin = float(np.min(half - np.abs(local)))
    excess = np.maximum(np.abs(local) - half, 0.0)
    nearest = obb_nearest_on_box_world(p_world, obb, margin=margin)
    p = np.asarray(p_world, dtype=np.float64)
    outside_euclid = float(np.linalg.norm(p - nearest))
    return {
        "signed_margin_m": round(signed_margin, 6),
        "outside_euclid_m": round(outside_euclid, 6),
        "outside_l2_excess_m": round(float(np.linalg.norm(excess)), 6),
        "outside_linf_excess_m": round(float(np.max(excess)), 6),
        "max_axis_excess_m": round(float(np.max(np.abs(local) - half)), 6),
    }


def enrich_run(row: dict[str, Any], objects: dict[str, Any], margin: float) -> dict[str, Any]:
    if row.get("error") or "P_world" not in row:
        return row
    spec = objects.get(row["object_key"])
    obb = spec.get("obb") if spec else None
    if obb is None:
        return row
    p = np.asarray(row["P_world"], dtype=np.float64)
    metrics = obb_outside_metrics(p, obb, margin=margin)
    row.update(metrics)
    row["obb_hit"] = obb_hit(p, obb, margin=margin)
    nearest = obb_nearest_on_box_world(p, obb, margin=margin)
    row["obb_nearest_on_box"] = [round(float(x), 6) for x in nearest]
    return row


def compare_base_lora(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_obj: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        if r.get("error"):
            continue
        by_obj.setdefault(r["object_key"], {})[r["group"]] = r

    pairs: list[dict[str, Any]] = []
    lora_better = base_better = tie = 0
    lora_better_miss_only = base_better_miss_only = tie_miss_only = 0
    base_outside: list[float] = []
    lora_outside: list[float] = []
    miss_base_outside: list[float] = []
    miss_lora_outside: list[float] = []

    for obj_key in sorted(by_obj):
        grp = by_obj[obj_key]
        b = grp.get("Base")
        l = grp.get("LoRA")
        if b is None or l is None:
            continue
        bo = float(b.get("outside_euclid_m", 0.0))
        lo = float(l.get("outside_euclid_m", 0.0))
        bh = bool(b.get("obb_hit"))
        lh = bool(l.get("obb_hit"))
        base_outside.append(bo)
        lora_outside.append(lo)
        if not bh:
            miss_base_outside.append(bo)
        if not lh:
            miss_lora_outside.append(lo)

        if lh and not bh:
            winner = "LoRA"
            lora_better += 1
        elif bh and not lh:
            winner = "Base"
            base_better += 1
        elif abs(bo - lo) < 1e-9:
            winner = "tie"
            tie += 1
        elif lo < bo:
            winner = "LoRA"
            lora_better += 1
        else:
            winner = "Base"
            base_better += 1

        both_miss = (not bh) and (not lh)
        if both_miss:
            if abs(bo - lo) < 1e-9:
                tie_miss_only += 1
            elif lo < bo:
                lora_better_miss_only += 1
            else:
                base_better_miss_only += 1

        pairs.append(
            {
                "object_key": obj_key,
                "Base_hit": bh,
                "LoRA_hit": lh,
                "Base_outside_euclid_m": round(bo, 6),
                "LoRA_outside_euclid_m": round(lo, 6),
                "delta_base_minus_lora_m": round(bo - lo, 6),
                "winner_closer_or_hit": winner,
                "both_miss": both_miss,
            }
        )

    def _mean(xs: list[float]) -> float | None:
        return round(float(np.mean(xs)), 6) if xs else None

    return {
        "n_objects": len(pairs),
        "winner_count": {"LoRA": lora_better, "Base": base_better, "tie": tie},
        "both_miss_winner_count": {
            "LoRA": lora_better_miss_only,
            "Base": base_better_miss_only,
            "tie": tie_miss_only,
        },
        "mean_outside_euclid_m": {"Base": _mean(base_outside), "LoRA": _mean(lora_outside)},
        "mean_outside_euclid_m_miss_only": {
            "Base": _mean(miss_base_outside),
            "LoRA": _mean(miss_lora_outside),
        },
        "per_object": pairs,
        "interpretation": (
            "outside_euclid_m=0 when inside OBB; lower is better. "
            "winner_closer_or_hit: LoRA if only LoRA hits or both miss but LoRA is closer."
        ),
    }


def _set_axes_equal_3d(ax: Any, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius < 1e-6:
        radius = 0.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_obb_panel(
    ax: Any,
    *,
    obb: dict[str, Any],
    p_world: np.ndarray,
    title: str,
    hit: bool,
    outside_m: float,
    signed_margin_m: float,
    base_p: np.ndarray | None = None,
) -> None:
    corners = obb_corners_world(obb)
    for i, j in OBB_WIREFRAME_EDGES:
        seg = corners[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#3498db", linewidth=1.4, alpha=0.9)

    nearest = obb_nearest_on_box_world(p_world, obb)
    if not hit and outside_m > 0:
        seg = np.stack([nearest, p_world], axis=0)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.scatter(*nearest, c="#f39c12", s=36, depthshade=False, label="nearest on box")

    color = "#27ae60" if hit else "#e74c3c"
    ax.scatter(*p_world, c=color, s=80, depthshade=False, label="LoRA P_world", zorder=5)
    if base_p is not None and np.linalg.norm(base_p - p_world) > 1e-6:
        ax.scatter(*base_p, c="#7f8c8d", s=55, depthshade=False, label="Base P_world", zorder=4)

    status = "HIT (inside)" if hit else f"MISS (+{outside_m:.3f} m)"
    ax.set_title(f"{title}\n{status}; signed margin {signed_margin_m:+.3f} m", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    all_pts = corners
    if base_p is not None:
        all_pts = np.vstack([all_pts, base_p])
    all_pts = np.vstack([all_pts, p_world, nearest])
    _set_axes_equal_3d(ax, all_pts)


def plot_lora_in_out_examples(
    *,
    objects: dict[str, Any],
    results_by_key: dict[tuple[str, str], dict[str, Any]],
    out_path: Path,
    in_object: str,
    out_object: str,
) -> None:
    import matplotlib.pyplot as plt

    in_row = results_by_key[(in_object, "LoRA")]
    out_row = results_by_key[(out_object, "LoRA")]
    out_base = results_by_key.get((out_object, "Base"))

    in_obb = objects[in_object]["obb"]
    out_obb = objects[out_object]["obb"]
    in_p = np.asarray(in_row["P_world"], dtype=np.float64)
    out_p = np.asarray(out_row["P_world"], dtype=np.float64)
    base_p = np.asarray(out_base["P_world"], dtype=np.float64) if out_base else None

    fig = plt.figure(figsize=(11, 5.2), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_obb_panel(
        ax1,
        obb=in_obb,
        p_world=in_p,
        title=f"LoRA (ray): in-box {in_object}",
        hit=bool(in_row["obb_hit"]),
        outside_m=float(in_row["outside_euclid_m"]),
        signed_margin_m=float(in_row["signed_margin_m"]),
    )
    _draw_obb_panel(
        ax2,
        obb=out_obb,
        p_world=out_p,
        title=f"LoRA (ray): out-of-box {out_object}",
        hit=bool(out_row["obb_hit"]),
        outside_m=float(out_row["outside_euclid_m"]),
        signed_margin_m=float(out_row["signed_margin_m"]),
        base_p=base_p,
    )

    fig.suptitle("LoRA (ray depth) fused P_world vs OBB (data2)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_miss_offset_bars(comparison: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in comparison["per_object"] if not r["Base_hit"] or not r["LoRA_hit"]]
    if not rows:
        return
    labels = [r["object_key"] for r in rows]
    base_v = [r["Base_outside_euclid_m"] for r in rows]
    lora_v = [r["LoRA_outside_euclid_m"] for r in rows]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4.2), dpi=150)
    ax.bar(x - w / 2, base_v, width=w, label="Base", color="#7f8c8d", edgecolor="#2c3e50")
    ax.bar(x + w / 2, lora_v, width=w, label="LoRA", color="#8e44ad", edgecolor="#2c3e50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Outside offset to OBB (m)")
    ax.set_title("Miss or hit mismatch: Euclidean distance outside OBB (0 = inside)")
    ax.legend()
    ymax = max(base_v + lora_v + [0.01]) * 1.2
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="OBB outside offset + Base vs LoRA + plots")
    ap.add_argument("--bbox", type=Path, default=_REPO / "docs/bbox_data2.json")
    ap.add_argument("--runs-root", type=Path, default=_REPO / "3DGS/test2/runs")
    ap.add_argument("--table-json", type=Path, default=None)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument(
        "--lora-fused",
        default=DEFAULT_FUSED_BY_GROUP["LoRA"],
        help="Fused JSON for LoRA runs (default fused_ray.json)",
    )
    ap.add_argument("--base-fused", default=DEFAULT_FUSED_BY_GROUP["Base"])
    ap.add_argument(
        "--refuse-lora-ray",
        action="store_true",
        help="Re-fuse LoRA with depth_mode=ray before eval",
    )
    ap.add_argument(
        "--ply",
        type=Path,
        default=_REPO / "3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply",
    )
    ap.add_argument("--out", type=Path, default=_REPO / "docs/results_3d_obb_offset.json")
    ap.add_argument(
        "--plot",
        type=Path,
        default=_REPO / "docs/obb_lora_in_out.png",
        help="LoRA in-box vs out-of-box 3D figure",
    )
    ap.add_argument(
        "--plot-miss-bar",
        type=Path,
        default=_REPO / "docs/obb_miss_offset_compare.png",
        help="Bar chart of outside offset on non-perfect objects",
    )
    ap.add_argument(
        "--example-in",
        default="brown_rabbit",
        help="Object key for LoRA in-box panel",
    )
    ap.add_argument(
        "--example-out",
        default="hair_clip",
        help="Object key for LoRA out-of-box panel",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    fused_by_group = {"Base": args.base_fused, "LoRA": args.lora_fused}

    if args.refuse_lora_ray:
        from refuse_fused import refuse_rows  # noqa: WPS433

        rows_pre = load_rows(args.table_json)
        refuse_rows(
            rows_pre,
            runs_root=args.runs_root,
            group="LoRA",
            ply=args.ply,
            output_name=args.lora_fused,
        )

    objects = load_bbox(args.bbox)
    rows = load_rows(args.table_json)

    results = [
        enrich_run(
            eval_run(
                object_key=key,
                display_name=name,
                group=group,
                run_id=run_id,
                runs_root=args.runs_root,
                objects=objects,
                margin=args.margin,
                fused_by_group=fused_by_group,
            ),
            objects,
            args.margin,
        )
        for key, name, group, run_id in rows
    ]
    hit_summary = summarize(results)
    comparison = compare_base_lora(results)
    payload = {
        "summary": {
            "obb_hit": hit_summary,
            "offset_comparison": comparison,
        },
        "runs": results,
        "bbox": str(args.bbox.as_posix()),
        "margin_m": args.margin,
        "fused_by_group": fused_by_group,
        "plot_examples": {"in_object": args.example_in, "out_object": args.example_out},
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")

    print(text)
    print("\n# Outside offset (m), 0 = inside")
    print(f"{'object_key':<22} {'B_hit':>5} {'L_hit':>5} {'B_out':>8} {'L_out':>8} {'winner':>8}")
    for item in comparison["per_object"]:
        print(
            f"{item['object_key']:<22} "
            f"{int(item['Base_hit']):>5} "
            f"{int(item['LoRA_hit']):>5} "
            f"{item['Base_outside_euclid_m']:>8.4f} "
            f"{item['LoRA_outside_euclid_m']:>8.4f} "
            f"{item['winner_closer_or_hit']:>8}"
        )
    oc = comparison
    print(
        f"\nCloser / sole hit: LoRA {oc['winner_count']['LoRA']} | "
        f"Base {oc['winner_count']['Base']} | tie {oc['winner_count']['tie']}"
    )
    print(
        f"Both miss only: LoRA closer {oc['both_miss_winner_count']['LoRA']} | "
        f"Base closer {oc['both_miss_winner_count']['Base']} | "
        f"tie {oc['both_miss_winner_count']['tie']}"
    )
    print(
        f"Mean outside offset (all): Base {oc['mean_outside_euclid_m']['Base']} m | "
        f"LoRA {oc['mean_outside_euclid_m']['LoRA']} m"
    )
    print(
        f"Mean outside offset (miss only): Base {oc['mean_outside_euclid_m_miss_only']['Base']} m | "
        f"LoRA {oc['mean_outside_euclid_m_miss_only']['LoRA']} m"
    )

    if not args.no_plot:
        by_key = {(r["object_key"], r["group"]): r for r in results if not r.get("error")}
        plot_lora_in_out_examples(
            objects=objects,
            results_by_key=by_key,
            out_path=args.plot,
            in_object=args.example_in,
            out_object=args.example_out,
        )
        plot_miss_offset_bars(comparison, args.plot_miss_bar)
        print(f"\nWrote plots: {args.plot} , {args.plot_miss_bar}")


if __name__ == "__main__":
    main()
