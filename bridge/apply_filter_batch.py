#!/usr/bin/env python3
"""Apply per-object view filtering: manual reject lists (large) or ray filter (small)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Manual rejects from user (large objects, --skip-ray-filter + explicit view ids)
MANUAL_REJECT: dict[str, list[str]] = {
    "data2_rabbit": ["024", "045"],
    "data2_golden_retriever": ["025", "051"],
    "data2_umbrella": ["022", "025", "051"],
}

# Per-object ray-filter overrides (appended to filter_views_3dgs.py CLI).
RAY_FILTER_OVERRIDES: dict[str, list[str]] = {
    "data2_medicine_bottle": ["--filter-preset", "relaxed_plus"],
}

FUSED_RUNS: dict[str, str] = {
    "data2_shaver": "20260516_111848_4c3b9a32",
    "data2_rabbit": "20260516_111341_147bac82",
    "data2_golden_retriever": "20260516_111616_f8dbfcc3",
    "data2_umbrella": "20260516_111101_d7bab60f",
    "data2_hair_clip": "20260516_112347_65bf02a5",
    "data2_medicine_bottle": "20260516_112749_04149b86",
    "data2_bracelet": "20260516_113151_cb2e562f",
    "data2_cookie": "20260516_113618_e51c780a",
    "data2_bowl": "20260516_114029_8d83a715",
    "data2_toy_cake": "20260516_114443_7dd80c38",
}


def apply_manual(out_dir: Path, reject_ids: set[str]) -> dict[str, Any]:
    proj_path = out_dir / "projections.json"
    if not proj_path.is_file():
        raise FileNotFoundError(proj_path)
    projections = json.loads(proj_path.read_text(encoding="utf-8"))
    kept: list[dict] = []
    rejected: list[dict] = []
    per_view: list[dict] = []

    for rec in projections:
        vid = str(rec["view_id"]).zfill(3)
        entry: dict[str, Any] = {
            "view_id": vid,
            "status": "rejected" if vid in reject_ids else "kept",
            "reason": "manual_reject" if vid in reject_ids else "manual_keep",
            "skip_ray_filter": True,
        }
        per_view.append(entry)
        if vid in reject_ids:
            rejected.append({**rec, "filter": entry})
        else:
            kept.append(dict(rec))

    report = {
        "filter_mode": "manual",
        "manual_reject_ids": sorted(reject_ids),
        "projections_input": len(projections),
        "counts": {"kept": len(kept), "rejected": len(rejected), "skipped": 0},
        "views": per_view,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "projections_kept.json").write_text(json.dumps(kept, indent=2), encoding="utf-8")
    (out_dir / "projections_rejected.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    (out_dir / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_ray_filter(
    obj_dir: str,
    *,
    views_root: Path,
    ply: Path,
    python: str,
    extra_args: list[str] | None = None,
) -> None:
    run_id = FUSED_RUNS[obj_dir]
    out_dir = REPO / "training_data" / obj_dir
    fused = REPO / "3DGS" / "test2" / "runs" / run_id / "fused.json"
    cmd = [
        python,
        str(REPO / "bridge" / "filter_views_3dgs.py"),
        "--projections",
        str(out_dir / "projections.json"),
        "--fused",
        str(fused),
        "--ply",
        str(ply),
        "--views-root",
        str(views_root),
        "--out-dir",
        str(out_dir),
        "--no-overlays",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[ray] {obj_dir} ...", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root", default="3DGS/test2")
    ap.add_argument(
        "--ply",
        default="3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply",
    )
    ap.add_argument("--only", nargs="*", help="Limit to data2_* dir names")
    ap.add_argument("--manual-only", action="store_true")
    ap.add_argument("--ray-only", action="store_true")
    args = ap.parse_args()

    views_root = (REPO / args.views_root).resolve()
    ply = (REPO / args.ply).resolve()
    python = sys.executable
    objects = sorted(FUSED_RUNS.keys())
    if args.only:
        objects = [o for o in objects if o in args.only]

    summary: list[str] = []
    for obj in objects:
        if obj in MANUAL_REJECT and not args.ray_only:
            reject = {v.zfill(3) for v in MANUAL_REJECT[obj]}
            r = apply_manual(REPO / "training_data" / obj, reject)
            summary.append(f"{obj}: manual kept={r['counts']['kept']} rejected={r['counts']['rejected']}")
        elif obj not in MANUAL_REJECT and not args.manual_only:
            overrides = RAY_FILTER_OVERRIDES.get(obj, [])
            run_ray_filter(
                obj,
                views_root=views_root,
                ply=ply,
                python=python,
                extra_args=overrides,
            )
            kept = json.loads((REPO / "training_data" / obj / "projections_kept.json").read_text())
            rej = json.loads((REPO / "training_data" / obj / "projections_rejected.json").read_text())
            summary.append(f"{obj}: ray kept={len(kept)} rejected={len(rej)}")

    print("\n=== summary ===")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
