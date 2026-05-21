#!/usr/bin/env python3
"""Batch DAV2 vs 3DGS depth compare from predictions.json (2 views × 10 objects)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

OBJECTS = [
    ("剃须刀", "20260519_170540_4c3b9a32"),
    ("棕兔", "20260519_171359_147bac82"),
    ("金毛", "20260519_172627_f8dbfcc3"),
    ("金碗", "20260519_173958_8d83a715"),
    ("雨伞", "20260519_174835_d7bab60f"),
    ("玩具蛋糕", "20260519_175733_7dd80c38"),
    ("饼干袋", "20260519_180800_e51c780a"),
    ("药瓶", "20260519_182106_04149b86"),
    ("手链", "20260520_001018_cb2e562f"),
    ("发夹", "20260519_183039_65bf02a5"),
]


def valid_candidates(views: list) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    for v in views:
        if not v.get("parse_ok") or v.get("error"):
            continue
        pts = v.get("points") or []
        if not pts:
            continue
        nx, ny = float(pts[0]["nx"]), float(pts[0]["ny"])
        if nx <= 0.02 or nx >= 0.98 or ny <= 0.05 or ny >= 0.98:
            continue
        out.append((int(v["view_id"]), nx, ny))
    return out


def pick_two_views(candidates: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    if len(candidates) <= 2:
        return candidates
    by_ny = sorted(candidates, key=lambda x: x[2])
    low = by_ny[0]
    high = by_ny[-1]
    if low[0] == high[0]:
        return [candidates[0], candidates[-1]]
    return [low, high]


def parse_compare_output(text: str) -> dict:
    out: dict[str, float | None] = {
        "nn_3dgs": None,
        "nn_dav2_raw": None,
        "nn_dav2_affine": None,
        "nn_dav2_inv": None,
        "depth_rms": None,
    }
    for line in text.splitlines():
        m = re.search(r"3DGS NN\s+([\d.]+)", line)
        if m:
            out["nn_3dgs"] = float(m.group(1))
        m = re.search(r"DAV2 no align NN\s+([\d.]+)", line)
        if m:
            out["nn_dav2_raw"] = float(m.group(1))
        m = re.search(r"DAV2 affine z NN\s+([\d.]+)", line)
        if m:
            out["nn_dav2_affine"] = float(m.group(1))
        m = re.search(r"DAV2 inv-fit NN\s+([\d.]+)", line)
        if m:
            out["nn_dav2_inv"] = float(m.group(1))
        m = re.search(r"depth map RMS on valid mask after affine z\]\s*=\s*([\d.]+)", line)
        if m:
            out["depth_rms"] = float(m.group(1))
    return out


def run_one(python: str, views_root: Path, ply: Path, view_id: int, nx: float, ny: float) -> dict:
    cmd = [
        python,
        str(_REPO / "bridge" / "compare_depth_sources.py"),
        "--views-root",
        str(views_root),
        "--view",
        f"{view_id:03d}",
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--ply",
        str(ply),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO))
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return {"ok": False, "error": merged.strip()[-500:]}
    metrics = parse_compare_output(merged)
    if metrics["nn_3dgs"] is None:
        return {"ok": False, "error": "failed to parse output", "raw_tail": merged[-800:]}
    return {"ok": True, **metrics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=_REPO / "3DGS" / "test2" / "runs")
    ap.add_argument("--views-root", type=Path, default=_REPO / "3DGS" / "test2")
    ap.add_argument(
        "--ply",
        type=Path,
        default=_REPO
        / "3DGS"
        / "gaussian-splatting"
        / "output"
        / "data2"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply",
    )
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--output", type=Path, default=_REPO / "docs" / "depth_compare_batch.json")
    args = ap.parse_args()

    jobs: list[dict] = []
    for name, run_id in OBJECTS:
        pred_path = args.runs_root / run_id / "predictions.json"
        if not pred_path.is_file():
            sys.exit(f"missing predictions: {pred_path}")
        data = json.loads(pred_path.read_text(encoding="utf-8"))
        picks = pick_two_views(valid_candidates(data.get("views", [])))
        if len(picks) < 2:
            sys.exit(f"{name}: need 2 valid views, got {len(picks)} from {pred_path}")
        for view_id, nx, ny in picks:
            jobs.append(
                {
                    "object": name,
                    "run_id": run_id,
                    "view_id": view_id,
                    "nx": nx,
                    "ny": ny,
                }
            )

    print(f"Running {len(jobs)} compare_depth_sources jobs ...")
    results: list[dict] = []
    for i, job in enumerate(jobs, 1):
        label = f"[{i}/{len(jobs)}] {job['object']} view_{job['view_id']:03d}"
        print(label, flush=True)
        metrics = run_one(
            args.python, args.views_root, args.ply, job["view_id"], job["nx"], job["ny"]
        )
        row = {**job, **metrics}
        results.append(row)
        if row.get("ok"):
            print(
                f"  3DGS={row['nn_3dgs']:.3f}  DAV2raw={row['nn_dav2_raw']:.3f}  "
                f"affine={row['nn_dav2_affine']:.3f}",
                flush=True,
            )
        else:
            print(f"  FAILED: {row.get('error', '')[:200]}", flush=True)

    payload = {
        "views_root": str(args.views_root),
        "ply": str(args.ply),
        "n_jobs": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output}")

    ok_rows = [r for r in results if r.get("ok")]
    if ok_rows:
        print("\nMarkdown table rows:")
        print("| 物体 | view | (nx, ny) | 3DGS NN | DAV2 raw | DAV2 affine | DAV2 inv | RMS |")
        print("|------|------|----------|---------|----------|-------------|----------|-----|")
        for r in ok_rows:
            print(
                f"| {r['object']} | {r['view_id']:03d} | ({r['nx']:.3f},{r['ny']:.3f}) | "
                f"{r['nn_3dgs']:.3f} | {r['nn_dav2_raw']:.3f} | {r['nn_dav2_affine']:.3f} | "
                f"{r['nn_dav2_inv']:.3f} | {r['depth_rms']:.2f} |"
            )


if __name__ == "__main__":
    main()
