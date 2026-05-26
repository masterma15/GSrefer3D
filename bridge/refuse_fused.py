#!/usr/bin/env python3
"""Re-fuse predictions.json with depth_mode=ray (ray pull-in along Gaussians).

Writes fused_ray.json next to each run (does not overwrite legacy fused.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BRIDGE = Path(__file__).resolve().parent
_REPO = _BRIDGE.parent
if str(_BRIDGE) not in sys.path:
    sys.path.insert(0, str(_BRIDGE))

from eval_3d_obb_hit import DEFAULT_ROWS, load_rows  # noqa: E402
import fuse_multiview as fm  # noqa: E402


DEFAULT_PLY = _REPO / "3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply"


def _fuse_params_from_legacy(run_dir: Path) -> dict[str, Any]:
    legacy = run_dir / "fused.json"
    if not legacy.is_file():
        return {}
    try:
        meta = json.loads(legacy.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, Any] = {}
    if "inlier_radius" in meta:
        out["inlier_radius"] = float(meta["inlier_radius"])
    return out


def refuse_run_dir(
    run_dir: Path,
    *,
    ply: Path,
    output_name: str = "fused_ray.json",
    inlier_radius: float = 1.0,
    min_inv: float = 1e-3,
    refine_k: float = 1.35,
    no_refine: bool = False,
    depth_mode: str = "ray",
    ray_perp_radius: float = 0.06,
    ray_pixel_radius: float = 6.0,
    ray_z_band: float = 2.0,
    ray_min_alpha: float = 0.45,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    pred_path = run_dir / "predictions.json"
    out_path = run_dir / output_name
    if not pred_path.is_file():
        return {"run_dir": str(run_dir), "error": f"missing {pred_path}"}
    if out_path.is_file() and not force:
        fused = json.loads(out_path.read_text(encoding="utf-8"))
        return {
            "run_dir": str(run_dir),
            "output": str(out_path),
            "skipped": True,
            "P_world": fused.get("P_world"),
            "depth_mode": fused.get("depth_mode"),
        }

    legacy_params = _fuse_params_from_legacy(run_dir)
    ir = float(legacy_params.get("inlier_radius", inlier_radius))

    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    result = fm.fuse(
        predictions,
        inlier_radius=ir,
        min_inv=min_inv,
        refine=not no_refine,
        refine_k=refine_k,
        ply_path=ply,
        depth_mode=depth_mode,
        ray_perp_radius=ray_perp_radius,
        ray_pixel_radius=ray_pixel_radius,
        ray_z_band=ray_z_band,
        ray_min_alpha=ray_min_alpha,
    )
    payload = result.to_dict()
    payload["depth_mode"] = depth_mode
    payload["refused_from"] = "predictions.json"
    if (run_dir / "fused.json").is_file():
        payload["legacy_fused"] = "fused.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    p = result.P_world
    return {
        "run_dir": str(run_dir),
        "output": str(out_path),
        "P_world": [round(float(x), 6) for x in p],
        "support": result.support,
        "inlier_radius": ir,
        "depth_mode": depth_mode,
    }


def refuse_rows(
    rows: list[tuple[str, str, str, str]],
    *,
    runs_root: Path,
    group: str,
    ply: Path,
    output_name: str = "fused_ray.json",
    force: bool = False,
    **fuse_kw: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for key, name, grp, run_id in rows:
        if grp != group:
            continue
        run_dir = runs_root / run_id
        rec = refuse_run_dir(
            run_dir,
            ply=ply,
            output_name=output_name,
            force=force,
            **fuse_kw,
        )
        rec["object_key"] = key
        rec["object"] = name
        rec["group"] = grp
        rec["run_id"] = run_id
        results.append(rec)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-fuse runs with depth_mode=ray -> fused_ray.json")
    ap.add_argument("--runs-root", type=Path, default=_REPO / "3DGS/test2/runs")
    ap.add_argument("--table-json", type=Path, default=None)
    ap.add_argument("--group", default="LoRA", help="Which group to re-fuse (default LoRA)")
    ap.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    ap.add_argument("--output-name", default="fused_ray.json")
    ap.add_argument("--inlier-radius", type=float, default=1.0,
                    help="Default if legacy fused.json has no inlier_radius")
    ap.add_argument("--refine-k", type=float, default=1.35)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--ray-perp-radius", type=float, default=0.06)
    ap.add_argument("--ray-pixel-radius", type=float, default=6.0)
    ap.add_argument("--ray-z-band", type=float, default=2.0)
    ap.add_argument("--ray-min-alpha", type=float, default=0.45)
    ap.add_argument("--force", action="store_true", help="Overwrite existing fused_ray.json")
    args = ap.parse_args()

    if not args.ply.is_file():
        raise SystemExit(f"ply not found: {args.ply}")

    rows = load_rows(args.table_json)
    results = refuse_rows(
        rows,
        runs_root=args.runs_root,
        group=args.group,
        ply=args.ply,
        output_name=args.output_name,
        inlier_radius=args.inlier_radius,
        refine_k=args.refine_k,
        no_refine=args.no_refine,
        ray_perp_radius=args.ray_perp_radius,
        ray_pixel_radius=args.ray_pixel_radius,
        ray_z_band=args.ray_z_band,
        ray_min_alpha=args.ray_min_alpha,
        force=args.force,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    n_ok = sum(1 for r in results if not r.get("error"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    print(f"\n[refuse] group={args.group} ok={n_ok} skipped={n_skip} total={len(results)}")


if __name__ == "__main__":
    main()
