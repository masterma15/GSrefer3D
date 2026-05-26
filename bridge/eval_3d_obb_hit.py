#!/usr/bin/env python3
"""OBB hit rate for fused P_world vs docs/bbox_data2.json (Base vs LoRA runs).

Hit test (CloudCompare Cross Section OBB):
  local = R^T @ (P - center)   where R columns are box local axes
  hit   iff |local[i]| <= half_extent[i] (+ optional margin) for all i

Also reports fused support and optional AABB legacy hit for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# bbox_data2.json object key
OBJECT_ZH_TO_KEY: dict[str, str] = {
    "剃须刀": "electric_shaver",
    "棕兔": "brown_rabbit",
    "金毛": "golden_retriever",
    "金碗": "golden_bowl",
    "雨伞": "umbrella",
    "玩具蛋糕": "toy_cake",
    "饼干袋": "cookie_bag",
    "药瓶": "medicine_bottle",
    "手链": "bracelet",
    "发夹": "hair_clip",
    "胶带": "double_sided_tape",
    "胶带 hold-out": "double_sided_tape",
}

DEFAULT_FUSED_BY_GROUP: dict[str, str] = {
    "Base": "fused.json",
    "LoRA": "fused_ray.json",
}

DEFAULT_ROWS: list[tuple[str, str, str, str]] = [
    ("electric_shaver", "剃须刀", "Base", "20260519_170540_4c3b9a32"),
    ("electric_shaver", "剃须刀", "LoRA", "20260519_143457_4c3b9a32"),
    ("brown_rabbit", "棕兔", "Base", "20260519_171359_147bac82"),
    ("brown_rabbit", "棕兔", "LoRA", "20260519_144845_147bac82"),
    ("golden_retriever", "金毛", "Base", "20260519_172627_f8dbfcc3"),
    ("golden_retriever", "金毛", "LoRA", "20260519_154013_f8dbfcc3"),
    ("golden_bowl", "金碗", "Base", "20260519_173958_8d83a715"),
    ("golden_bowl", "金碗", "LoRA", "20260519_154649_8d83a715"),
    ("umbrella", "雨伞", "Base", "20260519_174835_d7bab60f"),
    ("umbrella", "雨伞", "LoRA", "20260519_160219_d7bab60f"),
    ("toy_cake", "玩具蛋糕", "Base", "20260519_175733_7dd80c38"),
    ("toy_cake", "玩具蛋糕", "LoRA", "20260519_161222_7dd80c38"),
    ("cookie_bag", "饼干袋", "Base", "20260519_180800_e51c780a"),
    ("cookie_bag", "饼干袋", "LoRA", "20260519_162019_e51c780a"),
    ("medicine_bottle", "药瓶", "Base", "20260519_182106_04149b86"),
    ("medicine_bottle", "药瓶", "LoRA", "20260519_163004_04149b86"),
    ("bracelet", "手链", "Base", "20260520_001018_cb2e562f"),
    ("bracelet", "手链", "LoRA", "20260519_163546_cb2e562f"),
    ("hair_clip", "发夹", "Base", "20260519_183039_65bf02a5"),
    ("hair_clip", "发夹", "LoRA", "20260519_164312_65bf02a5"),
    ("double_sided_tape", "胶带 hold-out", "Base", "20260519_000313_6c883d56"),
    ("double_sided_tape", "胶带 hold-out", "LoRA", "20260519_132142_6c883d56"),
]


def load_bbox(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("objects", data)


def load_rows(table_json: Path | None) -> list[tuple[str, str, str, str]]:
    if table_json is None or not table_json.is_file():
        return list(DEFAULT_ROWS)
    raw = json.loads(table_json.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, str, str]] = []
    for r in raw:
        obj_zh = r["object"]
        key = r.get("object_key") or OBJECT_ZH_TO_KEY.get(obj_zh) or OBJECT_ZH_TO_KEY.get(
            obj_zh.replace(" hold-out", "")
        )
        if key is None:
            continue
        rows.append((key, obj_zh, r["group"], r["run_id"]))
    return rows


def _half_extent(obb: dict[str, Any]) -> np.ndarray:
    if "half_extent" in obb:
        return np.asarray(obb["half_extent"], dtype=np.float64)
    return np.asarray(obb["width"], dtype=np.float64) / 2.0


def obb_local(p_world: np.ndarray, obb: dict[str, Any]) -> np.ndarray:
    center = np.asarray(obb["center"], dtype=np.float64)
    rot = np.asarray(obb["rotation_columns"], dtype=np.float64)
    return rot.T @ (p_world - center)


def obb_hit(p_world: np.ndarray, obb: dict[str, Any], *, margin: float = 0.0) -> bool:
    local = obb_local(p_world, obb)
    half = _half_extent(obb) + margin
    return bool(np.all(np.abs(local) <= half))


def aabb_hit(p_world: np.ndarray, bmin: list[float], bmax: list[float]) -> bool:
    p = np.asarray(p_world, dtype=np.float64)
    lo = np.asarray(bmin, dtype=np.float64)
    hi = np.asarray(bmax, dtype=np.float64)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def resolve_aabb(obj_spec: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    if "bbox_min" in obj_spec and "bbox_max" in obj_spec:
        return obj_spec["bbox_min"], obj_spec["bbox_max"]
    legacy = obj_spec.get("aabb_legacy")
    if isinstance(legacy, dict) and "bbox_min" in legacy:
        return legacy["bbox_min"], legacy["bbox_max"]
    return None


def fused_filename_for_group(group: str, fused_by_group: dict[str, str] | None) -> str:
    mapping = fused_by_group or {"Base": "fused.json", "LoRA": "fused.json"}
    return mapping.get(group, "fused.json")


def eval_run(
    *,
    object_key: str,
    display_name: str,
    group: str,
    run_id: str,
    runs_root: Path,
    objects: dict[str, Any],
    margin: float,
    fused_by_group: dict[str, str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "object_key": object_key,
        "object": display_name,
        "group": group,
        "run_id": run_id,
    }
    spec = objects.get(object_key)
    if spec is None:
        row["error"] = f"no bbox spec for {object_key}"
        return row

    run_dir = runs_root / run_id
    fused_name = fused_filename_for_group(group, fused_by_group)
    fused_path = run_dir / fused_name
    if not fused_path.is_file():
        row["error"] = f"missing {fused_path}"
        return row

    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    p = np.asarray(fused["P_world"], dtype=np.float64)
    row["fused_file"] = fused_name
    row["depth_mode"] = fused.get("depth_mode")
    row["P_world"] = [round(float(x), 6) for x in p]
    row["support"] = fused.get("support")
    row["hold_out"] = bool(spec.get("hold_out", False))

    obb = spec.get("obb")
    if obb is not None:
        local = obb_local(p, obb)
        half = _half_extent(obb)
        row["obb_local"] = [round(float(x), 6) for x in local]
        row["obb_half_extent"] = [round(float(x), 6) for x in half]
        row["obb_max_axis_excess"] = round(float(np.max(np.abs(local) - half)), 6)
        row["obb_hit"] = obb_hit(p, obb, margin=margin)
        if margin > 0:
            row["obb_hit_margin_m"] = margin

    aabb = resolve_aabb(spec)
    if aabb is not None:
        row["aabb_hit"] = aabb_hit(p, aabb[0], aabb[1])

    return row


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[bool]] = {"Base": [], "LoRA": []}
    by_object: dict[str, dict[str, bool | None]] = {}

    for r in results:
        hit = r.get("obb_hit")
        if hit is None or r.get("error"):
            continue
        grp = r["group"]
        if grp in by_group:
            by_group[grp].append(bool(hit))
        obj = r["object_key"]
        by_object.setdefault(obj, {})[grp] = bool(hit)

    def _rate(vals: list[bool]) -> float | None:
        return round(100.0 * sum(vals) / len(vals), 1) if vals else None

    pairs = []
    for obj, hits in sorted(by_object.items()):
        b, l = hits.get("Base"), hits.get("LoRA")
        pairs.append({"object_key": obj, "Base": b, "LoRA": l})

    return {
        "obb_hit_rate_pct": {g: _rate(v) for g, v in by_group.items()},
        "n_evaluated": {g: len(v) for g, v in by_group.items()},
        "per_object": pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OBB hit rate for fused.json vs bbox_data2.json")
    ap.add_argument("--bbox", type=Path, default=Path("docs/bbox_data2.json"))
    ap.add_argument("--runs-root", type=Path, default=Path("3DGS/test2/runs"))
    ap.add_argument(
        "--table-json",
        type=Path,
        default=None,
        help="Optional run table (e.g. docs/results_2d_eval.json); maps object name -> bbox key",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Extra half-extent (m) on each OBB axis (e.g. 0.01)",
    )
    ap.add_argument("--out", type=Path, default=Path("docs/results_3d_obb_hit.json"))
    ap.add_argument(
        "--lora-fused",
        default="fused_ray.json",
        help="Fused JSON filename for LoRA runs (default fused_ray.json; use fused.json for legacy invdepth+snap)",
    )
    ap.add_argument(
        "--base-fused",
        default="fused.json",
        help="Fused JSON filename for Base runs",
    )
    ap.add_argument(
        "--refuse-lora-ray",
        action="store_true",
        help="Re-fuse LoRA predictions with depth_mode=ray before eval (writes fused_ray.json)",
    )
    ap.add_argument(
        "--ply",
        type=Path,
        default=Path("3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply"),
        help="point_cloud.ply for --refuse-lora-ray",
    )
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
        eval_run(
            object_key=key,
            display_name=name,
            group=group,
            run_id=run_id,
            runs_root=args.runs_root,
            objects=objects,
            margin=args.margin,
            fused_by_group=fused_by_group,
        )
        for key, name, group, run_id in rows
    ]
    summary = summarize(results)
    payload = {
        "summary": summary,
        "runs": results,
        "bbox": str(args.bbox.as_posix()),
        "margin_m": args.margin,
        "fused_by_group": fused_by_group,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")

    print(text)

    # Compact table for terminal
    print("\n# OBB hit (1=hit, 0=miss, -=error)")
    print(f"{'object_key':<22} {'Base':>6} {'LoRA':>6}")
    for item in summary["per_object"]:
        b = item["Base"]
        l = item["LoRA"]
        bs = "-" if b is None else str(int(b))
        ls = "-" if l is None else str(int(l))
        print(f"{item['object_key']:<22} {bs:>6} {ls:>6}")
    print(
        f"\nHit rate: Base {summary['obb_hit_rate_pct'].get('Base')}% "
        f"({summary['n_evaluated'].get('Base')} objs) | "
        f"LoRA {summary['obb_hit_rate_pct'].get('LoRA')}% "
        f"({summary['n_evaluated'].get('LoRA')} objs)"
    )


if __name__ == "__main__":
    main()
