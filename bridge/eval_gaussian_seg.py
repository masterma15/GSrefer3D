#!/usr/bin/env python3
"""P1：高斯分割相对手标 OBB，以及选中点重投影到 2D mask 的精度。

读 gaussian_seg.json（indices + 投票质心）。P_world 只是 mask 种子，不进主指标。
三维：投票质心是否在手标 OBB 内；选中高斯落在 OBB 内的比例（精确率，无高斯级 GT 不算召回）。
二维：各 used 视角上，画幅内选中 μ 落在该视 mask 内的比例（均值）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from obb_geom import half_extent, load_bbox, obb_hit, obb_hits, obb_local  # noqa: E402
from frustum_segment import (  # noqa: E402
    OBJ_DIR_TO_BBOX_KEY,
    frustum_hit_mask,
    load_ply_xyz,
    world_to_cam_uvz,
)
from gen_training_data import in_frame  # noqa: E402


def eval_obb(
    xyz_sel: np.ndarray,
    centroid: np.ndarray,
    obb: dict[str, Any],
    *,
    margin: float,
) -> dict[str, Any]:
    n = int(xyz_sel.shape[0])
    in_box = obb_hits(xyz_sel, obb, margin=margin) if n else np.zeros(0, dtype=bool)
    frac = float(in_box.mean()) if n else 0.0
    return {
        "n_selected": n,
        "centroid_obb_hit": obb_hit(centroid, obb, margin=margin),
        "frac_selected_in_obb": frac,
        "n_selected_in_obb": int(in_box.sum()) if n else 0,
        "centroid_obb_local": [round(float(x), 6) for x in obb_local(centroid, obb)],
        "obb_half_extent": [round(float(x), 6) for x in half_extent(obb)],
        "margin_m": margin,
    }


def eval_mask_precision(
    xyz_sel: np.ndarray,
    per_view: list[dict[str, Any]],
    obj_dir: Path,
    views_root: Path,
) -> dict[str, Any]:
    """各 kept 视角：画幅内选中点落在 mask 内的比例。"""
    from PIL import Image

    rows: list[dict[str, Any]] = []
    precs: list[float] = []
    for item in per_view:
        vid = str(item["view_id"]).zfill(3)
        mask_path = obj_dir / "mask" / f"view_{vid}.png"
        cam_path = views_root / "camera_params" / f"view_{vid}.json"
        if not mask_path.is_file() or not cam_path.is_file():
            continue
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        mask = np.array(Image.open(mask_path).convert("L"))
        if xyz_sel.shape[0] == 0:
            rows.append({"view_id": vid, "in_frame": 0, "in_mask": 0, "precision": None})
            continue
        hit, z = frustum_hit_mask(xyz_sel, cam, mask)
        u, v, _ = world_to_cam_uvz(xyz_sel, cam)
        w, h = int(cam["width"]), int(cam["height"])
        framed = np.array(
            [z[i] > 1e-8 and in_frame(float(u[i]), float(v[i]), w, h, margin=0) for i in range(len(z))],
            dtype=bool,
        )
        n_frame = int(framed.sum())
        n_mask = int((hit & framed).sum())
        prec = (n_mask / n_frame) if n_frame else None
        if prec is not None:
            precs.append(prec)
        rows.append({"view_id": vid, "in_frame": n_frame, "in_mask": n_mask, "precision": prec})
    return {
        "mean_view_mask_precision": float(np.mean(precs)) if precs else None,
        "n_views_eval": len(rows),
        "views": rows,
    }


def eval_one(seg_path: Path, bbox_path: Path, *, object_key: str | None, views_root: Path | None) -> dict[str, Any]:
    seg = json.loads(seg_path.read_text(encoding="utf-8"))
    objects = load_bbox(bbox_path)
    key = object_key or seg.get("bbox_key") or OBJ_DIR_TO_BBOX_KEY.get(Path(seg.get("object_dir", "")).name)
    row: dict[str, Any] = {
        "seg_path": str(seg_path.as_posix()),
        "object_dir": seg.get("object_dir"),
        "object_key": key,
        "n_selected": seg.get("n_selected"),
        "n_views_used": seg.get("n_views_used"),
    }
    if key is None or key not in objects:
        row["error"] = f"无 OBB：key={key!r}"
        return row
    spec = objects[key]
    obb = spec.get("obb")
    if obb is None:
        row["error"] = f"{key} 无 obb"
        return row

    ply = Path(seg["ply"])
    idx = np.asarray(seg.get("indices") or [], dtype=np.int64)
    xyz = load_ply_xyz(ply)
    xyz_sel = xyz[idx] if idx.size else np.zeros((0, 3), dtype=np.float64)
    centroid = np.asarray(seg["centroid"], dtype=np.float64)
    margin = float(json.loads(bbox_path.read_text(encoding="utf-8")).get("margin_default_m") or 0.015)

    row.update(eval_obb(xyz_sel, centroid, obb, margin=margin))
    obj_dir = seg_path.parent
    vr = Path(views_root) if views_root else Path(seg.get("views_root") or _REPO / "3DGS" / "test2")
    row["mask2d"] = eval_mask_precision(xyz_sel, seg.get("per_view") or [], obj_dir, vr)
    return row


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [bool(r["centroid_obb_hit"]) for r in results if "centroid_obb_hit" in r and not r.get("error")]
    fracs = [float(r["frac_selected_in_obb"]) for r in results if "frac_selected_in_obb" in r and not r.get("error")]
    precs = [
        float(r["mask2d"]["mean_view_mask_precision"])
        for r in results
        if r.get("mask2d") and r["mask2d"].get("mean_view_mask_precision") is not None
    ]
    return {
        "n_ok": len(hits),
        "centroid_obb_hit_rate": round(100.0 * sum(hits) / len(hits), 1) if hits else None,
        "mean_frac_selected_in_obb": round(float(np.mean(fracs)), 4) if fracs else None,
        "mean_view_mask_precision": round(float(np.mean(precs)), 4) if precs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate gaussian_seg.json vs OBB and 2D masks")
    ap.add_argument("--seg", type=Path, action="append", dest="segs", help="gaussian_seg.json (repeatable)")
    ap.add_argument("--obj-dir", type=Path, action="append", dest="obj_dirs")
    ap.add_argument("--obj-glob", type=str, default=None, help="如 training_data/data2_*（排除 data2_sft）")
    ap.add_argument("--bbox", type=Path, default=_REPO / "docs" / "bbox_data2.json")
    ap.add_argument("--object-key", type=str, default=None)
    ap.add_argument("--views-root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    paths: list[Path] = list(args.segs or [])
    for d in args.obj_dirs or []:
        p = d / "gaussian_seg.json"
        if p.is_file():
            paths.append(p)
    if args.obj_glob:
        skip = {"data2_sft", "data2_tape"}
        for d in sorted(p for p in Path().glob(args.obj_glob) if p.is_dir() and p.name not in skip):
            p = d / "gaussian_seg.json"
            if p.is_file():
                paths.append(p)
    if not paths:
        ap.error("provide --seg / --obj-dir / --obj-glob containing gaussian_seg.json")

    results = [eval_one(p, args.bbox, object_key=args.object_key, views_root=args.views_root) for p in paths]
    summary = summarize(results)
    payload = {"summary": summary, "results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[eval] wrote {args.output}")
    print(json.dumps(summary, ensure_ascii=False))
    for r in results:
        if r.get("error"):
            print(f"  {r.get('object_dir')}: {r['error']}")
        else:
            print(
                f"  {r.get('object_dir')}: centroid_hit={r['centroid_obb_hit']} "
                f"frac_in_obb={r['frac_selected_in_obb']:.3f} "
                f"mask2d={r['mask2d'].get('mean_view_mask_precision')}"
            )


if __name__ == "__main__":
    main()
