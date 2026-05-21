#!/usr/bin/env python3
"""Generate Location-task training data (project -> filter -> mask -> refine).

See module stages in ``main()`` choices: project, mask, refine.
Mask reads ``projections_kept.json`` when present. Default mask mode: Grounding DINO + SAM2.
Refine snaps answers to mask geometry.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _intrinsics(d: dict) -> tuple[float, float, float, float]:
    w, h = int(d["width"]), int(d["height"])
    fx = w / (2.0 * math.tan(float(d["fov_x"]) / 2.0))
    fy = h / (2.0 * math.tan(float(d["fov_y"]) / 2.0))
    return fx, fy, w / 2.0, h / 2.0


def world_to_image(p_world: np.ndarray, cam: dict) -> tuple[float, float, float]:
    """Return (u, v, z_cam). z_cam <= 0 means behind camera."""
    R_c2w = np.asarray(cam["rotation"], dtype=np.float64)
    C = np.asarray(cam["position"], dtype=np.float64)
    p_cam = R_c2w.T @ (p_world - C)
    z = p_cam[2]
    if z <= 0:
        return -1.0, -1.0, z
    fx, fy, cx, cy = _intrinsics(cam)
    return fx * p_cam[0] / z + cx, fy * p_cam[1] / z + cy, z


def in_frame(u: float, v: float, w: int, h: int, margin: int = 10) -> bool:
    return margin <= u < w - margin and margin <= v < h - margin


# ---------------------------------------------------------------------------
# Stage 1: project
# ---------------------------------------------------------------------------

def stage_project(args: argparse.Namespace) -> None:
    fused = json.loads(Path(args.fused).read_text())
    p_world = np.array(fused["P_world"], dtype=np.float64)

    views_root = Path(args.views_root)
    cam_dir = views_root / "camera_params"
    rgb_dir = views_root / "rgb"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for cam_path in sorted(cam_dir.glob("view_*.json")):
        view_id = re.search(r"view_(\d+)", cam_path.name).group(1)
        rgb_path = rgb_dir / f"view_{view_id}.png"
        if not rgb_path.exists():
            continue
        cam = json.loads(cam_path.read_text())
        u, v, z = world_to_image(p_world, cam)
        w, h = int(cam["width"]), int(cam["height"])
        if z <= 0 or not in_frame(u, v, w, h):
            continue
        records.append({
            "view_id": view_id,
            "u": u, "v": v,
            "nx": float(np.clip(u / w, 0.0, 1.0)),
            "ny": float(np.clip(v / h, 0.0, 1.0)),
            "width": w, "height": h,
            "rgb_path": str(rgb_path.resolve()),
        })

    proj_path = out_dir / "projections.json"
    proj_path.write_text(json.dumps(records, indent=2))
    print(f"[project] {len(records)} views -> {proj_path}")


# ---------------------------------------------------------------------------
# Stage 2: mask  (run in WSL roborefer env)
# ---------------------------------------------------------------------------

def _resolve_rgb_path(path: str) -> Path:
    """Resolve rgb_path; map Windows E:\\... to WSL /mnt/e/... when needed."""
    p = Path(path)
    if p.is_file():
        return p
    raw = path.strip()
    if len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha():
        drive = raw[0].lower()
        rest = raw[2:].lstrip("\\/")
        wsl = Path("/mnt") / drive / Path(rest.replace("\\", "/"))
        if wsl.is_file():
            return wsl
    return p


def _load_projections(out_dir: Path) -> list[dict]:
    kept = out_dir / "projections_kept.json"
    proj = out_dir / "projections.json"
    if kept.is_file():
        return json.loads(kept.read_text(encoding="utf-8"))
    if proj.is_file():
        return json.loads(proj.read_text(encoding="utf-8"))
    sys.exit(f"[error] no projections_kept.json or projections.json in {out_dir}")


def _import_mask_grounding():
    bridge_dir = Path(__file__).resolve().parent
    if str(bridge_dir) not in sys.path:
        sys.path.insert(0, str(bridge_dir))
    from mask_grounding import (  # noqa: WPS433
        GroundingSamConfig,
        grounding_sam_mask,
        load_grounding_dino,
        load_sam2_predictor,
        segment_point,
    )

    return (
        GroundingSamConfig,
        grounding_sam_mask,
        load_grounding_dino,
        load_sam2_predictor,
        segment_point,
    )


def stage_mask(args: argparse.Namespace) -> None:
    from PIL import Image

    out_dir = Path(args.out)
    records = _load_projections(out_dir)
    mask_dir = out_dir / "mask"
    mask_dir.mkdir(exist_ok=True)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = args.mask_mode

    gs_cfg = None
    gdino_model = None
    predictor = None

    if mode == "grounding":
        (
            GroundingSamConfig,
            grounding_sam_mask,
            load_grounding_dino,
            load_sam2_predictor,
            _segment_point,
        ) = _import_mask_grounding()
        gs_cfg = GroundingSamConfig(
            box_threshold=args.grounding_box_threshold,
            text_threshold=args.grounding_text_threshold,
            anchor_box_radius=args.anchor_box_radius,
            min_mask_ratio=args.min_mask_ratio,
            max_box_area_ratio=args.max_box_area_ratio,
            max_point_box_dist=args.max_point_box_dist,
            min_box_area_ratio=args.min_box_area_ratio,
            dino_score_weight=args.dino_score_weight,
            near_score_weight=args.near_score_weight,
            near_dist_sigma=args.near_dist_sigma,
            contain_bonus=args.contain_bonus,
            area_penalty=args.area_penalty,
            tiny_box_penalty=args.tiny_box_penalty,
            compact_box_weight=args.compact_box_weight,
            large_box_penalty=args.large_box_penalty,
            prefer_containing_point=args.prefer_containing_point,
            use_sam_point_prompt=args.sam_point_prompt,
            min_box_mask_iou=args.min_box_mask_iou,
            fallback_point_sam=not args.no_fallback_point_sam,
            fallback_point_sam_if_box_miss=args.fallback_point_sam_if_box_miss,
        )
        gdino_model = load_grounding_dino(
            args.grounding_config, args.grounding_checkpoint, device=device
        )
        predictor = load_sam2_predictor(args.sam2_checkpoint, args.sam2_config)
        print(f"[mask] mode=grounding caption={args.object!r} device={device}")
    else:
        _, _, _, load_sam2_predictor, segment_point = _import_mask_grounding()
        predictor = load_sam2_predictor(args.sam2_checkpoint, args.sam2_config)
        print(f"[mask] mode=point device={device}")

    # Match RefSpatial-Expand-Bench / use_api.py / export_spatial_train.DEFAULT_SUFFIX
    suffix = (
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
        "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
    )

    qa_records = []
    skips = []
    for rec in records:
        view_id = rec["view_id"]
        rgb_path = _resolve_rgb_path(rec["rgb_path"])
        if not rgb_path.exists():
            print(f"[skip] rgb not found: {rgb_path}")
            skips.append({"view_id": view_id, "reason": "rgb_missing"})
            continue

        img_np = np.array(Image.open(rgb_path).convert("RGB"))
        w, h = rec["width"], rec["height"]
        u, v = float(rec["u"]), float(rec["v"])
        meta = {}

        if mode == "grounding":
            mask_bool, meta = grounding_sam_mask(
                img_np,
                u,
                v,
                args.object,
                w,
                h,
                gdino_model,
                predictor,
                gs_cfg,
                device=device,
            )
            if mask_bool is None:
                reason = meta.get("reason", "grounding_failed")
                print(f"[skip] view_{view_id}: {reason}")
                skips.append({"view_id": view_id, "reason": reason, **meta})
                continue
            mask = mask_bool.astype(np.uint8) * 255
        else:
            mask_bool, _sam_score = segment_point(predictor, img_np, u, v)
            if mask_bool.sum() / (w * h) < args.min_mask_ratio:
                print(f"[skip] view_{view_id}: mask too small")
                skips.append({"view_id": view_id, "reason": "mask_too_small"})
                continue
            mask = mask_bool.astype(np.uint8) * 255

        mask_fname = f"view_{view_id}.png"
        Image.fromarray(mask, mode="L").save(mask_dir / mask_fname)

        entry = {
            "id": len(qa_records),
            "view_id": view_id,
            "object": args.object,
            "prompt": args.prompt,
            "suffix": suffix,
            "rgb_path": str(rgb_path.resolve()),
            "mask_path": f"mask/{mask_fname}",
            "answer": [[rec["nx"], rec["ny"]]],
            "u": rec["u"],
            "v": rec["v"],
            "width": w,
            "height": h,
            "category": "object",
            "step": 1,
            "scene": args.scene,
            "mask_mode": mode,
        }
        if meta:
            entry["mask_meta"] = meta
        qa_records.append(entry)

    (out_dir / "question.json").write_text(
        json.dumps(qa_records, indent=2, ensure_ascii=False)
    )
    if skips:
        skip_path = out_dir / "mask_skips.json"
        skip_path.write_text(json.dumps(skips, indent=2), encoding="utf-8")
        print(f"[mask] skips {len(skips)} -> {skip_path}")
    print(f"[mask] {len(qa_records)} samples -> {out_dir}")


def stage_refine(args: argparse.Namespace) -> None:
    """Move answer to mask centroid (fallback: nearest in-mask pixel to prompt point)."""
    from PIL import Image

    out_dir = Path(args.out)
    q_path = out_dir / "question.json"
    if not q_path.is_file():
        sys.exit(f"[error] {q_path} not found. Run --stage mask first.")

    records = json.loads(q_path.read_text(encoding="utf-8"))
    refined = 0
    for rec in records:
        mask_rel = rec.get("mask_path", "")
        mask_path = out_dir / mask_rel if not Path(mask_rel).is_absolute() else Path(mask_rel)
        if not mask_path.is_file():
            continue
        w = int(rec.get("width", 0))
        h = int(rec.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        m = np.array(Image.open(mask_path).convert("L"))
        fg = m >= 128
        if not np.any(fg):
            continue
        ys, xs = np.nonzero(fg)
        cu = float(xs.mean())
        cv = float(ys.mean())
        if args.refine_mode == "nearest":
            u0, v0 = float(rec.get("u", rec["answer"][0][0] * w)), float(
                rec.get("v", rec["answer"][0][1] * h)
            )
            d2 = (xs - u0) ** 2 + (ys - v0) ** 2
            j = int(np.argmin(d2))
            cu, cv = float(xs[j]), float(ys[j])
        nx = float(np.clip(cu / w, 0.0, 1.0))
        ny = float(np.clip(cv / h, 0.0, 1.0))
        rec["answer"] = [[nx, ny]]
        rec["nx"] = nx
        rec["ny"] = ny
        rec["u"] = cu
        rec["v"] = cv
        rec["refined"] = True
        refined += 1

    q_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[refine] updated {refined}/{len(records)} answers -> {q_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["project", "mask", "refine"])
    # project args
    ap.add_argument("--fused", help="fused.json path")
    ap.add_argument("--views-root", help="dir with rgb/ and camera_params/")
    # mask args
    ap.add_argument("--prompt", help="RoboRefer prompt string")
    ap.add_argument("--object", help="short object description")
    ap.add_argument(
        "--mask-mode",
        choices=("grounding", "point"),
        default="grounding",
        help="grounding: DINO box + SAM2; point: legacy single-point SAM2",
    )
    ap.add_argument("--sam2-checkpoint")
    ap.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2.1 small for 8GB; large: configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    ap.add_argument(
        "--grounding-config",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Grounding DINO model config .py",
    )
    ap.add_argument(
        "--grounding-checkpoint",
        help="Grounding DINO .pth (required for --mask-mode grounding)",
    )
    ap.add_argument("--grounding-box-threshold", type=float, default=0.10)
    ap.add_argument("--grounding-text-threshold", type=float, default=0.12)
    ap.add_argument("--anchor-box-radius", type=int, default=64)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.35)
    ap.add_argument(
        "--max-point-box-dist",
        type=float,
        default=-1.0,
        help="Drop DINO boxes farther than this (px) from projection; <0 => 0.12*diag",
    )
    ap.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.003,
        help="Ignore fragment boxes below this image area ratio unless none left",
    )
    ap.add_argument("--dino-score-weight", type=float, default=0.50)
    ap.add_argument("--near-score-weight", type=float, default=0.40)
    ap.add_argument("--near-dist-sigma", type=float, default=50.0)
    ap.add_argument("--contain-bonus", type=float, default=0.06)
    ap.add_argument("--area-penalty", type=float, default=0.12)
    ap.add_argument("--tiny-box-penalty", type=float, default=0.40)
    ap.add_argument("--compact-box-weight", type=float, default=0.28)
    ap.add_argument("--large-box-penalty", type=float, default=0.38)
    ap.add_argument(
        "--prefer-containing-point",
        action="store_true",
        help="Prefer boxes containing projection (legacy; usually worse for floating 3D anchors)",
    )
    ap.add_argument(
        "--sam-point-prompt",
        action="store_true",
        help="Also pass projection as SAM positive point (default: box-only SAM)",
    )
    ap.add_argument("--min-box-mask-iou", type=float, default=0.12)
    ap.add_argument("--no-fallback-point-sam", action="store_true")
    ap.add_argument(
        "--fallback-point-sam-if-box-miss",
        action="store_true",
        help="If all box SAM fail IoU check, fall back to point SAM (not recommended)",
    )
    ap.add_argument("--min-mask-ratio", type=float, default=0.001)
    ap.add_argument("--scene", default="indoor")
    ap.add_argument(
        "--refine-mode",
        choices=("centroid", "nearest"),
        default="centroid",
        help="refine: mask centroid or nearest in-mask pixel to SAM prompt",
    )
    # shared
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    if args.stage == "project":
        for req in ("fused", "views_root"):
            if not getattr(args, req):
                ap.error(f"--stage project requires --{req.replace('_','-')}")
        stage_project(args)
    elif args.stage == "mask":
        for req in ("prompt", "object", "sam2_checkpoint"):
            if not getattr(args, req):
                ap.error(f"--stage mask requires --{req.replace('_','-')}")
        if args.mask_mode == "grounding" and not args.grounding_checkpoint:
            ap.error("--stage mask with --mask-mode grounding requires --grounding-checkpoint")
        stage_mask(args)
    else:
        stage_refine(args)


if __name__ == "__main__":
    main()
