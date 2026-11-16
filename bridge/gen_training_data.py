#!/usr/bin/env python3
"""生成 Location 任务训练数据（project -> mask -> refine）。

各阶段见 ``main()`` 的 choices：project、mask、refine。
mask 阶段默认只读 ``projections.json``。加上 ``--use-kept`` 时只读射线滤帧留下的 ``projections_kept.json``。
refine 把答案贴到 mask 几何上。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from repo_paths import resolve_data_path, to_repo_relative
from unproject import in_frame, world_to_image


# ---------------------------------------------------------------------------
# 阶段 1：project
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
            "rgb_path": to_repo_relative(rgb_path),
        })

    proj_path = out_dir / "projections.json"
    proj_path.write_text(json.dumps(records, indent=2))
    print(f"[project] {len(records)} views -> {proj_path}")


# ---------------------------------------------------------------------------
# 阶段 2：mask（在 WSL roborefer 环境中运行）
# ---------------------------------------------------------------------------

def _resolve_rgb_path(path: str) -> Path:
    """解析 rgb_path：相对仓库根、Windows 绝对路径或 /mnt 路径。"""
    return resolve_data_path(path)


def _load_projections(out_dir: Path, *, use_kept: bool = False) -> list[dict]:
    """默认只读画面内投影。``use_kept`` 时只读射线滤帧留下的视角。"""
    if use_kept:
        kept = out_dir / "projections_kept.json"
        if not kept.is_file():
            sys.exit(f"[error] no projections_kept.json in {out_dir}")
        return json.loads(kept.read_text(encoding="utf-8"))
    proj = out_dir / "projections.json"
    if proj.is_file():
        return json.loads(proj.read_text(encoding="utf-8"))
    sys.exit(f"[error] no projections.json in {out_dir}")


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
    records = _load_projections(out_dir, use_kept=bool(getattr(args, "use_kept", False)))
    view_filter: set[str] | None = None
    raw_ids = getattr(args, "view_ids", None)
    if raw_ids:
        view_filter = {str(x).strip().zfill(3) for x in str(raw_ids).split(",") if str(x).strip()}
        records = [r for r in records if str(r["view_id"]).zfill(3) in view_filter]
        if not records:
            raise SystemExit(f"--view-ids {raw_ids!r} 在投影里没有匹配视角")
        print(f"[mask] view filter: {sorted(view_filter)} ({len(records)} projections)")
    mask_dir = out_dir / "mask"
    mask_dir.mkdir(exist_ok=True)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        large_box_area_start=args.large_box_area_start,
        large_box_penalty=args.large_box_penalty,
        prefer_containing_point=args.prefer_containing_point,
        use_sam_point_prompt=args.sam_point_prompt,
        min_box_mask_iou=args.min_box_mask_iou,
        fallback_point_sam=not args.no_fallback_point_sam,
        fallback_point_sam_if_box_miss=args.fallback_point_sam_if_box_miss,
        text_only=args.text_only,
    )
    gdino_model = load_grounding_dino(
        args.grounding_config, args.grounding_checkpoint, device=device
    )
    predictor = load_sam2_predictor(args.sam2_checkpoint, args.sam2_config)
    mode = "text_only" if gs_cfg.text_only else "grounding"
    print(f"[mask] mode={mode} caption={args.object!r} device={device}")

    # 与 RefSpatial-Expand-Bench / use_api.py / export_spatial_train.DEFAULT_SUFFIX 对齐
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

        mask_fname = f"view_{view_id}.png"
        Image.fromarray(mask, mode="L").save(mask_dir / mask_fname)

        entry = {
            "id": len(qa_records),
            "view_id": view_id,
            "object": args.object,
            "prompt": args.prompt,
            "suffix": suffix,
            "rgb_path": to_repo_relative(rgb_path),
            "mask_path": f"mask/{mask_fname}",
            "answer": [[rec["nx"], rec["ny"]]],
            "u": rec["u"],
            "v": rec["v"],
            "width": w,
            "height": h,
            "category": "object",
            "step": 1,
            "scene": args.scene,
            "mask_mode": "grounding",
        }
        if meta:
            entry["mask_meta"] = meta
        qa_records.append(entry)

    q_path = out_dir / "question.json"
    if view_filter is not None and q_path.is_file():
        existing = json.loads(q_path.read_text(encoding="utf-8"))
        by_vid = {str(r.get("view_id")).zfill(3): r for r in existing}
        for entry in qa_records:
            by_vid[str(entry["view_id"]).zfill(3)] = entry
        merged = list(by_vid.values())
        for i, rec in enumerate(merged):
            rec["id"] = i
        q_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[mask] merged {len(qa_records)} views into {q_path} (total {len(merged)})")
    else:
        q_path.write_text(json.dumps(qa_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[mask] {len(qa_records)} samples -> {out_dir}")
    if skips:
        skip_path = out_dir / "mask_skips.json"
        skip_path.write_text(json.dumps(skips, indent=2), encoding="utf-8")
        print(f"[mask] skips {len(skips)} -> {skip_path}")


def stage_refine(args: argparse.Namespace) -> None:
    """把答案移到 mask 质心（回退：距提示点最近的 mask 内像素）。"""
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
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["project", "mask", "refine"])
    # project 参数
    ap.add_argument("--fused", help="fused.json path")
    ap.add_argument("--views-root", help="dir with rgb/ and camera_params/")
    # mask 参数
    ap.add_argument(
        "--use-kept",
        action="store_true",
        help="mask only the views in projections_kept.json",
    )
    ap.add_argument("--prompt", help="RoboRefer prompt string")
    ap.add_argument("--object", help="short object description")
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
        help="Grounding DINO .pth",
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
    ap.add_argument("--contain-bonus", type=float, default=0.18)
    ap.add_argument("--area-penalty", type=float, default=0.12)
    ap.add_argument("--tiny-box-penalty", type=float, default=0.40)
    ap.add_argument("--compact-box-weight", type=float, default=0.10)
    ap.add_argument(
        "--large-box-area-start",
        type=float,
        default=0.15,
        help="Box area / image area above this starts the soft large-box deduction",
    )
    ap.add_argument(
        "--large-box-penalty",
        type=float,
        default=0.12,
        help="轻扣大于中位数的框；过大惩罚会把整机拆成零件小框",
    )
    ap.add_argument(
        "--view-ids",
        type=str,
        default=None,
        help="仅重跑这些视角（逗号分隔，如 041,030）；写入 mask 并合并进已有 question.json",
    )
    ap.add_argument(
        "--prefer-containing-point",
        action="store_true",
        help="Prefer boxes containing projection (legacy; usually worse for floating 3D anchors)",
    )
    ap.add_argument(
        "--text-only",
        action="store_true",
        help="Rank remaining DINO boxes by score only; the referring point does not filter, score, or prompt SAM",
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
    # 共用
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
        if not args.grounding_checkpoint:
            ap.error("--stage mask requires --grounding-checkpoint")
        stage_mask(args)
    else:
        stage_refine(args)


if __name__ == "__main__":
    main()
