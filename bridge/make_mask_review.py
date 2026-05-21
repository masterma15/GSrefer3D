#!/usr/bin/env python3
"""Export RGB + mask + point overlays for manual QA (one PNG per view).

Reads ``question.json`` under each ``training_data/data2_*`` dir (after mask, before
or after refine). Writes ``<out>/review/view_XXX.png``.

Example::

    python bridge/make_mask_review.py --out training_data/data2_medicine_bottle
    python bridge/make_mask_review.py --inputs training_data/data2_bowl training_data/data2_shaver
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def _resolve_path(path: str, base: Path, views_root: Path | None) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_absolute():
        cand = base / p
        if cand.is_file():
            return cand
    raw = path.strip()
    if len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha():
        drive = raw[0].lower()
        rest = raw[2:].lstrip("\\/")
        wsl = Path("/mnt") / drive / Path(rest.replace("\\", "/"))
        if wsl.is_file():
            return wsl
        win = Path(raw)
        if win.is_file():
            return win
    if views_root is not None:
        m = re.search(r"view_(\d+)\.png$", raw.replace("\\", "/"))
        if m:
            cand = views_root / "rgb" / f"view_{m.group(1)}.png"
            if cand.is_file():
                return cand
    return p


def _draw_review(
    rgb: np.ndarray,
    mask: np.ndarray,
    u: float,
    v: float,
    *,
    box: list[float] | None,
    candidate_boxes: list[dict] | None,
    alpha: float,
    title: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    base = Image.fromarray(rgb).convert("RGBA")
    fg = mask >= 128
    tint = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    tint[fg, 1] = 200
    tint[fg, 3] = int(255 * alpha)
    tint_img = Image.fromarray(tint, mode="RGBA")
    out_img = Image.alpha_composite(base, tint_img).convert("RGB")
    draw = ImageDraw.Draw(out_img)

    selected_idx = None
    if candidate_boxes:
        for rb in candidate_boxes:
            b = rb.get("box")
            if b is None or len(b) != 4:
                continue
            x1, y1, x2, y2 = b
            is_sel = box is not None and len(box) == 4 and all(
                abs(float(a) - float(b_)) < 1.5 for a, b_ in zip(box, b)
            )
            if is_sel:
                selected_idx = rb.get("idx")
            color = (255, 200, 0) if is_sel else (80, 180, 255)
            width = 3 if is_sel else 1
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = f"#{rb.get('idx', '?')} d={rb.get('dino_score', 0):.2f}"
            draw.text((x1 + 2, max(40, y1 - 14)), label, fill=color)
    elif box is not None and len(box) == 4:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 200, 0), width=3)

    r = 10
    ui, vi = int(round(u)), int(round(v))
    draw.ellipse((ui - r, vi - r, ui + r, vi + r), fill=(255, 40, 40), outline=(180, 0, 0), width=2)
    draw.line([(ui - 14, vi), (ui + 14, vi)], fill=(255, 255, 255), width=2)
    draw.line([(ui, vi - 14), (ui, vi + 14)], fill=(255, 255, 255), width=2)

    bar_h = 36
    draw.rectangle([0, 0, out_img.width, bar_h], fill=(30, 30, 30))
    draw.text((8, 8), title, fill=(255, 255, 255))
    return np.array(out_img)


def export_object(
    obj_dir: Path,
    *,
    review_dir: Path | None,
    views_root: Path | None,
    alpha: float,
) -> int:
    q_path = obj_dir / "question.json"
    if not q_path.is_file():
        print(f"[skip] no question.json: {obj_dir}")
        return 0

    records = json.loads(q_path.read_text(encoding="utf-8"))
    out_review = review_dir or (obj_dir / "review")
    out_review.mkdir(parents=True, exist_ok=True)

    n = 0
    for rec in records:
        view_id = str(rec.get("view_id", rec.get("id", 0))).zfill(3)
        mask_rel = rec.get("mask_path", f"mask/view_{view_id}.png")
        mask_path = _resolve_path(mask_rel, obj_dir, views_root)
        rgb_rel = rec.get("rgb_path", "")
        rgb_path = _resolve_path(rgb_rel, obj_dir, views_root) if rgb_rel else None
        if rgb_path is None or not rgb_path.is_file():
            if views_root:
                rgb_path = views_root / "rgb" / f"view_{view_id}.png"
        if rgb_path is None or not rgb_path.is_file():
            print(f"[skip] {obj_dir.name} view_{view_id}: rgb missing")
            continue
        if not mask_path.is_file():
            print(f"[skip] {obj_dir.name} view_{view_id}: mask missing")
            continue

        from PIL import Image

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        w = int(rec.get("width", rgb.shape[1]))
        h = int(rec.get("height", rgb.shape[0]))
        ans = rec.get("answer", [[0.0, 0.0]])
        nx, ny = float(ans[0][0]), float(ans[0][1])
        u = float(rec.get("u", nx * w))
        v = float(rec.get("v", ny * h))

        meta = rec.get("mask_meta") or {}
        box = meta.get("box")
        ranked = meta.get("ranked_boxes") or []
        n_boxes = int(meta.get("num_boxes", len(ranked)))
        parts = [f"view_{view_id}", f"dino_boxes={n_boxes}"]
        if rec.get("refined"):
            parts.append("refined")
        if meta.get("point_in_mask") is False:
            parts.append("pt_outside_mask")
        if "dino_score" in meta:
            parts.append(f"sel_dino={meta['dino_score']:.2f}")
        if "box_mask_iou" in meta:
            parts.append(f"iou={meta['box_mask_iou']:.2f}")
        if meta.get("select_reason"):
            parts.append(str(meta["select_reason"]))
        title = " | ".join(parts)

        out_np = _draw_review(
            rgb,
            mask,
            u,
            v,
            box=box,
            candidate_boxes=ranked if ranked else None,
            alpha=alpha,
            title=title,
        )
        out_path = out_review / f"review_view_{view_id}.png"
        Image.fromarray(out_np).save(out_path)
        n += 1

    print(f"[review] {obj_dir.name}: {n} -> {out_review}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Export mask QA overlays to review/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path, help="Single object dir, e.g. training_data/data2_bowl")
    g.add_argument(
        "--inputs",
        nargs="+",
        help="Multiple dirs or globs, e.g. training_data/data2_*",
    )
    ap.add_argument(
        "--review-dir",
        type=Path,
        help="Override output dir (default: <out>/review)",
    )
    ap.add_argument(
        "--views-root",
        type=Path,
        default="3DGS/test2",
        help="Fallback RGB root if rgb_path in JSON is stale",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Mask green overlay opacity",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    views_root = (repo / args.views_root).resolve() if args.views_root else None
    if views_root is not None and not views_root.is_dir():
        views_root = None

    dirs: list[Path] = []
    if args.out is not None:
        dirs.append(args.out.resolve())
    else:
        for pattern in args.inputs:
            p = Path(pattern)
            if "*" in pattern or "?" in pattern:
                dirs.extend(sorted(p.parent.glob(p.name)))
            elif p.is_dir():
                dirs.append(p.resolve())
            else:
                print(f"[skip] not found: {pattern}")
        dirs = sorted(set(dirs))

    total = 0
    for d in dirs:
        if not d.is_dir():
            continue
        rd = args.review_dir.resolve() if args.review_dir else None
        total += export_object(d, review_dir=rd, views_root=views_root, alpha=args.alpha)

    if total == 0:
        sys.exit("[error] no review images written")
    print(f"[done] {total} images total")


if __name__ == "__main__":
    main()
