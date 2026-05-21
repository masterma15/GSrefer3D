#!/usr/bin/env python3
"""Export refine before/after overlays: green=projection, red=refined centroid.

Reads ``question.json`` + ``projections_kept.json`` (original nx,ny). Writes
``<out>/review_refine/refine_view_XXX.png``.

Example::

    python bridge/make_refine_review.py --out training_data/data2_medicine_bottle
    python bridge/make_refine_review.py --inputs training_data/data2_*
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from make_mask_review import _resolve_path  # noqa: E402


def _load_proj_map(obj_dir: Path) -> dict[str, dict]:
    for name in ("projections_kept.json", "projections.json"):
        p = obj_dir / name
        if p.is_file():
            rows = json.loads(p.read_text(encoding="utf-8"))
            return {str(r["view_id"]).zfill(3): r for r in rows}
    return {}


def _draw(
    rgb,
    mask,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    *,
    alpha: float,
    title: str,
):
    from PIL import Image, ImageDraw
    import numpy as np

    base = Image.fromarray(rgb).convert("RGBA")
    fg = mask >= 128
    tint = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    tint[fg, 1] = 180
    tint[fg, 3] = int(255 * alpha)
    img = Image.alpha_composite(base, Image.fromarray(tint, mode="RGBA")).convert("RGB")
    draw = ImageDraw.Draw(img)

    def _pt(u: float, v: float, color: tuple[int, int, int], label: str) -> None:
        r = 9
        ui, vi = int(round(u)), int(round(v))
        draw.ellipse((ui - r, vi - r, ui + r, vi + r), fill=color, outline=(0, 0, 0), width=2)
        draw.text((ui + 12, vi - 8), label, fill=color)

    _pt(u0, v0, (0, 220, 0), "proj")
    _pt(u1, v1, (255, 50, 50), "ref")
    draw.line([(u0, v0), (u1, v1)], fill=(255, 255, 0), width=2)

    bar_h = 40
    draw.rectangle([0, 0, img.width, bar_h], fill=(25, 25, 25))
    draw.text((6, 6), title, fill=(255, 255, 255))
    return np.array(img)


def export_object(obj_dir: Path, *, views_root: Path | None, alpha: float) -> int:
    q_path = obj_dir / "question.json"
    if not q_path.is_file():
        print(f"[skip] no question.json: {obj_dir}")
        return 0

    proj_map = _load_proj_map(obj_dir)
    records = json.loads(q_path.read_text(encoding="utf-8"))
    out_dir = obj_dir / "review_refine"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for rec in records:
        vid = str(rec.get("view_id", "")).zfill(3)
        proj = proj_map.get(vid)
        if proj is None:
            print(f"[skip] {obj_dir.name} view_{vid}: not in projections_kept")
            continue

        mask_path = _resolve_path(rec.get("mask_path", f"mask/view_{vid}.png"), obj_dir, views_root)
        rgb_path = _resolve_path(rec.get("rgb_path", ""), obj_dir, views_root)
        if not rgb_path.is_file() and views_root:
            rgb_path = views_root / "rgb" / f"view_{vid}.png"
        if not rgb_path.is_file() or not mask_path.is_file():
            continue

        from PIL import Image

        rgb = __import__("numpy").array(Image.open(rgb_path).convert("RGB"))
        mask = __import__("numpy").array(Image.open(mask_path).convert("L"))

        w = int(rec.get("width", proj.get("width", rgb.shape[1])))
        h = int(rec.get("height", proj.get("height", rgb.shape[0])))
        u0 = float(proj["u"])
        v0 = float(proj["v"])
        ans = rec.get("answer", [[proj["nx"], proj["ny"]]])[0]
        u1 = float(rec.get("u", float(ans[0]) * w))
        v1 = float(rec.get("v", float(ans[1]) * h))
        dist = math.hypot(u1 - u0, v1 - v0)

        parts = [f"view_{vid}", f"move={dist:.1f}px"]
        if rec.get("refined"):
            parts.append("refined")
        title = " | ".join(parts)

        out_np = _draw(rgb, mask, u0, v0, u1, v1, alpha=alpha, title=title)
        Image.fromarray(out_np).save(out_dir / f"refine_view_{vid}.png")
        n += 1

    print(f"[refine-review] {obj_dir.name}: {n} -> {out_dir}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--inputs", nargs="+")
    ap.add_argument("--views-root", type=Path, default="3DGS/test2")
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    views_root = (repo / args.views_root).resolve()
    if not views_root.is_dir():
        views_root = None

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    dirs: list[Path] = []
    if args.out:
        dirs = [args.out.resolve()]
    else:
        for pattern in args.inputs:
            p = Path(pattern)
            if "*" in str(pattern):
                dirs.extend(sorted(p.parent.glob(p.name)))
            elif p.is_dir():
                dirs.append(p.resolve())

    total = sum(export_object(d, views_root=views_root, alpha=args.alpha) for d in dirs)
    print(f"[done] {total} images")


if __name__ == "__main__":
    main()
