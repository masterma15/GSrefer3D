#!/usr/bin/env python3
"""Draw RoboRefer normalized (nx, ny) on each rendered RGB (same convention as fuse).

Uses ``CameraView`` + ``Unprojector.normalized_to_pixel`` so circles sit on the same
integer pixel used for ``depth_raw`` sampling in ``fuse_multiview.gather_candidates``.

Optional ``--fused`` tints inlier candidates green and others orange (matched by
``view_id`` + order in ``predictions`` points list).

Requires: pip install pillow

Example::

  python bridge/overlay_predictions_rgb.py \\
    --root E:/3DGS-VLM/3DGS/test2 \\
    --predictions E:/3DGS-VLM/3DGS/test2/predictions.json \\
    --fused E:/3DGS-VLM/3DGS/test2/fused.json \\
    --out-dir E:/3DGS-VLM/3DGS/test2/overlays_rgb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "bridge"))

from unproject import CameraView, Unprojector  # noqa: E402


def _rel(root: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def _inlier_keys(fused: dict[str, Any] | None) -> set[tuple[int, int]] | None:
    if fused is None:
        return None
    cands = fused.get("candidates") or []
    idxs = fused.get("inlier_indices") or []
    out: set[tuple[int, int]] = set()
    for i in idxs:
        if 0 <= i < len(cands):
            c = cands[i]
            vid = int(c["view_id"])
            pidx = int(c.get("point_idx", 0))
            out.add((vid, pidx))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay RoboRefer (nx,ny) on rgb/view_*.png (matches fuse pixel convention).",
    )
    ap.add_argument("--root", type=Path, required=True, help="Custom-view root (parent of rgb/)")
    ap.add_argument("--predictions", type=Path, required=True, help="predictions.json")
    ap.add_argument("--fused", type=Path, default=None, help="Optional fused.json for inlier coloring")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output folder (default: <root>/overlays_rgb)")
    ap.add_argument("--radius", type=int, default=14, help="Circle radius in pixels")
    args = ap.parse_args()

    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise SystemExit("install pillow: pip install pillow") from e

    root = args.root.resolve()
    out_dir = (args.out_dir or (root / "overlays_rgb")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fused: dict[str, Any] | None = None
    if args.fused is not None:
        with args.fused.open("r", encoding="utf-8") as f:
            fused = json.load(f)
    inlier_keys = _inlier_keys(fused)

    with args.predictions.open("r", encoding="utf-8") as f:
        pred = json.load(f)

    n_written = 0
    for v in pred.get("views", []):
        vid = int(v["view_id"])
        rgb_rel = v.get("rgb_path", f"rgb/view_{vid:03d}.png")
        cam_rel = v.get("camera_path", f"camera_params/view_{vid:03d}.json")
        rgb_path = _rel(root, rgb_rel)
        cam_path = _rel(root, cam_rel)
        if not rgb_path.is_file():
            print(f"[skip] view_{vid:03d}: missing rgb {rgb_path}")
            continue
        if not cam_path.is_file():
            print(f"[skip] view_{vid:03d}: missing camera {cam_path}")
            continue

        view = CameraView.from_json(cam_path)
        unp = Unprojector(view)
        im = Image.open(rgb_path).convert("RGB")
        if im.size != (view.width, view.height):
            print(
                f"[warn] view_{vid:03d}: RGB size {im.size} != camera {view.width}x{view.height}; "
                "drawing with camera convention anyway.",
                file=sys.stderr,
            )
        draw = ImageDraw.Draw(im)

        points = v.get("points") or []
        if not points:
            label = (v.get("error") or "no points")[:60]
            draw.rectangle([4, 4, min(800, view.width - 4), 28], fill=(40, 40, 40))
            draw.text((8, 8), f"v{vid} {label}", fill=(255, 200, 100))
        else:
            for pi, pt in enumerate(points):
                nx = float(pt["nx"])
                ny = float(pt["ny"])
                u, vv = unp.normalized_to_pixel(nx, ny)
                if inlier_keys is not None:
                    is_in = (vid, pi) in inlier_keys
                    fill = (80, 220, 120) if is_in else (255, 140, 40)
                else:
                    fill = (255, 60, 40)
                r = args.radius
                draw.ellipse([u - r, vv - r, u + r, vv + r], outline=(255, 255, 255), width=3, fill=fill)
                draw.text((u + r + 2, vv - 6), f"({nx:.3f},{ny:.3f})", fill=(255, 255, 0))

        out_path = out_dir / f"overlay_view_{vid:03d}.png"
        im.save(out_path)
        n_written += 1
        print(f"[ok] {out_path}")

    print(f"[summary] wrote {n_written} overlay(s) under {out_dir}")


if __name__ == "__main__":
    main()
