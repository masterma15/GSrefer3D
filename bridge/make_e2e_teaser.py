#!/usr/bin/env python3
"""Build demo/teaser_base_lora_<object>.png from two e2e runs' overlays_rgb/.

Layout matches ``demo/teaser_base_lora_tape.png``: *n* rows (views) × 2 columns
(Base | LoRA). Reads ``overlay_view_XXX.png`` produced by ``run_bridge_e2e.py``.

Requires: pip install pillow

Examples::

  # Hold-out tape (same runs as README §3)
  python bridge/make_e2e_teaser.py --preset tape --output demo/teaser_base_lora_tape.png

  # In-domain umbrella — auto-pick 3 views with largest LoRA 2D gain vs GT
  python bridge/make_e2e_teaser.py --preset umbrella --output demo/teaser_base_lora_umbrella.png

  # Explicit runs + views
  python bridge/make_e2e_teaser.py \\
    --base-run 20260519_170540_4c3b9a32 \\
    --lora-run 20260519_143457_4c3b9a32 \\
    --slug shaver --views 18 33 55 \\
    --output demo/teaser_base_lora_shaver.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# suffix run_id pairs from docs/RESULTS.md §2–§3 (Base, LoRA, GT slug or None)
PRESETS: dict[str, tuple[str, str, str | None]] = {
    "tape": ("20260519_000313_6c883d56", "20260519_132142_6c883d56", None),
    "shaver": ("20260519_170540_4c3b9a32", "20260519_143457_4c3b9a32", "shaver"),
    "rabbit": ("20260519_171359_147bac82", "20260519_144845_147bac82", "rabbit"),
    "umbrella": ("20260519_174835_d7bab60f", "20260519_160219_d7bab60f", "umbrella"),
    "golden_retriever": ("20260519_172627_f8dbfcc3", "20260519_154013_f8dbfcc3", "golden_retriever"),
}

IMAGE_RE = re.compile(r"^(.+)_view_(\d+)\.png$")


def _resolve_run(runs_root: Path, run_id: str) -> Path:
    p = Path(run_id)
    if p.is_dir():
        return p.resolve()
    full = runs_root / run_id
    if not full.is_dir():
        raise SystemExit(f"[error] run dir not found: {full}")
    return full.resolve()


def _overlay_path(run_dir: Path, view_id: int) -> Path:
    p = run_dir / "overlays_rgb" / f"overlay_view_{view_id:03d}.png"
    if not p.is_file():
        raise SystemExit(f"[error] missing overlay: {p}")
    return p


def _load_prompt(run_dir: Path) -> str:
    p = run_dir / "prompt.txt"
    if not p.is_file():
        return ""
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    return lines[-1].strip() if lines else ""


def _load_gt(gt_path: Path, slug: str) -> dict[int, tuple[float, float]]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    out: dict[int, tuple[float, float]] = {}
    for rec in data:
        m = IMAGE_RE.match(rec["image"])
        if not m or m.group(1) != slug:
            continue
        vid = int(m.group(2))
        ans = rec["conversations"][-1]["value"]
        inner = ans.strip().removeprefix("[").removesuffix("]").strip()
        if inner.startswith("("):
            parts = inner.strip("()").split(",")
            out[vid] = (float(parts[0]), float(parts[1]))
    return out


def _load_pred(run_dir: Path) -> dict[int, tuple[float, float]]:
    pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    out: dict[int, tuple[float, float]] = {}
    for v in pred.get("views", []):
        if not v.get("parse_ok") or not v.get("points"):
            continue
        p = v["points"][0]
        out[int(v["view_id"])] = (float(p["nx"]), float(p["ny"]))
    return out


def _l2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pick_views_gt(
    base_dir: Path,
    lora_dir: Path,
    slug: str,
    gt_path: Path,
    n: int,
) -> list[int]:
    gt = _load_gt(gt_path, slug)
    base_pred = _load_pred(base_dir)
    lora_pred = _load_pred(lora_dir)
    scored: list[tuple[float, float, int]] = []
    for vid, g in gt.items():
        if vid not in base_pred or vid not in lora_pred:
            continue
        gain = _l2(base_pred[vid], g) - _l2(lora_pred[vid], g)
        scored.append((gain, _l2(base_pred[vid], g), vid))
    if not scored:
        raise SystemExit(f"[error] no overlapping GT views for slug={slug!r}")
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [vid for _, _, vid in scored[:n]]


def _inlier_view_ids(run_dir: Path) -> list[int]:
    fused_path = run_dir / "fused.json"
    if not fused_path.is_file():
        return []
    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    cands = fused.get("candidates") or []
    idxs = fused.get("inlier_indices") or []
    out: list[int] = []
    for i in idxs:
        if 0 <= i < len(cands):
            out.append(int(cands[i]["view_id"]))
    return sorted(set(out))


def _pick_views_holdout(base_dir: Path, lora_dir: Path, n: int) -> list[int]:
    ids = sorted(set(_inlier_view_ids(base_dir)) | set(_inlier_view_ids(lora_dir)))
    if not ids:
        ids = list(range(72))
    if len(ids) <= n:
        return ids[:n]
    step = len(ids) / n
    return [ids[int(round(i * step + step / 2 - 1))] for i in range(n)]


def _fit_panel(im, max_w: int, max_h: int):
    from PIL import Image

    w, h = im.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return im


def _draw_header(width: int, title: str, col_labels: tuple[str, str]):
    from PIL import Image, ImageDraw, ImageFont

    bar_h = 52
    img = Image.new("RGB", (width, bar_h), (28, 28, 28))
    draw = ImageDraw.Draw(img)
    draw.text((8, 6), title[:120], fill=(255, 255, 255))
    half = width // 2
    draw.line([(half, 0), (half, bar_h)], fill=(80, 80, 80), width=2)
    draw.text((half // 2 - 20, 28), col_labels[0], fill=(255, 180, 80))
    draw.text((half + half // 2 - 20, 28), col_labels[1], fill=(80, 220, 120))
    return img


def build_teaser(
    *,
    base_dir: Path,
    lora_dir: Path,
    view_ids: list[int],
    output: Path,
    panel_max_w: int,
    panel_max_h: int,
    title: str | None,
) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("install pillow: pip install pillow") from e

    base_panels = [_fit_panel(Image.open(_overlay_path(base_dir, v)), panel_max_w, panel_max_h) for v in view_ids]
    lora_panels = [_fit_panel(Image.open(_overlay_path(lora_dir, v)), panel_max_w, panel_max_h) for v in view_ids]

    pw = max(p.width for p in base_panels + lora_panels)
    ph = max(p.height for p in base_panels + lora_panels)
    gap = 4

    grid_w = pw * 2 + gap
    grid_h = ph * len(view_ids) + gap * (len(view_ids) - 1)
    grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))

    for row, (bp, lp) in enumerate(zip(base_panels, lora_panels)):
        y = row * (ph + gap)
        grid.paste(bp, (0, y))
        grid.paste(lp, (pw + gap, y))

    prompt = title or _load_prompt(lora_dir) or _load_prompt(base_dir)
    header = _draw_header(grid_w, prompt, ("Base", "LoRA"))
    out = Image.new("RGB", (grid_w, header.height + grid_h), (255, 255, 255))
    out.paste(header, (0, 0))
    out.paste(grid, (0, header.height))

    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)
    print(f"[ok] {output}  views={view_ids}  base={base_dir.name}  lora={lora_dir.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collage Base vs LoRA overlays_rgb into a README teaser PNG.")
    ap.add_argument("--runs-root", type=Path, default=_REPO / "3DGS" / "test2" / "runs")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="Known Base/LoRA run pair")
    ap.add_argument("--base-run", type=str, help="Base run_id or run directory")
    ap.add_argument("--lora-run", type=str, help="LoRA run_id or run directory")
    ap.add_argument("--views", type=int, nargs="+", help="View ids (default: auto-pick 3)")
    ap.add_argument("--num-views", type=int, default=3, help="Auto-pick count when --views omitted")
    ap.add_argument("--slug", type=str, help="GT object slug (shaver, umbrella, …); inferred from --preset")
    ap.add_argument("--gt", type=Path, default=_REPO / "training_data" / "data2_sft" / "location_point.json")
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO / "demo" / "teaser_base_lora_umbrella.png",
        help="Output PNG (default: demo/teaser_base_lora_umbrella.png)",
    )
    ap.add_argument("--title", type=str, default=None, help="Header prompt line (default: prompt.txt)")
    ap.add_argument("--panel-max-w", type=int, default=640)
    ap.add_argument("--panel-max-h", type=int, default=480)
    args = ap.parse_args()

    slug: str | None = args.slug
    base_id = args.base_run
    lora_id = args.lora_run
    if args.preset:
        base_id, lora_id, preset_slug = PRESETS[args.preset]
        if slug is None:
            slug = preset_slug

    if not base_id or not lora_id:
        raise SystemExit("provide --preset or both --base-run and --lora-run")

    runs_root = args.runs_root.resolve()
    base_dir = _resolve_run(runs_root, base_id)
    lora_dir = _resolve_run(runs_root, lora_id)

    if args.views:
        view_ids = list(args.views)
    elif slug and args.gt.is_file():
        view_ids = _pick_views_gt(base_dir, lora_dir, slug, args.gt.resolve(), args.num_views)
        print(f"[auto] GT slug={slug!r} -> views {view_ids} (largest LoRA gain)")
    else:
        view_ids = _pick_views_holdout(base_dir, lora_dir, args.num_views)
        print(f"[auto] hold-out / no GT -> views {view_ids} (spread over fused inliers)")

    build_teaser(
        base_dir=base_dir,
        lora_dir=lora_dir,
        view_ids=view_ids,
        output=args.output.resolve(),
        panel_max_w=args.panel_max_w,
        panel_max_h=args.panel_max_h,
        title=args.title,
    )


if __name__ == "__main__":
    main()
