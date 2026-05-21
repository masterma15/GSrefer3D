#!/usr/bin/env python3
"""Export data2_* packs into RoboRefer ``spatialdataset`` JSON (no 357GB RefSpatial download).

Reads each ``training_data/data2_<object>/question.json`` (or ``projections.json``)
and writes a merged training tree::

    <out>/
      image/          # RGB copies (unique filenames)
      depth/          # depth PNG from --views-root (optional)
      location_point.json

Each JSON record matches ``LazySupervisedSpatialDataset`` expectations::

    {
      "image": "bowl_view_012.png",
      "depth": "bowl_view_012.png",   # omit field for RGB-only
      "conversations": [
        {"from": "human", "value": "<prompt> <suffix>"},
        {"from": "gpt",   "value": "[(0.76, 0.57)]"}
      ]
    }

Training code injects ``<image>`` / ``<depth>`` tokens; do not add them here.

Example (Windows envGS)::

    python bridge/export_spatial_train.py ^
        --inputs training_data/data2_* ^
        --views-root 3DGS/test2 ^
        --out training_data/data2_sft

Registered in ``RoboRefer-main/llava/data/datasets_mixture.py`` as ``data2_location``
(folder on disk remains ``training_data/data2_sft/``). Train with::

    --data_mixture data2_location
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Align with RefSpatial-Expand-Bench Location + RoboRefer API (use_api.py).
DEFAULT_SUFFIX = (
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
    "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
    "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
)

# Per-object RoboRefer prompts (match e2e run_manifest.json)
OBJECT_PROMPTS: dict[str, str] = {
    "data2_bowl": "Please point to the gold-colored bowl on the desk.",
    "data2_bracelet": "Please point to the white beaded bracelet on the desk.",
    "data2_cookie": "Please point to the green square cookie package on the desk.",
    "data2_golden_retriever": "Please point to the light yellow plush golden retriever toy.",
    "data2_hair_clip": "Please point to the purple hair clip on the desk.",
    "data2_medicine_bottle": "Please point to the medicine bottle on the desk.",
    "data2_rabbit": "Please point to the brown plush rabbit.",
    "data2_shaver": "Please point to the electric shaver on the desk.",
    "data2_toy_cake": "Please point to the toy cake held by the brown plush rabbit.",
    "data2_umbrella": "Please point to the black and red umbrella on the desk.",
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "obj"


def _format_answer(nx: float, ny: float) -> str:
    return f"[({nx:.4f}, {ny:.4f})]"


def _load_records(src_dir: Path, prefer: str, *, default_prompt: str = "") -> list[dict]:
    q_path = src_dir / "question.json"
    kept_path = src_dir / "projections_kept.json"
    p_path = src_dir / "projections.json"

    if prefer == "question" or (prefer == "auto" and q_path.exists()):
        if not q_path.exists():
            return []
        return json.loads(q_path.read_text(encoding="utf-8"))

    proj_path: Path | None = None
    if prefer == "kept" or (prefer == "auto" and kept_path.is_file()):
        proj_path = kept_path if kept_path.is_file() else None
    if proj_path is None and p_path.is_file():
        proj_path = p_path
    if proj_path is None:
        return []

    proj = json.loads(proj_path.read_text(encoding="utf-8"))
    prompt = default_prompt or OBJECT_PROMPTS.get(src_dir.name, "")
    out = []
    for rec in proj:
        out.append({
            "view_id": rec["view_id"],
            "nx": rec["nx"],
            "ny": rec["ny"],
            "rgb_path": rec.get("rgb_path", ""),
            "prompt": rec.get("prompt") or prompt,
            "suffix": rec.get("suffix") or DEFAULT_SUFFIX,
        })
    return out


def export(args: argparse.Namespace) -> None:
    out = Path(args.out)
    img_out = out / "image"
    depth_out = out / "depth"
    img_out.mkdir(parents=True, exist_ok=True)
    if not args.rgb_only:
        depth_out.mkdir(parents=True, exist_ok=True)

    views_root = Path(args.views_root) if args.views_root else None
    train_records: list[dict] = []
    global_id = 0

    input_dirs: list[Path] = []
    for pattern in args.inputs:
        p = Path(pattern)
        if "*" in pattern or "?" in pattern:
            input_dirs.extend(sorted(p.parent.glob(p.name)))
        elif p.is_dir():
            input_dirs.append(p)
        else:
            print(f"[skip] not found: {pattern}")
    input_dirs = sorted(set(input_dirs))
    for src_dir in input_dirs:
        if not src_dir.is_dir():
            print(f"[skip] not a directory: {src_dir}")
            continue
        slug = _slug(src_dir.name.replace("data2_", "", 1))
        obj_prompt = OBJECT_PROMPTS.get(src_dir.name, args.prompt or "")
        records = _load_records(src_dir, args.source, default_prompt=obj_prompt)
        if not records:
            print(f"[skip] no question.json / projections.json in {src_dir}")
            continue

        prompt_default = args.prompt or ""
        if not prompt_default:
            # try prompt.txt in parent runs — not required
            pass

        for rec in records:
            view_id = str(rec.get("view_id", rec.get("id", global_id))).zfill(3)
            prompt = rec.get("prompt") or prompt_default
            if not prompt:
                print(f"[skip] {src_dir.name} view_{view_id}: missing prompt")
                continue

            ans = rec.get("answer")
            if ans and isinstance(ans, list) and len(ans) > 0:
                nx, ny = float(ans[0][0]), float(ans[0][1])
            elif "nx" in rec and "ny" in rec:
                nx, ny = float(rec["nx"]), float(rec["ny"])
            else:
                print(f"[skip] {src_dir.name} view_{view_id}: no coordinates")
                continue

            # Always Bench/API suffix (ignore per-record suffix in question.json).
            human = f"{prompt.rstrip()} {DEFAULT_SUFFIX}".strip()

            rgb_rel = rec.get("rgb_path", f"image/view_{view_id}.png")
            src_rgb = Path(rgb_rel)
            if not src_rgb.is_file():
                src_rgb = src_dir / rgb_rel
            if not src_rgb.is_file() and views_root:
                src_rgb = views_root / "rgb" / f"view_{view_id}.png"
            if not src_rgb.is_file():
                print(f"[skip] rgb missing: {rgb_rel}")
                continue

            out_name = f"{slug}_view_{view_id}.png"
            shutil.copy2(src_rgb, img_out / out_name)

            entry: dict = {
                "id": global_id,
                "image": out_name,
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": _format_answer(nx, ny)},
                ],
            }

            if not args.rgb_only and views_root:
                depth_src = views_root / "depth" / f"view_{view_id}.png"
                if depth_src.is_file():
                    shutil.copy2(depth_src, depth_out / out_name)
                    entry["depth"] = out_name
                elif not args.allow_missing_depth:
                    print(f"[skip] depth missing for view_{view_id}")
                    continue

            train_records.append(entry)
            global_id += 1

    json_path = out / args.json_name
    json_path.write_text(
        json.dumps(train_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[export] {len(train_records)} samples -> {json_path}")
    print(f"         image/: {img_out}")
    if not args.rgb_only:
        print(f"         depth/: {depth_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export data2 packs to RoboRefer spatialdataset JSON.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Source dirs, e.g. training_data/data2_bowl training_data/data2_shaver",
    )
    ap.add_argument("--out", required=True, help="Output root, e.g. training_data/data2_sft")
    ap.add_argument("--views-root", default="3DGS/test2", help="RGB/depth source if paths in JSON are stale")
    ap.add_argument(
        "--source",
        choices=["auto", "question", "projections", "kept"],
        default="auto",
        help="auto: question.json if present else projections_kept.json else projections.json",
    )
    ap.add_argument("--rgb-only", action="store_true", help="Omit depth field (register without depth_path)")
    ap.add_argument("--allow-missing-depth", action="store_true", help="Keep sample even if depth PNG missing")
    ap.add_argument("--json-name", default="location_point.json")
    ap.add_argument("--prompt", help="Fallback prompt if record has none (projections-only mode)")
    args = ap.parse_args()
    export(args)


if __name__ == "__main__":
    main()
