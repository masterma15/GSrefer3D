#!/usr/bin/env python3
"""Compare e2e predictions.json against training GT in location_point.json (2D L2)."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# slug in location_point.json image names
OBJECT_SLUGS: dict[str, str] = {
    "剃须刀": "shaver",
    "棕兔": "rabbit",
    "金毛": "golden_retriever",
    "金碗": "bowl",
    "雨伞": "umbrella",
    "玩具蛋糕": "toy_cake",
    "饼干袋": "cookie",
    "药瓶": "medicine_bottle",
    "手链": "bracelet",
    "发夹": "hair_clip",
}

IMAGE_RE = re.compile(r"^(.+)_view_(\d+)\.png$")


def load_gt(gt_path: Path, slug: str) -> dict[int, tuple[float, float]]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    out: dict[int, tuple[float, float]] = {}
    for rec in data:
        m = IMAGE_RE.match(rec["image"])
        if not m or m.group(1) != slug:
            continue
        vid = int(m.group(2))
        ans = rec["conversations"][-1]["value"]
        # [(0.7214, 0.5814)]
        inner = ans.strip().removeprefix("[").removesuffix("]").strip()
        if inner.startswith("("):
            parts = inner.strip("()").split(",")
            nx, ny = float(parts[0]), float(parts[1])
            out[vid] = (nx, ny)
    return out


def load_pred(run_dir: Path) -> dict[int, tuple[float, float]]:
    pred = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    out: dict[int, tuple[float, float]] = {}
    for v in pred.get("views", []):
        if not v.get("parse_ok") or not v.get("points"):
            continue
        p = v["points"][0]
        out[int(v["view_id"])] = (float(p["nx"]), float(p["ny"]))
    return out


def metrics(gt: dict[int, tuple[float, float]], pred: dict[int, tuple[float, float]]) -> dict:
    errs: list[float] = []
    for vid, g in gt.items():
        if vid not in pred:
            continue
        px, py = pred[vid]
        gx, gy = g
        errs.append(math.hypot(px - gx, py - gy))
    if not errs:
        return {"n": 0, "median_l2": None, "mean_l2": None, "pct_lt_0.05": None, "pct_lt_0.10": None}
    errs.sort()
    n = len(errs)
    med = errs[n // 2] if n % 2 else (errs[n // 2 - 1] + errs[n // 2]) / 2
    return {
        "n": n,
        "median_l2": round(med, 4),
        "mean_l2": round(sum(errs) / n, 4),
        "pct_lt_0.05": round(100 * sum(e < 0.05 for e in errs) / n, 1),
        "pct_lt_0.10": round(100 * sum(e < 0.10 for e in errs) / n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path("training_data/data2_sft/location_point.json"))
    ap.add_argument("--runs-root", type=Path, default=Path("3DGS/test2/runs"))
    ap.add_argument("--runs", type=str, nargs="*", help="run_id=物体:组别 entries via JSON file instead")
    ap.add_argument("--table-json", type=Path, help="JSON list of {object, group, run_id}")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Default table from EXPERIMENT_DATA_INDEX §2.3
    rows = [
        ("剃须刀", "Base", "20260519_170540_4c3b9a32"),
        ("剃须刀", "LoRA", "20260519_143457_4c3b9a32"),
        ("棕兔", "Base", "20260519_171359_147bac82"),
        ("棕兔", "LoRA", "20260519_144845_147bac82"),
        ("金毛", "Base", "20260519_172627_f8dbfcc3"),
        ("金毛", "LoRA", "20260519_154013_f8dbfcc3"),
        ("金碗", "Base", "20260519_173958_8d83a715"),
        ("金碗", "LoRA", "20260519_154649_8d83a715"),
        ("雨伞", "Base", "20260519_174835_d7bab60f"),
        ("雨伞", "LoRA", "20260519_160219_d7bab60f"),
        ("玩具蛋糕", "Base", "20260519_175733_7dd80c38"),
        ("玩具蛋糕", "LoRA", "20260519_161222_7dd80c38"),
        ("饼干袋", "Base", "20260519_180800_e51c780a"),
        ("饼干袋", "LoRA", "20260519_162019_e51c780a"),
        ("药瓶", "Base", "20260519_182106_04149b86"),
        ("药瓶", "LoRA", "20260519_163004_04149b86"),
        ("手链", "Base", "20260520_001018_cb2e562f"),
        ("手链", "LoRA", "20260519_163546_cb2e562f"),
        ("发夹", "Base", "20260519_183039_65bf02a5"),
        ("发夹", "LoRA", "20260519_164312_65bf02a5"),
        ("胶带 hold-out", "Base", "20260519_000313_6c883d56"),
        ("胶带 hold-out", "LoRA", "20260519_132142_6c883d56"),
    ]
    if args.table_json and args.table_json.is_file():
        rows = [(r["object"], r["group"], r["run_id"]) for r in json.loads(args.table_json.read_text(encoding="utf-8"))]

    gt_cache: dict[str, dict[int, tuple[float, float]]] = {}
    results = []
    for obj, group, run_id in rows:
        slug = OBJECT_SLUGS.get(obj.replace(" hold-out", ""))
        run_dir = args.runs_root / run_id
        row = {"object": obj, "group": group, "run_id": run_id, "slug": slug}
        if slug is None:
            row.update({"n_gt_views": 0, "note": "无 SFT GT（hold-out）"})
            if run_dir.is_dir():
                fused = run_dir / "fused.json"
                if fused.is_file():
                    f = json.loads(fused.read_text(encoding="utf-8"))
                    row["support"] = f.get("support")
            results.append(row)
            continue
        if slug not in gt_cache:
            gt_cache[slug] = load_gt(args.gt, slug)
        gt = gt_cache[slug]
        pred = load_pred(run_dir) if run_dir.is_dir() else {}
        m = metrics(gt, pred)
        row.update(m)
        row["n_gt_views"] = len(gt)
        if (run_dir / "fused.json").is_file():
            row["support"] = json.loads((run_dir / "fused.json").read_text(encoding="utf-8")).get("support")
        results.append(row)

    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
