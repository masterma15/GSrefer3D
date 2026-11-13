#!/usr/bin/env python3
"""显式编辑视角列表。e2e 不按物体名调用这里。

``kept``：射线拒帧之后，从 projections_kept.json 去掉列出的视角。
``labels``：mask 之后，从 question.json 和 mask/review 图去掉列出的视角。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# 投票前从保留集去掉的视角（大物体人工列表）
KEPT_REJECT: dict[str, list[str]] = {
    "data2_rabbit": ["024", "045"],
    "data2_golden_retriever": ["025", "051"],
    "data2_umbrella": ["022", "025", "051"],
}

# mask 之后从训练标签去掉的视角
LABEL_CULL: dict[str, list[str]] = {
    "data2_bowl": [],
    "data2_bracelet": ["007", "032", "044", "070"],
    "data2_cookie": [],
    "data2_golden_retriever": ["017", "022", "037", "057", "063", "069"],
    "data2_hair_clip": ["012", "057"],
    "data2_medicine_bottle": [],
    "data2_rabbit": ["007", "016", "023", "037", "050", "057", "066"],
    "data2_shaver": ["009", "036", "037"],
    "data2_toy_cake": ["001", "032", "034", "042"],
    "data2_umbrella": ["052", "058", "063", "065"],
}


def reject_kept_views(out_dir: Path, reject_ids: set[str]) -> dict[str, Any]:
    """从 projections_kept.json 去掉这些 view id。"""
    kept_path = out_dir / "projections_kept.json"
    if not kept_path.is_file():
        raise FileNotFoundError(kept_path)
    records = json.loads(kept_path.read_text(encoding="utf-8"))
    kept: list[dict] = []
    removed: list[str] = []
    for rec in records:
        vid = str(rec["view_id"]).zfill(3)
        if vid in reject_ids:
            removed.append(vid)
        else:
            kept.append(rec)
    kept_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    return {"object": out_dir.name, "before": len(records), "after": len(kept), "removed": removed}


def cull_question(out_dir: Path, reject_ids: set[str]) -> dict[str, Any]:
    """从 question.json 以及对应 mask / review 图去掉这些 view id。"""
    q_path = out_dir / "question.json"
    if not q_path.is_file():
        raise FileNotFoundError(q_path)
    recs = json.loads(q_path.read_text(encoding="utf-8"))
    before = len(recs)
    kept: list[dict] = []
    removed: list[str] = []
    for rec in recs:
        vid = str(rec.get("view_id", rec.get("id", ""))).zfill(3)
        if vid in reject_ids:
            removed.append(vid)
            for rel in (
                f"mask/view_{vid}.png",
                f"review/review_view_{vid}.png",
                f"review_refine/refine_view_{vid}.png",
            ):
                p = out_dir / rel
                if p.is_file():
                    p.unlink()
        else:
            kept.append(rec)
    for i, rec in enumerate(kept):
        rec["id"] = i
    q_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    skips_path = out_dir / "mask_skips.json"
    if skips_path.is_file() and reject_ids:
        skips = json.loads(skips_path.read_text(encoding="utf-8"))
        skips = [s for s in skips if str(s.get("view_id", "")).zfill(3) not in reject_ids]
        skips_path.write_text(json.dumps(skips, indent=2), encoding="utf-8")
    return {"object": out_dir.name, "before": before, "after": len(kept), "removed": removed}


def _ids_for(table: dict[str, list[str]], name: str, views: list[str] | None) -> set[str]:
    if views:
        return {v.zfill(3) for v in views}
    return {v.zfill(3) for v in table.get(name, [])}


def main() -> None:
    ap = argparse.ArgumentParser(description="显式删视角：kept 或 labels。不由 e2e 自动调用。")
    ap.add_argument("kind", choices=("kept", "labels"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--views", nargs="*", default=None, help="不写则用该目录名上登记的列表")
    args = ap.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO / args.out_dir).resolve()
    table = KEPT_REJECT if args.kind == "kept" else LABEL_CULL
    reject = _ids_for(table, out_dir.name, args.views)
    if not reject:
        raise SystemExit(f"{out_dir.name}: 没有要删除的 view id，请传 --views")
    if args.kind == "kept":
        result = reject_kept_views(out_dir, reject)
    else:
        result = cull_question(out_dir, reject)
    print(
        f"{result['object']}: {result['before']} -> {result['after']} "
        f"(removed {', '.join(result['removed'])})"
    )


if __name__ == "__main__":
    main()
