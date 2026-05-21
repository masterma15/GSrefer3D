#!/usr/bin/env python3
"""Remove rejected views from question.json + mask/review PNGs (manual QA cull)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# From 人工复筛.txt (view_id zero-padded to 3 digits)
DEFAULT_CULLS: dict[str, list[str]] = {
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


def cull_object(out_dir: Path, reject_ids: set[str]) -> dict:
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

    return {
        "object": out_dir.name,
        "before": before,
        "after": len(kept),
        "removed": removed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=Path, default=REPO / "training_data")
    ap.add_argument("--only", nargs="*", help="Limit to data2_* names")
    args = ap.parse_args()

    culls = DEFAULT_CULLS
    if args.only:
        culls = {k: v for k, v in culls.items() if k in args.only}

    total_removed = 0
    for obj, ids in sorted(culls.items()):
        out_dir = args.training_root / obj
        if not out_dir.is_dir():
            print(f"[skip] {obj}: dir missing")
            continue
        reject = {v.zfill(3) for v in ids}
        if not reject:
            print(f"[skip] {obj}: no rejects listed")
            continue
        r = cull_object(out_dir, reject)
        total_removed += len(r["removed"])
        print(
            f"{r['object']}: {r['before']} -> {r['after']} "
            f"(removed {len(r['removed'])}: {', '.join(r['removed'])})"
        )
    print(f"\n[done] total removed entries: {total_removed}")


if __name__ == "__main__":
    main()
