#!/usr/bin/env python3
"""mask 之后按登记的视角列表删 question.json。不由 e2e 自动调用。

数据在 ``view_lists.LABEL_CULL``。也可直接::

    python bridge/view_lists.py labels --out-dir training_data/data2_shaver
"""
from __future__ import annotations

import argparse
from pathlib import Path

from view_lists import LABEL_CULL, cull_question

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=Path, default=REPO / "training_data")
    ap.add_argument("--only", nargs="*", help="Limit to data2_* names")
    args = ap.parse_args()

    culls = LABEL_CULL
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
        result = cull_question(out_dir, reject)
        total_removed += len(result["removed"])
        print(
            f"{result['object']}: {result['before']} -> {result['after']} "
            f"(removed {len(result['removed'])}: {', '.join(result['removed'])})"
        )
    print(f"\n[done] total removed entries: {total_removed}")


if __name__ == "__main__":
    main()
