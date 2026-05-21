#!/usr/bin/env python3
"""Rewrite Windows E:\\ paths in projections_kept.json to /mnt/e/... for WSL mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def patch_file(path: Path) -> int:
    recs = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for r in recs:
        rp = r.get("rgb_path", "")
        if len(rp) >= 3 and rp[1] == ":" and rp[0].isalpha():
            drive = rp[0].lower()
            rest = rp[2:].lstrip("\\/")
            r["rgb_path"] = f"/mnt/{drive}/{rest.replace(chr(92), '/')}"
            changed += 1
    if changed:
        path.write_text(json.dumps(recs, indent=2), encoding="utf-8")
    print(f"[patch] {path} records={len(recs)} changed={changed}")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kept_json", type=Path)
    args = ap.parse_args()
    patch_file(args.kept_json.resolve())


if __name__ == "__main__":
    main()
