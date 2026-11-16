#!/usr/bin/env python3
"""对 training_data 里 data3 softedge 核批量跑门控后处理（现行：mask-jury）。

发现顺序（每个 ``data3_*`` 物体目录，跳过 ``_s0xx`` 变体目录名）：
1. ``<obj>/试验_pair_softedge/{gaussian_seg,gaussian_seg_white}.json``
2. 否则 ``by_object/<obj>/试验_pair_softedge/...``（``--obj-dir`` 仍用 ``training_data/<obj>``）

门控：``n_views ≤ 30`` 且 ``n_selected < 7000`` 才开启。
命中 → ``试验_gated_maskclose/gaussian_seg.json``（内容为 mask-jury）+ 可选注入；
未命中 → 同目录写出 softedge 副本（mode=gated_skip），默认不注入。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from postprocess_seg import (  # noqa: E402
    MASKCLOSE_FEW_VIEWS_MAX,
    MASKCLOSE_FEW_VIEWS_N_SELECTED_MAX,
    maskclose_gate_from_core,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_variant_dir(name: str) -> bool:
    """跳过 ``data3_foo_s015`` 这类阈值试验目录，避免和主物体重复。"""
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return False
    suf = parts[1]
    return len(suf) >= 2 and suf[0] == "s" and suf[1:].isdigit()


def discover_jobs(training_root: Path) -> list[dict[str, Path | str]]:
    jobs: list[dict[str, Path | str]] = []
    by_object = training_root / "by_object"
    for obj_dir in sorted(training_root.iterdir()):
        if not obj_dir.is_dir() or obj_dir.name == "by_object":
            continue
        if not obj_dir.name.startswith("data3_"):
            continue
        if _is_variant_dir(obj_dir.name):
            continue
        if not (obj_dir / "question.json").is_file() or not (obj_dir / "mask").is_dir():
            continue
        core = obj_dir / "试验_pair_softedge" / "gaussian_seg.json"
        white = obj_dir / "试验_pair_softedge" / "gaussian_seg_white.json"
        if not (core.is_file() and white.is_file()):
            alt = by_object / obj_dir.name / "试验_pair_softedge"
            core = alt / "gaussian_seg.json"
            white = alt / "gaussian_seg_white.json"
        if not (core.is_file() and white.is_file()):
            continue
        jobs.append(
            {
                "name": obj_dir.name,
                "obj_dir": obj_dir,
                "core": core,
                "white": white,
                "out": obj_dir / "试验_gated_maskclose" / "gaussian_seg.json",
            }
        )
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="批量门控后处理（softedge → 少视角且选点偏少时 mask-jury）"
    )
    ap.add_argument(
        "--training-root",
        type=Path,
        default=_REPO / "training_data",
        help="training_data 根目录",
    )
    ap.add_argument("--inject", action="store_true", help="门控命中时注入 SIBR")
    ap.add_argument(
        "--inject-on-skip",
        action="store_true",
        help="门控跳过时也注入（内容为 softedge；iteration 名仍用 _maskclose 后缀需自辨）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印门控结果，不跑后处理",
    )
    ap.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="只跑这些物体目录名；默认全部发现到的 data3",
    )
    args = ap.parse_args()
    jobs = discover_jobs(args.training_root.resolve())
    if args.names:
        want = set(args.names)
        jobs = [j for j in jobs if j["name"] in want]
    if not jobs:
        print("[batch] no jobs", flush=True)
        sys.exit(1)

    print(
        f"[batch] gate: n_views<={MASKCLOSE_FEW_VIEWS_MAX} and "
        f"n_selected<{MASKCLOSE_FEW_VIEWS_N_SELECTED_MAX} → mask-jury",
        flush=True,
    )
    py = sys.executable
    script = Path(__file__).resolve().parent / "postprocess_seg.py"
    ok = fail = enable_n = skip_n = 0
    for job in jobs:
        name = str(job["name"])
        core = Path(job["core"])
        payload = _load_json(core)
        if not isinstance(payload, dict):
            print(f"[batch] FAIL {name}: core not object", flush=True)
            fail += 1
            continue
        enable, reason, n_sel, n_views = maskclose_gate_from_core(
            payload, object_dir=job["obj_dir"]
        )
        tag = "ENABLE" if enable else "SKIP"
        print(
            f"[batch] {tag} {name} reason={reason} n_selected={n_sel} n_views={n_views}",
            flush=True,
        )
        if args.dry_run:
            if enable:
                enable_n += 1
            else:
                skip_n += 1
            ok += 1
            continue
        cmd = [
            py,
            str(script),
            "--gated-maskclose",
            "--core",
            str(core),
            "--white",
            str(job["white"]),
            "--obj-dir",
            str(job["obj_dir"]),
            "--out",
            str(job["out"]),
        ]
        if args.inject and (enable or args.inject_on_skip):
            cmd.extend(["--inject", "--iteration", f"seg_{name}_maskclose"])
            if args.inject_on_skip and not enable:
                cmd.append("--inject-on-skip")
        try:
            subprocess.run(cmd, check=True)
            ok += 1
            if enable:
                enable_n += 1
            else:
                skip_n += 1
        except subprocess.CalledProcessError as exc:
            print(f"[batch] FAIL {name}: exit={exc.returncode}", flush=True)
            fail += 1

    print(
        f"[batch] done ok={ok} fail={fail} enable={enable_n} skip={skip_n} total={len(jobs)}",
        flush=True,
    )
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
