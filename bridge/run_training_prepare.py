#!/usr/bin/env python3
"""One entry: filter (3DGS ray) -> mask (DINO+SAM2) -> review PNGs for manual QA.

Windows (envGS)::

    python bridge/run_training_prepare.py --object data2_bracelet --stage all

Mask needs WSL roborefer + CUDA; use ``--use-wsl`` (default on win32)::

    python bridge/run_training_prepare.py --object data2_bracelet --stage all --use-wsl

Stages: filter | mask | review | all
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FUSED_RUNS: dict[str, str] = {
    "data2_shaver": "20260516_111848_4c3b9a32",
    "data2_rabbit": "20260516_111341_147bac82",
    "data2_golden_retriever": "20260516_111616_f8dbfcc3",
    "data2_umbrella": "20260516_111101_d7bab60f",
    "data2_hair_clip": "20260516_112347_65bf02a5",
    "data2_medicine_bottle": "20260516_112749_04149b86",
    "data2_bracelet": "20260516_113151_cb2e562f",
    "data2_cookie": "20260516_113618_e51c780a",
    "data2_bowl": "20260516_114029_8d83a715",
    "data2_toy_cake": "20260516_114443_7dd80c38",
}

MANUAL_REJECT: dict[str, list[str]] = {
    "data2_rabbit": ["024", "045"],
    "data2_golden_retriever": ["025", "051"],
    "data2_umbrella": ["022", "025", "051"],
}

FILTER_PRESET: dict[str, str] = {
    "data2_medicine_bottle": "relaxed_plus",
}

OBJECT_SPECS: dict[str, dict[str, str]] = {
    "data2_bowl": {
        "prompt": "Please point to the gold-colored bowl on the desk.",
        "object": "gold bowl",
    },
    "data2_bracelet": {
        "prompt": "Please point to the white beaded bracelet on the desk.",
        "object": "white beaded bracelet",
    },
    "data2_cookie": {
        "prompt": "Please point to the green square cookie package on the desk.",
        "object": "green cookie package",
    },
    "data2_golden_retriever": {
        "prompt": "Please point to the light yellow plush golden retriever toy.",
        "object": "plush golden retriever",
    },
    "data2_hair_clip": {
        "prompt": "Please point to the purple hair clip on the desk.",
        "object": "purple hair clip",
    },
    "data2_medicine_bottle": {
        "prompt": "Please point to the medicine bottle on the desk.",
        "object": "medicine bottle",
    },
    "data2_rabbit": {
        "prompt": "Please point to the brown plush rabbit.",
        "object": "plush rabbit",
    },
    "data2_shaver": {
        "prompt": "Please point to the electric shaver on the desk.",
        "object": "electric shaver",
    },
    "data2_toy_cake": {
        "prompt": "Please point to the toy cake held by the brown plush rabbit.",
        # DINO only: avoid "rabbit/plush" — triggers whole-doll boxes; target small cake
        "object": "small decorative toy cake",
    },
    "data2_umbrella": {
        "prompt": "Please point to the black and red umbrella on the desk.",
        "object": "black and red folded umbrella",
    },
}


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or REPO))


def stage_filter(obj: str, args: argparse.Namespace) -> None:
    if obj in MANUAL_REJECT and not args.ray_only:
        _run(
            [
                sys.executable,
                str(REPO / "bridge" / "apply_filter_batch.py"),
                "--manual-only",
                "--only",
                obj,
            ]
        )
        return

    cmd = [
        sys.executable,
        str(REPO / "bridge" / "apply_filter_batch.py"),
        "--ray-only",
        "--only",
        obj,
        "--views-root",
        str(args.views_root),
        "--ply",
        str(args.ply),
    ]
    if args.filter_preset:
        pass  # per-object preset in apply_filter_batch.RAY_FILTER_OVERRIDES
    _run(cmd)


def _wsl_path(p: Path | str) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:]
    return s


def _mask_cmd(obj: str, args: argparse.Namespace, *, wsl: bool) -> list[str]:
    spec = OBJECT_SPECS[obj]
    out = _wsl_path(REPO / "training_data" / obj) if wsl else str(REPO / "training_data" / obj)
    py = "python" if wsl else sys.executable
    script = (
        f"{_wsl_path(REPO / 'bridge/gen_training_data.py')}"
        if wsl
        else str(REPO / "bridge" / "gen_training_data.py")
    )
    return [
        py,
        script,
        "--stage",
        "mask",
        "--mask-mode",
        "grounding",
        "--out",
        out,
        "--prompt",
        spec["prompt"],
        "--object",
        spec["object"],
        "--sam2-checkpoint",
        _wsl_path(REPO / args.sam2_checkpoint) if wsl else str((REPO / args.sam2_checkpoint).resolve()),
        "--sam2-config",
        args.sam2_config,
        "--grounding-config",
        _wsl_path(REPO / args.grounding_config) if wsl else str((REPO / args.grounding_config).resolve()),
        "--grounding-checkpoint",
        _wsl_path(REPO / args.grounding_checkpoint)
        if wsl
        else str((REPO / args.grounding_checkpoint).resolve()),
        "--grounding-box-threshold",
        str(args.grounding_box_threshold),
        "--grounding-text-threshold",
        str(args.grounding_text_threshold),
        "--anchor-box-radius",
        str(args.anchor_box_radius),
        "--max-box-area-ratio",
        str(args.max_box_area_ratio),
    ]


def _patch_kept_paths_cmd(*, wsl: bool) -> list[str]:
    script = (
        f"{_wsl_path(REPO / 'bridge/patch_kept_rgb_paths.py')}"
        if wsl
        else str(REPO / "bridge" / "patch_kept_rgb_paths.py")
    )
    py = "python3" if wsl else sys.executable
    return [py, script]


def stage_mask(obj: str, args: argparse.Namespace) -> None:
    kept = REPO / "training_data" / obj / "projections_kept.json"
    if args.use_wsl:
        wsl_repo = "/mnt/e/3DGS-VLM"
        mask_cmd = _mask_cmd(obj, args, wsl=True)
        patch_cmd = _patch_kept_paths_cmd(wsl=True) + [
            f"{wsl_repo}/training_data/{obj}/projections_kept.json",
        ]
        inner = " && ".join(
            [
                f"cd {wsl_repo}",
                "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh",
                "conda activate roborefer",
                f"export PYTHONPATH={wsl_repo}/GroundingDINO:$PYTHONPATH",
                "export HF_HUB_OFFLINE=1",
                " ".join(patch_cmd),
                " ".join(f'"{a}"' if " " in a else a for a in mask_cmd),
            ]
        )
        _run(["wsl", "bash", "-lc", inner])
    else:
        from bridge.patch_kept_rgb_paths import patch_file

        if kept.is_file():
            patch_file(kept)
        _run(_mask_cmd(obj, args, wsl=False))


def stage_review(obj: str, args: argparse.Namespace) -> None:
    out = REPO / "training_data" / obj
    _run(
        [
            sys.executable,
            str(REPO / "bridge" / "make_mask_review.py"),
            "--out",
            str(out),
            "--views-root",
            str((REPO / args.views_root).resolve()),
            "--alpha",
            str(args.review_alpha),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="filter -> mask -> review for one data2 object")
    ap.add_argument("--object", required=True, choices=sorted(OBJECT_SPECS.keys()))
    ap.add_argument("--stage", choices=("filter", "mask", "review", "all"), default="all")
    ap.add_argument("--views-root", default="3DGS/test2")
    ap.add_argument(
        "--ply",
        default="3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply",
    )
    ap.add_argument("--filter-preset", default="", help="Override ray filter preset")
    ap.add_argument("--ray-only", action="store_true")
    ap.add_argument(
        "--use-wsl",
        action=argparse.BooleanOptionalAction,
        default=platform.system().lower().startswith("win"),
        help="Run mask in WSL (default True on Windows)",
    )
    ap.add_argument(
        "--sam2-checkpoint",
        default="weights/sam2.1_hiera_large.pt",
    )
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument(
        "--grounding-config",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    ap.add_argument("--grounding-checkpoint", default="weights/groundingdino_swint_ogc.pth")
    ap.add_argument("--grounding-box-threshold", type=float, default=0.10)
    ap.add_argument("--grounding-text-threshold", type=float, default=0.12)
    ap.add_argument("--anchor-box-radius", type=int, default=64)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.35)
    ap.add_argument("--review-alpha", type=float, default=0.45)
    args = ap.parse_args()

    obj = args.object
    stages = ["filter", "mask", "review"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"\n========== {obj} :: {st} ==========", flush=True)
        if st == "filter":
            stage_filter(obj, args)
        elif st == "mask":
            stage_mask(obj, args)
        else:
            stage_review(obj, args)
    print(f"\n[done] {obj} stages={stages}")
    if args.stage in ("all", "review"):
        print(f"  review -> training_data/{obj}/review/")
    print("  After manual cull: edit question.json, then refine + export")


if __name__ == "__main__":
    main()
