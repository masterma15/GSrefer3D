#!/usr/bin/env python3
"""一条命令：3DGS 模型 + 提示词 → 物体微调包 + 视锥投票点云 + SIBR。

跨环境：本进程（envGS）只 subprocess / WSL，不 import 3DGS raster 或 SAM。
``P_world`` / ``fused.json`` 只作 mask 种子，3D 结果是 ``gaussian_seg.json``。

产物::

    <out-dir>/                  question.json, mask/, gaussian_seg.json, fused.json
    <views>/runs/<run_id>/      predictions.json, fused.json, overlays, run_manifest.json
    <model>/point_cloud/iteration_seg_<name>/
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "bridge"

if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from e2e_stages import (  # noqa: E402
    STAGES,
    guess_ply,
    preflight_api,
    preflight_wsl,
    require_views,
    run_mask_local,
    run_mask_wsl,
    stage_fuse,
    stage_render,
)
from frustum_segment import OBJ_DIR_TO_BBOX_KEY  # noqa: E402


def _abs(p: Path) -> Path:
    p = p.expanduser()
    return p.resolve() if p.is_absolute() else (REPO / p).resolve()


def _safe_run_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    s = s.strip("._")
    return s[:120] if s else "run"


def _default_run_id(prompt: str, run_name: str | None) -> str:
    if run_name:
        return _safe_run_name(run_name)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{ts}_{h}"


def _allocate_run_dir(views_root: Path, base_run_id: str) -> tuple[str, Path]:
    n = 0
    while True:
        rid = base_run_id if n == 0 else f"{base_run_id}_{n}"
        run_dir = views_root / "runs" / rid
        if not run_dir.exists():
            return rid, run_dir
        n += 1


def _subprocess_rc(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[e2e] $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(cwd) if cwd else None)
    if rc != 0:
        raise SystemExit(f"command failed with exit code {rc}")


def _should_run(stage: str, from_stage: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(from_stage)


def _count_query_ok(predictions_path: Path) -> tuple[int, int]:
    data = json.loads(predictions_path.read_text(encoding="utf-8"))
    views = data.get("views") or []
    ok = sum(1 for v in views if v.get("parse_ok"))
    return ok, len(views)


def _json_len(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return len(data)
    return 0


def _resolve_bbox_key(name: str, bbox_key: str | None) -> str | None:
    if bbox_key:
        return bbox_key
    return OBJ_DIR_TO_BBOX_KEY.get(name)


def _bbox_exists(bbox_path: Path, key: str | None) -> bool:
    if not key or not bbox_path.is_file():
        return False
    objects = json.loads(bbox_path.read_text(encoding="utf-8"))
    spec = objects.get(key) if isinstance(objects, dict) else None
    if spec is None and isinstance(objects, dict):
        spec = (objects.get("objects") or {}).get(key)
    return isinstance(spec, dict) and "obb" in spec


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_run_dir(
    *,
    views_root: Path,
    out_dir: Path,
    prompt: str,
    run_name: str | None,
    run_dir_arg: Path | None,
    from_stage: str,
) -> tuple[str, Path]:
    if run_dir_arg is not None:
        run_dir = _abs(run_dir_arg)
        return run_dir.name, run_dir

    pointer = out_dir / "e2e_manifest.json"
    if from_stage in ("fuse", "filter", "mask", "vote") and pointer.is_file():
        meta = json.loads(pointer.read_text(encoding="utf-8"))
        hinted = Path(meta.get("run_dir") or "")
        if hinted.is_dir():
            return str(meta.get("run_id") or hinted.name), hinted

    if run_name:
        cand = views_root / "runs" / _safe_run_name(run_name)
        if cand.is_dir() and from_stage != "render":
            return cand.name, cand

    rid, run_dir = _allocate_run_dir(views_root, _default_run_id(prompt, run_name))
    return rid, run_dir


def _resolve_fused(out_dir: Path, run_dir: Path) -> Path:
    for p in (out_dir / "fused.json", run_dir / "fused.json"):
        if p.is_file():
            return p
    raise SystemExit(f"fused.json not found in {out_dir} or {run_dir}")


def _clean_debug(run_dir: Path, out_dir: Path) -> None:
    for d in (
        run_dir / "overlays_rgb",
        out_dir / "review",
        out_dir / "filter_overlays",
    ):
        if d.is_dir():
            shutil.rmtree(d)
            print(f"[clean] removed {d}")
    rejected = out_dir / "projections_rejected.json"
    if rejected.is_file():
        rejected.unlink()
        print(f"[clean] removed {rejected}")


def _stage_query(
    *,
    views_root: Path,
    predictions_out: Path,
    url: str,
    prompt: str,
    retry: int,
    no_depth: bool,
    views: list[int] | None,
) -> None:
    cmd: list[str] = [
        sys.executable,
        str(BRIDGE / "roborefer_client.py"),
        "--root",
        str(views_root),
        "--url",
        url,
        "--prompt",
        prompt,
        "--output",
        str(predictions_out),
        "--retry",
        str(retry),
    ]
    if no_depth:
        cmd.append("--no-depth")
    if views:
        cmd.extend(["--views", *[str(v) for v in views]])
    _subprocess_rc(cmd, cwd=REPO)
    ok, n = _count_query_ok(predictions_out)
    print(f"[e2e] query ok={ok}/{n}")
    if ok == 0:
        raise SystemExit("query: every view failed")


def _stage_overlay(views_root: Path, predictions_path: Path, fused_path: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(BRIDGE / "overlay_predictions_rgb.py"),
        "--root",
        str(views_root),
        "--predictions",
        str(predictions_path),
        "--fused",
        str(fused_path),
        "--out-dir",
        str(out_dir),
    ]
    _subprocess_rc(cmd, cwd=REPO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="3DGS model + prompt → referring Gaussians + per-object SFT pack.",
    )
    ap.add_argument("--model-path", type=Path, required=True, help="3DGS trained model dir (same as render.py -m)")
    ap.add_argument(
        "--custom-views-out",
        type=Path,
        default=Path("3DGS/test2"),
        help="Multi-view root (rgb/, camera_params/, depth_raw/). Overridable; default 3DGS/test2.",
    )
    ap.add_argument("--prompt", type=str, required=True, help="RoboRefer instruction")
    ap.add_argument("--object", type=str, required=True, help="Grounding DINO short caption")
    ap.add_argument("--name", type=str, required=True, help="Object pack slug, e.g. data2_shaver")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Object pack dir (default: training_data/<name>)",
    )
    ap.add_argument("--url", type=str, default="http://127.0.0.1:25547")
    ap.add_argument("--run-name", type=str, default=None, help="Reuse/set run_id under views/runs/")
    ap.add_argument("--run-dir", type=Path, default=None, help="Existing run dir (predictions/fused)")

    ap.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGES,
        default="render",
        help="Resume from this stage. --skip-render is an alias for --from query.",
    )
    ap.add_argument("--skip-render", action="store_true", help="Same as --from query")
    ap.add_argument("--pause-after-mask", action="store_true", help="Stop after mask+review; continue with --from vote")
    ap.add_argument("--bbox-key", type=str, default=None, help="Key in docs/bbox_data2.json for OBB overlay/eval")
    ap.add_argument("--bbox", type=Path, default=Path("docs/bbox_data2.json"))
    ap.add_argument("--clean", action="store_true", help="After success, delete overlays / review / rejected dumps")

    ap.add_argument("--iteration", type=int, default=None, help="Passed to render.py --iteration")
    ap.add_argument("--num-custom-views", type=int, default=36, dest="num_custom_views")

    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--views", type=int, nargs="+", default=None)

    ap.add_argument("--inlier-radius", type=float, default=5.0)
    ap.add_argument("--min-inv", type=float, default=1e-3)
    ap.add_argument("--refine-k", type=float, default=1.35)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--ply", type=Path, default=None, help="point_cloud.ply; default: latest numeric iteration")
    ap.add_argument("--depth-dir", type=Path, default=None)
    ap.add_argument("--depth-mode", choices=("invdepth", "ray"), default="ray")
    ap.add_argument("--ray-perp-radius", type=float, default=0.06)
    ap.add_argument("--ray-pixel-radius", type=float, default=6.0)
    ap.add_argument("--ray-z-band", type=float, default=2.0)
    ap.add_argument("--ray-min-alpha", type=float, default=0.45)

    ap.add_argument("--skip-overlay", action="store_true")
    ap.add_argument("--filter-preset", default="default")
    ap.add_argument(
        "--use-wsl",
        action=argparse.BooleanOptionalAction,
        default=platform.system().lower().startswith("win"),
        help="Run DINO+SAM in WSL roborefer (default True on Windows)",
    )
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path("weights/sam2.1_hiera_large.pt"))
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument(
        "--grounding-config",
        type=Path,
        default=Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    )
    ap.add_argument("--grounding-checkpoint", type=Path, default=Path("weights/groundingdino_swint_ogc.pth"))
    ap.add_argument("--grounding-box-threshold", type=float, default=0.10)
    ap.add_argument("--grounding-text-threshold", type=float, default=0.12)
    ap.add_argument("--anchor-box-radius", type=int, default=64)
    ap.add_argument("--max-box-area-ratio", type=float, default=0.35)
    ap.add_argument("--review-alpha", type=float, default=0.45)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from_stage = args.from_stage
    if args.skip_render and from_stage == "render":
        from_stage = "query"

    model_path = _abs(args.model_path)
    views_root = _abs(args.custom_views_out)
    out_dir = _abs(args.out_dir) if args.out_dir is not None else _abs(Path("training_data") / args.name)
    bbox_path = _abs(args.bbox)
    sam2_ckpt = _abs(args.sam2_checkpoint)
    gdino_ckpt = _abs(args.grounding_checkpoint)
    gdino_cfg = _abs(args.grounding_config)

    if from_stage == "render":
        views_root.mkdir(parents=True, exist_ok=True)
    else:
        if not views_root.is_dir():
            raise SystemExit(f"custom-views-out is not a directory: {views_root}")
        require_views(views_root)

    if _should_run("query", from_stage):
        preflight_api(args.url)
    if _should_run("mask", from_stage) and args.use_wsl:
        preflight_wsl(
            sam2_checkpoint=sam2_ckpt,
            grounding_checkpoint=gdino_ckpt,
            grounding_config=gdino_cfg,
        )

    ply_used = _abs(args.ply) if args.ply is not None else guess_ply(model_path)
    if ply_used is None or not ply_used.is_file():
        if _should_run("fuse", from_stage) or _should_run("filter", from_stage) or _should_run("vote", from_stage):
            raise SystemExit("--ply missing and no numeric point_cloud/iteration_*/point_cloud.ply under --model-path")

    run_id, run_dir = _resolve_run_dir(
        views_root=views_root,
        out_dir=out_dir,
        prompt=args.prompt,
        run_name=args.run_name,
        run_dir_arg=args.run_dir,
        from_stage=from_stage,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.json"
    fused_run = run_dir / "fused.json"
    fused_obj = out_dir / "fused.json"
    overlays_dir = run_dir / "overlays_rgb"

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "from_stage": from_stage,
        "prompt": args.prompt,
        "object": args.object,
        "name": args.name,
        "url": args.url,
        "model_path": str(model_path).replace("\\", "/"),
        "views_root": str(views_root).replace("\\", "/"),
        "run_dir": str(run_dir).replace("\\", "/"),
        "out_dir": str(out_dir).replace("\\", "/"),
        "fuse_ply": str(ply_used).replace("\\", "/") if ply_used else None,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    (run_dir / "prompt.txt").write_text(
        "# 溯源：与本 run 的 RoboRefer --prompt 完全一致（UTF-8）\n"
        f"# run_id: {run_id}\n"
        f"# name: {args.name}\n"
        f"# object: {args.object}\n"
        f"# url: {args.url}\n"
        "# ----------------------------------------\n"
        f"{args.prompt}\n",
        encoding="utf-8",
    )

    if _should_run("render", from_stage):
        rargs = argparse.Namespace(
            model_path=model_path,
            custom_views_out=views_root,
            iteration=args.iteration,
            num_custom_views=args.num_custom_views,
        )
        stage_render(rargs)
        require_views(views_root)

    if _should_run("query", from_stage):
        _stage_query(
            views_root=views_root,
            predictions_out=predictions_path,
            url=args.url,
            prompt=args.prompt,
            retry=args.retry,
            no_depth=args.no_depth,
            views=args.views,
        )
    elif _should_run("fuse", from_stage) and not predictions_path.is_file():
        raise SystemExit(f"--from {from_stage} needs {predictions_path} (pass --run-dir)")

    if _should_run("fuse", from_stage):
        fuse_ns = argparse.Namespace(
            predictions=predictions_path,
            inlier_radius=args.inlier_radius,
            min_inv=args.min_inv,
            no_refine=args.no_refine,
            refine_k=args.refine_k,
            ply=ply_used,
            fused_output=fused_run,
            depth_dir=_abs(args.depth_dir) if args.depth_dir else None,
            depth_mode=args.depth_mode,
            ray_perp_radius=args.ray_perp_radius,
            ray_pixel_radius=args.ray_pixel_radius,
            ray_z_band=args.ray_z_band,
            ray_min_alpha=args.ray_min_alpha,
        )
        stage_fuse(fuse_ns)
        shutil.copy2(fused_run, fused_obj)
        if not args.skip_overlay:
            overlays_dir.mkdir(parents=True, exist_ok=True)
            _stage_overlay(views_root, predictions_path, fused_run, overlays_dir)
        else:
            print("[e2e] skip-overlay")

    if _should_run("filter", from_stage):
        fused_path = _resolve_fused(out_dir, run_dir)
        _subprocess_rc(
            [
                sys.executable,
                str(BRIDGE / "gen_training_data.py"),
                "--stage",
                "project",
                "--fused",
                str(fused_path),
                "--views-root",
                str(views_root),
                "--out",
                str(out_dir),
            ],
            cwd=REPO,
        )
        proj = out_dir / "projections.json"
        if not proj.is_file() or _json_len(proj) == 0:
            raise SystemExit("project produced no in-frame views")
        filter_cmd = [
            sys.executable,
            str(BRIDGE / "filter_views_3dgs.py"),
            "--projections",
            str(proj),
            "--fused",
            str(fused_path),
            "--ply",
            str(ply_used),
            "--views-root",
            str(views_root),
            "--out-dir",
            str(out_dir),
            "--filter-preset",
            args.filter_preset,
        ]
        if args.skip_overlay:
            filter_cmd.append("--no-overlays")
        _subprocess_rc(filter_cmd, cwd=REPO)
        kept = out_dir / "projections_kept.json"
        if not kept.is_file() or _json_len(kept) == 0:
            raise SystemExit("filter: no views kept")

    if _should_run("mask", from_stage):
        kept = out_dir / "projections_kept.json"
        if not kept.is_file():
            raise SystemExit(f"--from mask needs {kept}")
        mask_kwargs = dict(
            out_dir=out_dir,
            prompt=args.prompt,
            object_name=args.object,
            sam2_checkpoint=sam2_ckpt,
            sam2_config=args.sam2_config,
            grounding_config=gdino_cfg,
            grounding_checkpoint=gdino_ckpt,
            grounding_box_threshold=args.grounding_box_threshold,
            grounding_text_threshold=args.grounding_text_threshold,
            anchor_box_radius=args.anchor_box_radius,
            max_box_area_ratio=args.max_box_area_ratio,
        )
        if args.use_wsl:
            run_mask_wsl(**mask_kwargs)
        else:
            run_mask_local(**mask_kwargs)
        q_path = out_dir / "question.json"
        if not q_path.is_file() or _json_len(q_path) == 0:
            raise SystemExit("mask: question.json empty")
        _subprocess_rc(
            [
                sys.executable,
                str(BRIDGE / "make_mask_review.py"),
                "--out",
                str(out_dir),
                "--views-root",
                str(views_root),
                "--alpha",
                str(args.review_alpha),
            ],
            cwd=REPO,
        )
        _write_json(
            out_dir / "e2e_manifest.json",
            {
                "run_id": run_id,
                "run_dir": str(run_dir).replace("\\", "/"),
                "name": args.name,
                "object": args.object,
                "prompt": args.prompt,
            },
        )
        if args.pause_after_mask:
            print("\n[e2e] paused after mask. Review:")
            print(f"  {out_dir / 'review'}")
            print("Continue:")
            print(
                f"  python bridge/run_bridge_e2e.py --model-path {model_path} "
                f"--custom-views-out {views_root} --prompt {args.prompt!r} "
                f"--object {args.object!r} --name {args.name} --out-dir {out_dir} "
                f"--from vote --run-dir {run_dir}"
            )
            _write_json(run_dir / "run_manifest.json", manifest)
            return

    if _should_run("vote", from_stage):
        q_path = out_dir / "question.json"
        if not q_path.is_file():
            raise SystemExit(f"--from vote needs {q_path}")
        _subprocess_rc(
            [
                sys.executable,
                str(BRIDGE / "gen_training_data.py"),
                "--stage",
                "refine",
                "--out",
                str(out_dir),
            ],
            cwd=REPO,
        )
        fused_path = _resolve_fused(out_dir, run_dir)
        _subprocess_rc(
            [
                sys.executable,
                str(BRIDGE / "frustum_segment.py"),
                "--obj-dir",
                str(out_dir),
                "--views-root",
                str(views_root),
                "--ply",
                str(ply_used),
                "--fused",
                str(fused_path),
            ],
            cwd=REPO,
        )
        seg_path = out_dir / "gaussian_seg.json"
        if not seg_path.is_file():
            raise SystemExit("vote: gaussian_seg.json missing")
        seg = json.loads(seg_path.read_text(encoding="utf-8"))
        n_sel = int(seg.get("n_selected") or 0)
        print(f"[e2e] vote selected={n_sel}")
        if n_sel <= 0:
            raise SystemExit("vote selected 0 Gaussians")

        bbox_key = _resolve_bbox_key(args.name, args.bbox_key)
        have_obb = _bbox_exists(bbox_path, bbox_key)
        inject_dir = model_path / "point_cloud" / f"iteration_seg_{args.name}"
        inject_cmd = [
            sys.executable,
            str(BRIDGE / "inject_gaussian_seg.py"),
            "--seg",
            str(seg_path),
            "--ply",
            str(ply_used),
            "--model-path",
            str(model_path),
            "--out-iteration-dir",
            str(inject_dir),
        ]
        if have_obb:
            inject_cmd.extend(["--bbox", str(bbox_path), "--object", bbox_key])
        else:
            inject_cmd.append("--no-obb")
        _subprocess_rc(inject_cmd, cwd=REPO)
        manifest["sibr_iteration"] = f"seg_{args.name}"
        manifest["inject_dir"] = str(inject_dir).replace("\\", "/")

        if have_obb:
            eval_out = out_dir / "eval_gaussian_seg.json"
            _subprocess_rc(
                [
                    sys.executable,
                    str(BRIDGE / "eval_gaussian_seg.py"),
                    "--obj-dir",
                    str(out_dir),
                    "--bbox",
                    str(bbox_path),
                    "--object-key",
                    bbox_key,
                    "--views-root",
                    str(views_root),
                    "--output",
                    str(eval_out),
                ],
                cwd=REPO,
            )
            manifest["eval"] = str(eval_out).replace("\\", "/")

    if args.clean:
        _clean_debug(run_dir, out_dir)

    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(
        out_dir / "e2e_manifest.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir).replace("\\", "/"),
            "name": args.name,
            "object": args.object,
            "prompt": args.prompt,
            "sibr_iteration": manifest.get("sibr_iteration"),
        },
    )

    print("\n[e2e] done.")
    print(f"  out_dir = {out_dir}")
    print(f"  run_dir = {run_dir}")
    print(f"  SIBR: --iteration seg_{args.name}")
    print(f"  (from viewers/bin; model {model_path})")


if __name__ == "__main__":
    main()
