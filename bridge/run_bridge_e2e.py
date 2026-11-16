#!/usr/bin/env python3
"""一条命令：3DGS 模型 + 提示词 → 物体微调包 + 视锥投票点云 + SIBR。

跨环境：本进程（envGS）只 subprocess / WSL，不 import 3DGS raster 或 SAM。
``P_world`` / ``fused.json`` 只作 mask 种子，3D 结果是 ``gaussian_seg.json``。

vote 默认两次视线加权（softedge）。随后 **门控后处理**：``n_views≤30`` 且
``n_selected<7000`` 时跑 **mask-jury**（多帧高置信删杂散 + 密度补点），否则保持 softedge。
``--force-maskclose`` 绕过门控强制 mask-jury；``--no-maskclose`` 只 inject 投票结果。

带方位的指代用 ``--views`` 只送一张图，点按该像素深度反投影，不再选邻视角。
邻视角算法在 ``select_nearby_views.py``，端到端不调用，留作以后微调数据选视角。
掩码和投票仍用这颗种子落在画面内的全部视角。

产物::

    <out-dir>/                  question.json, mask/, gaussian_seg.json, fused.json
                                （另存 gaussian_seg_softedge.json；门控命中时 gaussian_seg 为 mask-jury）
    <views>/runs/<run_id>/      predictions.json, fused.json, gaussian_seg_white.json, overlays
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
from bbox_keys import OBJ_DIR_TO_BBOX_KEY  # noqa: E402


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


# 射线滤帧留下的视角少于此数时，检测和投票改用全部画面内视角。
RAY_FILTER_MIN_KEPT = 40


def mask_uses_kept_views(kept: int, *, fallback: bool, min_kept: int = RAY_FILTER_MIN_KEPT) -> bool:
    """滤帧开着时，默认检测只读留下的视角。回退开关打开且留下的太少，则不滤。"""
    if not fallback:
        return True
    return int(kept) >= int(min_kept)


def resolve_mask_use_kept(out_dir: Path, *, ray_filter: bool, fallback: bool) -> bool:
    if not ray_filter:
        return False
    if not fallback:
        return True
    report_path = out_dir / "filter_report.json"
    if not report_path.is_file():
        raise SystemExit(f"--ray-filter-fallback 需要 {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = report.get("counts") or {}
    kept = int(counts.get("kept") or 0)
    total = int(counts.get("input") or 0)
    use_kept = mask_uses_kept_views(kept, fallback=True)
    if use_kept:
        print(
            f"[e2e] ray filter kept {kept}/{total} >= {RAY_FILTER_MIN_KEPT}; "
            "mask and vote use kept views"
        )
    else:
        print(
            f"[e2e] ray filter kept {kept}/{total} < {RAY_FILTER_MIN_KEPT}; "
            "mask and vote use all in-frame views"
        )
    return use_kept


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
    from overlay_predictions_rgb import write_prediction_overlays

    write_prediction_overlays(views_root, predictions_path, fused_path, out_dir)


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
    ap.add_argument(
        "--ray-filter",
        action="store_true",
        help="After projection, drop views whose ray transmittance is below 0.45, then mask and vote only the kept views",
    )
    ap.add_argument(
        "--ray-filter-fallback",
        action="store_true",
        help=(
            "With --ray-filter: if fewer than 40 views remain, "
            "mask and vote use every in-frame view instead of the kept list"
        ),
    )
    ap.add_argument("--bbox-key", type=str, default=None, help="Key in docs/bbox_data2.json for OBB overlay/eval")
    ap.add_argument("--bbox", type=Path, default=Path("docs/bbox_data2.json"))
    ap.add_argument("--clean", action="store_true", help="After success, delete overlays / review / rejected dumps")

    ap.add_argument("--iteration", type=int, default=None, help="Passed to render.py --iteration")
    ap.add_argument("--num-custom-views", type=int, default=108, dest="num_custom_views")

    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument(
        "--views",
        type=int,
        nargs="+",
        default=None,
        help="只把这些视角送给指代。带方位时传一张，例如 --views 11。",
    )

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
    # 选框打分默认与 mask_grounding / gen_training_data 一致（整机框优先，避免零件小框）
    ap.add_argument("--contain-bonus", type=float, default=0.18)
    ap.add_argument("--compact-box-weight", type=float, default=0.10)
    ap.add_argument("--large-box-penalty", type=float, default=0.12)
    ap.add_argument("--large-box-area-start", type=float, default=0.15)
    ap.add_argument("--area-penalty", type=float, default=0.12)
    ap.add_argument("--tiny-box-penalty", type=float, default=0.40)
    ap.add_argument("--dino-score-weight", type=float, default=0.50)
    ap.add_argument("--near-score-weight", type=float, default=0.40)
    ap.add_argument("--near-dist-sigma", type=float, default=50.0)
    ap.add_argument("--min-box-area-ratio", type=float, default=0.003)
    ap.add_argument("--review-alpha", type=float, default=0.45)
    ap.add_argument(
        "--vote-frac",
        type=float,
        default=None,
        help="Use the legacy global fraction vote. Default is the two-pass pair vote (white > 0.03, final > 0.15).",
    )
    ap.add_argument(
        "--force-maskclose",
        action="store_true",
        help="vote 后绕过门控，强制跑 mask-jury（需默认 pair 投票的白集）",
    )
    ap.add_argument(
        "--no-maskclose",
        action="store_true",
        help="vote 后不做门控后处理，直接 inject softedge/投票结果",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from_stage = args.from_stage
    if args.skip_render and from_stage == "render":
        from_stage = "query"
    if args.ray_filter_fallback and not args.ray_filter:
        raise SystemExit("--ray-filter-fallback 要和 --ray-filter 一起用")
    if args.force_maskclose and args.no_maskclose:
        raise SystemExit("不要同时指定 --force-maskclose 与 --no-maskclose")

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
    scene = None
    if (
        ply_used is not None
        and ply_used.is_file()
        and (
            _should_run("fuse", from_stage)
            or _should_run("filter", from_stage)
            or _should_run("vote", from_stage)
        )
    ):
        from ray_visibility import RayOcclusionModel

        # 融合查询用 0.06 m / 0.025 m；拒帧在每次查询里另传 0.12 m / 0.02 m。
        scene = RayOcclusionModel.from_ply(ply_used, perp_radius=args.ray_perp_radius, sample_step=0.025)

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
        "query_views": args.views,
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

    query_views = args.views

    if _should_run("query", from_stage):
        _stage_query(
            views_root=views_root,
            predictions_out=predictions_path,
            url=args.url,
            prompt=args.prompt,
            retry=args.retry,
            no_depth=args.no_depth,
            views=query_views,
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
        stage_fuse(fuse_ns, ray_model=scene if args.depth_mode == "ray" else None)
        shutil.copy2(fused_run, fused_obj)
        if not args.skip_overlay:
            overlays_dir.mkdir(parents=True, exist_ok=True)
            _stage_overlay(views_root, predictions_path, fused_run, overlays_dir)
        else:
            print("[e2e] skip-overlay")

    if (
        not args.ray_filter
        and (_should_run("filter", from_stage) or _should_run("mask", from_stage))
        and (out_dir / "projections_kept.json").is_file()
    ):
        raise SystemExit(
            f"{out_dir} 含有射线滤帧留下的 projections_kept.json。"
            "未加 --ray-filter 时不会按透射率丢视角。请换一个新的 --out-dir，避免覆盖现有掩码和高斯。"
        )

    if _should_run("filter", from_stage):
        fused_path = _resolve_fused(out_dir, run_dir)
        from gen_training_data import stage_project

        stage_project(
            argparse.Namespace(fused=fused_path, views_root=views_root, out=out_dir)
        )
        proj = out_dir / "projections.json"
        if not proj.is_file() or _json_len(proj) == 0:
            raise SystemExit("project produced no in-frame views")
        if args.ray_filter:
            from filter_views_3dgs import filter_namespace, filter_views

            # 不复用融合时加载的射线模型。滤帧用自己的 0.12 m 半径和 0.02 m 步长。
            filter_views(
                filter_namespace(
                    projections=proj,
                    fused=fused_path,
                    ply=ply_used,
                    views_root=views_root,
                    out_dir=out_dir,
                    write_overlays=False,
                )
            )
            kept = out_dir / "projections_kept.json"
            if (
                not args.ray_filter_fallback
                and (not kept.is_file() or _json_len(kept) == 0)
            ):
                raise SystemExit("filter: no views kept")

    use_kept = False
    if _should_run("mask", from_stage) or _should_run("vote", from_stage):
        use_kept = resolve_mask_use_kept(
            out_dir,
            ray_filter=args.ray_filter,
            fallback=args.ray_filter_fallback,
        )

    if _should_run("mask", from_stage):
        proj_name = "projections_kept.json" if use_kept else "projections.json"
        proj = out_dir / proj_name
        if not proj.is_file():
            raise SystemExit(f"--from mask needs {proj}")
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
            contain_bonus=args.contain_bonus,
            compact_box_weight=args.compact_box_weight,
            large_box_penalty=args.large_box_penalty,
            large_box_area_start=args.large_box_area_start,
            area_penalty=args.area_penalty,
            tiny_box_penalty=args.tiny_box_penalty,
            dino_score_weight=args.dino_score_weight,
            near_score_weight=args.near_score_weight,
            near_dist_sigma=args.near_dist_sigma,
            min_box_area_ratio=args.min_box_area_ratio,
            use_kept=use_kept,
        )
        if args.use_wsl:
            run_mask_wsl(**mask_kwargs)
        else:
            run_mask_local(**mask_kwargs)
        q_path = out_dir / "question.json"
        if not q_path.is_file() or _json_len(q_path) == 0:
            raise SystemExit("mask: question.json empty")
        from make_mask_review import export_object

        export_object(out_dir, review_dir=None, views_root=views_root, alpha=args.review_alpha)
        _write_json(
            out_dir / "e2e_manifest.json",
            {
                "run_id": run_id,
                "run_dir": str(run_dir).replace("\\", "/"),
                "name": args.name,
                "object": args.object,
                "prompt": args.prompt,
                "ray_filter": bool(args.ray_filter),
                "ray_filter_fallback": bool(args.ray_filter_fallback),
                "mask_views": "kept" if use_kept else "in_frame",
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
                + (" --ray-filter" if args.ray_filter else "")
                + (" --ray-filter-fallback" if args.ray_filter_fallback else "")
            )
            _write_json(run_dir / "run_manifest.json", manifest)
            return

    if _should_run("vote", from_stage):
        q_path = out_dir / "question.json"
        if not q_path.is_file():
            raise SystemExit(f"--from vote needs {q_path}")
        from gen_training_data import stage_refine

        stage_refine(argparse.Namespace(out=out_dir, refine_mode="centroid"))
        fused_path = _resolve_fused(out_dir, run_dir)
        from frustum_segment import run_default_pair_vote, run_object

        if args.vote_frac is not None:
            run_object(
                out_dir,
                views_root=views_root,
                ply=ply_used,
                fused=fused_path,
                min_vote=0.0,
                min_vote_frac=float(args.vote_frac),
                use_t_weight=True,
                cluster_radius=None,
                z_slack=0.15,
                apply_depth_slice=False,
                out_dir=None,
                xyz=None if scene is None else scene.xyz,
                ignore_kept=bool(args.ray_filter_fallback and not use_kept),
            )
        else:
            run_default_pair_vote(
                out_dir,
                white_json=run_dir / "gaussian_seg_white.json",
                final_out_dir=out_dir,
                views_root=views_root,
                ply=ply_used,
                fused=fused_path,
                cluster_radius=None,
                z_slack=0.15,
                ignore_kept=bool(args.ray_filter_fallback and not use_kept),
                xyz=None if scene is None else scene.xyz,
            )
        seg_path = out_dir / "gaussian_seg.json"
        if not seg_path.is_file():
            raise SystemExit("vote: gaussian_seg.json missing")
        seg = json.loads(seg_path.read_text(encoding="utf-8"))
        n_sel = int(seg.get("n_selected") or 0)
        print(f"[e2e] vote selected={n_sel}")
        if n_sel <= 0:
            raise SystemExit("vote selected 0 Gaussians")

        # softedge 备份；门控命中时 gaussian_seg.json 会被 mask-jury 覆盖。
        softedge_bak = out_dir / "gaussian_seg_softedge.json"
        shutil.copy2(seg_path, softedge_bak)
        white_path = run_dir / "gaussian_seg_white.json"
        maskclose_meta: dict[str, Any] = {
            "mode": "off",
            "enabled": False,
            "postprocess": "mask_jury",
            "softedge_backup": str(softedge_bak).replace("\\", "/"),
            "white": str(white_path).replace("\\", "/"),
        }
        if args.no_maskclose:
            maskclose_meta["mode"] = "disabled"
            print("[e2e] postprocess skipped (--no-maskclose)", flush=True)
        elif args.vote_frac is not None:
            maskclose_meta["mode"] = "skipped_legacy_vote_frac"
            print("[e2e] postprocess skipped (legacy --vote-frac has no softedge white)", flush=True)
        elif not white_path.is_file():
            maskclose_meta["mode"] = "skipped_no_white"
            print(f"[e2e] postprocess skipped (missing {white_path})", flush=True)
        else:
            from postprocess_seg import mask_jury_vote, maskclose_gate_from_core

            enable, reason, n_gate_sel, n_gate_views = maskclose_gate_from_core(
                seg, object_dir=out_dir
            )
            if args.force_maskclose:
                enable = True
                reason = f"force_maskclose({reason})"
                maskclose_meta["mode"] = "force"
            else:
                maskclose_meta["mode"] = "gated"
            maskclose_meta["gate_reason"] = reason
            maskclose_meta["n_selected"] = n_gate_sel
            maskclose_meta["n_views_used"] = n_gate_views
            print(
                f"[e2e] postprocess gate enable={enable} reason={reason} "
                f"n_selected={n_gate_sel} n_views={n_gate_views}",
                flush=True,
            )
            if enable:
                out = mask_jury_vote(
                    softedge_bak,
                    white_path,
                    obj_dir=out_dir,
                    out_json=seg_path,
                )
                pp = out.get("postprocess") if isinstance(out, dict) else None
                if isinstance(pp, dict):
                    pp["gate_reason"] = reason
                    pp["gate_n_selected_core"] = n_gate_sel
                    pp["gate_n_views_used"] = n_gate_views
                    seg_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
                seg = out
                n_sel = int(seg.get("n_selected") or 0)
                maskclose_meta["enabled"] = True
                maskclose_meta["n_selected_after"] = n_sel
                print(f"[e2e] mask-jury selected={n_sel}", flush=True)
            else:
                print("[e2e] keep softedge (gate skip)", flush=True)
        manifest["maskclose"] = maskclose_meta

        bbox_key = _resolve_bbox_key(args.name, args.bbox_key)
        have_obb = _bbox_exists(bbox_path, bbox_key)
        inject_dir = model_path / "point_cloud" / f"iteration_seg_{args.name}"
        from inject_gaussian_seg import inject_gaussian_seg
        from obb_geom import load_bbox

        obb = None
        if have_obb and bbox_key is not None:
            obb = load_bbox(bbox_path)[bbox_key]["obb"]
        inject_gaussian_seg(
            ply=ply_used,
            seg=seg,
            out_iteration_dir=inject_dir,
            obb=obb,
            object_key=bbox_key,
            add_obb=have_obb,
        )
        manifest["sibr_iteration"] = f"seg_{args.name}"
        manifest["inject_dir"] = str(inject_dir).replace("\\", "/")

        if have_obb:
            from eval_gaussian_seg import eval_one, summarize

            eval_out = out_dir / "eval_gaussian_seg.json"
            result = eval_one(seg_path, bbox_path, object_key=bbox_key, views_root=views_root)
            payload = {"summary": summarize([result]), "results": [result]}
            _write_json(eval_out, payload)
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
