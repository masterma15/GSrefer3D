#!/usr/bin/env python3
"""One-shot bridge orchestration (B–F): render → RoboRefer → fuse → marker → overlays → model copy + inject.

Prerequisite (manual): start RoboRefer ``api.py`` in WSL/Linux; this script only probes ``--url`` before batch query.

Artifacts for each run live under::

    <custom-views-out>/runs/<run_id>/{predictions.json,fused.json,marker.ply,overlays_rgb/,run_manifest.json,prompt.txt}

A full copy of the trained model (for SIBR) is written to::

    <model-path.parent>/<model-path.name>_runs/<run_id>/

with markers injected into ``point_cloud/iteration_35000/point_cloud.ply`` (or ``iteration_36000`` if snap base would collide).

Run from repo root (or pass absolute paths), in the same conda env as ``render.py`` / ``fuse`` (e.g. envGS)::

  cd E:/GSrefer3D
  python bridge/run_bridge_e2e.py \\
    --model-path 3DGS/gaussian-splatting/output/data2 \\
    --custom-views-out 3DGS/test2 \\
    --prompt "Please point to the brown stuffed rabbit." \\
    --snap \\
    --ply 3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "bridge"

# Fixed per user preference (SIBR); if snap ply lives here too, use _INJECT_COLLISION_FALLBACK.
_INJECT_ITER_DIR = "iteration_35000"
_INJECT_COLLISION_FALLBACK = "iteration_36000"


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


def _allocate_run_paths(
    views_root: Path,
    model_path: Path,
    base_run_id: str,
) -> tuple[str, Path, Path]:
    n = 0
    while True:
        rid = base_run_id if n == 0 else f"{base_run_id}_{n}"
        run_dir = views_root / "runs" / rid
        model_copy = model_path.parent / f"{model_path.name}_runs" / rid
        if not run_dir.exists() and not model_copy.exists():
            return rid, run_dir, model_copy
        n += 1


def _probe_api(url: str, timeout: float = 5.0) -> bool:
    try:
        import requests

        u = url.rstrip("/")
        requests.get(u, timeout=timeout)
        return True
    except Exception:
        return False


def _subprocess_rc(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[e2e] $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(cwd) if cwd else None)
    if rc != 0:
        raise SystemExit(f"command failed with exit code {rc}")


def _stage_roborefer_client(
    *,
    views_root: Path,
    predictions_out: Path,
    url: str,
    prompt: str,
    retry: int,
    no_depth: bool,
    views: list[int] | None,
    visibility_check: bool,
    visibility_permissive: bool,
    visibility_strict: bool,
    visibility_base_url: str | None,
    visibility_model: str | None,
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
    if visibility_check:
        cmd.append("--visibility-check")
    if visibility_permissive:
        cmd.append("--visibility-permissive")
    if visibility_strict:
        cmd.append("--visibility-strict")
    if visibility_base_url:
        cmd.extend(["--visibility-base-url", visibility_base_url])
    if visibility_model:
        cmd.extend(["--visibility-model", visibility_model])
    _subprocess_rc(cmd, cwd=REPO)


def _stage_overlay(
    *,
    views_root: Path,
    predictions_path: Path,
    fused_path: Path,
    out_dir: Path,
) -> None:
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


def _stage_inject(
    *,
    base_ply_in_copy: Path,
    fused_json: Path,
    out_iteration_dir: Path,
    all_candidates: bool,
    extra_args: list[str] | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(BRIDGE / "inject_gaussian_markers.py"),
        "--ply",
        str(base_ply_in_copy),
        "--fused-json",
        str(fused_json),
        "--out-iteration-dir",
        str(out_iteration_dir),
    ]
    if all_candidates:
        cmd.append("--all-candidates")
    if extra_args:
        cmd.extend(extra_args)
    _subprocess_rc(cmd, cwd=REPO)


def _print_sibr_footer(*, model_copy_dir: Path, inject_iteration_folder: str) -> None:
    """Print copy-paste PowerShell lines for SIBR using paths from this run."""
    viewers_bin = (REPO / "3DGS" / "gaussian-splatting" / "viewers" / "bin").resolve()
    model_m = model_copy_dir.resolve()
    vb = str(viewers_bin)
    mm = str(model_m)
    iter_suffix = inject_iteration_folder.removeprefix("iteration_")
    sep = "-" * 68
    print(f"\n{sep}")
    print("下一步：用 SIBR 查看刚生成的模型副本（PowerShell 中复制执行以下两行）")
    print(sep)
    print(f'Set-Location "{vb}"')
    print(f'.\\SIBR_gaussianViewer_app.exe -m "{mm}"')
    print(
        f"\n在 SIBR 界面中将 point_cloud 的 iteration 选为 {iter_suffix} "
        f"（目录名 {inject_iteration_folder}），即可看到注入的红色标记点。\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bridge B–F in one command: render, RoboRefer batch, fuse, marker PLY, overlays, "
        "optional full model copy + marker inject for SIBR.",
    )
    ap.add_argument("--model-path", type=Path, required=True, help="3DGS trained model dir (same as render.py -m)")
    ap.add_argument(
        "--custom-views-out",
        type=Path,
        default=Path("3DGS/test2"),
        help="Multi-view root (contains rgb/, camera_params/, depth_raw/ …).",
    )
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--url", type=str, default="http://127.0.0.1:25547")
    ap.add_argument("--run-name", type=str, default=None, help="If set, used as run_id (sanitized). Else timestamp+hash.")

    ap.add_argument("--skip-render", action="store_true", help="Assume rgb/camera_params already exist under views root.")
    ap.add_argument("--iteration", type=int, default=None, help="Passed to render.py --iteration")
    ap.add_argument("--num-custom-views", type=int, default=36, dest="num_custom_views")

    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--views", type=int, nargs="+", default=None)

    ap.add_argument("--visibility-check", action="store_true")
    ap.add_argument("--visibility-permissive", action="store_true")
    ap.add_argument("--visibility-strict", action="store_true")
    ap.add_argument("--visibility-base-url", type=str, default=None)
    ap.add_argument("--visibility-model", type=str, default=None)

    ap.add_argument("--inlier-radius", type=float, default=1.0)
    ap.add_argument("--min-inv", type=float, default=1e-3)
    ap.add_argument("--refine-k", type=float, default=1.75)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--ply", type=Path, default=None, help="point_cloud.ply used for snap-to-gaussian (fuse + inject base).")
    ap.add_argument("--snap", action="store_true", help="If --ply omitted, use latest point_cloud under --model-path.")
    ap.add_argument("--exclude", type=int, nargs="+", default=None)
    ap.add_argument("--depth-dir", type=Path, default=None, help="Override depth_raw dir (e.g. views_root/depth_raw_dav2).")

    ap.add_argument("--skip-overlay", action="store_true")
    ap.add_argument("--no-model-bundle", action="store_true",
                    help="Skip copying the full model + inject (only write run_dir artifacts).")

    ap.add_argument("--inject-all-candidates", action="store_true",
                    help="Pass --all-candidates to inject_gaussian_markers.py (debug overlay in Gaussians).")

    ap.add_argument(
        "--inject-surface-push",
        type=float,
        default=0.0,
        metavar="M",
        help=(
            "Meters (scene units): nudge injected marker center out of dense plush interior "
            "(0 default; try 0.03–0.1 if markers are invisible in SIBR)."
        ),
    )
    ap.add_argument("--inject-surface-push-k", type=int, default=48, help="k neighbours for inject surface push.")
    ap.add_argument(
        "--inject-marker-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Extra world translation for marker center (scene units), e.g. 0 0.06 0 to lift along +Y.",
    )
    ap.add_argument("--inject-log-scale", type=float, default=None,
                    help="Override marker --log-scale (less negative = larger splats).")
    ap.add_argument("--inject-marker-count", type=int, default=None, help="Override number of marker Gaussians.")
    ap.add_argument("--inject-opacity", type=float, default=None, help="Override --opacity-sigmoid for markers.")
    ap.add_argument("--inject-jitter", type=float, default=None, help="Override marker position jitter.")

    args = ap.parse_args()

    if args.visibility_permissive and args.visibility_strict:
        ap.error("--visibility-permissive and --visibility-strict are mutually exclusive.")

    model_path = _abs(args.model_path)
    views_root = _abs(args.custom_views_out)
    if not views_root.is_dir():
        raise SystemExit(f"custom-views-out is not a directory: {views_root}")

    base_id = _default_run_id(args.prompt, args.run_name)
    run_id, run_dir, model_copy_dir = _allocate_run_paths(views_root, model_path, base_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.json"
    fused_path = run_dir / "fused.json"
    marker_path = run_dir / "marker.ply"
    overlays_dir = run_dir / "overlays_rgb"

    ply_used = _abs(args.ply) if args.ply is not None else None
    if ply_used is None and not args.snap:
        ap.error("Provide --ply and/or --snap (use --snap to auto-pick latest point_cloud.ply under --model-path).")
    if ply_used is None:
        sys.path.insert(0, str(BRIDGE))
        from pipeline import _guess_ply  # noqa: WPS433

        ply_used = _guess_ply(model_path)
        if ply_used is None or not ply_used.is_file():
            raise SystemExit(f"--snap set but no point_cloud.ply found under {model_path}")

    try:
        rel_ply = ply_used.relative_to(model_path)
    except ValueError as e:
        raise SystemExit(
            f"--ply must live under --model-path for bundle inject ({ply_used} vs {model_path})"
        ) from e

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "prompt": args.prompt,
        "url": args.url,
        "model_path": str(model_path).replace("\\", "/"),
        "views_root": str(views_root).replace("\\", "/"),
        "run_dir": str(run_dir).replace("\\", "/"),
        "snap_ply": str(ply_used).replace("\\", "/"),
        "model_copy_dir": None if args.no_model_bundle else str(model_copy_dir).replace("\\", "/"),
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    trace = run_dir / "prompt.txt"
    trace.write_text(
        "# 溯源：与本 run 的 RoboRefer --prompt 完全一致（UTF-8）\n"
        f"# run_id: {run_id}\n"
        f"# url: {args.url}\n"
        "# ----------------------------------------\n"
        f"{args.prompt}\n",
        encoding="utf-8",
    )

    # --- B render ---
    if not args.skip_render:
        sys.path.insert(0, str(BRIDGE))
        from pipeline import stage_render  # noqa: WPS433

        rargs = argparse.Namespace(
            model_path=model_path,
            custom_views_out=views_root,
            iteration=args.iteration,
            num_custom_views=args.num_custom_views,
        )
        stage_render(rargs)
    else:
        print("[e2e] skip-render: using existing views under", views_root)

    # --- C RoboRefer ---
    if not _probe_api(args.url):
        print(
            "\n请先启动 RoboRefer API，并确认本机可访问该地址（例如 WSL 中已运行 api.py，端口与 --url 一致）。\n"
            f"当前无法连接: {args.url}\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    _stage_roborefer_client(
        views_root=views_root,
        predictions_out=predictions_path,
        url=args.url,
        prompt=args.prompt,
        retry=args.retry,
        no_depth=args.no_depth,
        views=args.views,
        visibility_check=args.visibility_check,
        visibility_permissive=args.visibility_permissive,
        visibility_strict=args.visibility_strict,
        visibility_base_url=args.visibility_base_url,
        visibility_model=args.visibility_model,
    )

    # --- D+E fuse + marker ply ---
    sys.path.insert(0, str(BRIDGE))
    from pipeline import stage_fuse  # noqa: WPS433

    depth_dir_resolved = _abs(args.depth_dir) if args.depth_dir else None
    fuse_ns = argparse.Namespace(
        predictions=predictions_path,
        custom_views_out=views_root,
        model_path=model_path,
        inlier_radius=args.inlier_radius,
        min_inv=args.min_inv,
        no_refine=args.no_refine,
        refine_k=args.refine_k,
        ply=ply_used,
        snap=True,
        fused_output=fused_path,
        marker_output=marker_path,
        exclude=args.exclude,
        depth_dir=depth_dir_resolved,
    )
    stage_fuse(fuse_ns)

    # --- overlay ---
    if not args.skip_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)
        _stage_overlay(
            views_root=views_root,
            predictions_path=predictions_path,
            fused_path=fused_path,
            out_dir=overlays_dir,
        )
    else:
        print("[e2e] skip-overlay")

    inject_iteration_folder: str | None = None

    # --- copy model + inject ---
    if not args.no_model_bundle:
        if model_copy_dir.exists():
            raise SystemExit(f"refuse to overwrite existing model copy: {model_copy_dir}")
        print(f"[e2e] copying model tree -> {model_copy_dir} (may take a while)…")
        shutil.copytree(model_path, model_copy_dir, symlinks=False)
        base_in_copy = model_copy_dir / rel_ply
        if rel_ply.parent.name == _INJECT_ITER_DIR:
            inject_folder_name = _INJECT_COLLISION_FALLBACK
            print(
                f"[e2e] inject -> {inject_folder_name} "
                f"(snap base is {_INJECT_ITER_DIR}/; avoid same-file overwrite on Windows)",
            )
        else:
            inject_folder_name = _INJECT_ITER_DIR
        inject_out = model_copy_dir / "point_cloud" / inject_folder_name
        inject_iteration_folder = inject_out.name
        if not base_in_copy.is_file():
            raise SystemExit(f"copied base ply missing: {base_in_copy}")
        inj_extra: list[str] = [
            "--surface-push",
            str(args.inject_surface_push),
            "--surface-push-k",
            str(args.inject_surface_push_k),
        ]
        if args.inject_marker_offset is not None:
            inj_extra.extend(["--marker-offset", *[str(x) for x in args.inject_marker_offset]])
        if args.inject_log_scale is not None:
            inj_extra.extend(["--log-scale", str(args.inject_log_scale)])
        if args.inject_marker_count is not None:
            inj_extra.extend(["--marker-count", str(args.inject_marker_count)])
        if args.inject_opacity is not None:
            inj_extra.extend(["--opacity-sigmoid", str(args.inject_opacity)])
        if args.inject_jitter is not None:
            inj_extra.extend(["--jitter", str(args.inject_jitter)])
        _stage_inject(
            base_ply_in_copy=base_in_copy,
            fused_json=fused_path,
            out_iteration_dir=inject_out,
            all_candidates=args.inject_all_candidates,
            extra_args=inj_extra,
        )
        manifest["inject_marker_options"] = {
            "surface_push": args.inject_surface_push,
            "surface_push_k": args.inject_surface_push_k,
            "marker_offset": list(args.inject_marker_offset) if args.inject_marker_offset is not None else None,
            "extra_cli": inj_extra,
        }
        manifest["model_copy_dir"] = str(model_copy_dir).replace("\\", "/")
        manifest["inject_iteration_dir"] = inject_iteration_folder
        manifest["sibr_hint"] = (
            f'Set-Location "{REPO / "3DGS" / "gaussian-splatting" / "viewers" / "bin"}" ; '
            f'.\\SIBR_gaussianViewer_app.exe -m "{model_copy_dir}"'
        )
        iter_suffix = inject_iteration_folder.removeprefix("iteration_")
        manifest["sibr_iteration_note"] = (
            f"In SIBR pick point_cloud iteration {iter_suffix} (folder {inject_iteration_folder})."
        )
        with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    else:
        print("[e2e] no-model-bundle: skipped full model copy + inject")

    print("\n[e2e] done.")
    print(f"  run_dir       = {run_dir}")
    if not args.no_model_bundle and inject_iteration_folder is not None:
        print(f"  model_copy    = {model_copy_dir}")
        _print_sibr_footer(model_copy_dir=model_copy_dir, inject_iteration_folder=inject_iteration_folder)
    elif args.no_model_bundle:
        print("  (未复制模型) 融合点可打开 MeshLab / CloudCompare 查看:")
        print(f"    {marker_path.resolve()}")


if __name__ == "__main__":
    main()
