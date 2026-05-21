#!/usr/bin/env python3
"""End-to-end orchestrator: 3DGS multi-view render -> RoboRefer -> fuse -> marker.

The pipeline crosses two conda envs that cannot share a Python process:
    1. render        (gaussian_splatting env) -> custom-view dir
    2. roborefer     (roborefer env, separate API server) -> predictions.json
    3. fuse          (gaussian_splatting env) -> fused.json
    4. visualize     (gaussian_splatting env) -> marker.ply

Modes:
  --stage render     run only step 1 (subprocess to gaussian-splatting/render.py)
  --stage query      run only step 2 (in-process import; must be in roborefer env)
  --stage fuse       run only step 3 + 4
  --stage all        try every step in order; will STOP and print instructions
                     if it detects the wrong env for step 2

Detection: we import bridge.roborefer_client and try ``_import_query_server``;
if it fails, the orchestrator prints exactly which command to run in the
roborefer env and exits with code 10. After you finish step 2, re-run with
``--stage fuse`` (or ``--stage all`` again) from gaussian_splatting env.

Default scene layout:
    --model-path 3DGS/gaussian-splatting/output/<run_id>
    --custom-views-out 3DGS/test1
The orchestrator never invents paths; everything is explicit.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "bridge"


# ---------------------------------------------------------------------------
# Step 1: render (subprocess into gaussian-splatting/render.py)
# ---------------------------------------------------------------------------


def stage_render(args: argparse.Namespace) -> Path:
    inner = REPO / "3DGS" / "gaussian-splatting"
    render_py = inner / "render.py"
    if not render_py.is_file():
        # Older layout used during this project: gaussian-splatting/gaussian-splatting/
        legacy = REPO / "gaussian-splatting" / "gaussian-splatting" / "render.py"
        if legacy.is_file():
            inner = legacy.parent
            render_py = legacy
        else:
            root_render = REPO / "3DGS" / "render.py"
            if root_render.is_file():
                inner = root_render.parent
                render_py = root_render
            else:
                raise SystemExit(
                    f"render.py not found at {inner / 'render.py'}, legacy {legacy}, or {root_render}"
                )

    out_dir = args.custom_views_out.resolve()
    cmd = [
        sys.executable, str(render_py),
        "-m", str(args.model_path.resolve()),
        "--custom_views",
        "--output_path", str(out_dir),
        "--num_custom_views", str(int(args.num_custom_views)),
    ]
    if args.iteration is not None:
        cmd.extend(["--iteration", str(args.iteration)])

    print(f"[stage render] cwd={inner}")
    print(f"[stage render] $ {' '.join(shlex.quote(c) for c in cmd)}")
    rc = subprocess.call(cmd, cwd=str(inner))
    if rc != 0:
        raise SystemExit(f"render.py exited with code {rc}")

    rgb = out_dir / "rgb"
    if not rgb.is_dir() or not any(rgb.glob("view_*.png")):
        raise SystemExit(f"render produced no RGB views in {rgb}")
    print(f"[stage render] ok: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Step 2: query RoboRefer
# ---------------------------------------------------------------------------


_ROBOREFER_HINT = """
== Stage 2 needs the roborefer env ==

This Python process is in the wrong env (vila / RoboRefer API not importable).
Switch to the roborefer env, make sure the API server is running, then run:

  conda activate roborefer
  cd RoboRefer-main/API
  python api.py --port 25547 \\
      --depth_model_path /mnt/e/3DGS-VLM/weights/depth_anything_v2_vitl.pth \\
      --vlm_model_path  /mnt/e/3DGS-VLM/RoboRefer-2B-SFT

In a second roborefer-env shell, run the batch client:

  python {client} \\
      --root {root} \\
      --prompt {prompt!r} \\
      --url http://127.0.0.1:25547 \\
      --output {out}

When it finishes, switch back to gaussian_splatting and run:

  conda activate gaussian_splatting
  python {pipeline} --stage fuse \\
      --predictions {out} \\
      --ply {ply_hint}
"""


def _can_import_roborefer() -> bool:
    """Check if requests is available (the only real dependency now)."""
    try:
        import requests  # noqa: F401
        return True
    except ImportError:
        return False


def stage_query(args: argparse.Namespace) -> Path:
    out = args.predictions or (args.custom_views_out.resolve() / "predictions.json")

    if not _can_import_roborefer():
        ply_hint = _guess_ply(args.model_path) if args.model_path else "<point_cloud.ply>"
        print(_ROBOREFER_HINT.format(
            client=BRIDGE / "roborefer_client.py",
            root=args.custom_views_out.resolve(),
            prompt=args.prompt,
            out=out,
            pipeline=BRIDGE / "pipeline.py",
            ply_hint=ply_hint,
        ))
        sys.exit(10)

    sys.path.insert(0, str(BRIDGE))
    import roborefer_client as rc  # noqa: WPS433

    ns = argparse.Namespace(
        prompt=args.prompt,
        url=args.url,
        retry=args.retry,
        no_depth=args.no_depth,
        no_suffix=False,
        root=args.custom_views_out,
        views=args.views,
        output=out,
    )
    result = rc.run_batch(ns)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    ok = sum(1 for v in result["views"] if v["parse_ok"])
    print(f"[stage query] wrote {out}  ok={ok}/{len(result['views'])}")
    return out


# ---------------------------------------------------------------------------
# Step 3 + 4: fuse + visualize
# ---------------------------------------------------------------------------


def _guess_ply(model_path: Path | None) -> Path | None:
    if model_path is None:
        return None
    candidates = sorted(
        Path(model_path).glob("point_cloud/iteration_*/point_cloud.ply"),
        key=lambda p: int(p.parent.name.split("_")[1]),
    )
    return candidates[-1] if candidates else None


def stage_fuse(args: argparse.Namespace) -> tuple[Path, Path]:
    sys.path.insert(0, str(BRIDGE))
    import fuse_multiview as fm  # noqa: WPS433
    import visualize as viz  # noqa: WPS433

    pred_path = args.predictions or (args.custom_views_out.resolve() / "predictions.json")
    if not pred_path.is_file():
        raise SystemExit(f"predictions.json not found: {pred_path}; run --stage query first")

    with pred_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    if getattr(args, "exclude", None):
        exclude_set = set(args.exclude)
        for v in predictions["views"]:
            if v.get("view_id") in exclude_set:
                v["visible"] = False
                print(f"[stage fuse] exclude view_id={v['view_id']}")

    depth_dir = getattr(args, "depth_dir", None)
    if depth_dir is not None:
        depth_dir = Path(depth_dir).resolve()

    ply = args.ply or _guess_ply(args.model_path) if (args.snap or args.ply) else None
    result = fm.fuse(
        predictions,
        inlier_radius=args.inlier_radius,
        min_inv=args.min_inv,
        refine=not args.no_refine,
        refine_k=args.refine_k,
        ply_path=ply,
        depth_dir=depth_dir,
    )

    fused_out = args.fused_output or (pred_path.parent / "fused.json")
    fused_out.parent.mkdir(parents=True, exist_ok=True)
    with fused_out.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    p = result.P_world
    print(f"[stage fuse] P_world=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
          f"support={result.support}/{len(result.candidates)}")
    if result.snapped_to_gaussian:
        print(f"             snap_distance={result.snap_distance:.4f}")
    print(f"[stage fuse] wrote {fused_out}")

    marker_out = args.marker_output or (fused_out.parent / "marker.ply")
    viz.write_marker_ply(json.loads(fused_out.read_text(encoding="utf-8")), marker_out)
    print(f"[stage visualize] wrote {marker_out}")
    return fused_out, marker_out


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end 3DGS + RoboRefer orchestrator.")
    ap.add_argument("--stage", choices=["render", "query", "fuse", "all"], default="all")

    # render-stage args
    ap.add_argument("--model-path", type=Path, default=None,
                    help="3DGS trained model dir (e.g. 3DGS/gaussian-splatting/output/<run_id>)")
    ap.add_argument("--custom-views-out", type=Path, default=Path("3DGS/test1"),
                    help="Where render.py --custom_views writes / where query/fuse read from.")
    ap.add_argument("--iteration", type=int, default=None, help="--iteration to render.py")
    ap.add_argument(
        "--num-custom-views",
        type=int,
        default=36,
        dest="num_custom_views",
        help="Passed to render.py --num_custom_views (default 36).",
    )

    # query-stage args
    ap.add_argument("--prompt", type=str, default=None, help="RoboRefer prompt (required for query/all)")
    ap.add_argument("--url", type=str, default="http://127.0.0.1:25547")
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--views", type=int, nargs="+", default=None)
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Override predictions.json path (default: <custom-views-out>/predictions.json)")

    # fuse-stage args
    ap.add_argument("--inlier-radius", type=float, default=0.5)
    ap.add_argument("--min-inv", type=float, default=1e-3)
    ap.add_argument("--ply", type=Path, default=None)
    ap.add_argument("--no-refine", action="store_true", help="Disable iterative refinement")
    ap.add_argument("--refine-k", type=float, default=2.0, help="Refinement threshold: k * median_dist")
    ap.add_argument("--snap", action="store_true",
                    help="Auto-discover point_cloud.ply under --model-path and snap fused point to nearest gaussian")
    ap.add_argument("--exclude", type=int, nargs="+", default=None,
                    help="View IDs to treat as invisible during fusion (same as fuse_multiview --exclude)")
    ap.add_argument("--depth-dir", type=Path, default=None,
                    help="Override depth_raw directory (e.g. depth_raw_dav2/)")
    ap.add_argument("--fused-output", type=Path, default=None)
    ap.add_argument("--marker-output", type=Path, default=None)

    args = ap.parse_args()

    if args.stage in ("render", "all"):
        if args.model_path is None:
            ap.error("--model-path is required for render/all stage")
        stage_render(args)

    if args.stage in ("query", "all"):
        if args.prompt is None:
            ap.error("--prompt is required for query/all stage")
        stage_query(args)

    if args.stage in ("fuse", "all"):
        stage_fuse(args)


if __name__ == "__main__":
    main()
