#!/usr/bin/env python3
"""e2e 编排用的阶段实现：渲染、融合、WSL mask 预检。不单独当用户入口。"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "bridge"

EXIT_API = 10
EXIT_WSL = 11

STAGES = ("render", "query", "fuse", "filter", "mask", "vote")


def guess_ply(model_path: Path | None) -> Path | None:
    """取 --model-path 下编号最大的 numeric iteration（跳过 iteration_seg_*）。"""
    if model_path is None:
        return None
    found: list[tuple[int, Path]] = []
    for p in Path(model_path).glob("point_cloud/iteration_*/point_cloud.ply"):
        suffix = p.parent.name[len("iteration_") :]
        if suffix.isdigit():
            found.append((int(suffix), p))
    if not found:
        return None
    found.sort()
    return found[-1][1]


def stage_render(args: argparse.Namespace) -> Path:
    inner = REPO / "3DGS"
    render_py = inner / "render.py"
    if not render_py.is_file():
        raise SystemExit(f"render.py not found at {render_py}")

    out_dir = args.custom_views_out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(render_py),
        "-m",
        str(args.model_path.resolve()),
        "--custom_views",
        "--output_path",
        str(out_dir),
        "--num_custom_views",
        str(int(args.num_custom_views)),
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


def stage_fuse(args: argparse.Namespace, ray_model: Any | None = None) -> Path:
    sys.path.insert(0, str(BRIDGE))
    import fuse_multiview as fm  # noqa: WPS433

    pred_path = Path(args.predictions)
    if not pred_path.is_file():
        raise SystemExit(f"predictions.json not found: {pred_path}")

    with pred_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    depth_dir = getattr(args, "depth_dir", None)
    if depth_dir is not None:
        depth_dir = Path(depth_dir).resolve()

    ply = args.ply
    depth_mode = getattr(args, "depth_mode", "ray")
    result = fm.fuse(
        predictions,
        inlier_radius=args.inlier_radius,
        min_inv=args.min_inv,
        refine=not args.no_refine,
        refine_k=args.refine_k,
        ply_path=ply,
        depth_dir=depth_dir,
        depth_mode=depth_mode,
        ray_perp_radius=getattr(args, "ray_perp_radius", 0.06),
        ray_pixel_radius=getattr(args, "ray_pixel_radius", 6.0),
        ray_z_band=getattr(args, "ray_z_band", 2.0),
        ray_min_alpha=getattr(args, "ray_min_alpha", 0.45),
        ray_model=ray_model,
    )

    fused_out = Path(args.fused_output)
    fused_out.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["depth_mode"] = depth_mode
    with fused_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    p = result.P_world
    print(
        f"[stage fuse] P_world=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
        f"support={result.support}/{len(result.candidates)}"
    )
    print(f"[stage fuse] wrote {fused_out}（种子，不是 3D 结果）")
    if result.support <= 0:
        raise SystemExit("fuse produced no inliers")
    return fused_out


def to_wsl_path(p: Path | str) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:]
    return s


def _bash_join(parts: list[str]) -> str:
    return " ".join(f'"{a}"' if (" " in a or "\\" in a) else a for a in parts)


def require_views(views_root: Path) -> None:
    missing = [name for name in ("rgb", "camera_params", "depth_raw") if not (views_root / name).is_dir()]
    if missing:
        raise SystemExit(f"{views_root}: missing {', '.join(missing)}")
    if not any((views_root / "rgb").glob("view_*.png")):
        raise SystemExit(f"{views_root}/rgb: no view_*.png")


def probe_api(url: str, timeout: float = 5.0) -> bool:
    try:
        import requests

        requests.get(url.rstrip("/"), timeout=timeout)
        return True
    except Exception:
        return False


def preflight_api(url: str) -> None:
    if probe_api(url):
        print(f"[preflight] API ok: {url}")
        return
    print(
        "\n请先启动 RoboRefer API，并确认本机可访问该地址（WSL api.py 或 SSH 隧道）。\n"
        f"当前无法连接: {url}\n",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_API)


def preflight_wsl(
    *,
    sam2_checkpoint: Path,
    grounding_checkpoint: Path,
    grounding_config: Path,
) -> None:
    missing = [p for p in (sam2_checkpoint, grounding_checkpoint, grounding_config) if not p.is_file()]
    if missing:
        print("[preflight] missing mask weights/config:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        raise SystemExit(EXIT_WSL)
    if shutil.which("wsl") is None:
        print("[preflight] wsl not found on PATH", file=sys.stderr)
        raise SystemExit(EXIT_WSL)
    inner = (
        "set +u; "
        "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null "
        "|| source ~/anaconda3/etc/profile.d/conda.sh; "
        "conda activate roborefer; "
        "python -c 'print(1)'"
    )
    rc = subprocess.call(["wsl", "bash", "-lc", inner])
    if rc != 0:
        print("[preflight] WSL roborefer env check failed", file=sys.stderr)
        raise SystemExit(EXIT_WSL)
    print("[preflight] WSL roborefer ok")


def _mask_box_score_cli(
    *,
    contain_bonus: float,
    compact_box_weight: float,
    large_box_penalty: float,
    large_box_area_start: float,
    area_penalty: float,
    tiny_box_penalty: float,
    dino_score_weight: float,
    near_score_weight: float,
    near_dist_sigma: float,
    min_box_area_ratio: float,
) -> list[str]:
    """与 mask_grounding / gen_training_data 默认对齐的选框打分 CLI。"""
    return [
        "--contain-bonus",
        str(contain_bonus),
        "--compact-box-weight",
        str(compact_box_weight),
        "--large-box-penalty",
        str(large_box_penalty),
        "--large-box-area-start",
        str(large_box_area_start),
        "--area-penalty",
        str(area_penalty),
        "--tiny-box-penalty",
        str(tiny_box_penalty),
        "--dino-score-weight",
        str(dino_score_weight),
        "--near-score-weight",
        str(near_score_weight),
        "--near-dist-sigma",
        str(near_dist_sigma),
        "--min-box-area-ratio",
        str(min_box_area_ratio),
    ]


def run_mask_wsl(
    *,
    out_dir: Path,
    prompt: str,
    object_name: str,
    sam2_checkpoint: Path,
    sam2_config: str,
    grounding_config: Path,
    grounding_checkpoint: Path,
    grounding_box_threshold: float,
    grounding_text_threshold: float,
    anchor_box_radius: int,
    max_box_area_ratio: float,
    contain_bonus: float = 0.18,
    compact_box_weight: float = 0.10,
    large_box_penalty: float = 0.12,
    large_box_area_start: float = 0.15,
    area_penalty: float = 0.12,
    tiny_box_penalty: float = 0.40,
    dino_score_weight: float = 0.50,
    near_score_weight: float = 0.40,
    near_dist_sigma: float = 50.0,
    min_box_area_ratio: float = 0.003,
    use_kept: bool = False,
) -> None:
    wsl_repo = to_wsl_path(REPO)
    out_wsl = to_wsl_path(out_dir)
    script = to_wsl_path(BRIDGE / "gen_training_data.py")
    mask_cmd = [
        "python",
        script,
        "--stage",
        "mask",
        "--out",
        out_wsl,
        "--prompt",
        prompt,
        "--object",
        object_name,
        "--sam2-checkpoint",
        to_wsl_path(sam2_checkpoint),
        "--sam2-config",
        sam2_config,
        "--grounding-config",
        to_wsl_path(grounding_config),
        "--grounding-checkpoint",
        to_wsl_path(grounding_checkpoint),
        "--grounding-box-threshold",
        str(grounding_box_threshold),
        "--grounding-text-threshold",
        str(grounding_text_threshold),
        "--anchor-box-radius",
        str(anchor_box_radius),
        "--max-box-area-ratio",
        str(max_box_area_ratio),
        *_mask_box_score_cli(
            contain_bonus=contain_bonus,
            compact_box_weight=compact_box_weight,
            large_box_penalty=large_box_penalty,
            large_box_area_start=large_box_area_start,
            area_penalty=area_penalty,
            tiny_box_penalty=tiny_box_penalty,
            dino_score_weight=dino_score_weight,
            near_score_weight=near_score_weight,
            near_dist_sigma=near_dist_sigma,
            min_box_area_ratio=min_box_area_ratio,
        ),
    ]
    if use_kept:
        mask_cmd.append("--use-kept")
    inner = " && ".join(
        [
            f"cd {wsl_repo}",
            "set +u",
            "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh",
            "conda activate roborefer",
            f"export PYTHONPATH={wsl_repo}/GroundingDINO:$PYTHONPATH",
            "export HF_HUB_OFFLINE=1",
            _bash_join(mask_cmd),
        ]
    )
    print(f"[e2e] wsl mask -> {out_dir}")
    rc = subprocess.call(["wsl", "bash", "-lc", inner])
    if rc != 0:
        raise SystemExit(f"WSL mask failed with exit code {rc}")


def run_mask_local(
    *,
    out_dir: Path,
    prompt: str,
    object_name: str,
    sam2_checkpoint: Path,
    sam2_config: str,
    grounding_config: Path,
    grounding_checkpoint: Path,
    grounding_box_threshold: float,
    grounding_text_threshold: float,
    anchor_box_radius: int,
    max_box_area_ratio: float,
    contain_bonus: float = 0.18,
    compact_box_weight: float = 0.10,
    large_box_penalty: float = 0.12,
    large_box_area_start: float = 0.15,
    area_penalty: float = 0.12,
    tiny_box_penalty: float = 0.40,
    dino_score_weight: float = 0.50,
    near_score_weight: float = 0.40,
    near_dist_sigma: float = 50.0,
    min_box_area_ratio: float = 0.003,
    use_kept: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(BRIDGE / "gen_training_data.py"),
        "--stage",
        "mask",
        "--out",
        str(out_dir),
        "--prompt",
        prompt,
        "--object",
        object_name,
        "--sam2-checkpoint",
        str(sam2_checkpoint),
        "--sam2-config",
        sam2_config,
        "--grounding-config",
        str(grounding_config),
        "--grounding-checkpoint",
        str(grounding_checkpoint),
        "--grounding-box-threshold",
        str(grounding_box_threshold),
        "--grounding-text-threshold",
        str(grounding_text_threshold),
        "--anchor-box-radius",
        str(anchor_box_radius),
        "--max-box-area-ratio",
        str(max_box_area_ratio),
        *_mask_box_score_cli(
            contain_bonus=contain_bonus,
            compact_box_weight=compact_box_weight,
            large_box_penalty=large_box_penalty,
            large_box_area_start=large_box_area_start,
            area_penalty=area_penalty,
            tiny_box_penalty=tiny_box_penalty,
            dino_score_weight=dino_score_weight,
            near_score_weight=near_score_weight,
            near_dist_sigma=near_dist_sigma,
            min_box_area_ratio=min_box_area_ratio,
        ),
    ]
    if use_kept:
        cmd.append("--use-kept")
    print(f"[e2e] $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(REPO))
    if rc != 0:
        raise SystemExit(f"mask failed with exit code {rc}")
