#!/usr/bin/env python3
"""Filter multi-view projections using 3DGS ray transmittance.

**Reject** (default): beam of 5 rays through ``P_world`` (+/- offsets); reject if
``min(T) < --ray-min-transmittance``. Use ``--skip-ray-filter`` for manual / large
objects (keeps all in-frustum views, no scipy ply load).

**Yellow overlays** — depth-ratio cluster points (visualization only).

Outputs: projections_kept.json, projections_rejected.json, filter_report.json,
optional filter_overlays/.

Point relabel after SAM mask: ``gen_training_data.py --stage refine``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from fuse_multiview import load_ply_xyz  # noqa: E402
from gen_training_data import in_frame, world_to_image  # noqa: E402
from ray_visibility import (  # noqa: E402
    RayOcclusionModel,
    cluster_depth_band,
    ray_reject_reason,
)
from unproject import Unprojector, CameraView  # noqa: E402

# Preset bundles (applied in main() before filter_views). CLI flags override preset.
FILTER_PRESETS: dict[str, dict[str, float | int | bool]] = {
    "default": {},
    "relaxed": {
        "ray_min_transmittance": 0.32,
        "ray_perp_radius": 0.10,
        "min_cluster_in_frame": 5,
        "ray_beam_offset_m": 0.06,
    },
  # ~+10 kept vs relaxed on medicine_bottle (tau from rejected T distribution)
    "relaxed_plus": {
        "ray_min_transmittance": 0.12,
        "ray_perp_radius": 0.08,
        "min_cluster_in_frame": 4,
        "ray_beam_offset_m": 0.05,
    },
}


def apply_filter_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Merge named preset into args (only keys listed in FILTER_PRESETS)."""
    name = getattr(args, "filter_preset", "default") or "default"
    if name not in FILTER_PRESETS:
        sys.exit(f"[error] unknown --filter-preset {name!r}; choose from {list(FILTER_PRESETS)}")
    preset = FILTER_PRESETS[name]
    for key, val in preset.items():
        setattr(args, key, val)
    return args


def select_cluster_indices(
    xyz: np.ndarray,
    p_world: np.ndarray,
    *,
    radius: float,
    k_max: int,
    min_cluster: int = 0,
    max_radius: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Return (indices, effective_radius). Grows radius until min_cluster or max_radius."""
    d = np.linalg.norm(xyz - p_world.reshape(1, 3), axis=1)
    r_eff = float(radius)
    while True:
        in_radius = np.flatnonzero(d <= r_eff)
        if min_cluster > 0 and in_radius.size < min_cluster and r_eff < max_radius:
            r_eff = min(r_eff * 1.25, max_radius)
            continue
        break
    if in_radius.size == 0:
        k = min(k_max, len(d))
        return np.argpartition(d, k - 1)[:k], r_eff
    if in_radius.size > k_max:
        sub = d[in_radius]
        keep = np.argpartition(sub, k_max - 1)[:k_max]
        return in_radius[keep], r_eff
    return in_radius, r_eff


def _sample_z_pix(unp: Unprojector, depth_path: Path, u: float, v: float) -> float | None:
    w, h = unp.view.width, unp.view.height
    ui = int(np.clip(round(u), 0, w - 1))
    vi = int(np.clip(round(v), 0, h - 1))
    try:
        _, z_pix = unp.sample_depth_raw(depth_path, ui, vi)
    except Exception:
        return None
    if z_pix <= 0:
        return None
    return float(z_pix)


def ratio_depth_visible(
    ratio: float,
    median_ratio: float,
    *,
    rel_slack: float,
) -> bool:
    if ratio <= 0 or median_ratio <= 0:
        return False
    return ratio >= median_ratio * (1.0 - rel_slack)


def project_cluster_yellow(
    cluster_xyz: np.ndarray,
    cam: dict,
    depth_path: Path,
    unp: Unprojector,
    *,
    frame_margin: int,
    rel_slack: float,
) -> tuple[np.ndarray, int, float | None, float]:
    w, h = unp.view.width, unp.view.height
    entries: list[tuple[float, float, float]] = []
    raw_count = 0
    for p in cluster_xyz:
        u, v, z_cam = world_to_image(p, cam)
        if z_cam <= 0 or not in_frame(u, v, w, h, margin=frame_margin):
            continue
        raw_count += 1
        z_pix = _sample_z_pix(unp, depth_path, u, v)
        if z_pix is None:
            continue
        entries.append((u, v, z_pix / z_cam))

    if not entries:
        return np.zeros((0, 2), dtype=np.float64), raw_count, None, 0.0

    ratios = np.array([e[2] for e in entries], dtype=np.float64)
    median_ratio = float(np.median(ratios))
    ratio_std = float(np.std(ratios))
    visible_uv = [
        [u, v]
        for u, v, ratio in entries
        if ratio_depth_visible(ratio, median_ratio, rel_slack=rel_slack)
    ]
    if visible_uv:
        return np.asarray(visible_uv, dtype=np.float64), raw_count, median_ratio, ratio_std
    return np.zeros((0, 2), dtype=np.float64), raw_count, median_ratio, ratio_std


def draw_overlay(
    rgb_path: Path,
    out_path: Path,
    *,
    orig_uv: tuple[float, float],
    final_uv: tuple[float, float],
    visible_uv: np.ndarray,
    max_yellow: int = 400,
) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    ou, ov = orig_uv
    fu, fv = final_uv
    if visible_uv.shape[0] > 0:
        step = max(1, visible_uv.shape[0] // max_yellow)
        for i in range(0, visible_uv.shape[0], step):
            u, v = visible_uv[i]
            r = 3
            draw.ellipse((u - r, v - r, u + r, v + r), fill=(255, 220, 0))
    rg = 9
    draw.ellipse((ou - rg, ov - rg, ou + rg, ov + rg), fill=(0, 255, 0), outline=(0, 120, 0), width=2)
    same = math.hypot(ou - fu, ov - fv) < 4.0
    if same:
        rr = 5
        draw.ellipse((fu - rr, fv - rr, fu + rr, fv + rr), fill=(255, 40, 40))
    else:
        rr = 8
        draw.ellipse((fu - rr, fv - rr, fu + rr, fv + rr), fill=(255, 40, 40), outline=(180, 0, 0), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def filter_views(args: argparse.Namespace) -> dict[str, Any]:
    fused = json.loads(Path(args.fused).read_text(encoding="utf-8"))
    p_world = np.asarray(fused["P_world"], dtype=np.float64)
    fuse_support = int(fused.get("support", 0))
    snap_distance = fused.get("snap_distance")
    cluster_radius = float(args.cluster_radius)
    if not getattr(args, "no_expand_radius_to_snap", False) and snap_distance is not None:
        cluster_radius = max(cluster_radius, float(snap_distance) * 1.2)

    projections = json.loads(Path(args.projections).read_text(encoding="utf-8"))
    xyz = load_ply_xyz(Path(args.ply))
    cluster_idx, cluster_radius = select_cluster_indices(
        xyz,
        p_world,
        radius=cluster_radius,
        k_max=args.k_max,
        min_cluster=args.min_cluster_gaussians,
        max_radius=args.max_cluster_radius,
    )
    cluster_xyz = xyz[cluster_idx]

    ray_model: RayOcclusionModel | None = None
    if not args.skip_ray_filter:
        ray_model = RayOcclusionModel.from_ply(
            Path(args.ply),
            perp_radius=args.ray_perp_radius,
            sample_step=args.ray_sample_step,
        )

    views_root = Path(args.views_root)
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "filter_overlays"
    if args.write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if fuse_support < args.support_warn_below:
        warnings.append(
            f"fused support={fuse_support} < {args.support_warn_below} (low-confidence fusion)"
        )

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    per_view: list[dict[str, Any]] = []

    def _maybe_overlay(
        view_id: str,
        u_orig: float,
        v_orig: float,
        u_fin: float,
        v_fin: float,
        vis_arr: np.ndarray,
    ) -> None:
        if not args.write_overlays:
            return
        draw_overlay(
            views_root / "rgb" / f"view_{view_id}.png",
            overlay_dir / f"overlay_view_{view_id}.png",
            orig_uv=(u_orig, v_orig),
            final_uv=(u_fin, v_fin),
            visible_uv=vis_arr,
        )

    for rec in projections:
        view_id = str(rec["view_id"]).zfill(3)
        w, h = int(rec["width"]), int(rec["height"])
        u0, v0 = float(rec["u"]), float(rec["v"])
        nx0, ny0 = float(rec["nx"]), float(rec["ny"])

        cam_path = views_root / "camera_params" / f"view_{view_id}.json"
        depth_path = views_root / "depth_raw" / f"view_{view_id}.npy"
        rgb_path = views_root / "rgb" / f"view_{view_id}.png"

        entry: dict[str, Any] = {
            "view_id": view_id,
            "u_orig": u0,
            "v_orig": v0,
            "nx_orig": nx0,
            "ny_orig": ny0,
        }

        if not cam_path.is_file() or not depth_path.is_file() or not rgb_path.is_file():
            entry["status"] = "skipped"
            entry["reason"] = "missing_camera_depth_or_rgb"
            skipped.append(entry)
            per_view.append(entry)
            continue

        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        view = CameraView.from_json(cam_path)
        unp = Unprojector(view, inv_eps=args.inv_eps)

        vis, raw_in_frustum, median_ratio, ratio_std = project_cluster_yellow(
            cluster_xyz,
            cam,
            depth_path,
            unp,
            frame_margin=args.frame_margin,
            rel_slack=args.depth_rel_slack,
        )
        entry["visible_count_raw"] = raw_in_frustum
        entry["visible_count_depth"] = int(vis.shape[0])
        if median_ratio is not None:
            entry["median_depth_ratio"] = median_ratio
        entry["ratio_std"] = ratio_std

        if raw_in_frustum < args.min_cluster_in_frame:
            entry["status"] = "rejected"
            entry["reason"] = "cluster_out_of_frame"
            rejected.append({**rec, "filter": entry})
            per_view.append(entry)
            _maybe_overlay(view_id, u0, v0, u0, v0, vis)
            continue

        reject_reason: str | None = None
        if args.skip_ray_filter:
            entry["T_at_z_lo"] = None
            entry["skip_ray_filter"] = True
        else:
            assert ray_model is not None
            band = cluster_depth_band(
                cluster_xyz,
                cam,
                frame_margin=args.frame_margin,
                w=w,
                h=h,
            )
            if band is None:
                reject_reason = "no_cluster_depth_band"
                entry["z_lo"] = None
                entry["z_hi"] = None
                entry["T_at_z_lo"] = None
            else:
                z_lo, z_hi = band
                z_cut = max(1e-4, z_lo - float(args.ray_z_slack))
                entry["z_lo"] = z_lo
                entry["z_hi"] = z_hi
                entry["z_cut"] = z_cut
                if args.ray_beam:
                    ray_stats = ray_model.transmittance_beam_min(
                        cam,
                        p_world,
                        z_cut,
                        z_hi=z_hi,
                        beam_offset_m=args.ray_beam_offset_m,
                    )
                else:
                    ray_stats = ray_model.transmittance_before(cam, p_world, z_cut, z_hi=z_hi)
                entry["T_at_z_lo"] = ray_stats["T_at_z_cut"]
                entry["ray_n_candidates"] = ray_stats["n_candidates"]
                entry["ray_n_foreground"] = ray_stats["n_foreground"]
                reject_reason = ray_reject_reason(
                    ray_stats["T_at_z_cut"],
                    min_transmittance=args.ray_min_transmittance,
                )

        if reject_reason is not None:
            entry["status"] = "rejected"
            entry["reason"] = reject_reason
            rejected.append({**rec, "filter": entry})
            per_view.append(entry)
            _maybe_overlay(view_id, u0, v0, u0, v0, vis)
            continue

        entry["status"] = "kept"
        entry["reason"] = "skip_ray_filter" if args.skip_ray_filter else "ok"

        out_rec = dict(rec)
        out_rec["u"] = u0
        out_rec["v"] = v0
        out_rec["nx"] = nx0
        out_rec["ny"] = ny0
        out_rec["rgb_path"] = str(rgb_path.resolve())
        kept.append(out_rec)
        per_view.append(entry)
        _maybe_overlay(view_id, u0, v0, u0, v0, vis)

    proj_ids = {str(r["view_id"]).zfill(3) for r in projections}
    all_cam_ids = sorted(
        re.search(r"view_(\d+)", p.name).group(1)  # type: ignore[union-attr]
        for p in views_root.glob("camera_params/view_*.json")
    )
    missing_from_project = [vid for vid in all_cam_ids if vid not in proj_ids]

    report = {
        "fused_path": str(Path(args.fused).as_posix()),
        "ply_path": str(Path(args.ply).as_posix()),
        "views_root": str(views_root.as_posix()),
        "P_world": p_world.tolist(),
        "fuse_support": fuse_support,
        "cluster_size": int(cluster_idx.size),
        "cluster_radius_effective": cluster_radius,
        "filter_preset": getattr(args, "filter_preset", "default"),
        "params": {
            "skip_ray_filter": args.skip_ray_filter,
            "filter_preset": getattr(args, "filter_preset", "default"),
            "min_cluster_in_frame": args.min_cluster_in_frame,
            "depth_rel_slack": args.depth_rel_slack,
            "ray_min_transmittance": args.ray_min_transmittance,
            "ray_perp_radius": args.ray_perp_radius,
            "ray_sample_step": args.ray_sample_step,
            "ray_z_slack": args.ray_z_slack,
            "ray_beam": args.ray_beam,
            "ray_beam_offset_m": args.ray_beam_offset_m,
        },
        "warnings": warnings,
        "cameras_total": len(all_cam_ids),
        "projections_input": len(projections),
        "missing_from_project_stage": missing_from_project,
        "counts": {
            "input": len(projections),
            "kept": len(kept),
            "rejected": len(rejected),
            "skipped": len(skipped),
        },
        "views": per_view,
        "skipped": skipped,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "projections_kept.json").write_text(
        json.dumps(kept, indent=2), encoding="utf-8"
    )
    (out_dir / "projections_rejected.json").write_text(
        json.dumps(rejected, indent=2),
        encoding="utf-8",
    )
    (out_dir / "filter_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    mode = "skip-ray" if args.skip_ray_filter else "ray-beam" if args.ray_beam else "ray"
    print(
        f"[filter] mode={mode} kept={len(kept)} rejected={len(rejected)} skipped={len(skipped)} "
        f"(input={len(projections)})"
    )
    print(f"         -> {out_dir / 'projections_kept.json'}")
    if args.write_overlays:
        print(f"         overlays -> {overlay_dir}")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="3DGS ray visibility filter for projections.")
    ap.add_argument("--projections", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--views-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--filter-preset",
        choices=tuple(FILTER_PRESETS.keys()),
        default="default",
        help="relaxed / relaxed_plus: looser ray T (relaxed_plus ~+10 views on small objects)",
    )
    ap.add_argument(
        "--skip-ray-filter",
        action="store_true",
        help="Keep all in-frustum views (large objects / manual curation); no ray T test",
    )
    ap.add_argument("--cluster-radius", type=float, default=0.05)
    ap.add_argument("--k-max", type=int, default=3000)
    ap.add_argument("--min-cluster-gaussians", type=int, default=150)
    ap.add_argument("--max-cluster-radius", type=float, default=0.35)
    ap.add_argument("--min-cluster-in-frame", type=int, default=8)
    ap.add_argument("--depth-rel-slack", type=float, default=0.15)
    ap.add_argument(
        "--ray-min-transmittance",
        type=float,
        default=0.45,
        help="Keep if min beam T >= this (lower = more lenient, keeps more views)",
    )
    ap.add_argument(
        "--ray-perp-radius",
        type=float,
        default=0.12,
        help="Cylinder radius (m) around each beam ray",
    )
    ap.add_argument("--ray-sample-step", type=float, default=0.02)
    ap.add_argument("--ray-z-slack", type=float, default=0.01)
    ap.add_argument(
        "--ray-beam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 5-ray min-T (default on; better for plush occluders)",
    )
    ap.add_argument(
        "--ray-beam-offset-m",
        type=float,
        default=0.05,
        help="Lateral offset (m) for side rays in camera plane",
    )
    ap.add_argument("--frame-margin", type=int, default=10)
    ap.add_argument("--inv-eps", type=float, default=1e-3)
    ap.add_argument("--support-warn-below", type=int, default=5)
    ap.add_argument("--no-expand-radius-to-snap", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()
    args = apply_filter_preset(args)
    args.write_overlays = not args.no_overlays
    filter_views(args)


if __name__ == "__main__":
    main()
