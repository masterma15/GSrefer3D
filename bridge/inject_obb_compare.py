#!/usr/bin/env python3
"""Inject OBB wireframe + Base/LoRA fused points into 3DGS point_cloud.ply for SIBR.

Colors (RGB):
  - OBB wireframe: cyan
  - Base P_world: blue
  - LoRA P_world: magenta

Example (double-sided tape hold-out):
  python bridge/inject_obb_compare.py \\
    --ply 3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply \\
    --bbox docs/bbox_data2.json \\
    --object double_sided_tape \\
    --base-fused 3DGS/test2/runs/20260519_000313_6c883d56/fused.json \\
    --lora-fused 3DGS/test2/runs/20260519_132142_6c883d56/fused_ray.json \\
    --out-iteration-dir 3DGS/gaussian-splatting/output/data2/point_cloud/iteration_obb_tape
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

_BRIDGE = Path(__file__).resolve().parent
_REPO = _BRIDGE.parent
if str(_BRIDGE) not in sys.path:
    sys.path.insert(0, str(_BRIDGE))

from eval_3d_obb_hit import DEFAULT_ROWS, load_bbox, obb_hit, obb_local, _half_extent  # noqa: E402
from eval_3d_obb_offset import OBB_WIREFRAME_EDGES, obb_corners_world  # noqa: E402

C0 = 0.28209479177387814

PRESETS: dict[str, dict[str, str]] = {
    "double_sided_tape": {
        "object_key": "double_sided_tape",
        "base_run": "20260519_000313_6c883d56",
        "lora_run": "20260519_132142_6c883d56",
        "out_name": "iteration_obb_tape",
    },
    "electric_shaver": {
        "object_key": "electric_shaver",
        "base_run": "20260519_170540_4c3b9a32",
        "lora_run": "20260519_143457_4c3b9a32",
        "out_name": "iteration_obb_shaver",
    },
    "brown_rabbit": {
        "object_key": "brown_rabbit",
        "base_run": "20260519_171359_147bac82",
        "lora_run": "20260519_144845_147bac82",
        "out_name": "iteration_obb_rabbit",
    },
}

# Same objects as demo/teaser_3d_*.gif (SIBR anchor recordings)
GIF_OBJECT_PRESETS = ("electric_shaver", "brown_rabbit")


def _inverse_sigmoid(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def _rgb_to_f_dc(rgb: np.ndarray) -> tuple[float, float, float]:
    rgb = np.asarray(rgb, dtype=np.float64).reshape(3)
    sh = (rgb - 0.5) / C0
    return float(sh[0]), float(sh[1]), float(sh[2])


def obb_wireframe_samples(obb: dict[str, Any], *, step_m: float = 0.025) -> np.ndarray:
    corners = obb_corners_world(obb)
    pts: list[np.ndarray] = []
    for i, j in OBB_WIREFRAME_EDGES:
        a, b = corners[i], corners[j]
        seg_len = float(np.linalg.norm(b - a))
        n = max(2, int(math.ceil(seg_len / step_m)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            pts.append(a + t * (b - a))
    return np.stack(pts, axis=0)


def _load_p_world(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["P_world"], dtype=np.float64)


def inject_obb_compare(
    *,
    ply: Path,
    obb: dict[str, Any],
    base_p: np.ndarray,
    lora_p: np.ndarray,
    out_iteration_dir: Path | None = None,
    output: Path | None = None,
    wireframe_step_m: float = 0.035,
    wireframe_per_point: int = 1,
    marker_count: int = 18,
    wireframe_log_scale: float = -5.0,
    marker_log_scale: float = -4.2,
    wireframe_jitter: float = 0.002,
    marker_jitter: float = 0.005,
    seed: int = 0,
) -> Path:
    src = ply.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"missing source ply: {src}")

    ply_data = PlyData.read(str(src))
    el0 = ply_data.elements[0]
    if el0.name != "vertex":
        raise ValueError(f"expected first element 'vertex', got {el0.name!r}")
    data = el0.data.copy()
    n = len(data)
    if n == 0:
        raise ValueError("empty vertex buffer")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    props = list(data.dtype.names)
    if "opacity" not in props or not all(f"scale_{i}" in props for i in range(3)):
        raise ValueError("unexpected ply schema (need opacity, scale_0..2)")

    op = _inverse_sigmoid(0.99)
    rng = np.random.default_rng(seed)
    new_rows: list[tuple[Any, ...]] = []

    def _template_index(center: np.ndarray) -> int:
        d2 = np.sum((xyz - center.reshape(1, 3)) ** 2, axis=1)
        return int(np.argmin(d2))

    def _append_points(
        centers: np.ndarray,
        rgb: tuple[float, float, float],
        *,
        count: int,
        jitter: float,
        log_scale: float,
    ) -> None:
        r, g, b = _rgb_to_f_dc(np.asarray(rgb, dtype=np.float64))
        for center in centers:
            ti = _template_index(center)
            for _ in range(count):
                row = data[ti].copy()
                j = rng.normal(scale=jitter, size=3)
                row["x"] = float(center[0] + j[0])
                row["y"] = float(center[1] + j[1])
                row["z"] = float(center[2] + j[2])
                row["nx"] = 0.0
                row["ny"] = 0.0
                row["nz"] = 0.0
                if "f_dc_0" in props:
                    row["f_dc_0"] = r
                    row["f_dc_1"] = g
                    row["f_dc_2"] = b
                for name in props:
                    if name.startswith("f_rest_"):
                        row[name] = 0.0
                row["opacity"] = op
                row["scale_0"] = log_scale
                row["scale_1"] = log_scale
                row["scale_2"] = log_scale
                if all(f"rot_{i}" in props for i in range(4)):
                    row["rot_0"] = 1.0
                    row["rot_1"] = 0.0
                    row["rot_2"] = 0.0
                    row["rot_3"] = 0.0
                new_rows.append(tuple(row))

    wf = obb_wireframe_samples(obb, step_m=wireframe_step_m)
    _append_points(
        wf, (0.05, 0.85, 0.95),
        count=wireframe_per_point, jitter=wireframe_jitter, log_scale=wireframe_log_scale,
    )
    _append_points(
        base_p.reshape(1, 3), (0.15, 0.45, 0.95),
        count=marker_count, jitter=marker_jitter, log_scale=marker_log_scale,
    )
    _append_points(
        lora_p.reshape(1, 3), (0.90, 0.15, 0.85),
        count=marker_count, jitter=marker_jitter, log_scale=marker_log_scale,
    )

    extra = np.array(new_rows, dtype=data.dtype)
    merged = np.empty(n + len(extra), dtype=data.dtype)
    merged[:n] = data
    merged[n:] = extra
    out_el = PlyElement.describe(merged, "vertex")
    out_ply = PlyData([out_el])

    out_path: Path | None = None
    if out_iteration_dir is not None:
        out_dir = out_iteration_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "point_cloud.ply"
        out_ply.write(str(out_path))
    if output is not None:
        outp = output.resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_ply.write(str(outp))
        out_path = outp
    if out_path is None:
        raise ValueError("set --out-iteration-dir and/or --output")

    half = _half_extent(obb)
    base_hit = obb_hit(base_p, obb)
    lora_hit = obb_hit(lora_p, obb)
    print(
        f"[inject_obb_compare] vertices={len(merged)} (+{len(extra)}) "
        f"base_hit={base_hit} lora_hit={lora_hit}"
    )
    print(f"[inject_obb_compare] base_local={obb_local(base_p, obb).round(4).tolist()}")
    print(f"[inject_obb_compare] lora_local={obb_local(lora_p, obb).round(4).tolist()}")
    print(f"[inject_obb_compare] half_extent={half.round(4).tolist()}")
    print(f"[ok] wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject OBB wireframe + Base/LoRA points for SIBR")
    ap.add_argument("--ply", type=Path, default=_REPO / "3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply")
    ap.add_argument("--bbox", type=Path, default=_REPO / "docs/bbox_data2.json")
    ap.add_argument("--object", "--object-key", dest="object_key", default=None)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--runs-root", type=Path, default=_REPO / "3DGS/test2/runs")
    ap.add_argument("--base-fused", type=Path, default=None)
    ap.add_argument("--lora-fused", type=Path, default=None)
    ap.add_argument("--out-iteration-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--model-path", type=Path, default=_REPO / "3DGS/gaussian-splatting/output/data2")
    ap.add_argument("--wireframe-step", type=float, default=0.035, help="Sample spacing (m) along OBB edges")
    ap.add_argument("--marker-count", type=int, default=18, help="Gaussians per Base/LoRA point")
    ap.add_argument("--wireframe-log-scale", type=float, default=-5.0, help="log scale for wireframe (more negative = thinner)")
    ap.add_argument("--marker-log-scale", type=float, default=-4.2, help="log scale for Base/LoRA markers")
    ap.add_argument("--all-presets", action="store_true", help="Inject tape + shaver + rabbit presets")
    ap.add_argument(
        "--gif-presets",
        action="store_true",
        help="Inject electric_shaver + brown_rabbit (same as demo/teaser_3d_*.gif)",
    )
    args = ap.parse_args()

    objects = load_bbox(args.bbox)
    presets = list(PRESETS.values()) if args.all_presets else []
    if args.gif_presets:
        presets = [PRESETS[k] for k in GIF_OBJECT_PRESETS]
    if args.preset:
        presets = [PRESETS[args.preset]]
    if not presets:
        if args.object_key is None or args.base_fused is None or args.lora_fused is None:
            ap.error("provide --preset / --all-presets, or --object + --base-fused + --lora-fused")
        presets = [{
            "object_key": args.object_key,
            "base_run": None,
            "lora_run": None,
            "out_name": "iteration_obb_custom",
        }]

    written: list[Path] = []
    for spec in presets:
        obj_key = spec["object_key"]
        obj = objects.get(obj_key)
        if obj is None or "obb" not in obj:
            raise SystemExit(f"no OBB for {obj_key} in {args.bbox}")

        if spec.get("base_run"):
            base_fused = args.runs_root / spec["base_run"] / "fused.json"
            lora_fused = args.runs_root / spec["lora_run"] / "fused_ray.json"
            out_dir = args.model_path / "point_cloud" / spec["out_name"]
        else:
            base_fused = args.base_fused
            lora_fused = args.lora_fused
            out_dir = args.out_iteration_dir or (args.model_path / "point_cloud" / spec["out_name"])

        out_path = inject_obb_compare(
            ply=args.ply,
            obb=obj["obb"],
            base_p=_load_p_world(base_fused),
            lora_p=_load_p_world(lora_fused),
            out_iteration_dir=out_dir,
            output=args.output,
            wireframe_step_m=args.wireframe_step,
            marker_count=args.marker_count,
            wireframe_log_scale=args.wireframe_log_scale,
            marker_log_scale=args.marker_log_scale,
        )
        written.append(out_path)
        iter_name = out_dir.name
        model = args.model_path.resolve()
        print(
            "\nSIBR (from viewers/bin):\n"
            f'  .\\SIBR_gaussianViewer_app.exe -m "{model}" --iteration {iter_name}\n'
            "Legend: cyan=OBB wireframe, blue=Base fused.json, magenta=LoRA fused_ray.json\n"
        )

    if args.all_presets or len(presets) > 1:
        print(f"[done] wrote {len(written)} injected point clouds")


if __name__ == "__main__":
    main()
