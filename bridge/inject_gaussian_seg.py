#!/usr/bin/env python3
"""把视锥投票选中的高斯原地改色，并叠加手标 OBB 线框，写出 SIBR 可读 ply。

不写回 iteration_30000。选中高斯的位置 / 尺度 / 不透明度保持原样，只改 0 阶球谐颜色。

颜色（线性 RGB 0–1）：
  - 选中且在 OBB 内：亮品红
  - 选中但在 OBB 外：亮绿（假阳性，方便查验漏桌）
  - OBB 线框：青色（追加细高斯，不改原场景）
  - 可选种子 P_world：蓝色

示例：
  python bridge/inject_gaussian_seg.py \\
    --seg training_data/data2_shaver/gaussian_seg.json \\
    --out-iteration-dir 3DGS/gaussian-splatting/output/data2/point_cloud/iteration_seg_electric_shaver
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

from frustum_segment import OBJ_DIR_TO_BBOX_KEY  # noqa: E402
from obb_geom import load_bbox, obb_hits, obb_wireframe_samples, rgb_to_f_dc  # noqa: E402

# 查验默认色：品红=框内选中，亮绿=框外选中，青=OBB，蓝=融合种子
COLOR_IN_OBB = (0.95, 0.10, 0.85)
COLOR_OUT_OBB = (0.15, 0.95, 0.20)
COLOR_WIRE = (0.05, 0.85, 0.95)
COLOR_SEED = (0.15, 0.45, 0.95)

DEFAULT_PLY = _REPO / "3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply"
DEFAULT_MODEL = _REPO / "3DGS/gaussian-splatting/output/data2"
DEFAULT_BBOX = _REPO / "docs/bbox_data2.json"
DEFAULT_MIN_VOTE_FRAC = 0.5


def _inverse_sigmoid(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def sibr_iteration_arg(out_iteration_dir: Path) -> str:
    """SIBR ``--iteration X`` 对应目录 ``point_cloud/iteration_X``。"""
    name = out_iteration_dir.name
    prefix = "iteration_"
    if name.startswith(prefix) and len(name) > len(prefix):
        return name[len(prefix):]
    return name


def resolve_object_key(seg: dict[str, Any], object_key: str | None) -> str | None:
    if object_key:
        return object_key
    key = seg.get("bbox_key")
    if key:
        return str(key)
    obj_dir = Path(str(seg.get("object_dir") or "")).name
    return OBJ_DIR_TO_BBOX_KEY.get(obj_dir)


def _set_f_dc(data: np.ndarray, idx: np.ndarray, rgb: tuple[float, float, float]) -> None:
    if idx.size == 0:
        return
    props = data.dtype.names or ()
    r, g, b = rgb_to_f_dc(rgb)
    if "f_dc_0" in props:
        data["f_dc_0"][idx] = r
        data["f_dc_1"][idx] = g
        data["f_dc_2"][idx] = b
    for name in props:
        if name.startswith("f_rest_"):
            data[name][idx] = 0.0


def _append_markers(
    data: np.ndarray,
    xyz: np.ndarray,
    centers: np.ndarray,
    rgb: tuple[float, float, float],
    *,
    count: int,
    jitter: float,
    log_scale: float,
    opacity: float,
    rng: np.random.Generator,
) -> list[tuple[Any, ...]]:
    if centers.size == 0:
        return []
    props = list(data.dtype.names or ())
    r, g, b = rgb_to_f_dc(rgb)
    op = _inverse_sigmoid(opacity)
    rows: list[tuple[Any, ...]] = []
    for center in np.asarray(centers, dtype=np.float64).reshape(-1, 3):
        ti = int(np.argmin(np.sum((xyz - center.reshape(1, 3)) ** 2, axis=1)))
        for _ in range(count):
            row = data[ti].copy()
            j = rng.normal(scale=jitter, size=3)
            row["x"] = float(center[0] + j[0])
            row["y"] = float(center[1] + j[1])
            row["z"] = float(center[2] + j[2])
            if "nx" in props:
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
            rows.append(tuple(row))
    return rows


def inject_gaussian_seg(
    *,
    ply: Path,
    seg: dict[str, Any],
    out_iteration_dir: Path,
    obb: dict[str, Any] | None,
    object_key: str | None,
    in_color: tuple[float, float, float] = COLOR_IN_OBB,
    out_color: tuple[float, float, float] = COLOR_OUT_OBB,
    wire_color: tuple[float, float, float] = COLOR_WIRE,
    seed_color: tuple[float, float, float] = COLOR_SEED,
    add_obb: bool = True,
    add_seed: bool = True,
    max_recolor: int | None = None,
    margin: float = 0.0,
    wireframe_step_m: float = 0.035,
    wireframe_log_scale: float = -5.0,
    seed_count: int = 18,
    seed_log_scale: float = -4.2,
    seed: int = 0,
) -> dict[str, Any]:
    src = ply.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"缺少源 ply: {src}")
    out_dir_check = out_iteration_dir.resolve()
    if out_dir_check.name == "iteration_30000" or (
        out_dir_check == src.parent.resolve() and src.parent.name == "iteration_30000"
    ):
        raise ValueError("拒绝写回 iteration_30000，请换 --out-iteration-dir")

    ply_data = PlyData.read(str(src))
    el0 = ply_data.elements[0]
    if el0.name != "vertex":
        raise ValueError(f"期望首元素为 vertex，实际 {el0.name!r}")
    data = el0.data.copy()
    n = len(data)
    if n == 0:
        raise ValueError("vertex 为空")
    props = list(data.dtype.names or ())
    if "opacity" not in props or not all(f"scale_{i}" in props for i in range(3)):
        raise ValueError("ply schema 异常（需要 opacity, scale_0..2）")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    idx_all = np.asarray(seg.get("indices") or [], dtype=np.int64)
    idx_all = idx_all[(idx_all >= 0) & (idx_all < n)]
    rng = np.random.default_rng(seed)
    if max_recolor is not None and idx_all.size > int(max_recolor):
        idx = np.sort(rng.choice(idx_all, size=int(max_recolor), replace=False))
    else:
        idx = idx_all

    if obb is not None and idx.size:
        inside = obb_hits(xyz[idx], obb, margin=margin)
    elif idx.size:
        inside = np.ones(idx.size, dtype=bool)
    else:
        inside = np.zeros((0,), dtype=bool)
    idx_in = idx[inside]
    idx_out = idx[~inside]
    _set_f_dc(data, idx_in, in_color)
    _set_f_dc(data, idx_out, out_color)

    extra_rows: list[tuple[Any, ...]] = []
    n_wire = 0
    if add_obb:
        if obb is None:
            raise ValueError("叠加 OBB 需要 bbox，或改用 --no-obb")
        wf = obb_wireframe_samples(obb, step_m=wireframe_step_m)
        extra_rows.extend(
            _append_markers(
                data, xyz, wf, wire_color,
                count=1, jitter=0.002, log_scale=wireframe_log_scale,
                opacity=0.99, rng=rng,
            )
        )
        n_wire = len(wf)

    n_seed = 0
    if add_seed:
        pw = seg.get("P_world") or seg.get("centroid")
        if pw is not None:
            extra_rows.extend(
                _append_markers(
                    data, xyz, np.asarray(pw, dtype=np.float64).reshape(1, 3), seed_color,
                    count=seed_count, jitter=0.005, log_scale=seed_log_scale,
                    opacity=0.99, rng=rng,
                )
            )
            n_seed = seed_count

    if extra_rows:
        extra = np.array(extra_rows, dtype=data.dtype)
        merged = np.empty(n + len(extra), dtype=data.dtype)
        merged[:n] = data
        merged[n:] = extra
    else:
        merged = data

    out_dir = out_iteration_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "point_cloud.ply"
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(out_path))

    if obb is not None and idx_all.size:
        n_sel_in = int(obb_hits(xyz[idx_all], obb, margin=margin).sum())
    elif obb is None:
        n_sel_in = int(idx_all.size)
    else:
        n_sel_in = 0

    manifest: dict[str, Any] = {
        "object_key": object_key,
        "ply_src": str(src.as_posix()),
        "out_ply": str(out_path.as_posix()),
        "sibr_iteration": sibr_iteration_arg(out_dir),
        "n_gaussians": n,
        "n_selected": int(idx_all.size),
        "n_recolored": int(idx.size),
        "n_selected_in_obb": n_sel_in,
        "n_recolored_in_obb": int(idx_in.size),
        "n_recolored_out_obb": int(idx_out.size),
        "n_wireframe": n_wire,
        "n_seed_markers": n_seed,
        "min_vote": seg.get("min_vote"),
        "min_vote_frac": seg.get("min_vote_frac"),
        "vertices_written": int(len(merged)),
    }
    (out_dir / "inject_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _warn_vote_frac(seg: dict[str, Any]) -> None:
    frac = seg.get("min_vote_frac")
    if frac is None:
        print(
            f"[warn] gaussian_seg.json 无 min_vote_frac；当前默认投票门槛是 {DEFAULT_MIN_VOTE_FRAC}。"
            " 查验后若要改门槛，请重跑 frustum_segment 再注入。",
            flush=True,
        )
        return
    if abs(float(frac) - DEFAULT_MIN_VOTE_FRAC) > 1e-9:
        print(
            f"[warn] 该 seg 的 min_vote_frac={frac}，与默认 {DEFAULT_MIN_VOTE_FRAC} 不同。",
            flush=True,
        )


def _load_seg(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _print_sibr(model_path: Path, out_dir: Path, manifest: dict[str, Any]) -> None:
    model = model_path.resolve()
    it = sibr_iteration_arg(out_dir)
    print(
        f"[ok] wrote {manifest['out_ply']}  "
        f"recolor={manifest['n_recolored']}/{manifest['n_selected']} "
        f"in_obb={manifest['n_recolored_in_obb']} out_obb={manifest['n_recolored_out_obb']} "
        f"+wire={manifest['n_wireframe']}",
        flush=True,
    )
    print(
        "\nSIBR（必须先进入 viewers/bin）:\n"
        "  Set-Location E:\\GSrefer3D\\3DGS\\gaussian-splatting\\viewers\\bin\n"
        f'  .\\SIBR_gaussianViewer_app.exe -m "{model}" --iteration {it}\n'
        "图例: 品红=选中且在 OBB 内, 亮绿=选中但框外, 青=手标 OBB, 蓝=P_world 种子\n",
        flush=True,
    )


def _collect_seg_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = list(args.segs or [])
    for d in args.obj_dirs or []:
        paths.append(d / "gaussian_seg.json")
    if args.obj_glob:
        for d in sorted(p for p in Path().glob(args.obj_glob) if p.is_dir() and p.name != "data2_sft"):
            paths.append(d / "gaussian_seg.json")
    # 去重且保持顺序
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recolor frustum-voted Gaussians + overlay manual OBB for SIBR",
    )
    ap.add_argument("--seg", type=Path, action="append", dest="segs", help="gaussian_seg.json（可重复）")
    ap.add_argument("--obj-dir", type=Path, action="append", dest="obj_dirs", help="含 gaussian_seg.json 的物体目录")
    ap.add_argument("--obj-glob", type=str, default=None, help="如 training_data/data2_*（排除 data2_sft）")
    ap.add_argument("--ply", type=Path, default=None, help="源 point_cloud.ply；默认读 seg['ply'] 或 iteration_30000")
    ap.add_argument("--bbox", type=Path, default=DEFAULT_BBOX)
    ap.add_argument("--object", "--object-key", dest="object_key", default=None)
    ap.add_argument("--out-iteration-dir", type=Path, default=None, help="仅单物体时可用；勿指向 iteration_30000")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--in-color", type=float, nargs=3, default=COLOR_IN_OBB, metavar=("R", "G", "B"))
    ap.add_argument("--out-color", type=float, nargs=3, default=COLOR_OUT_OBB, metavar=("R", "G", "B"))
    ap.add_argument("--no-obb", action="store_true", help="只改色，不叠加 OBB 线框")
    ap.add_argument("--no-seed", action="store_true", help="不插入 P_world 蓝点")
    ap.add_argument("--max-recolor", type=int, default=None, help="选中过多时随机抽这么多个上色")
    ap.add_argument("--margin", type=float, default=0.0, help="OBB 判定外扩（米）")
    ap.add_argument("--wireframe-step", type=float, default=0.035)
    ap.add_argument("--wireframe-log-scale", type=float, default=-5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seg_paths = _collect_seg_paths(args)
    if not seg_paths:
        ap.error("请提供 --seg / --obj-dir / --obj-glob")
    if args.out_iteration_dir is not None and len(seg_paths) != 1:
        ap.error("--out-iteration-dir 只能配单个 --seg / --obj-dir")

    objects = load_bbox(args.bbox) if args.bbox.is_file() else {}
    written = 0
    for seg_path in seg_paths:
        try:
            seg = _load_seg(seg_path)
        except FileNotFoundError as e:
            print(f"[skip] {e}", flush=True)
            continue
        _warn_vote_frac(seg)
        key = resolve_object_key(seg, args.object_key)
        ply = args.ply
        if ply is None:
            ply = Path(seg["ply"]) if seg.get("ply") else DEFAULT_PLY
        if args.out_iteration_dir is not None:
            out_dir = args.out_iteration_dir
        elif key:
            out_dir = args.model_path / "point_cloud" / f"iteration_seg_{key}"
        else:
            raise SystemExit(f"{seg_path}: 无法推断 object key，请传 --object 或 --out-iteration-dir")

        obb = None
        if not args.no_obb:
            if key is None:
                raise SystemExit(f"{seg_path}: 叠加 OBB 需要 --object 或 json 里的 bbox_key")
            spec = objects.get(key)
            if spec is None or "obb" not in spec:
                raise SystemExit(f"{args.bbox} 中没有 {key} 的 OBB")
            obb = spec["obb"]

        manifest = inject_gaussian_seg(
            ply=ply,
            seg=seg,
            out_iteration_dir=out_dir,
            obb=obb,
            object_key=key,
            in_color=tuple(args.in_color),
            out_color=tuple(args.out_color),
            add_obb=not args.no_obb,
            add_seed=not args.no_seed,
            max_recolor=args.max_recolor,
            margin=args.margin,
            wireframe_step_m=args.wireframe_step,
            wireframe_log_scale=args.wireframe_log_scale,
            seed=args.seed,
        )
        _print_sibr(args.model_path, out_dir, manifest)
        written += 1
    if written == 0:
        raise SystemExit("没有写出任何注入 ply")


if __name__ == "__main__":
    main()
