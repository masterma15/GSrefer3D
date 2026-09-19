#!/usr/bin/env python3
"""多视 mask 视锥过滤 + T(z) 加权投票 + 簇深度切片 → 目标高斯集合（P0/P1）。

输入（物体目录，如 training_data/data2_shaver）：
  projections_kept.json、mask/view_*.png、filter_report.json（含 P_world 与 T_at_z_lo）。
相机与 ply 默认读 filter_report，可用 CLI 覆盖。

流程：
  1. 只用 kept 视角（T(z) 已拒帧）。
  2. 将全体高斯中心 μ 投到该视 mask：落在前景即视锥命中。
  3. 投票：命中则加上该视权重（P1：权重=T_at_z_lo，缺省 1）。
  4. 深度切片：至少在一个命中视角上，z_cam 落在簇深度带 [z_lo,z_hi]（可放宽）。
  5. 写出 gaussian_seg.json 与选中点的 xyzrgb ply（不是语言场，也不上 BEV/虚相机）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from apply_filter_batch import FUSED_RUNS  # noqa: E402
from gen_training_data import _intrinsics  # noqa: E402
from ray_visibility import cluster_depth_band  # noqa: E402

# 训练目录名 → bbox_data2.json 的 object key
OBJ_DIR_TO_BBOX_KEY: dict[str, str] = {
    "data2_shaver": "electric_shaver",
    "data2_rabbit": "brown_rabbit",
    "data2_golden_retriever": "golden_retriever",
    "data2_bowl": "golden_bowl",
    "data2_umbrella": "umbrella",
    "data2_toy_cake": "toy_cake",
    "data2_cookie": "cookie_bag",
    "data2_medicine_bottle": "medicine_bottle",
    "data2_bracelet": "bracelet",
    "data2_hair_clip": "hair_clip",
    "data2_tape": "double_sided_tape",
}

# apply_filter_batch 的 20260516 run 若不在磁盘，用 §7 评测 run
_FALLBACK_RUN_IDS: dict[str, tuple[str, ...]] = {
    "data2_shaver": ("20260519_143457_4c3b9a32", "20260519_170540_4c3b9a32"),
    "data2_rabbit": ("20260519_144845_147bac82", "20260519_171359_147bac82"),
    "data2_golden_retriever": ("20260519_154013_f8dbfcc3", "20260519_172627_f8dbfcc3"),
    "data2_bowl": ("20260519_154649_8d83a715", "20260519_173958_8d83a715"),
    "data2_umbrella": ("20260519_160219_d7bab60f", "20260519_174835_d7bab60f"),
    "data2_toy_cake": ("20260519_161222_7dd80c38", "20260519_175733_7dd80c38"),
    "data2_cookie": ("20260519_162019_e51c780a", "20260519_180800_e51c780a"),
    "data2_medicine_bottle": ("20260519_163004_04149b86", "20260519_182106_04149b86"),
    "data2_bracelet": ("20260519_163546_cb2e562f", "20260520_001018_cb2e562f"),
    "data2_hair_clip": ("20260519_164312_65bf02a5", "20260519_183039_65bf02a5"),
    "data2_tape": ("20260519_132142_6c883d56", "20260519_000313_6c883d56"),
}

# apply_filter_batch 的 20260516 run 若不在磁盘，用 §7 评测 run
_FALLBACK_RUN_IDS: dict[str, tuple[str, ...]] = {
    "data2_shaver": ("20260519_143457_4c3b9a32", "20260519_170540_4c3b9a32"),
    "data2_rabbit": ("20260519_144845_147bac82", "20260519_171359_147bac82"),
    "data2_golden_retriever": ("20260519_154013_f8dbfcc3", "20260519_172627_f8dbfcc3"),
    "data2_bowl": ("20260519_154649_8d83a715", "20260519_173958_8d83a715"),
    "data2_umbrella": ("20260519_160219_d7bab60f", "20260519_174835_d7bab60f"),
    "data2_toy_cake": ("20260519_161222_7dd80c38", "20260519_175733_7dd80c38"),
    "data2_cookie": ("20260519_162019_e51c780a", "20260519_180800_e51c780a"),
    "data2_medicine_bottle": ("20260519_163004_04149b86", "20260519_182106_04149b86"),
    "data2_bracelet": ("20260519_163546_cb2e562f", "20260520_001018_cb2e562f"),
    "data2_hair_clip": ("20260519_164312_65bf02a5", "20260519_183039_65bf02a5"),
    "data2_tape": ("20260519_132142_6c883d56", "20260519_000313_6c883d56"),
}


def load_ply_xyz(path: Path) -> np.ndarray:
    """只读 ply 的 xyz，供视锥投影。"""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply.elements[0]
    return np.stack(
        [np.asarray(v["x"], dtype=np.float64), np.asarray(v["y"], dtype=np.float64), np.asarray(v["z"], dtype=np.float64)],
        axis=1,
    )


def world_to_cam_uvz(xyz: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量：世界点 → (u, v, z_cam)。z<=0 的 u/v 为 nan。

    与 ``gen_training_data.world_to_image`` 同一外参约定：JSON rotation 为 R_c2w。
    """
    r_c2w = np.asarray(cam["rotation"], dtype=np.float64)
    c = np.asarray(cam["position"], dtype=np.float64)
    p_cam = (xyz - c.reshape(1, 3)) @ r_c2w
    z = p_cam[:, 2]
    fx, fy, cx, cy = _intrinsics(cam)
    z_ok = z > 1e-8
    z_safe = np.where(z_ok, z, np.nan)
    u = fx * p_cam[:, 0] / z_safe + cx
    v = fy * p_cam[:, 1] / z_safe + cy
    return u, v, z


def frustum_hit_mask(
    xyz: np.ndarray,
    cam: dict,
    mask: np.ndarray,
    *,
    fg_thresh: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """视锥命中：μ 投到像平面且落在 mask 前景。

    返回 (hit, z_cam)，hit 为 (N,) bool。
    """
    u, v, z = world_to_cam_uvz(xyz, cam)
    h, w = int(mask.shape[0]), int(mask.shape[1])
    valid = np.isfinite(u) & np.isfinite(v) & (z > 1e-8)
    hit = np.zeros(xyz.shape[0], dtype=bool)
    if np.any(valid):
        ui = np.floor(u[valid]).astype(np.int64)
        vi = np.floor(v[valid]).astype(np.int64)
        in_img = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        sel = np.nonzero(valid)[0][in_img]
        hit[sel] = mask[vi[in_img], ui[in_img]] >= fg_thresh
    return hit, z


def select_cluster_xyz(
    xyz: np.ndarray,
    p_world: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """P_world 邻域内的高斯中心，用于算各视簇深度带。"""
    d = np.linalg.norm(xyz - p_world.reshape(1, 3), axis=1)
    idx = np.nonzero(d <= radius)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(d))], dtype=np.int64)
    return xyz[idx]


def view_weight_from_report(entry: dict[str, Any] | None, *, use_t_weight: bool) -> float:
    """P1：kept 视角权重用 T(z)；无 T 或关闭加权则为 1。拒帧权重为 0。"""
    if entry is None:
        return 1.0
    if entry.get("status") == "rejected":
        return 0.0
    if not use_t_weight:
        return 1.0
    t = entry.get("T_at_z_lo")
    if t is None:
        return 1.0
    return float(np.clip(float(t), 0.0, 1.0))


def load_kept_view_ids(obj_dir: Path) -> list[str]:
    kept_path = obj_dir / "projections_kept.json"
    if not kept_path.is_file():
        raise FileNotFoundError(f"缺少 {kept_path}，请先跑 filter_views_3dgs / apply_filter_batch")
    recs = json.loads(kept_path.read_text(encoding="utf-8"))
    return [str(r["view_id"]).zfill(3) for r in recs]


def load_filter_view_index(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for v in report.get("views") or []:
        out[str(v["view_id"]).zfill(3)] = v
    return out


def resolve_mask_path(obj_dir: Path, view_id: str) -> Path | None:
    p = obj_dir / "mask" / f"view_{view_id}.png"
    return p if p.is_file() else None


def segment_gaussians(
    xyz: np.ndarray,
    *,
    p_world: np.ndarray,
    views: list[dict[str, Any]],
    min_vote: float,
    cluster_radius: float,
    z_slack: float,
    apply_depth_slice: bool,
) -> dict[str, Any]:
    """对已装好相机/mask/权重的视角做投票与深度切片。

    views 每项：view_id, cam, mask, weight, width, height。
    """
    n = xyz.shape[0]
    votes = np.zeros(n, dtype=np.float64)
    # 深度：任一命中视角上 z 落在该视簇带内
    depth_ok = np.zeros(n, dtype=bool)
    per_view_stats: list[dict[str, Any]] = []

    cluster_xyz = select_cluster_xyz(xyz, p_world, radius=cluster_radius)

    for item in views:
        cam = item["cam"]
        mask = item["mask"]
        w = float(item["weight"])
        view_id = item["view_id"]
        hit, z_cam = frustum_hit_mask(xyz, cam, mask)
        votes += w * hit.astype(np.float64)
        n_hit = int(hit.sum())
        band = cluster_depth_band(
            cluster_xyz,
            cam,
            frame_margin=0,
            w=int(cam["width"]),
            h=int(cam["height"]),
        )
        z_lo = z_hi = None
        if apply_depth_slice and band is not None:
            z_lo, z_hi = band
            lo = z_lo * (1.0 - z_slack)
            hi = z_hi * (1.0 + z_slack)
            depth_ok |= hit & (z_cam >= lo) & (z_cam <= hi)
        per_view_stats.append(
            {
                "view_id": view_id,
                "weight": w,
                "frustum_hits": n_hit,
                "z_lo": z_lo,
                "z_hi": z_hi,
            }
        )

    voted = votes >= float(min_vote)
    if apply_depth_slice:
        # 没有任何视角算出深度带时，不因 depth_ok 全假而清空
        if np.any(depth_ok) or any(s["z_lo"] is not None for s in per_view_stats):
            selected = voted & depth_ok
        else:
            selected = voted
    else:
        selected = voted

    idx = np.nonzero(selected)[0]
    if idx.size:
        centroid = xyz[idx].mean(axis=0)
    else:
        centroid = p_world.copy()

    return {
        "indices": idx,
        "votes": votes,
        "selected": selected,
        "centroid": centroid,
        "per_view": per_view_stats,
        "n_selected": int(idx.size),
        "cluster_size": int(cluster_xyz.shape[0]),
    }


def write_xyzrgb_ply(path: Path, xyz: np.ndarray, rgb: tuple[int, int, int]) -> None:
    """选中高斯中心写成彩色点云，供 MeshLab / CloudCompare 叠看。"""
    from plyfile import PlyData, PlyElement

    n = xyz.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.empty(n, dtype=dtype)
    arr["x"] = xyz[:, 0].astype(np.float32)
    arr["y"] = xyz[:, 1].astype(np.float32)
    arr["z"] = xyz[:, 2].astype(np.float32)
    arr["red"] = np.uint8(rgb[0])
    arr["green"] = np.uint8(rgb[1])
    arr["blue"] = np.uint8(rgb[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def load_views_for_object(
    obj_dir: Path,
    views_root: Path,
    report_index: dict[str, dict[str, Any]],
    *,
    use_t_weight: bool,
) -> list[dict[str, Any]]:
    """组装 kept 且有 mask、相机的视角列表。"""
    from PIL import Image

    out: list[dict[str, Any]] = []
    for vid in load_kept_view_ids(obj_dir):
        mask_path = resolve_mask_path(obj_dir, vid)
        cam_path = views_root / "camera_params" / f"view_{vid}.json"
        if mask_path is None or not cam_path.is_file():
            continue
        entry = report_index.get(vid)
        weight = view_weight_from_report(entry, use_t_weight=use_t_weight)
        if weight <= 0.0:
            continue
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        mask = np.array(Image.open(mask_path).convert("L"))
        out.append(
            {
                "view_id": vid,
                "cam": cam,
                "mask": mask,
                "weight": weight,
                "mask_path": str(mask_path.as_posix()),
            }
        )
    return out


def resolve_p_world(
    obj_dir: Path,
    report: dict[str, Any],
    fused: Path | None,
) -> tuple[np.ndarray, str | None]:
    """P_world：--fused > filter_report > 已知 e2e run 的 fused.json。"""
    candidates: list[Path] = []
    if fused is not None:
        candidates.append(fused)
    fp = report.get("fused_path")
    if fp:
        candidates.append(Path(fp))
    run_id = FUSED_RUNS.get(obj_dir.name)
    if run_id:
        run_dir = _REPO / "3DGS" / "test2" / "runs" / run_id
        candidates.append(run_dir / "fused.json")
        candidates.append(run_dir / "fused_ray.json")
    # 训练用 20260516 run 可能已删，回退到 3D OBB 表里的 LoRA / Base run
    for rid in _FALLBACK_RUN_IDS.get(obj_dir.name, ()):
        run_dir = _REPO / "3DGS" / "test2" / "runs" / rid
        candidates.append(run_dir / "fused_ray.json")
        candidates.append(run_dir / "fused.json")
    for path in candidates:
        if path.is_file():
            p = np.asarray(json.loads(path.read_text(encoding="utf-8"))["P_world"], dtype=np.float64)
            return p, str(path.as_posix())
    if report.get("P_world"):
        return np.asarray(report["P_world"], dtype=np.float64), report.get("fused_path")
    raise FileNotFoundError(f"{obj_dir.name} 无 P_world（无 fused.json / filter_report.P_world）")


def run_object(
    obj_dir: Path,
    *,
    views_root: Path | None,
    ply: Path | None,
    fused: Path | None,
    min_vote: float,
    min_vote_frac: float | None,
    use_t_weight: bool,
    cluster_radius: float | None,
    z_slack: float,
    apply_depth_slice: bool,
    out_dir: Path | None,
) -> dict[str, Any]:
    obj_dir = obj_dir.resolve()
    report_path = obj_dir / "filter_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report_index = load_filter_view_index(report)

    if views_root is None:
        vr = report.get("views_root")
        views_root = Path(vr) if vr else _REPO / "3DGS" / "test2"
    views_root = views_root.resolve()

    if ply is None:
        pp = report.get("ply_path")
        ply = Path(pp) if pp else _REPO / "3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply"
    ply = ply.resolve()

    p_world, fused_used = resolve_p_world(obj_dir, report, fused)

    if cluster_radius is None:
        cluster_radius = float(report.get("cluster_radius_effective") or 0.2)

    views = load_views_for_object(obj_dir, views_root, report_index, use_t_weight=use_t_weight)
    if not views:
        raise SystemExit(f"{obj_dir.name}: 没有同时具备 kept + mask + 相机的视角")

    if min_vote_frac is not None and float(min_vote_frac) >= 0.0:
        min_vote = float(min_vote_frac) * len(views)
    print(f"[seg] {obj_dir.name} views={len(views)} min_vote={min_vote:.2f} ply={ply}", flush=True)
    xyz = load_ply_xyz(ply)
    result = segment_gaussians(
        xyz,
        p_world=p_world,
        views=views,
        min_vote=min_vote,
        cluster_radius=cluster_radius,
        z_slack=z_slack,
        apply_depth_slice=apply_depth_slice,
    )

    out_dir = (out_dir or obj_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = result["indices"]
    payload = {
        "object_dir": obj_dir.name,
        "bbox_key": OBJ_DIR_TO_BBOX_KEY.get(obj_dir.name),
        "ply": str(ply.as_posix()),
        "views_root": str(views_root.as_posix()),
        "fused": fused_used,
        "P_world": p_world.tolist(),
        "centroid": result["centroid"].tolist(),
        "n_gaussians": int(xyz.shape[0]),
        "n_selected": result["n_selected"],
        "n_views_used": len(views),
        "min_vote": min_vote,
        "min_vote_frac": min_vote_frac,
        "use_t_weight": use_t_weight,
        "cluster_radius": cluster_radius,
        "cluster_size": result["cluster_size"],
        "z_slack": z_slack,
        "apply_depth_slice": apply_depth_slice,
        "indices": idx.astype(int).tolist(),
        "per_view": result["per_view"],
    }
    json_path = out_dir / "gaussian_seg.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    ply_out = out_dir / "gaussian_seg.ply"
    if idx.size:
        write_xyzrgb_ply(ply_out, xyz[idx], (220, 40, 40))
    else:
        write_xyzrgb_ply(ply_out, p_world.reshape(1, 3), (220, 40, 40))

    print(
        f"[seg] selected={result['n_selected']}/{xyz.shape[0]} "
        f"centroid={result['centroid'].tolist()} -> {json_path}",
        flush=True,
    )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Frustum-filter 3DGS Gaussians from multi-view masks")
    ap.add_argument("--obj-dir", type=Path, action="append", dest="obj_dirs", help="training_data/data2_* (repeatable)")
    ap.add_argument("--obj-glob", type=str, default=None, help="e.g. training_data/data2_* (excludes data2_sft)")
    ap.add_argument("--views-root", type=Path, default=None)
    ap.add_argument("--ply", type=Path, default=None)
    ap.add_argument("--fused", type=Path, default=None)
    ap.add_argument("--min-vote", type=float, default=2.0, help="Keep Gaussian if weighted vote >= this")
    ap.add_argument(
        "--min-vote-frac",
        type=float,
        default=0.5,
        help="min_vote = this * n_used_views (overrides --min-vote). 默认 0.5；查验后可改 0.4/0.6。负数则退回 --min-vote。",
    )
    ap.add_argument("--no-t-weight", action="store_true", help="Each kept view votes 1 (ignore T(z))")
    ap.add_argument("--cluster-radius", type=float, default=None, help="Override filter_report cluster radius")
    ap.add_argument("--z-slack", type=float, default=0.15, help="Expand [z_lo,z_hi] by this fraction")
    ap.add_argument("--no-depth-slice", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None, help="Only valid with a single --obj-dir")
    args = ap.parse_args()

    dirs: list[Path] = list(args.obj_dirs or [])
    if args.obj_glob:
        dirs.extend(sorted(p for p in Path().glob(args.obj_glob) if p.is_dir() and p.name != "data2_sft"))
    if not dirs:
        ap.error("provide --obj-dir and/or --obj-glob")
    if args.out_dir is not None and len(dirs) != 1:
        ap.error("--out-dir requires exactly one --obj-dir")

    failed = 0
    for d in dirs:
        try:
            run_object(
                d,
                views_root=args.views_root,
                ply=args.ply,
                fused=args.fused,
                min_vote=args.min_vote,
                min_vote_frac=args.min_vote_frac,
                use_t_weight=not args.no_t_weight,
                cluster_radius=args.cluster_radius,
                z_slack=args.z_slack,
                apply_depth_slice=not args.no_depth_slice,
                out_dir=args.out_dir,
            )
        except (FileNotFoundError, SystemExit) as e:
            failed += 1
            print(f"[skip] {d}: {e}", flush=True)
    if failed:
        print(f"[seg] done with {failed} skipped", flush=True)


if __name__ == "__main__":
    main()
