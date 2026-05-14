#!/usr/bin/env python3
"""Fuse per-view RoboRefer predictions into a single 3D point.

Input: predictions.json produced by bridge/roborefer_client.py.

Pipeline:
  1. For every (view, point) pair, look up the matching depth_raw .npy and
     camera JSON, then unproject (nx, ny) -> world coordinates. Points whose
     stored expected_invdepth is below ``--min-inv`` (==> behind the far plane,
     z_cam blowing up) are dropped.
  2. RANSAC vote: for each candidate world point P_i, count how many other
     candidates are within ``--inlier-radius`` of P_i. Keep the candidate with
     the largest support; ties broken by smallest mean distance.
  3. Refine: take the geometric median of inliers (Weiszfeld iterations),
     which is robust to remaining outliers vs a plain mean.

Optional: snap to nearest Gaussian in point_cloud.ply when --ply is given.

Output: fused.json with the final P_world, inlier set, per-view candidates,
and provenance fields.

Run inside the gaussian_splatting conda env (numpy + plyfile).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "bridge"))

from unproject import CameraView, Unprojector  # noqa: E402


@dataclass
class Candidate:
    view_id: int
    point_idx: int
    nx: float
    ny: float
    pixel: tuple[int, int]
    invdepth: float
    z_cam: float
    P_world: np.ndarray
    rgb_path: str
    camera_path: str
    depth_raw_path: str

    def to_record(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "point_idx": self.point_idx,
            "nx": self.nx,
            "ny": self.ny,
            "pixel": list(self.pixel),
            "invdepth": self.invdepth,
            "z_cam": self.z_cam,
            "P_world": self.P_world.tolist(),
            "rgb_path": self.rgb_path,
            "camera_path": self.camera_path,
            "depth_raw_path": self.depth_raw_path,
        }


@dataclass
class FusionResult:
    P_world: np.ndarray
    inlier_indices: list[int]
    support: int
    inlier_radius: float
    snapped_to_gaussian: bool = False
    snap_distance: float | None = None
    candidates: list[Candidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "P_world": self.P_world.tolist(),
            "support": self.support,
            "num_candidates": len(self.candidates),
            "inlier_indices": self.inlier_indices,
            "inlier_radius": self.inlier_radius,
            "snapped_to_gaussian": self.snapped_to_gaussian,
            "snap_distance": self.snap_distance,
            "candidates": [c.to_record() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# Candidate gathering
# ---------------------------------------------------------------------------


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / rel)


def gather_candidates(
    predictions: dict[str, Any],
    *,
    min_inv: float,
    inv_eps: float = 1e-6,
    depth_dir: Path | None = None,
) -> list[Candidate]:
    root = Path(predictions["root"])
    out: list[Candidate] = []
    for v in predictions["views"]:
        if not v.get("parse_ok"):
            continue
        if "visible" in v and not v["visible"]:
            continue
        cam_path = _resolve(root, v["camera_path"])
        if depth_dir is not None:
            view_stem = Path(v["depth_raw_path"]).stem
            depth_path = depth_dir / f"{view_stem}.npy"
        else:
            depth_path = _resolve(root, v["depth_raw_path"])
        rgb_path = _resolve(root, v["rgb_path"])
        if not cam_path.is_file() or not depth_path.is_file():
            print(f"[warn] view {v['view_id']}: missing camera/depth artifact, skip", file=sys.stderr)
            continue
        view = CameraView.from_json(cam_path)
        unp = Unprojector(view, inv_eps=inv_eps)
        for pi, pt in enumerate(v["points"]):
            nx = float(pt["nx"])
            ny = float(pt["ny"])
            u, v_pix = unp.normalized_to_pixel(nx, ny)
            inv, z_cam = unp.sample_depth_raw(depth_path, u, v_pix, kind="expected_invdepth")
            if inv < min_inv:
                # depth too small -> z explodes; treat as miss
                continue
            _, p_world, _ = unp.normalized_to_world(nx, ny, z_cam)
            out.append(Candidate(
                view_id=int(v["view_id"]),
                point_idx=pi,
                nx=nx,
                ny=ny,
                pixel=(u, v_pix),
                invdepth=float(inv),
                z_cam=float(z_cam),
                P_world=np.asarray(p_world, dtype=np.float64),
                rgb_path=str(rgb_path).replace("\\", "/"),
                camera_path=str(cam_path).replace("\\", "/"),
                depth_raw_path=str(depth_path).replace("\\", "/"),
            ))
    return out


# ---------------------------------------------------------------------------
# RANSAC + geometric median
# ---------------------------------------------------------------------------


def ransac_select_inliers(
    points: np.ndarray,  # (N, 3)
    radius: float,
) -> tuple[list[int], int]:
    """Pick the candidate with the most neighbours within ``radius``.

    Returns (inlier_indices, support) where support == len(inlier_indices).
    Ties broken by the candidate with the smallest mean distance to its inliers.
    """
    n = points.shape[0]
    if n == 0:
        return [], 0
    # pairwise distance matrix is fine for typical N <= a few dozens
    diff = points[:, None, :] - points[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    within = d <= radius
    counts = within.sum(axis=1)
    best = -1
    best_count = -1
    best_mean = float("inf")
    for i in range(n):
        c = int(counts[i])
        if c > best_count:
            best, best_count = i, c
            best_mean = float(d[i, within[i]].mean())
            continue
        if c == best_count:
            mean_d = float(d[i, within[i]].mean())
            if mean_d < best_mean:
                best, best_mean = i, mean_d
    inliers = [j for j in range(n) if within[best, j]]
    return inliers, len(inliers)


def geometric_median(points: np.ndarray, *, iters: int = 64, eps: float = 1e-7) -> np.ndarray:
    """Weiszfeld's algorithm. Returns the L1 (median) center of `points`."""
    if points.shape[0] == 1:
        return points[0].copy()
    x = points.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(points - x, axis=1)
        if np.any(d < eps):
            # if x already coincides with one point, return that point (median)
            return points[int(np.argmin(d))].copy()
        w = 1.0 / d
        x_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < eps:
            return x_new
        x = x_new
    return x


# ---------------------------------------------------------------------------
# Optional snap-to-gaussian
# ---------------------------------------------------------------------------


def load_ply_xyz(path: Path) -> np.ndarray:
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply.elements[0]
    return np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1
    ).astype(np.float64)


def snap_to_gaussian(p: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz)
        dist, idx = tree.query(p, k=1)
        return xyz[int(idx)].copy(), float(dist)
    except ImportError:
        d = np.linalg.norm(xyz - p, axis=1)
        i = int(np.argmin(d))
        return xyz[i].copy(), float(d[i])


# ---------------------------------------------------------------------------
# Top-level fuse()
# ---------------------------------------------------------------------------


def iterative_refine(
    pts: np.ndarray,
    inlier_indices: list[int],
    *,
    k: float = 2.0,
    max_iters: int = 5,
    min_points: int = 2,
) -> tuple[np.ndarray, list[int]]:
    """Iteratively refine inliers: compute geometric median, drop points
    farther than k * median_distance, repeat until stable."""
    indices = list(inlier_indices)
    for _ in range(max_iters):
        if len(indices) < min_points:
            break
        subset = pts[indices]
        center = geometric_median(subset)
        dists = np.linalg.norm(subset - center, axis=1)
        med_dist = float(np.median(dists))
        threshold = max(med_dist * k, 1e-6)
        keep = [indices[j] for j, d in enumerate(dists) if d <= threshold]
        if len(keep) == len(indices):
            break
        if len(keep) < min_points:
            break
        indices = keep
    center = geometric_median(pts[indices])
    return center, indices


def fuse(
    predictions: dict[str, Any],
    *,
    inlier_radius: float,
    min_inv: float,
    refine: bool = True,
    refine_k: float = 2.0,
    ply_path: Path | None = None,
    depth_dir: Path | None = None,
) -> FusionResult:
    cands = gather_candidates(predictions, min_inv=min_inv, depth_dir=depth_dir)
    if not cands:
        raise SystemExit("no usable candidates after gathering (check parse_ok / depth_raw / min-inv)")
    pts = np.stack([c.P_world for c in cands], axis=0)
    inliers, support = ransac_select_inliers(pts, inlier_radius)

    if refine and len(inliers) >= 2:
        fused, inliers = iterative_refine(pts, inliers, k=refine_k)
        support = len(inliers)
    else:
        fused = geometric_median(pts[inliers]) if inliers else pts.mean(axis=0)

    snapped = False
    snap_d: float | None = None
    if ply_path is not None:
        if not ply_path.is_file():
            print(f"[warn] ply not found, skip snap: {ply_path}", file=sys.stderr)
        else:
            xyz = load_ply_xyz(ply_path)
            fused, snap_d = snap_to_gaussian(fused, xyz)
            snapped = True

    return FusionResult(
        P_world=fused,
        inlier_indices=inliers,
        support=support,
        inlier_radius=inlier_radius,
        snapped_to_gaussian=snapped,
        snap_distance=snap_d,
        candidates=cands,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse per-view RoboRefer predictions into a single 3D point.")
    ap.add_argument("--predictions", type=Path, required=True, help="predictions.json from roborefer_client.py")
    ap.add_argument("--inlier-radius", type=float, default=0.5, help="RANSAC inlier threshold in scene units")
    ap.add_argument("--min-inv", type=float, default=1e-3,
                    help="Drop candidates whose expected_invdepth is below this (z_cam too far / sky)")
    ap.add_argument("--ply", type=Path, default=None, help="Optional point_cloud.ply for snap-to-gaussian")
    ap.add_argument("--no-refine", action="store_true", help="Disable iterative refinement after RANSAC")
    ap.add_argument("--refine-k", type=float, default=2.0, help="Refinement threshold multiplier (k * median_dist)")
    ap.add_argument("--output", type=Path, default=None, help="Where to write fused.json (default: predictions sibling)")
    ap.add_argument("--depth-dir", type=Path, default=None,
                    help="Override depth_raw directory (e.g. depth_raw_dav2/ for DAv2-aligned depth)")
    ap.add_argument("--exclude", type=int, nargs="+", default=None,
                    help="View IDs to exclude from fusion (e.g. --exclude 5 24)")
    args = ap.parse_args()

    with args.predictions.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    if args.exclude:
        exclude_set = set(args.exclude)
        for v in predictions["views"]:
            if v.get("view_id") in exclude_set:
                v["visible"] = False
                print(f"[exclude] view_id={v['view_id']}")

    result = fuse(
        predictions,
        inlier_radius=args.inlier_radius,
        min_inv=args.min_inv,
        refine=not args.no_refine,
        refine_k=args.refine_k,
        ply_path=args.ply,
        depth_dir=args.depth_dir,
    )

    out = args.output or (args.predictions.parent / "fused.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    p = result.P_world
    print(f"[summary] P_world=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
    print(f"          support={result.support}/{len(result.candidates)} inlier_radius={result.inlier_radius}")
    if result.snapped_to_gaussian:
        print(f"          snap_distance={result.snap_distance:.4f}")
    print(f"[summary] wrote {out}")


if __name__ == "__main__":
    main()
