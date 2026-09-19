#!/usr/bin/env python3
"""把二维像素反投影到 3DGS 场景的世界坐标系。

约定与 render.py 写入 camera_params/view_XXX.json 的字段一致：
- ``rotation``：JSON 里实际是 3DGS 的 R_c2w（camera-to-world）；``from_json`` 会转置成 R_w2c。
- ``position``：世界系相机中心 C（== view.camera_center）。
- ``fov_x`` / ``fov_y``：弧度；内参按主点在 (W/2, H/2)、方形像素推导。

更新后的 render.py 保存的深度栅格（``depth_raw/view_XXX.npy``，
kind="expected_invdepth"）用 ``z_cam = 1 / max(inv, eps)``。
与 view_000（nx=0.458, ny=0.298）端到端验证一致：P_world 落在高斯附近
（场景对角约 189 时 NN 距离约 0.25）。

作为脚本运行时保持旧 ``minimal_unproject.py`` 的单像素打印行为，
方便已有笔记和 verify_unproject_vs_pointcloud.py 继续加载。
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraView:
    """针孔反投影所需的单视角 JSON 字段子集。"""

    width: int
    height: int
    fov_x: float
    fov_y: float
    R_w2c: np.ndarray  # (3, 3)
    camera_center: np.ndarray  # (3,)

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraView":
        with Path(path).open("r", encoding="utf-8") as f:
            d = json.load(f)
        # render.py 保存的 view.R 是 3DGS 约定下的 R_c2w（相机到世界），反投影需要 R_w2c，故转置。
        R_c2w = np.asarray(d["rotation"], dtype=np.float64)
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fov_x=float(d["fov_x"]),
            fov_y=float(d["fov_y"]),
            R_w2c=R_c2w.T,
            camera_center=np.asarray(d["position"], dtype=np.float64),
        )

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        fx = self.width / (2.0 * math.tan(self.fov_x / 2.0))
        fy = self.height / (2.0 * math.tan(self.fov_y / 2.0))
        cx = self.width / 2.0
        cy = self.height / 2.0
        return fx, fy, cx, cy


class Unprojector:
    """按 3DGS render.py 约定，由像素 + 深度得到世界点。"""

    def __init__(self, view: CameraView, *, inv_eps: float = 1e-6) -> None:
        self.view = view
        self.inv_eps = inv_eps

    # --- 深度辅助 ----------------------------------------------------------

    def z_from_invdepth(self, inv: float) -> float:
        return 1.0 / max(float(inv), self.inv_eps)

    def sample_depth_raw(
        self,
        depth_raw_npy: str | Path,
        u: int,
        v: int,
        *,
        kind: str = "expected_invdepth",
    ) -> tuple[float, float]:
        """返回 ``(栅格存值, z_cam)``。

        ``kind="expected_invdepth"`` 与 render.py 写出的一致；若将来渲染器
        直接写相机系 z，则传 ``kind="linear_z"``。
        """
        arr = np.load(depth_raw_npy)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        val = float(arr[v, u])
        if kind == "expected_invdepth":
            return val, self.z_from_invdepth(val)
        if kind == "linear_z":
            return val, val
        raise ValueError(f"unknown depth kind: {kind!r}")

    # --- 核心反投影 --------------------------------------------------------

    def normalized_to_pixel(self, nx: float, ny: float) -> tuple[int, int]:
        w, h = self.view.width, self.view.height
        u = int(np.clip(nx * w, 0, w - 1))
        v = int(np.clip(ny * h, 0, h - 1))
        return u, v

    def pixel_to_world(self, u: float, v: float, z_cam: float) -> tuple[np.ndarray, np.ndarray]:
        fx, fy, cx, cy = self.view.intrinsics
        x = (u - cx) * z_cam / fx
        y = (v - cy) * z_cam / fy
        p_cam = np.array([x, y, z_cam], dtype=np.float64)
        p_world = self.view.R_w2c.T @ p_cam + self.view.camera_center
        return p_cam, p_world

    def normalized_to_world(
        self,
        nx: float,
        ny: float,
        z_cam: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        u, v = self.normalized_to_pixel(nx, ny)
        # 投影用亚像素浮点 (nx*W, ny*H)；取整后的 (u,v) 只给调用方索引栅格。
        fu = nx * self.view.width
        fv = ny * self.view.height
        p_cam, p_world = self.pixel_to_world(fu, fv, z_cam)
        return p_cam, p_world, (u, v)

    def normalized_with_depth_raw(
        self,
        nx: float,
        ny: float,
        depth_raw_npy: str | Path,
        *,
        kind: str = "expected_invdepth",
    ) -> dict:
        u, v = self.normalized_to_pixel(nx, ny)
        stored, z_cam = self.sample_depth_raw(depth_raw_npy, u, v, kind=kind)
        p_cam, p_world, _ = self.normalized_to_world(nx, ny, z_cam)
        return {
            "pixel": (u, v),
            "stored_value": stored,
            "depth_kind": kind,
            "z_cam": z_cam,
            "P_cam": p_cam,
            "P_world": p_world,
        }


# ---------------------------------------------------------------------------
# 兼容旧加载器 / CLI（verify_unproject_vs_pointcloud.py）
# ---------------------------------------------------------------------------


def load_camera(path: str | Path) -> dict:
    """旧式加载器，供 verify_unproject_vs_pointcloud.py 继续使用。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def linear_z_from_raster_npy(
    path: str | Path,
    u: int,
    v: int,
    inv_to_z: bool,
    eps: float,
) -> tuple[float | None, float]:
    arr = np.load(path)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    val = float(arr[v, u])
    if inv_to_z:
        return val, 1.0 / max(val, eps)
    return None, val


def unproject(
    nx: float,
    ny: float,
    cam: dict,
    depth_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    view = CameraView(
        width=int(cam["width"]),
        height=int(cam["height"]),
        fov_x=float(cam["fov_x"]),
        fov_y=float(cam["fov_y"]),
        R_w2c=np.asarray(cam["rotation"], dtype=np.float64),
        camera_center=np.asarray(cam["position"], dtype=np.float64),
    )
    fu = nx * view.width
    fv = ny * view.height
    return Unprojector(view).pixel_to_world(fu, fv, depth_z)


def _cli(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Unproject normalized (nx, ny) using a 3DGS custom-view JSON + depth source."
    )
    p.add_argument("--camera", type=Path, required=True, help="camera_params/view_XXX.json")
    p.add_argument("--nx", type=float, required=True)
    p.add_argument("--ny", type=float, required=True)
    p.add_argument(
        "--depth-npy",
        type=Path,
        default=None,
        help="depth_raw/view_XXX.npy (expected_invdepth by default)",
    )
    p.add_argument(
        "--depth-linear-z",
        action="store_true",
        help="If set, the .npy stores camera-space z instead of expected_invdepth.",
    )
    p.add_argument("--depth-png", type=Path, default=None, help="8-bit depth PNG (legacy)")
    p.add_argument("--z-near", type=float, default=0.5)
    p.add_argument("--z-far", type=float, default=5.0)
    p.add_argument("--z-constant", type=float, default=None)
    p.add_argument("--inv-eps", type=float, default=1e-6)
    args = p.parse_args(list(argv) if argv is not None else None)

    view = CameraView.from_json(args.camera)
    unp = Unprojector(view, inv_eps=args.inv_eps)
    u, v = unp.normalized_to_pixel(args.nx, args.ny)

    stored = None
    if args.z_constant is not None:
        z = float(args.z_constant)
    elif args.depth_npy is not None:
        kind = "linear_z" if args.depth_linear_z else "expected_invdepth"
        stored, z = unp.sample_depth_raw(args.depth_npy, u, v, kind=kind)
    elif args.depth_png is not None:
        from PIL import Image

        img = np.array(Image.open(args.depth_png))
        g = float(img[v, u, 0]) if img.ndim == 3 else float(img[v, u])
        z = float(args.z_near + (g / 255.0) * (args.z_far - args.z_near))
    else:
        z = 1.0
        print("[info] no depth source and no --z-constant: using z=1.0 (arbitrary unit)")

    p_cam, p_world, _ = unp.normalized_to_world(args.nx, args.ny, z)
    print("pixel (u,v):", u, v, "  depth_z (linear z_cam):", z)
    if stored is not None and not args.depth_linear_z:
        print("raster buffer value at pixel (expected_invdepth):", stored)
    print("P_cam:", p_cam)
    print("P_world:", p_world)


if __name__ == "__main__":
    _cli()
