#!/usr/bin/env python3
"""把 data3 厨房的投票高斯收成可见质心微调包。

目录对齐 ``training_data/data2_sft_ray``：``image/``、``depth/``、``location_point.json``。
标签是全场景染色权重 ≥ 0.5 的像素质心，不是 ``question.json`` 里的指代点。
提示词用各物体 question.json 里的短句。不写 ``data2_sft_ray``。

投票点云用 ``by_object/data3_*/定稿_pair_occ_from003_s015``（这份是软边缘之前的白集 > 0.03、最终 > 0.15）。代码默认最终门槛现为 0.15，这里仍读定稿，不改成软边缘重投的结果。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "bridge") not in sys.path:
    sys.path.insert(0, str(_REPO / "bridge"))

from export_ray_visible import _restore, _with_selected  # noqa: E402
from export_spatial_train import DEFAULT_SUFFIX, _format_answer  # noqa: E402
from render_object_weight import (  # noqa: E402
    VISIBLE_WEIGHT,
    _prepare_white_set,
    _restore_appearance,
    render_object_weight,
)
from visible_mask import (  # noqa: E402
    DEFAULT_HIGH_RATIO,
    DEFAULT_LOW_RATIO,
    classify_occlusion,
    mask_centroid,
    occlusion_ratio,
)

VIEWS = _REPO / "3DGS" / "test3"
SFT_OUT = _REPO / "training_data" / "data3_sft_ray"

# slug, 定稿投票目录, 与 question.json 相同的短句。
OBJECTS = [
    ("air_fryer", "by_object/data3_air_fryer/定稿_pair_occ_from003_s015", "Please point to the air fryer."),
    ("induction_cooktop", "by_object/data3_induction_cooktop/定稿_pair_occ_from003_s015", "Please point to the flat induction cooktop."),
    ("pink_charger_case", "by_object/data3_pink_charger_case/定稿_pair_occ_from003_s015", "Please point to the pink charger case."),
    ("white_peeler", "by_object/data3_white_peeler/定稿_pair_occ_from003_s015", "Please point to the white peeler."),
    ("white_thermos", "by_object/data3_white_thermos/定稿_pair_occ_from003_s015", "Please point to the white thermos."),
    ("steel_pot", "by_object/data3_steel_pot/定稿_pair_occ_from003_s015", "Please point to the stainless steel pot."),
    ("blue_bowl", "by_object/data3_blue_bowl/定稿_pair_occ_from003_s015", "Please point to the blue bowl."),
    ("blue_plate", "by_object/data3_blue_plate/定稿_pair_occ_from003_s015", "Please point to the blue plate."),
    ("pink_plate", "by_object/data3_pink_plate/定稿_pair_occ_from003_s015", "Please point to the pink plate."),
    ("ceramic_spoon", "by_object/data3_ceramic_spoon/定稿_pair_occ_from003_s015", "Please point to the small ceramic spoon."),
]


def _cameras() -> list[tuple[str, dict]]:
    cameras = []
    for cam_path in sorted((VIEWS / "camera_params").glob("view_*.json")):
        view_id = cam_path.stem.split("_")[-1].zfill(3)
        cameras.append((view_id, json.loads(cam_path.read_text(encoding="utf-8"))))
    if not cameras:
        raise RuntimeError(f"{VIEWS / 'camera_params'} 没有 view_*.json")
    return cameras


def _labels_path(slug: str) -> Path:
    return SFT_OUT / "_labels" / f"{slug}.json"


def _cache(slug: str) -> Path:
    return SFT_OUT / "_cache" / slug


def _labels_complete(path: Path, n_views: int) -> bool:
    if not path.is_file():
        return False
    rows = json.loads(path.read_text(encoding="utf-8"))
    return len(rows) == n_views and all("kind" in row for row in rows)


def _cached_map(cache: Path, kind: str, view_id: str, render_fn) -> str:
    npy = cache / f"{kind}_{view_id}.npy"
    started = cache / f"{kind}_{view_id}.started"
    failed = cache / f"{kind}_{view_id}.failed"
    if npy.is_file():
        if started.exists():
            started.unlink()
        return "ok"
    if failed.is_file():
        return "failed"
    if started.is_file():
        failed.write_text("cuda", encoding="utf-8")
        started.unlink()
        print(f"[sft] skip {kind} view_{view_id}: 上次光栅失败", flush=True)
        return "failed"
    with open(started, "w", encoding="utf-8") as handle:
        handle.write(view_id)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"[sft] {kind} view_{view_id}", flush=True)
    arr = render_fn()
    tmp = cache / f"{kind}_{view_id}.tmp.npy"
    with open(tmp, "wb") as handle:
        np.save(handle, arr)
    os.replace(tmp, npy)
    started.unlink()
    return "ok"


def _find(slug: str) -> tuple[str, str]:
    for item_slug, seg_dir, prompt in OBJECTS:
        if item_slug == slug:
            return seg_dir, prompt
    raise SystemExit(f"未知物体 {slug}")


def export_object(slug: str) -> None:
    from render_object_invdepth import load_gaussian_model

    seg_dir, prompt = _find(slug)
    cameras = _cameras()
    out = _labels_path(slug)
    if _labels_complete(out, len(cameras)):
        print(f"[sft] {slug} already {len(cameras)}", flush=True)
        return
    seg_path = _REPO / "training_data" / seg_dir / "gaussian_seg.json"
    seg = json.loads(seg_path.read_text(encoding="utf-8"))
    indices = np.asarray(seg.get("indices") or [], dtype=np.int64)
    ply = Path(seg["ply"])
    cache = _cache(slug)
    cache.mkdir(parents=True, exist_ok=True)
    model = load_gaussian_model(ply)
    saved = None
    selected = None
    try:
        _, saved_dc, saved_rest, saved_opacity = _prepare_white_set(model, indices)
        saved = (saved_dc, saved_rest, saved_opacity)
        selected = _with_selected(model, indices)
        for view_id, cam in cameras:
            _cached_map(cache, "alone", view_id, lambda cam=cam: render_object_weight(model, cam))
        _restore(model, selected)
        selected = None
        for view_id, cam in cameras:
            _cached_map(cache, "weight", view_id, lambda cam=cam: render_object_weight(model, cam))
    finally:
        if selected is not None:
            _restore(model, selected)
        if saved is not None:
            _restore_appearance(model, *saved)
        del model
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    n_label = 0
    kinds: dict[str, int] = {}
    for view_id, cam in cameras:
        weight_path = cache / f"weight_{view_id}.npy"
        alone_path = cache / f"alone_{view_id}.npy"
        if not weight_path.is_file():
            rows.append(
                {
                    "view_id": view_id,
                    "kind": "weight_failed",
                    "label": None,
                    "prompt": prompt,
                    "failed": True,
                }
            )
            kinds["weight_failed"] = kinds.get("weight_failed", 0) + 1
            continue
        green = np.load(weight_path) >= VISIBLE_WEIGHT
        centroid = mask_centroid(green)
        width = int(cam["width"])
        height = int(cam["height"])
        label = None
        if centroid is not None and width > 0 and height > 0:
            u, v = centroid
            label = {"u": u, "v": v, "nx": u / width, "ny": v / height}
            n_label += 1
        if not alone_path.is_file():
            kind = "silhouette_failed"
            ratio = None
            silhouette_px = None
        else:
            silhouette = np.load(alone_path) >= VISIBLE_WEIGHT
            ratio_value = occlusion_ratio(silhouette.astype(np.uint8) * 255, green)
            kind = classify_occlusion(ratio_value, low=DEFAULT_LOW_RATIO, high=DEFAULT_HIGH_RATIO)
            ratio = None if ratio_value != ratio_value else float(ratio_value)
            silhouette_px = int(silhouette.sum())
        kinds[kind] = kinds.get(kind, 0) + 1
        rows.append(
            {
                "view_id": view_id,
                "kind": kind,
                "ratio": ratio,
                "green_px": int(green.sum()),
                "silhouette_px": silhouette_px,
                "label": label,
                "prompt": prompt,
                "failed": False,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows), encoding="utf-8")
    shutil.rmtree(cache, ignore_errors=True)
    print(f"[sft] {slug} views={len(rows)} labels={n_label} {kinds} -> {out}", flush=True)


def write_sft() -> None:
    img_out = SFT_OUT / "image"
    depth_out = SFT_OUT / "depth"
    img_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)
    records = []
    for slug, _seg_dir, _prompt in OBJECTS:
        rows = json.loads(_labels_path(slug).read_text(encoding="utf-8"))
        for row in rows:
            label = row.get("label")
            if not label:
                continue
            view_id = row["view_id"]
            rgb = VIEWS / "rgb" / f"view_{view_id}.png"
            depth = VIEWS / "depth" / f"view_{view_id}.png"
            if not rgb.is_file() or not depth.is_file():
                print(f"[skip] {slug} view_{view_id}: 缺 rgb 或 depth")
                continue
            out_name = f"{slug}_view_{view_id}.png"
            shutil.copy2(rgb, img_out / out_name)
            shutil.copy2(depth, depth_out / out_name)
            human = f"{row['prompt'].rstrip()} {DEFAULT_SUFFIX}".strip()
            records.append(
                {
                    "id": len(records),
                    "image": out_name,
                    "depth": out_name,
                    "conversations": [
                        {"from": "human", "value": human},
                        {"from": "gpt", "value": _format_answer(label["nx"], label["ny"])},
                    ],
                }
            )
    out_json = SFT_OUT / "location_point.json"
    out_json.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[sft] {len(records)} -> {out_json}", flush=True)


def _empty_silhouette(seg_dir: str, view_id: str, xyz: np.ndarray, log_scale: np.ndarray) -> bool:
    """物体高斯的半值范围都进不了画面时，0.5 轮廓为空。"""
    import math

    from frustum_segment import world_to_cam_uvz

    seg = json.loads((_REPO / "training_data" / seg_dir / "gaussian_seg.json").read_text(encoding="utf-8"))
    indices = np.asarray(seg.get("indices") or [], dtype=np.int64)
    cam = json.loads((VIEWS / "camera_params" / f"view_{view_id}.json").read_text(encoding="utf-8"))
    sel = xyz[indices]
    scale = np.exp(log_scale[indices]).max(axis=1)
    u, v, z = world_to_cam_uvz(sel, cam)
    width, height = int(cam["width"]), int(cam["height"])
    fx = 0.5 * width / math.tan(float(cam["fov_x"]) / 2.0)
    dx = np.where(u < 0, -u, np.where(u > width - 1, u - (width - 1), 0.0))
    dy = np.where(v < 0, -v, np.where(v > height - 1, v - (height - 1), 0.0))
    dist = np.hypot(dx, dy)
    front = np.isfinite(u) & np.isfinite(v) & (z > 1e-8)
    # exp(-0.5 * d^2) = 0.5 时 d ≈ 1.177。更远的高斯在画面里到不了 0.5。
    reach = 1.177 * fx * scale / np.maximum(z, 1e-6)
    return not bool(np.any(front & (reach > dist)))


def reclass_failed() -> None:
    """光栅在空轮廓上会非法访问。可见像素已是 0 时，用投影判断轮廓是不是空的。"""
    from plyfile import PlyData

    ply_path = _REPO / "3DGS/gaussian-splatting/output/data3/point_cloud/iteration_30000/point_cloud.ply"
    vertex = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=1)
    log_scale = np.stack(
        [np.asarray(vertex["scale_0"]), np.asarray(vertex["scale_1"]), np.asarray(vertex["scale_2"])],
        axis=1,
    )
    del vertex
    for slug, seg_dir, _prompt in OBJECTS:
        path = _labels_path(slug)
        rows = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for row in rows:
            if row.get("kind") != "silhouette_failed":
                continue
            view_id = str(row["view_id"])
            if int(row.get("green_px") or 0) != 0:
                raise SystemExit(f"{slug} view_{view_id} 可见像素不是 0，不能按空轮廓分类")
            if not _empty_silhouette(seg_dir, view_id, xyz, log_scale):
                raise SystemExit(f"{slug} view_{view_id} 轮廓能伸进画面，光栅失败不能当成空轮廓")
            row["kind"] = "empty_mask"
            row["ratio"] = None
            row["silhouette_px"] = 0
            changed = True
            print(f"[sft] {slug} view_{view_id} empty_mask", flush=True)
        if changed:
            path.write_text(json.dumps(rows), encoding="utf-8")
            print(f"[sft] updated {path}", flush=True)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--reclass-failed":
        reclass_failed()
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        export_object(sys.argv[2])
        return
    n_views = len(_cameras())
    for slug, _seg_dir, _prompt in OBJECTS:
        if _labels_complete(_labels_path(slug), n_views):
            print(f"[sft] {slug} reuse", flush=True)
            continue
        for _attempt in range(80):
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--worker", slug])
            if rc == 0:
                break
            print(f"[sft] {slug} exit {rc}, 继续剩下的视角", flush=True)
        else:
            raise SystemExit(f"{slug} 多次光栅失败")
    write_sft()


if __name__ == "__main__":
    main()
