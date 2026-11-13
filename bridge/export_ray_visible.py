#!/usr/bin/env python3
"""用旧的视锥投票点云导出可见掩码和微调坐标。

点云来自覆盖 ``training_data/data2_*`` 之前的那批上色结果
（``iteration_seg_hair_clip`` 等，点数与当时的评测一致）。
不读后来的 ``data2_*`` / ``*_allviews`` 投票。

每个视角渲两次，阈值都是 0.5：

- 全图权重：不透明度不动。不低于 0.5 的像素是绿色，即新掩码。
- 只留物体高斯：其余不进这次光栅。不低于 0.5 的像素是无遮挡轮廓。

遮挡比例 = 轮廓里没有进绿色的像素 / 轮廓像素。低于 0.2 几乎无遮挡，
高于 0.8 完全遮挡，中间是部分遮挡。绿色掩码的质心是微调坐标。
没有绿色像素的视角不进数据集。

不覆盖 ``training_data/data2_sft``。胶带不写入这份训练 JSON。
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

from export_spatial_train import DEFAULT_SUFFIX, _format_answer, _slug  # noqa: E402
from inject_gaussian_seg import COLOR_IN_OBB, COLOR_OUT_OBB  # noqa: E402
from obb_geom import rgb_to_f_dc  # noqa: E402
from render_object_weight import VISIBLE_WEIGHT, render_object_weight  # noqa: E402
from visible_mask import (  # noqa: E402
    DEFAULT_HIGH_RATIO,
    DEFAULT_LOW_RATIO,
    classify_occlusion,
    mask_centroid,
    occlusion_ratio,
)

VIEWS = _REPO / "3DGS" / "test2"
PLY = _REPO / "3DGS/gaussian-splatting/output/data2/point_cloud/iteration_30000/point_cloud.ply"
PC_ROOT = _REPO / "3DGS/gaussian-splatting/output/data2/point_cloud"
SFT_OUT = _REPO / "training_data" / "data2_sft_ray"
OUT_ROOT = _REPO / "training_data" / "published_vote"
N_GAUSSIANS = 852945

# 评测时那批投票。iteration 名是当时的 SIBR 目录，prompt 仍用各物体 question.json。
PUBLISHED = [
    ("shaver", "iteration_seg_electric_shaver", "data2_shaver"),
    ("rabbit", "iteration_seg_brown_rabbit", "data2_rabbit"),
    ("golden_retriever", "iteration_seg_golden_retriever", "data2_golden_retriever"),
    ("bowl", "iteration_seg_golden_bowl", "data2_bowl"),
    ("umbrella", "iteration_seg_umbrella", "data2_umbrella"),
    ("toy_cake", "iteration_seg_toy_cake", "data2_toy_cake"),
    ("cookie", "iteration_seg_cookie_bag", "data2_cookie"),
    ("medicine_bottle", "iteration_seg_medicine_bottle", "data2_medicine_bottle"),
    ("bracelet", "iteration_seg_bracelet", "data2_bracelet"),
    ("hair_clip", "iteration_seg_hair_clip", "data2_hair_clip"),
    ("tape", "iteration_seg_double_sided_tape", "data2_tape"),
]


def _out_dir(key: str) -> Path:
    return OUT_ROOT / f"data2_{key}"


def _prompt(prompt_dir: str) -> str:
    question = _REPO / "training_data" / prompt_dir / "question.json"
    if not question.is_file():
        raise FileNotFoundError(f"缺少 {question}")
    records = json.loads(question.read_text(encoding="utf-8"))
    if not records or not records[0].get("prompt"):
        raise ValueError(f"{question} 没有 prompt")
    return str(records[0]["prompt"])


def recover_indices(key: str, iteration: str) -> np.ndarray:
    """从上色 ply 里找回投票选中的高斯编号。编号与 iteration_30000 对齐。"""
    from plyfile import PlyData

    manifest = json.loads((PC_ROOT / iteration / "inject_manifest.json").read_text(encoding="utf-8"))
    expect = int(manifest["n_selected"])
    n_keep = int(manifest["n_gaussians"])
    if n_keep != N_GAUSSIANS:
        raise RuntimeError(f"{iteration} 的高斯数 {n_keep} 与场景 {N_GAUSSIANS} 不一致")
    ply_path = PC_ROOT / iteration / "point_cloud.ply"
    vertex = PlyData.read(str(ply_path))["vertex"]
    if len(vertex) < n_keep:
        raise RuntimeError(f"{ply_path} 顶点 {len(vertex)} 少于 {n_keep}")
    dc = np.stack(
        [np.asarray(vertex[f"f_dc_{c}"][:n_keep], dtype=np.float32) for c in range(3)],
        axis=1,
    )
    del vertex
    targets = [
        np.asarray(rgb_to_f_dc(COLOR_IN_OBB), dtype=np.float32),
        np.asarray(rgb_to_f_dc(COLOR_OUT_OBB), dtype=np.float32),
    ]
    selected = np.zeros(n_keep, dtype=bool)
    for target in targets:
        selected |= np.max(np.abs(dc - target), axis=1) < 1e-3
    indices = np.flatnonzero(selected).astype(np.int64)
    if int(indices.size) != expect:
        raise RuntimeError(f"{iteration} 找回 {indices.size} 个高斯，清单是 {expect}")
    out = _out_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_iteration": iteration,
        "source_ply": str(ply_path).replace("\\", "/"),
        "n_selected": int(indices.size),
        "indices": indices.tolist(),
    }
    (out / "gaussian_seg.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"[vote] {key} {indices.size} <- {iteration}", flush=True)
    return indices


def load_indices(key: str, iteration: str) -> np.ndarray:
    path = _out_dir(key) / "gaussian_seg.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_iteration") == iteration and payload.get("indices"):
            indices = np.asarray(payload["indices"], dtype=np.int64)
            print(f"[vote] {key} reuse {indices.size}", flush=True)
            return indices
    return recover_indices(key, iteration)


def _with_selected(model, indices: np.ndarray):
    """暂时只留下物体高斯。调用方负责恢复返回的参数。"""
    import torch
    from torch import nn

    names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    saved = [(name, getattr(model, name)) for name in names]
    idx = torch.as_tensor(indices, device=saved[0][1].device, dtype=torch.long)
    for name, param in saved:
        setattr(
            model,
            name,
            nn.Parameter(param.data.index_select(0, idx).detach(), requires_grad=False),
        )
    return saved


def _restore(model, saved) -> None:
    for name, param in saved:
        setattr(model, name, param)


def _paint(model, indices: np.ndarray):
    import torch
    from render_object_weight import _DC_BLACK, _DC_WHITE

    saved_dc = model._features_dc.data.detach().clone()
    saved_rest = model._features_rest.data.detach().clone()
    model._features_rest.data.zero_()
    model._features_dc.data.fill_(_DC_BLACK)
    idx = torch.as_tensor(indices, device=model._features_dc.device, dtype=torch.long)
    model._features_dc.data[idx] = _DC_WHITE
    return saved_dc, saved_rest


def _cached_render(cache: Path, kind: str, view_id: str, render_fn):
    """成功写 npy。上次死在这个视角上就记失败，不再重试这一次光栅。"""
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
        print(f"[ray] skip {kind} view_{view_id}: 上次光栅失败", flush=True)
        return "failed"
    with open(started, "w", encoding="utf-8") as handle:
        handle.write(view_id)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"[ray] {kind} view_{view_id}", flush=True)
    arr = render_fn()
    tmp = cache / f"{kind}_{view_id}.tmp.npy"
    with open(tmp, "wb") as handle:
        np.save(handle, arr)
    os.replace(tmp, npy)
    started.unlink()
    return "ok"


def export_object(key: str, iteration: str, prompt_dir: str) -> None:
    from render_object_invdepth import load_gaussian_model

    out = _out_dir(key)
    done = out / "ray_occlusion.json"
    if done.is_file():
        rows = json.loads(done.read_text(encoding="utf-8"))
        if len(rows) == 72:
            print(f"[ray] {key} already {len(rows)}", flush=True)
            return
    indices = load_indices(key, iteration)
    prompt = _prompt(prompt_dir)
    cam_paths = sorted((VIEWS / "camera_params").glob("view_*.json"))
    cameras = []
    for cam_path in cam_paths:
        view_id = cam_path.stem.split("_")[-1].zfill(3)
        cameras.append((view_id, json.loads(cam_path.read_text(encoding="utf-8"))))
    if len(cameras) != 72:
        raise RuntimeError(f"视角数 {len(cameras)}，需要 72")
    cache = out / "_ray_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model = load_gaussian_model(PLY)
    saved_dc, saved_rest = _paint(model, indices)
    selected = _with_selected(model, indices)
    try:
        for view_id, cam in cameras:
            _cached_render(
                cache,
                "alone",
                view_id,
                lambda cam=cam: render_object_weight(model, cam),
            )
        _restore(model, selected)
        selected = None
        for view_id, cam in cameras:
            _cached_render(
                cache,
                "weight",
                view_id,
                lambda cam=cam: render_object_weight(model, cam),
            )
    finally:
        if selected is not None:
            _restore(model, selected)
        model._features_dc.data.copy_(saved_dc)
        model._features_rest.data.copy_(saved_rest)
        del model
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    from PIL import Image

    mask_dir = out / "mask_ray"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for view_id, cam in cameras:
        weight_path = cache / f"weight_{view_id}.npy"
        alone_path = cache / f"alone_{view_id}.npy"
        if not weight_path.is_file():
            raise RuntimeError(f"{key} view_{view_id} 没有权重图")
        weight = np.load(weight_path)
        green = weight >= VISIBLE_WEIGHT
        width = int(cam["width"])
        height = int(cam["height"])
        Image.fromarray(green.astype(np.uint8) * 255, mode="L").save(mask_dir / f"view_{view_id}.png")
        centroid = mask_centroid(green)
        label = None
        if centroid is not None and width > 0 and height > 0:
            u, v = centroid
            label = {"u": u, "v": v, "nx": u / width, "ny": v / height}
        if not alone_path.is_file():
            rows.append(
                {
                    "view_id": view_id,
                    "kind": "silhouette_failed",
                    "ratio": None,
                    "green_px": int(green.sum()),
                    "silhouette_px": None,
                    "label": label,
                    "prompt": prompt,
                }
            )
            continue
        alone = np.load(alone_path)
        silhouette = alone >= VISIBLE_WEIGHT
        ratio = occlusion_ratio(silhouette.astype(np.uint8) * 255, green)
        kind = classify_occlusion(ratio, low=DEFAULT_LOW_RATIO, high=DEFAULT_HIGH_RATIO)
        rows.append(
            {
                "view_id": view_id,
                "kind": kind,
                "ratio": None if ratio != ratio else float(ratio),
                "green_px": int(green.sum()),
                "silhouette_px": int(silhouette.sum()),
                "label": label,
                "prompt": prompt,
            }
        )
    done.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    shutil.rmtree(cache, ignore_errors=True)
    kinds: dict[str, int] = {}
    n_label = 0
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        if row.get("label"):
            n_label += 1
    print(f"[ray] {key} views={len(rows)} labels={n_label} {kinds} -> {mask_dir}", flush=True)


def write_sft() -> None:
    img_out = SFT_OUT / "image"
    depth_out = SFT_OUT / "depth"
    img_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)
    records = []
    for key, _iteration, _prompt_dir in PUBLISHED:
        if key == "tape":
            continue
        rows = json.loads((_out_dir(key) / "ray_occlusion.json").read_text(encoding="utf-8"))
        slug = _slug(key)
        for row in rows:
            label = row.get("label")
            if not label:
                continue
            view_id = row["view_id"]
            rgb = VIEWS / "rgb" / f"view_{view_id}.png"
            depth = VIEWS / "depth" / f"view_{view_id}.png"
            if not rgb.is_file() or not depth.is_file():
                print(f"[skip] {key} view_{view_id}: 缺 rgb 或 depth")
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


def _worker(key: str) -> None:
    for item_key, iteration, prompt_dir in PUBLISHED:
        if item_key == key:
            export_object(item_key, iteration, prompt_dir)
            return
    raise SystemExit(f"未知物体 {key}")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--recover-only":
        for key, iteration, _prompt_dir in PUBLISHED:
            load_indices(key, iteration)
        return
    for key, _iteration, _prompt_dir in PUBLISHED:
        done = _out_dir(key) / "ray_occlusion.json"
        if done.is_file() and len(json.loads(done.read_text(encoding="utf-8"))) == 72:
            print(f"[ray] {key} reuse", flush=True)
            continue
        for _attempt in range(80):
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--worker", key])
            if rc == 0:
                break
            print(f"[ray] {key} exit {rc}, 继续剩下的视角", flush=True)
        else:
            raise SystemExit(f"{key} 多次光栅失败")
    write_sft()


if __name__ == "__main__":
    main()
