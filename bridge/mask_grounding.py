"""Grounding DINO + SAM2 mask generation for training data."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GroundingSamConfig:
    box_threshold: float = 0.10
    text_threshold: float = 0.12
    anchor_box_radius: int = 64
    min_mask_ratio: float = 0.001
    # Box selection: weighted DINO confidence + proximity (no absolute smallest-box rule)
    max_box_area_ratio: float = 0.35
    max_point_box_dist: float = -1.0  # px; <0 => 0.15 * image diagonal
    min_box_area_ratio: float = 0.003  # drop tiny fragment boxes (part/neighbor clutter)
    dino_score_weight: float = 0.50
    near_score_weight: float = 0.40
    near_dist_sigma: float = 50.0  # exp(-pt_dist/sigma): smooth nearness, not lexicographic
    contain_bonus: float = 0.06  # mild boost when anchor lies inside box
    area_penalty: float = 0.12  # mild global area prior
    compact_box_weight: float = 0.28  # bonus when box area <= median among candidates
    large_box_penalty: float = 0.38  # penalty when box area >> median (DINO likes big boxes)
    tiny_box_penalty: float = 0.40  # extra score penalty for area_ratio below min_box_area_ratio
    prefer_containing_point: bool = False  # legacy lexicographic mode (off)
    # SAM: box-only by default (3D anchor may sit on background at object edge)
    use_sam_point_prompt: bool = False
    min_box_mask_iou: float = 0.12  # reject SAM mask that ignores the DINO box
    fallback_point_sam: bool = True  # only when DINO returns zero boxes
    fallback_point_sam_if_box_miss: bool = False  # do not replace box mask when point off object


def load_grounding_dino(config_path: str, checkpoint_path: str, device: str = "cuda"):
    try:
        from groundingdino.util.inference import load_model
    except ImportError as e:
        sys.exit(
            "[error] GroundingDINO not installed.\n"
            "  git clone https://github.com/IDEA-Research/GroundingDINO\n"
            "  pip install -e .   # in repo root\n"
            f"  ({e})"
        )
    return load_model(config_path, checkpoint_path, device=device)


def load_sam2_predictor(checkpoint: str, config: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        sys.exit(
            "[error] sam2 not installed.\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
        )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))


def _transform_image_rgb(image_rgb: np.ndarray):
    import groundingdino.datasets.transforms as T
    from PIL import Image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    pil = Image.fromarray(image_rgb)
    tensor, _ = transform(pil, None)
    return tensor


def _predict_boxes(model, image_rgb: np.ndarray, caption: str, cfg: GroundingSamConfig, device: str):
    from groundingdino.util.inference import predict
    import torch
    from torchvision.ops import box_convert

    h, w = image_rgb.shape[:2]
    img_t = _transform_image_rgb(image_rgb)

    boxes, logits, _phrases = predict(
        model=model,
        image=img_t,
        caption=caption,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
        device=device,
    )
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)

    scaled = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
    xyxy = box_convert(boxes=scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    scores = logits.cpu().numpy().astype(np.float64)
    return xyxy, scores


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_area_ratio(box: np.ndarray, w: int, h: int) -> float:
    return _box_area(box) / float(w * h)


def _box_contains_point(box: np.ndarray, u: float, v: float) -> bool:
    return box[0] <= u <= box[2] and box[1] <= v <= box[3]


def _point_to_box_distance(u: float, v: float, box: np.ndarray) -> float:
    """0 if (u,v) inside box; else Euclidean distance to rectangle edge."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x1 <= u <= x2 and y1 <= v <= y2:
        return 0.0
    dx = max(x1 - u, 0.0, u - x2)
    dy = max(y1 - v, 0.0, v - y2)
    return math.hypot(dx, dy)


def _max_point_box_dist_px(w: int, h: int, cfg: GroundingSamConfig) -> float:
    if cfg.max_point_box_dist >= 0:
        return float(cfg.max_point_box_dist)
    return 0.15 * math.hypot(w, h)


def _nearness_term(pt_dist: float, cfg: GroundingSamConfig) -> float:
    return math.exp(-pt_dist / max(cfg.near_dist_sigma, 1e-6))


def _composite_box_score(
    dino: float,
    pt_dist: float,
    area_ratio: float,
    contains: bool,
    cfg: GroundingSamConfig,
    *,
    median_area_ratio: float,
) -> float:
    near = _nearness_term(pt_dist, cfg)
    tiny = max(0.0, cfg.min_box_area_ratio - area_ratio) / max(cfg.min_box_area_ratio, 1e-9)

    dino_eff = dino
    compact = 0.0
    med = max(median_area_ratio, 1e-9)
    if area_ratio > med:
        # DINO often scores large boxes higher; down-weight oversized detections
        dino_eff = dino * min(1.0, math.sqrt(med / area_ratio))
        compact = -cfg.large_box_penalty * min(2.5, area_ratio / med - 1.0)
    else:
        compact = cfg.compact_box_weight * (1.0 - area_ratio / med)

    score = (
        cfg.dino_score_weight * dino_eff
        + cfg.near_score_weight * near
        - cfg.area_penalty * math.sqrt(max(area_ratio, 1e-9))
        - cfg.tiny_box_penalty * tiny
        + compact
    )
    if contains:
        score += cfg.contain_bonus
    return score


def _mask_box_iou(mask: np.ndarray, box: np.ndarray) -> float:
    """IoU between binary mask and axis-aligned DINO box."""
    x1 = int(max(0, math.floor(box[0])))
    y1 = int(max(0, math.floor(box[1])))
    x2 = int(min(mask.shape[1], math.ceil(box[2])))
    y2 = int(min(mask.shape[0], math.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float(mask[y1:y2, x1:x2].sum())
    box_area = float((x2 - x1) * (y2 - y1))
    mask_area = float(mask.sum())
    union = mask_area + box_area - inter
    return inter / union if union > 0 else 0.0


def _rank_boxes(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    u: float,
    v: float,
    w: int,
    h: int,
    cfg: GroundingSamConfig,
) -> list[tuple[int, float, bool, float, float]]:
    """Return (index, select_score, contains_point, area, point_to_box_dist) best-first.

    Weighted score = w_dino*DINO + w_near*exp(-dist/sigma) - mild area - tiny fragment penalty.
    Hard filters: too far, too large, optional drop of sub-min area (keep if only those remain).
    """
    max_dist = _max_point_box_dist_px(w, h, cfg)
    pool: list[tuple[int, float, bool, float, float, float]] = []

    for i, box in enumerate(boxes_xyxy):
        pt_dist = _point_to_box_distance(u, v, box)
        if pt_dist > max_dist:
            continue
        area_ratio = _box_area_ratio(box, w, h)
        if area_ratio > cfg.max_box_area_ratio:
            continue
        area = _box_area(box)
        contains = pt_dist <= 0.0
        dino = float(scores[i])
        pool.append((i, dino, contains, area, pt_dist, area_ratio))

    if not pool:
        return []

    non_tiny = [p for p in pool if p[5] >= cfg.min_box_area_ratio]
    candidates = non_tiny if non_tiny else pool

    if cfg.prefer_containing_point:
        containing = [p for p in candidates if p[2]]
        if containing:
            containing.sort(key=lambda p: (-p[1], p[4], p[3]))
            rest = [p for p in candidates if not p[2]]
            rest.sort(key=lambda p: (-p[1], p[4], p[3]))
            candidates = containing + rest

    median_ar = float(np.median([p[5] for p in candidates]))

    scored: list[tuple[int, float, bool, float, float]] = []
    for i, dino, contains, area, pt_dist, area_ratio in candidates:
        sel = _composite_box_score(
            dino, pt_dist, area_ratio, contains, cfg, median_area_ratio=median_ar
        )
        scored.append((i, sel, contains, area, pt_dist))

    scored.sort(key=lambda r: -r[1])
    return scored


def select_box(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    u: float,
    v: float,
    w: int,
    h: int,
    cfg: GroundingSamConfig,
) -> int | None:
    ranked = _rank_boxes(boxes_xyxy, scores, u, v, w, h, cfg)
    if not ranked:
        return None
    return ranked[0][0]


def segment_box(
    predictor,
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    u: float | None = None,
    v: float | None = None,
    *,
    use_point: bool = True,
) -> tuple[np.ndarray, float]:
    predictor.set_image(image_rgb)
    kwargs: dict[str, Any] = {
        "box": box_xyxy.astype(np.float32),
        "multimask_output": False,
    }
    if use_point and u is not None and v is not None:
        kwargs["point_coords"] = np.array([[u, v]], dtype=np.float32)
        kwargs["point_labels"] = np.array([1])
    masks, sam_scores, _ = predictor.predict(**kwargs)
    mask = masks[0].astype(bool)
    score = float(sam_scores[0]) if len(sam_scores) else 0.0
    return mask, score


def segment_point(predictor, image_rgb: np.ndarray, u: float, v: float) -> tuple[np.ndarray, float]:
    predictor.set_image(image_rgb)
    masks, sam_scores, _ = predictor.predict(
        point_coords=np.array([[u, v]], dtype=np.float32),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best = int(np.argmax(sam_scores))
    return masks[best].astype(bool), float(sam_scores[best])


def _point_in_mask(mask: np.ndarray, u: float, v: float) -> bool:
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or vi < 0 or ui >= mask.shape[1] or vi >= mask.shape[0]:
        return False
    return bool(mask[vi, ui])


def _dist_to_mask(mask: np.ndarray, u: float, v: float) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("inf")
    d2 = (xs - u) ** 2 + (ys - v) ** 2
    return float(np.sqrt(d2.min()))


def grounding_sam_mask(
    image_rgb: np.ndarray,
    u: float,
    v: float,
    caption: str,
    w: int,
    h: int,
    gdino_model,
    sam_predictor,
    cfg: GroundingSamConfig,
    device: str = "cuda",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return binary mask (H,W) or None, plus debug meta."""
    meta: dict[str, Any] = {"mode": "grounding"}

    boxes, scores = _predict_boxes(gdino_model, image_rgb, caption, cfg, device)
    meta["num_boxes"] = int(len(boxes))

    max_dist_px = _max_point_box_dist_px(w, h, cfg)
    ranked = _rank_boxes(boxes, scores, u, v, w, h, cfg) if len(boxes) else []
    meta["max_point_box_dist_px"] = float(max_dist_px)
    meta["ranked_boxes"] = []
    for i, sel, cont, area, pt_dist in ranked[:8]:
        ar = float(_box_area_ratio(boxes[i], w, h))
        meta["ranked_boxes"].append(
            {
                "idx": int(i),
                "dino_score": float(scores[i]),
                "select_score": float(sel),
                "near_term": float(_nearness_term(pt_dist, cfg)),
                "contains_point": bool(cont),
                "point_to_box_dist": float(pt_dist),
                "area": float(area),
                "area_ratio": ar,
                "box": boxes[i].tolist(),
            }
        )
    if ranked:
        meta["median_area_ratio"] = float(
            np.median([_box_area_ratio(boxes[r[0]], w, h) for r in ranked[:8]])
        )

    best_mask: np.ndarray | None = None
    best_meta: dict[str, Any] = {}

    def _try_box(idx: int) -> tuple[np.ndarray, dict[str, Any]] | None:
        box = boxes[idx]
        m, sam_score = segment_box(
            sam_predictor,
            image_rgb,
            box,
            u,
            v,
            use_point=cfg.use_sam_point_prompt,
        )
        iou = _mask_box_iou(m, box)
        pin = _point_in_mask(m, u, v)
        dist = 0.0 if pin else _dist_to_mask(m, u, v)
        return m, {
            "box_idx": int(idx),
            "dino_score": float(scores[idx]),
            "box": box.tolist(),
            "sam_score": sam_score,
            "box_mask_iou": float(iou),
            "point_in_mask": pin,
            "dist_to_mask": dist,
            "area_ratio": float(m.sum() / (w * h)),
        }

    for i, sel, _cont, _area, pt_dist in ranked:
        trial = _try_box(i)
        if trial is None:
            continue
        m, info = trial
        info["select_score"] = float(sel)
        info["point_to_box_dist"] = float(pt_dist)
        if info["area_ratio"] < cfg.min_mask_ratio:
            continue
        if info["box_mask_iou"] < cfg.min_box_mask_iou:
            continue
        best_mask, best_meta = m, info
        best_meta["select_reason"] = "weighted_near_dino_sam"
        break

    if best_mask is None and len(boxes) == 0 and cfg.fallback_point_sam:
        meta["reason"] = "no_dino_boxes"
        m, sam_score = segment_point(sam_predictor, image_rgb, u, v)
        meta["fallback"] = "point_sam"
        meta["sam_score"] = sam_score
        area_ratio = m.sum() / (w * h)
        if area_ratio < cfg.min_mask_ratio:
            meta["area_ratio"] = float(area_ratio)
            return None, meta
        return m, meta

    if best_mask is None and len(boxes) > 0:
        meta["reason"] = "no_valid_box_sam"
        return None, meta

    if best_mask is None:
        meta["reason"] = "no_mask"
        return None, meta

    meta.update(best_meta)
    area_ratio = float(best_mask.sum() / (w * h))
    meta["area_ratio"] = area_ratio
    if area_ratio < cfg.min_mask_ratio:
        meta["reason"] = "mask_too_small"
        return None, meta

    return best_mask, meta
