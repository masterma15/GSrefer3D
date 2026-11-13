#!/usr/bin/env python3
"""把深度修剪后的掩码盖到原渲染图上，并在图上写遮挡分类。

读 ``occlusion.json`` 和 ``mask_visible/view_XXX.png``，RGB 用 ``<views-root>/rgb``。
写出 ``<物体目录>/mask_visible_overlay/view_XXX.png``。不改掩码和 occlusion.json。

只盖修剪后还留下的像素。完全遮挡时掩码为空，画面保持原渲染，只在顶栏写分类。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]

KIND_LABEL = {
    "almost_unoccluded": "几乎无遮挡",
    "partial": "部分遮挡",
    "fully_occluded": "完全遮挡",
    "empty_mask": "空掩码",
}
KIND_COLOR = {
    "almost_unoccluded": (46, 160, 67),
    "partial": (214, 128, 18),
    "fully_occluded": (196, 42, 42),
    "empty_mask": (120, 120, 120),
}


def _font(size: int):
    from PIL import ImageFont

    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def cover_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """掩码前景像素改成纯色，其余保持原渲染。"""
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"掩码 {mask.shape} 与渲染图 {rgb.shape} 不一致")
    out = np.array(rgb, copy=True)
    out[np.asarray(mask) >= 128] = np.asarray(color, dtype=np.uint8)
    return out


def draw_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    kind: str,
    ratio: float | None,
    view_id: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    color = KIND_COLOR.get(kind, (180, 180, 180))
    label = KIND_LABEL.get(kind, kind)
    covered = cover_mask(rgb, mask, color)
    image = Image.fromarray(covered, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font(32)
    if ratio is None or ratio != ratio:
        ratio_text = "删掉 —"
    else:
        ratio_text = f"删掉 {ratio * 100:.0f}%"
    title = f"view_{view_id}  {label}  {ratio_text}"
    bar_h = 52
    draw.rectangle([0, 0, image.width, bar_h], fill=(24, 24, 24))
    draw.text((12, 8), title, fill=(255, 255, 255), font=font)
    return np.array(image)


def export_object(obj_dir: Path, views_root: Path) -> int:
    from PIL import Image

    occ_path = obj_dir / "occlusion.json"
    if not occ_path.is_file():
        raise FileNotFoundError(f"缺少 {occ_path}")
    records = json.loads(occ_path.read_text(encoding="utf-8"))
    out_dir = obj_dir / "mask_visible_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in records:
        view_id = str(rec["view_id"]).zfill(3)
        kind = str(rec.get("kind", ""))
        rgb_path = views_root / "rgb" / f"view_{view_id}.png"
        if not rgb_path.is_file():
            print(f"[skip] {obj_dir.name} view_{view_id}: 没有渲染图")
            continue
        mask_path = obj_dir / "mask_visible" / f"view_{view_id}.png"
        if not mask_path.is_file():
            print(f"[skip] {obj_dir.name} view_{view_id}: 没有修剪掩码")
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        if mask.shape[:2] != rgb.shape[:2]:
            mask_img = Image.fromarray(mask, mode="L").resize((rgb.shape[1], rgb.shape[0]))
            mask = np.array(mask_img)
        ratio = rec.get("ratio")
        out = draw_overlay(
            rgb,
            mask,
            kind=kind,
            ratio=None if ratio is None else float(ratio),
            view_id=view_id,
        )
        Image.fromarray(out).save(out_dir / f"view_{view_id}.png")
        n += 1
    print(f"[overlay] {obj_dir.name}: {n} -> {out_dir}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="把修剪掩码盖到原渲染图上并标注遮挡分类")
    ap.add_argument("--obj-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--views-root", type=Path, default=Path("3DGS/test2"))
    args = ap.parse_args()
    views_root = args.views_root if args.views_root.is_absolute() else (_REPO / args.views_root).resolve()
    total = 0
    for raw in args.obj_dir:
        obj_dir = raw if raw.is_absolute() else (_REPO / raw).resolve()
        total += export_object(obj_dir, views_root)
    if total == 0:
        sys.exit("[error] 没有写出叠加图")
    print(f"[done] {total}")


if __name__ == "__main__":
    main()
