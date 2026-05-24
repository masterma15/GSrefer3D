"""Compress demo SIBR GIFs to a target size band (default 5–10 MB)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageSequence


def _load_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    im = Image.open(path)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for frame in ImageSequence.Iterator(im):
        frames.append(frame.convert("RGB"))
        durations.append(frame.info.get("duration", im.info.get("duration", 40)))
    return frames, durations


def _resize(frames: list[Image.Image], max_w: int) -> list[Image.Image]:
    w0, h0 = frames[0].size
    if w0 <= max_w:
        return frames
    scale = max_w / w0
    nw, nh = int(w0 * scale), int(h0 * scale)
    return [f.resize((nw, nh), Image.Resampling.LANCZOS) for f in frames]


def _subsample(frames: list[Image.Image], durations: list[int], step: int) -> tuple[list[Image.Image], list[int]]:
    out_f, out_d = [], []
    for i, (f, d) in enumerate(zip(frames, durations)):
        if i % step == 0:
            out_f.append(f)
            out_d.append(d * step)
    return out_f, out_d


def _quantize(frames: list[Image.Image], colors: int) -> list[Image.Image]:
    return [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("P") for f in frames]


def _save(frames: list[Image.Image], durations: list[int], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out.stat().st_size


def compress_one(
    src: Path,
    dst: Path,
    *,
    target_min_mb: float,
    target_max_mb: float,
    backup: bool,
) -> int:
    frames, durations = _load_frames(src)
    if backup and dst.exists():
        bak = dst.with_suffix(dst.suffix + ".orig")
        if not bak.exists():
            shutil.copy2(dst, bak)

    # Search: coarser settings until within band or best effort under max.
    configs = [
        (960, 1, 128),
        (960, 2, 128),
        (854, 2, 96),
        (720, 2, 96),
        (720, 3, 64),
        (640, 3, 64),
        (540, 3, 48),
    ]
    min_b = int(target_min_mb * 1e6)
    max_b = int(target_max_mb * 1e6)
    best_size = 10**12
    best_cfg = configs[-1]

    for max_w, step, colors in configs:
        f = _resize(frames, max_w)
        f, d = _subsample(f, durations, step)
        f = _quantize(f, colors)
        size = _save(f, d, dst)
        if min_b <= size <= max_b:
            return size
        if size < best_size and size <= max_b * 1.15:
            best_size, best_cfg = size, (max_w, step, colors)

    max_w, step, colors = best_cfg
    f = _resize(frames, max_w)
    f, d = _subsample(f, durations, step)
    f = _quantize(f, colors)
    return _save(f, d, dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dir", type=Path, default=Path(__file__).resolve().parents[1] / "demo")
    p.add_argument("--min-mb", type=float, default=5.0)
    p.add_argument("--max-mb", type=float, default=10.0)
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()
    names = ["teaser_3d_electric_shaver.gif", "teaser_3d_brown_rabbit.gif"]
    for name in names:
        src = args.demo_dir / name
        size = compress_one(
            src,
            src,
            target_min_mb=args.min_mb,
            target_max_mb=args.max_mb,
            backup=not args.no_backup,
        )
        print(f"{name}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
