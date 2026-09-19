#!/usr/bin/env python3
"""针对 3DGS 自定义视角目录的批量 RoboRefer 客户端。

期望的目录布局（由 ``3DGS/render.py --custom_views`` 生成，默认 36 个视角）：

  <root>/
    rgb/view_NNN.png          # N 个视角（默认 36，来自 3DGS/render.py --custom_views）
    depth/view_NNN.png        # 8-bit 逆深度风格 PNG，送给 RoboRefer
    depth_raw/view_NNN.npy    # float32，留给下游反投影
    camera_params/view_NNN.json

对每个视角，本脚本将 (RGB, depth.png) POST 到 RoboRefer API，参数为
``enable_depth=1, depth_url=[depth.png]``，解析返回的 ``[(nx, ny), ...]``，
并写出一份 ``predictions.json``：

  {
    "prompt":   "Please point to ...",
    "suffix":   "Your answer should be formatted ...",
    "url":      "http://127.0.0.1:25547",
    "root":     "<absolute root>",
    "views": [
      {
        "view_id":         0,
        "rgb_path":        "rgb/view_000.png",
        "depth_path":      "depth/view_000.png",
        "depth_raw_path":  "depth_raw/view_000.npy",
        "camera_path":     "camera_params/view_000.json",
        "raw_answer":      "[(0.458, 0.298)]",
        "points":          [{"nx": 0.458, "ny": 0.298}],
        "parse_ok":        true,
        "error":           null
      },
      ...
    ]
  }

某视角返回 None 或无法解析的文本时，保留 ``points=[]`` 并记录
``error``；整次运行不会中止。单视角模式（``--rgb`` + ``--depth``）
保持旧版 minimal_roborefer_e2e.py 的行为。

在 RoboRefer conda 环境中运行（推荐 Linux/WSL）：
  conda activate roborefer
  # 一个终端：
  cd RoboRefer-main/API && python api.py --port 25547
  # 默认权重：<repo>/weights/depth_anything_v2_vitl.pth、<repo>/RoboRefer-2B-SFT
  # 另一个终端：
  python bridge/roborefer_client.py \
    --root 3DGS/test1 \
    --prompt "Please point to the most salient object in the center." \
    --url http://127.0.0.1:25547 \
    --output 3DGS/test1/predictions.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SUFFIX = (
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], "
    "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
    "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
)


def _encode_image(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _query_server(
    image_paths: list[str],
    prompt: str,
    *,
    url: str = "http://127.0.0.1:25547",
    enable_depth: int = 0,
    depth_paths: list[str] | None = None,
    retry: int = 3,
) -> str | None:
    """RoboRefer /query 端点的直接 HTTP 客户端（不依赖 openai）。"""
    import requests as _requests

    image_url_list = [_encode_image(p) for p in image_paths]
    depth_url_list = [_encode_image(p) for p in depth_paths] if depth_paths else []

    request_data = {
        "image_url": image_url_list,
        "depth_url": depth_url_list,
        "enable_depth": enable_depth,
        "text": prompt,
    }

    for attempt in range(1, retry + 1):
        try:
            resp = _requests.post(url.rstrip("/") + "/query", json=request_data, timeout=120)
            if resp.status_code == 200:
                return resp.json()["answer"]
            print(f"[warn] attempt {attempt}: status {resp.status_code}")
        except Exception as e:
            print(f"[warn] attempt {attempt}: {type(e).__name__}: {e}")
    return None


def _parse_points(answer: str) -> tuple[list[dict[str, float]], str | None]:
    """尽力把 RoboRefer 输出解析为 [{nx, ny}, ...]。"""
    if not isinstance(answer, str):
        return [], f"answer not a string: {type(answer).__name__}"
    txt = answer.strip()
    # 部分 RoboRefer 回答会把列表包在散文里；提取第一个 [...] 块。
    m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
    if m is None:
        return [], "no list literal found"
    try:
        raw = ast.literal_eval(m.group(0))
    except (SyntaxError, ValueError) as e:
        return [], f"literal_eval failed: {e}"
    if not isinstance(raw, (list, tuple)):
        return [], "parsed value is not a list"
    out: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            nx = float(item[0])
            ny = float(item[1])
        except (TypeError, ValueError):
            continue
        out.append({"nx": nx, "ny": ny})
    if not out:
        return [], "no parseable (x, y) tuples"
    return out, None


def discover_views(root: Path, view_ids: list[int] | None) -> list[int]:
    rgb_dir = root / "rgb"
    if not rgb_dir.is_dir():
        raise SystemExit(f"missing {rgb_dir}; did you run render.py --custom_views?")
    found: list[int] = []
    for p in sorted(rgb_dir.glob("view_*.png")):
        try:
            found.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not found:
        raise SystemExit(f"no view_*.png under {rgb_dir}")
    if view_ids is None:
        return found
    requested = set(view_ids)
    missing = requested - set(found)
    if missing:
        raise SystemExit(f"requested view ids missing on disk: {sorted(missing)}")
    return [vid for vid in found if vid in requested]


def view_paths(root: Path, vid: int) -> dict[str, Path]:
    name = f"view_{vid:03d}"
    return {
        "rgb": root / "rgb" / f"{name}.png",
        "depth": root / "depth" / f"{name}.png",
        "depth_raw": root / "depth_raw" / f"{name}.npy",
        "camera": root / "camera_params" / f"{name}.json",
    }


def _check_files(paths: dict[str, Path], require_depth: bool) -> str | None:
    for label in ("rgb", "camera"):
        if not paths[label].is_file():
            return f"missing {label} file: {paths[label]}"
    if require_depth and not paths["depth"].is_file():
        return f"missing depth file: {paths['depth']}"
    return None


def query_one(
    rgb: Path,
    depth: Path | None,
    *,
    text: str,
    url: str,
    enable_depth: int,
    retry: int,
) -> tuple[str | None, str | None]:
    try:
        answer = _query_server(
            [str(rgb.resolve())],
            text,
            url=url,
            enable_depth=enable_depth,
            depth_paths=[str(depth.resolve())] if (enable_depth and depth is not None) else None,
            retry=retry,
        )
    except Exception as e:
        return None, f"_query_server raised {type(e).__name__}: {e}"
    if answer is None:
        return None, "server returned None (down or non-200 after retries)"
    return answer, None


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    vids = discover_views(root, args.views)

    suffix = "" if args.no_suffix else DEFAULT_SUFFIX
    text = args.prompt + suffix
    enable_depth = 0 if args.no_depth else 1

    print(f"[info] root={root}")
    print(f"[info] views={vids}")
    print(f"[info] enable_depth={enable_depth}  url={args.url}")

    view_records: list[dict[str, Any]] = []
    for vid in vids:
        paths = view_paths(root, vid)
        err = _check_files(paths, require_depth=bool(enable_depth))
        if err is not None:
            print(f"[skip] view_{vid:03d}: {err}")
            view_records.append({
                "view_id": vid,
                "rgb_path": _rel(root, paths["rgb"]),
                "depth_path": _rel(root, paths["depth"]),
                "depth_raw_path": _rel(root, paths["depth_raw"]),
                "camera_path": _rel(root, paths["camera"]),
                "visible": None,
                "raw_answer": None,
                "points": [],
                "parse_ok": False,
                "error": err,
            })
            continue

        t0 = time.time()
        answer, err = query_one(
            paths["rgb"],
            paths["depth"] if enable_depth else None,
            text=text,
            url=args.url,
            enable_depth=enable_depth,
            retry=args.retry,
        )
        dt = time.time() - t0

        if err is not None:
            print(f"[fail] view_{vid:03d}: {err}  ({dt:.1f}s)")
            view_records.append({
                "view_id": vid,
                "rgb_path": _rel(root, paths["rgb"]),
                "depth_path": _rel(root, paths["depth"]),
                "depth_raw_path": _rel(root, paths["depth_raw"]),
                "camera_path": _rel(root, paths["camera"]),
                "visible": True,
                "raw_answer": None,
                "points": [],
                "parse_ok": False,
                "error": err,
            })
            continue

        points, perr = _parse_points(answer)
        ok = perr is None
        status = "ok" if ok else f"parse-fail ({perr})"
        n = len(points)
        print(f"[done] view_{vid:03d}: {status}  n_points={n}  ({dt:.1f}s)")
        view_records.append({
            "view_id": vid,
            "rgb_path": _rel(root, paths["rgb"]),
            "depth_path": _rel(root, paths["depth"]),
            "depth_raw_path": _rel(root, paths["depth_raw"]),
            "camera_path": _rel(root, paths["camera"]),
            "visible": True,
            "raw_answer": answer,
            "points": points,
            "parse_ok": ok,
            "error": perr,
        })

    summary = {
        "prompt": args.prompt,
        "suffix": suffix,
        "url": args.url,
        "enable_depth": enable_depth,
        "root": str(root).replace("\\", "/"),
        "views": view_records,
    }
    return summary


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    """兼容旧版的单视角模式（替代 minimal_roborefer_e2e.py）。"""
    rgb = args.rgb.resolve()
    depth = args.depth.resolve() if args.depth is not None else None
    enable_depth = 0 if (args.no_depth or depth is None) else 1
    if not rgb.is_file():
        raise SystemExit(f"missing rgb file: {rgb}")
    if enable_depth and not depth.is_file():
        raise SystemExit(f"missing depth file: {depth}")

    suffix = "" if args.no_suffix else DEFAULT_SUFFIX
    text = args.prompt + suffix
    answer, err = query_one(
        rgb, depth,
        text=text, url=args.url, enable_depth=enable_depth, retry=args.retry,
    )
    if err is not None:
        raise SystemExit(err)
    print("raw answer:", answer)
    points, perr = _parse_points(answer)
    if perr is not None:
        print(f"[warn] parse failed: {perr}", file=sys.stderr)
    print("parsed points:", points)

    if args.output_image is not None:
        # 用 RoboRefer 的辅助函数在原始 RGB 上画圈
        api_dir = Path(__file__).resolve().parents[1] / "RoboRefer-main" / "API"
        sys.path.insert(0, str(api_dir))
        import use_api as use_api_mod  # noqa: WPS433

        use_api_mod.denormalize_and_mark(
            str(rgb),
            [(p["nx"], p["ny"]) for p in points],
            output_path=str(args.output_image.resolve()),
        )

    return {
        "prompt": args.prompt,
        "suffix": suffix,
        "url": args.url,
        "enable_depth": enable_depth,
        "rgb_path": str(rgb).replace("\\", "/"),
        "depth_path": str(depth).replace("\\", "/") if depth is not None else None,
        "raw_answer": answer,
        "points": points,
        "parse_ok": perr is None,
        "error": perr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RoboRefer batch client over a 3DGS custom-view directory.")
    ap.add_argument("--prompt", type=str, required=True, help="Object/instruction (suffix appended unless --no-suffix).")
    ap.add_argument("--url", type=str, default="http://127.0.0.1:25547")
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--no-depth", action="store_true", help="Force RGB-only mode.")
    ap.add_argument("--no-suffix", action="store_true", help="Do not append the standard normalized-coordinates suffix.")

    # 批量模式
    ap.add_argument("--root", type=Path, default=None, help="Custom-view root (e.g. 3DGS/test1)")
    ap.add_argument("--views", type=int, nargs="+", default=None, help="Subset of view ids; default = all.")
    ap.add_argument("--output", type=Path, default=None, help="Where to write predictions.json (batch mode).")
    ap.add_argument("--rgb", type=Path, default=None)
    ap.add_argument("--depth", type=Path, default=None)
    ap.add_argument("--output-image", type=Path, default=None, help="Annotated RGB output (single-view only).")

    args = ap.parse_args()
    if (args.root is None) == (args.rgb is None):
        ap.error("specify exactly one of --root (batch) or --rgb (single)")

    if args.root is not None:
        result = run_batch(args)
        out = args.output or (args.root.resolve() / "predictions.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ok = sum(1 for v in result["views"] if v["parse_ok"])
        print(f"[summary] wrote {out}  ok={ok}/{len(result['views'])}")
    else:
        result = run_single(args)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[summary] wrote {args.output}")


if __name__ == "__main__":
    main()
