#!/usr/bin/env python3
"""Batch RoboRefer client for a 3DGS custom-view directory.

Layout expected (produced by ``3DGS/render.py --custom_views``, default 36 views):

  <root>/
    rgb/view_NNN.png          # N views (default 36 from 3DGS/render.py --custom_views)
    depth/view_NNN.png        # 8-bit invdepth-style PNG fed to RoboRefer
    depth_raw/view_NNN.npy    # float32, kept for downstream unprojection
    camera_params/view_NNN.json

For each view this script POSTs (RGB, depth.png) to the RoboRefer API with
``enable_depth=1, depth_url=[depth.png]``, parses the returned ``[(nx, ny), ...]``,
and writes a single ``predictions.json``:

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

A view that returned None or unparseable text keeps ``points=[]`` and records
``error``; the run does NOT abort. Single-view mode (``--rgb`` + ``--depth``)
preserves the old minimal_roborefer_e2e.py behaviour.

Run this inside the RoboRefer conda env (Linux/WSL recommended):
  conda activate roborefer
  # in one shell:
  cd RoboRefer-main/API && python api.py --port 25547
  # defaults: <repo>/weights/depth_anything_v2_vitl.pth, <repo>/RoboRefer-2B-SFT
  # in another:
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
    """Direct HTTP client for RoboRefer /query endpoint (no openai dependency)."""
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
    """Best-effort parse of RoboRefer output into [{nx, ny}, ...]."""
    if not isinstance(answer, str):
        return [], f"answer not a string: {type(answer).__name__}"
    txt = answer.strip()
    # Some RoboRefer answers wrap the list in prose; isolate the first [...] block.
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

    # --- visibility pre-filter (optional) ---
    vis_checker = None
    visibility_object: str | None = None
    if getattr(args, "visibility_check", False):
        try:
            from visibility_client import (
                check_visibility,
                object_phrase_for_visibility,
                resolve_visibility_base_url,
                resolve_visibility_model,
            )
        except ImportError:
            print(
                "[error] --visibility-check requires the openai package. "
                "In the same conda env you use for this script: pip install openai",
                file=sys.stderr,
            )
            raise SystemExit(2) from None
        vis_checker = check_visibility
        visibility_object = object_phrase_for_visibility(args.prompt)
        _vb = resolve_visibility_base_url(getattr(args, "visibility_base_url", None))
        if getattr(args, "visibility_strict", False):
            _vm = "strict"
        elif getattr(args, "visibility_permissive", False):
            _vm = "permissive"
        else:
            _vm = "relaxed"
        _vmdl = resolve_visibility_model(getattr(args, "visibility_model", None))
        print(
            f"[info] visibility check enabled  object={visibility_object!r}  "
            f"dashscope={_vb}  mode={_vm}  model={_vmdl}"
        )

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
                "visibility_raw": None,
                "raw_answer": None,
                "points": [],
                "parse_ok": False,
                "error": err,
            })
            continue

        # visibility check
        visible, vis_raw = True, None
        if vis_checker is not None:
            assert visibility_object is not None
            try:
                vis_kw: dict[str, Any] = {}
                vb = getattr(args, "visibility_base_url", None)
                if vb and str(vb).strip():
                    vis_kw["base_url"] = str(vb).strip()
                if getattr(args, "visibility_strict", False):
                    vis_kw["strict"] = True
                elif getattr(args, "visibility_permissive", False):
                    vis_kw["permissive"] = True
                vm = getattr(args, "visibility_model", None)
                if vm and str(vm).strip():
                    vis_kw["model"] = str(vm).strip()
                visible, vis_raw = vis_checker(paths["rgb"], visibility_object, **vis_kw)
                status = "visible" if visible else "not-visible"
                print(f"[vis]  view_{vid:03d}: {status}  ({vis_raw!r})")
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                c = e.__cause__
                d = 0
                while c is not None and d < 4:
                    detail += f" | {type(c).__name__}: {c}"
                    c = getattr(c, "__cause__", None)
                    d += 1
                hint = ""
                if "Connection" in type(e).__name__ or "connection" in str(e).lower():
                    hint = (
                        " [hint: outside China try "
                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1 "
                        "via --visibility-base-url or QWEN_VISIBILITY_BASE_URL]"
                    )
                print(f"[vis-warn] view_{vid:03d}: {detail}{hint}, skipping RoboRefer for this view")
                view_records.append({
                    "view_id": vid,
                    "rgb_path": _rel(root, paths["rgb"]),
                    "depth_path": _rel(root, paths["depth"]),
                    "depth_raw_path": _rel(root, paths["depth_raw"]),
                    "camera_path": _rel(root, paths["camera"]),
                    "visible": None,
                    "visibility_raw": None,
                    "raw_answer": None,
                    "points": [],
                    "parse_ok": False,
                    "error": f"visibility_check_api_error: {detail}",
                })
                continue

        if not visible:
            view_records.append({
                "view_id": vid,
                "rgb_path": _rel(root, paths["rgb"]),
                "depth_path": _rel(root, paths["depth"]),
                "depth_raw_path": _rel(root, paths["depth_raw"]),
                "camera_path": _rel(root, paths["camera"]),
                "visible": False,
                "visibility_raw": vis_raw,
                "raw_answer": None,
                "points": [],
                "parse_ok": False,
                "error": "filtered by visibility check",
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
                "visibility_raw": vis_raw,
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
            "visibility_raw": vis_raw,
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
        "visibility_check_requested": bool(getattr(args, "visibility_check", False)),
        "visibility_check_active": vis_checker is not None,
        "visibility_object": visibility_object,
        "visibility_dashscope_base_url": (
            resolve_visibility_base_url(getattr(args, "visibility_base_url", None))
            if getattr(args, "visibility_check", False) and vis_checker is not None
            else None
        ),
        "visibility_prompt_mode": (
            "strict"
            if getattr(args, "visibility_strict", False)
            else (
                "permissive"
                if getattr(args, "visibility_permissive", False)
                else "relaxed"
            )
        )
        if getattr(args, "visibility_check", False) and vis_checker is not None
        else None,
        "visibility_model": (
            resolve_visibility_model(getattr(args, "visibility_model", None))
            if getattr(args, "visibility_check", False) and vis_checker is not None
            else None
        ),
    }
    return summary


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    """Backwards-compatible single-view mode (replaces minimal_roborefer_e2e.py)."""
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
        # use RoboRefer's helper to draw circles on the original RGB
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

    # batch mode
    ap.add_argument("--root", type=Path, default=None, help="Custom-view root (e.g. 3DGS/test1)")
    ap.add_argument("--views", type=int, nargs="+", default=None, help="Subset of view ids; default = all.")
    ap.add_argument("--output", type=Path, default=None, help="Where to write predictions.json (batch mode).")
    ap.add_argument("--visibility-check", action="store_true",
                    help="Pre-filter views with Qwen2-VL visibility check (requires QWEN_API_KEY).")
    ap.add_argument(
        "--visibility-base-url",
        default=None,
        metavar="URL",
        help=(
            "DashScope OpenAI-compatible base URL for visibility (overrides env). "
            "Default China: https://dashscope.aliyuncs.com/compatible-mode/v1 — "
            "overseas often: https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        ),
    )
    vis_grp = ap.add_mutually_exclusive_group()
    vis_grp.add_argument(
        "--visibility-strict",
        action="store_true",
        help=(
            "Strict 'clearly visible' visibility prompt (more false negatives). "
            "Mutually exclusive with --visibility-permissive."
        ),
    )
    vis_grp.add_argument(
        "--visibility-permissive",
        action="store_true",
        help=(
            "Strongest bias toward keeping each view (tiny/distant/blurred/occluded still usually yes). "
            "Use when relaxed still drops too many plausible angles; may keep some useless views."
        ),
    )
    ap.add_argument(
        "--visibility-model",
        default=None,
        metavar="NAME",
        help=(
            "DashScope vision model for visibility (default: qwen-vl-plus). "
            "Try qwen-vl-max if too many false 'no'. Env QWEN_VISIBILITY_MODEL overrides when unset."
        ),
    )
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
