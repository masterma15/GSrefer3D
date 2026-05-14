#!/usr/bin/env python3
"""Visibility check via Qwen2-VL using OpenAI-compatible API.

Pre-filters views before a pointing VLM. Three prompt tiers: **permissive** (max recall),
**relaxed** (default), **strict** (clearly visible). VLMs are conservative; permissive
biases heavily toward ``yes`` for multi-view grounding.

Requires:
    pip install openai
    QWEN_API_KEY env var (DashScope key)

If you see ``Connection error`` from Windows or outside mainland China, try the
international OpenAI-compatible base (Singapore), for example::

    set QWEN_VISIBILITY_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

Or pass ``--visibility-base-url`` from ``roborefer_client.py``. Keys from some
consoles only work with the matching region endpoint.
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path


_VISIBILITY_PROMPT_PERMISSIVE = (
    "This is one camera view used for 3D visual grounding of a {object}. "
    "Should we keep this view for a downstream pointing model (even if the object is tiny, "
    "far away, blurry, at a steep angle, partly hidden, or only possibly present)? "
    "Answer with only 'yes' or 'no'. "
    "Bias strongly toward 'yes': say 'yes' if there is any plausible chance the {object} "
    "could appear in this frame or the view might still help triangulate it. "
    "Say 'no' only if you are confident this view cannot help (e.g. object class impossible in this scene, "
    "or the target is clearly absent from the entire frame with no ambiguity)."
)

_VISIBILITY_PROMPT_RELAXED = (
    "In this image, could a {object} plausibly be present anywhere in the frame for visual grounding "
    "(including very small, distant, dark, at an angle, partly occluded, or near the border)? "
    "Answer with only 'yes' or 'no'. "
    "Answer 'yes' if there is a reasonable chance it is there or a pointing model might still pick a meaningful location; "
    "answer 'no' only if you are fairly sure it is not in the image or would be a pure guess with no visual support."
)

_VISIBILITY_PROMPT_STRICT = (
    "Is there a {object} clearly visible in this image? "
    "Answer with only 'yes' or 'no'. "
    "Answer 'no' if it is absent or heavily occluded."
)

# English spatial prompts: use the object phrase only, not the full instruction.
_OBJECT_PHRASE_RE = re.compile(
    r"(?is)\b(?:please\s+)?(?:point|click)\s+to\s+(?:the\s+)?(.+?)\s*\.?\s*$",
)


def object_phrase_for_visibility(user_prompt: str) -> str:
    """Strip wrapper text so Qwen is asked about the object, not the whole instruction."""
    t = (user_prompt or "").strip()
    m = _OBJECT_PHRASE_RE.search(t)
    if m:
        return m.group(1).strip().rstrip(".")
    return t

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-vl-plus"


def resolve_visibility_model(override: str | None = None) -> str:
    """DashScope vision model id (compatible-mode). Override > env > default."""
    o = (override or "").strip()
    if o:
        return o
    v = (os.environ.get("QWEN_VISIBILITY_MODEL") or "").strip()
    return v or _DEFAULT_MODEL


def resolve_visibility_base_url(cli_override: str | None = None) -> str:
    """China (default) vs intl DashScope compatible endpoint."""
    o = (cli_override or "").strip()
    if o:
        return o.rstrip("/")
    for key in ("QWEN_VISIBILITY_BASE_URL", "DASHSCOPE_COMPATIBLE_BASE_URL"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v.rstrip("/")
    return _DEFAULT_BASE_URL.rstrip("/")


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def check_visibility(
    rgb_path: str | Path,
    object_description: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
    strict: bool = False,
    permissive: bool = False,
) -> tuple[bool, str]:
    """Return (visible, raw_answer). ``strict`` or ``permissive`` select prompt tier."""
    from openai import OpenAI

    key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set QWEN_API_KEY env var or pass api_key=")

    if strict and permissive:
        raise ValueError("strict and permissive are mutually exclusive")

    bu = resolve_visibility_base_url(base_url)
    mdl = resolve_visibility_model(model)
    client = OpenAI(api_key=key, base_url=bu, timeout=timeout)
    b64 = _encode_image(str(rgb_path))
    if strict:
        tmpl = _VISIBILITY_PROMPT_STRICT
    elif permissive:
        tmpl = _VISIBILITY_PROMPT_PERMISSIVE
    else:
        tmpl = _VISIBILITY_PROMPT_RELAXED
    prompt = tmpl.format(object=object_description)

    resp = client.chat.completions.create(
        model=mdl,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=16,
    )
    msg = resp.choices[0].message.content
    raw = (msg or "").strip()
    raw_l = raw.lower()
    # Strip a short leading label some models add.
    for prefix in ("answer:", "result:", "final:", "response:"):
        if raw_l.startswith(prefix):
            raw_l = raw_l[len(prefix):].strip()
            break
    visible = raw_l.startswith("yes")
    return visible, raw
