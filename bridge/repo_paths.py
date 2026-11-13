"""仓库内数据路径：新文件写相对仓库根，读取时认三种写法。

相对路径、Windows 盘符绝对路径、WSL ``/mnt/<盘>/...`` 都解析成当前机器上的路径。
不改已有 JSON。
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _is_windows_drive(raw: str) -> bool:
    return len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha()


def _windows_to_wsl(raw: str) -> Path:
    drive = raw[0].lower()
    rest = raw[2:].lstrip("\\/")
    return Path("/mnt") / drive / rest.replace("\\", "/")


def _wsl_to_windows(raw: str) -> Path | None:
    # /mnt/e/GSrefer3D/...
    if not raw.startswith("/mnt/") or len(raw) < 7 or raw[6] != "/":
        return None
    drive = raw[5]
    if not drive.isalpha():
        return None
    rest = raw[7:]
    return Path(f"{drive.upper()}:/{rest}")


def resolve_data_path(path: str | Path, *, root: Path | None = None) -> Path:
    """把 JSON 里的路径解析成当前系统上的 Path。文件可以尚不存在。"""
    raw = str(path).strip()
    if _is_windows_drive(raw):
        win = Path(raw)
        if win.exists():
            return win
        return _windows_to_wsl(raw)
    if raw.startswith("/mnt/"):
        wsl = Path(raw)
        if wsl.exists():
            return wsl
        win = _wsl_to_windows(raw)
        if win is not None and win.exists():
            return win
        return wsl
    p = Path(raw)
    if p.is_absolute():
        return p
    if root is not None:
        under_root = Path(root) / p
        if under_root.exists():
            return under_root
    under_repo = REPO / p
    if under_repo.exists():
        return under_repo
    if root is not None:
        return Path(root) / p
    return under_repo


def to_repo_relative(path: str | Path) -> str:
    """新写入 JSON 的路径：能落在仓库内就用正斜杠相对路径。"""
    resolved = resolve_data_path(path)
    try:
        rel = resolved.resolve().relative_to(REPO.resolve())
    except ValueError:
        return resolved.as_posix()
    return rel.as_posix()
