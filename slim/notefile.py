"""Note file names and safe writes, shared by both lanes."""
from __future__ import annotations

import re
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    try:
        partial.write_text(text, encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"[\s_]+", "-", s)[:60] or "recording"
