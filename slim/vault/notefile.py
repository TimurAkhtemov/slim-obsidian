"""Note file names and safe writes, shared by both lanes."""
from __future__ import annotations

import re
from datetime import datetime
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


def local_date(when: datetime) -> str:
    """The LOCAL calendar date of an instant — for filenames and the `date:` field.

    ⚠ NEVER `when.strftime("%Y-%m-%d")` DIRECTLY. `when` is `datetime.now(timezone.utc)`, so a
    bare strftime yields the UTC date and every recording made after 20:00 local is stamped
    TOMORROW — measured 2026-08-27 on three evening lectures that carried the next day's date.
    It also MISSORTS: the date is the filename prefix, so an evening lecture sorts ahead of the
    following morning's. `recorded_at` stays the UTC instant, which is a point in time and
    correct as stored. This is the CALENDAR DAY, which only exists in a timezone."""
    return when.astimezone().strftime("%Y-%m-%d")
