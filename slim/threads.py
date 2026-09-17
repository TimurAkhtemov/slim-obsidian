"""Local per-note copilot threads: save, load, list, delete. Called by `chat.py`.

Threads persist so reopening a note resumes its conversation, and **persistence is not
evidence** — chat transcripts never enter ingestion or retrieval. The store lives under
`config.DATA_DIR/threads/`, outside the vault, so ingestion is structurally out of reach of it
(a regression test pins that).

Deterministic bounds, enforced here: whitelisted thread ids (no traversal), a per-thread turn
cap and byte cap that REFUSE rather than truncate, and per-thread delete as the only
destructive operation.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

from . import trace
from .config import DATA_DIR

THREADS_DIR = DATA_DIR / "threads"
MAX_TURNS = 500
MAX_THREAD_BYTES = 4_000_000     # a resumable conversation, not an archive
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")


class ThreadError(RuntimeError):
    """A thread operation was refused. The message is safe to return to the page."""


@dataclass(frozen=True)
class ThreadSummary:
    id: str
    title: str
    updated_at: str
    turns: int
    source_id: str | None = None
    source_path: str | None = None
    reasoning_mode: str | None = None


def _validate_id(thread_id) -> str:
    value = (thread_id or "").strip() if isinstance(thread_id, str) else ""
    if not _ID_RE.fullmatch(value):
        raise ThreadError(
            "thread id must be 1-64 letters, digits, or hyphens — no paths, no dots")
    return value


def _clean_turns(raw) -> list[dict]:
    """Keep only well-formed turns. The page owns the transcript shape; the store only
    guarantees it is a bounded list of {role, text, turn?} objects."""
    if not isinstance(raw, list):
        raise ThreadError("turns must be a list")
    turns = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role, text = item.get("role"), item.get("text")
        if role not in ("you", "slim") or not isinstance(text, str):
            continue
        turn = {"role": role, "text": text}
        if role == "you" and item.get("attachments"):
            # Import here to keep the thread store usable on its own while letting the image
            # contract own the exact MIME/id/size whitelist.
            from .copilot_images import ImageError, clean_refs
            try:
                turn["attachments"] = clean_refs(item.get("attachments"))
            except ImageError as exc:
                raise ThreadError(str(exc)) from exc
        if isinstance(item.get("turn"), dict):
            turn["turn"] = item["turn"]      # the SlimTurn payload, for rich rehydration
        turns.append(turn)
    if len(turns) > MAX_TURNS:
        raise ThreadError(
            f"thread has {len(turns)} turns, over the {MAX_TURNS}-turn cap — start a new "
            f"conversation (nothing was saved; the previous save is intact)")
    return turns


def _write(payload: dict) -> ThreadSummary:
    """Atomically write one already-cleaned thread payload."""
    tid = payload["id"]
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_THREAD_BYTES:
        raise ThreadError(
            f"thread is {len(encoded)} bytes, over the {MAX_THREAD_BYTES}-byte cap — "
            f"start a new conversation (nothing was saved)")
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    target = THREADS_DIR / f"{tid}.json"
    partial = target.with_name(f".{tid}.part")
    partial.write_bytes(encoded)
    partial.replace(target)
    return ThreadSummary(
        id=tid, title=payload["title"], updated_at=payload["updated_at"],
        turns=len(payload["turns"]), source_id=payload.get("source_id"),
        source_path=payload.get("source_path"),
        reasoning_mode=payload.get("reasoning_mode"))


def save_copilot(thread_id: str, title: str, turns, *, source_id: str,
                 source_path: str, reasoning_mode: str) -> ThreadSummary:
    """Upsert one source-anchored copilot chat."""
    tid = _validate_id(thread_id)
    sid = (source_id or "").strip() if isinstance(source_id, str) else ""
    if not _ID_RE.fullmatch(sid):
        raise ThreadError("source id must be 1-64 letters, digits, or hyphens")
    path = PurePosixPath(source_path or "")
    if path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".md":
        raise ThreadError("source path must be a vault-relative Markdown path")
    if reasoning_mode not in {"quick", "deep"}:
        raise ThreadError("reasoning mode must be quick or deep")
    cleaned = _clean_turns(turns)
    payload = {
        "id": tid,
        "title": (title or "").strip()[:120] or "Conversation",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_id": sid,
        "source_path": path.as_posix(),
        "reasoning_mode": reasoning_mode,
        "turns": cleaned,
    }
    return _write(payload)


def load(thread_id: str) -> dict | None:
    tid = _validate_id(thread_id)
    path = THREADS_DIR / f"{tid}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ThreadError(
            f"thread {tid} exists but could not be read ({type(exc).__name__})") from None


def list_copilot_threads() -> list[ThreadSummary]:
    """Only valid source-anchored chats, newest first."""
    if not THREADS_DIR.exists():
        return []
    summaries = []
    for path in THREADS_DIR.glob("*.json"):
        # ⚠ The construction is INSIDE the try, not just the parse: a thread file whose JSON
        # is valid but whose shape is not (`turns` a string, say) raises in `len()`, and one
        # bad file on disk would then brick the whole sidebar listing.
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("source_id") or not data.get("source_path"):
                continue
            summaries.append(ThreadSummary(
                id=str(data.get("id") or path.stem),
                title=str(data.get("title") or "Conversation"),
                updated_at=str(data.get("updated_at") or ""),
                turns=len(data.get("turns") or []),
                source_id=data.get("source_id"), source_path=data.get("source_path"),
                reasoning_mode=data.get("reasoning_mode")))
        except (OSError, json.JSONDecodeError, ValueError, AttributeError, TypeError):
            continue                          # an unreadable thread hides, never bricks
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


def delete_thread(thread_id: str) -> bool:
    """Delete ONE thread by id (the copilot sidebar's delete). Traced: deleting conversational
    history is exactly the kind of action a trace log exists to record. A bad id
    is refused before touching the filesystem; a missing file returns False, not an error
    (delete is idempotent — nothing to remove is a valid outcome, not a failure)."""
    tid = _validate_id(thread_id)
    try:
        (THREADS_DIR / f"{tid}.json").unlink()
        removed = True
    except FileNotFoundError:
        removed = False
    image_dir = THREADS_DIR / f"{tid}.images"
    if image_dir.is_dir():
        shutil.rmtree(image_dir)
        removed = True
    trace.record("thread_delete",
                 {"status": "deleted" if removed else "absent", "id": tid})
    return removed
