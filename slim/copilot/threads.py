"""Local copilot threads: save, load, list and search, delete. Called by `server.py`.

A thread stands on its own: `source_*` only records the note it was started on, and `pins` the
notes and folders the owner attached. Threads persist so a conversation resumes, and **persistence is not
evidence** — chat transcripts never enter ingestion or retrieval. The store lives under
`config.DATA_DIR/threads/`, outside the vault, so ingestion is structurally out of reach of it
(a regression test pins that).

Deterministic bounds, enforced here: whitelisted thread ids (no traversal), a per-thread turn
cap and byte cap that REFUSE rather than truncate, and per-thread delete as the only
destructive operation.

Beside each thread lives the note as it was before each editing run (`save_before`), so See
before and Restore work after the note is closed. Only the copy's ref rides in the thread: a
note is up to 400k characters, and the thread's byte cap refuses rather than truncates.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

from .. import trace
from ..config import DATA_DIR

THREADS_DIR = DATA_DIR / "threads"
MAX_TURNS = 500
MAX_PINS = 12
MAX_NOTE_PATH = 512              # the open note's path, kept on each question turn
MAX_THREAD_BYTES = 16_000_000    # a resumable conversation, each turn's transcript in it
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")
_REF_RE = re.compile(r"^[0-9a-f]{32}$")


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
    pins: tuple = ()


def validate_id(thread_id) -> str:
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
            from .images import ImageError, clean_refs
            try:
                turn["attachments"] = clean_refs(item.get("attachments"))
            except ImageError as exc:
                raise ThreadError(str(exc)) from exc
        if (role == "you" and isinstance(item.get("note"), str)
                and len(item["note"]) <= MAX_NOTE_PATH):
            turn["note"] = item["note"]          # the open note's path when this was sent
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
        pins=tuple(payload.get("pins") or ()))


def _clean_pins(raw) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ThreadError("pins must be a list")
    if len(raw) > MAX_PINS:
        raise ThreadError(f"a thread holds at most {MAX_PINS} pins")
    pins = []
    for item in raw:
        kind = item.get("kind") if isinstance(item, dict) else None
        text = item.get("path") if isinstance(item, dict) else None
        if kind not in ("note", "folder") or not isinstance(text, str) or not text.strip():
            raise ThreadError("a pin is {kind: note or folder, path}")
        path = PurePosixPath(text)
        if path.is_absolute() or ".." in path.parts or len(text) > MAX_NOTE_PATH:
            raise ThreadError("a pin's path must be vault-relative, with no ..")
        pins.append({"kind": kind, "path": path.as_posix()})
    return pins


def save_copilot(thread_id: str, title: str, turns, *,
                 source_id: str | None = None, source_path: str | None = None,
                 pins: list[dict] | None = None) -> ThreadSummary:
    """Upsert one copilot chat. `source_*` is the note it was started on, and optional. Each
    answer's turn names the effort it ran at; a chat has none of its own (its old
    `reasoning_mode`, on disk, is read by nothing)."""
    tid = validate_id(thread_id)
    sid = (source_id or "").strip() if isinstance(source_id, str) else ""
    if sid and not _ID_RE.fullmatch(sid):
        raise ThreadError("source id must be 1-64 letters, digits, or hyphens")
    spath = None
    if isinstance(source_path, str) and source_path:
        path = PurePosixPath(source_path)
        if path.is_absolute() or ".." in path.parts or path.suffix.casefold() != ".md":
            raise ThreadError("source path must be a vault-relative Markdown path")
        spath = path.as_posix()
    cleaned = _clean_turns(turns)
    payload = {
        "id": tid,
        "title": (title or "").strip()[:120] or "Conversation",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_id": sid or None,
        "source_path": spath,
        "pins": _clean_pins(pins),
        "turns": cleaned,
    }
    return _write(payload)


def load(thread_id: str) -> dict | None:
    tid = validate_id(thread_id)
    path = THREADS_DIR / f"{tid}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ThreadError(
            f"thread {tid} exists but could not be read ({type(exc).__name__})") from None


# Each thread file's summary and the casefolded text search looks through, by (mtime, size): a
# thread holds every answer's transcript, and the history list and its search box (200 ms after
# each keystroke) would otherwise parse every file whole on every request.
_LISTED: dict[str, tuple[tuple[int, int], ThreadSummary, list[str]]] = {}


def _listed(path) -> tuple[ThreadSummary, list[str]]:
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    hit = _LISTED.get(str(path))
    if hit and hit[0] == key:
        return hit[1], hit[2]
    data = json.loads(path.read_text(encoding="utf-8"))
    title = str(data.get("title") or "Conversation")
    turns = data.get("turns") or []
    summary = ThreadSummary(
        id=str(data.get("id") or path.stem), title=title,
        updated_at=str(data.get("updated_at") or ""),
        turns=len(turns),
        source_id=data.get("source_id") or None,
        source_path=data.get("source_path") or None,
        pins=tuple(data.get("pins") or ()))
    texts = [title.casefold(), *(t["text"].casefold() for t in turns
                                 if isinstance(t, dict) and isinstance(t.get("text"), str))]
    _LISTED[str(path)] = (key, summary, texts)
    return summary, texts


def list_copilot_threads(query: str = "") -> list[ThreadSummary]:
    """Every readable thread, newest first. With `query`, only those whose title or any turn's
    text contains it, case-insensitively."""
    if not THREADS_DIR.exists():
        return []
    needle = (query or "").casefold()
    summaries = []
    for path in THREADS_DIR.glob("*.json"):
        # ⚠ The construction is INSIDE the try, not just the parse: a thread file whose JSON
        # is valid but whose shape is not (`turns` a string, say) raises in `len()`, and one
        # bad file on disk would then brick the whole sidebar listing.
        try:
            summary, texts = _listed(path)
        except (OSError, json.JSONDecodeError, ValueError, AttributeError, TypeError):
            continue                          # an unreadable thread hides, never bricks
        if needle and not any(needle in text for text in texts):
            continue
        summaries.append(summary)
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


def _before_dir(thread_id: str):
    return THREADS_DIR / f"{validate_id(thread_id)}.before"


def save_before(thread_id: str, text: str) -> str:
    """Keep the note as it was before an editing run, byte for byte; returns the copy's ref."""
    directory = _before_dir(thread_id)
    if not isinstance(text, str):
        raise ThreadError("the copy from before must be text")
    ref = uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / f".{ref}.part"
    partial.write_bytes(text.encode("utf-8"))
    partial.replace(directory / f"{ref}.md")
    return ref


def load_before(thread_id: str, ref: str) -> str | None:
    """The note as it was before a run, or None when this thread has no such copy."""
    directory = _before_dir(thread_id)
    if not isinstance(ref, str) or not _REF_RE.fullmatch(ref):
        raise ThreadError("a copy's ref is 32 hex digits")
    try:
        return (directory / f"{ref}.md").read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ThreadError(f"copy {ref} could not be read ({type(exc).__name__})") from None


def delete_thread(thread_id: str) -> bool:
    """Delete ONE thread by id (the copilot sidebar's delete). Traced: deleting conversational
    history is exactly the kind of action a trace log exists to record. A bad id
    is refused before touching the filesystem; a missing file returns False, not an error
    (delete is idempotent — nothing to remove is a valid outcome, not a failure)."""
    tid = validate_id(thread_id)
    try:
        (THREADS_DIR / f"{tid}.json").unlink()
        removed = True
    except FileNotFoundError:
        removed = False
    for kept in (f"{tid}.images", f"{tid}.before"):
        if (THREADS_DIR / kept).is_dir():
            shutil.rmtree(THREADS_DIR / kept)
            removed = True
    trace.record("thread_delete",
                 {"status": "deleted" if removed else "absent", "id": tid})
    return removed
