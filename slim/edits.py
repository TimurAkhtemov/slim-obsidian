"""EDITS — turn the copilot's proposed changes into exact file contents the owner can accept.

The model writes SEARCH/REPLACE blocks as plain text, never JSON: a JSON grammar loops on a
long string field and LaTeX backslashes do not survive it (CLAUDE.md, Ollama traps). Code
parses the blocks, applies them to the file ON DISK, and hands the plugin each file's new text
with the hash it was computed from. Nothing here writes: the plugin writes only what the owner
accepts, and only if the file still hashes the same.

What may be touched is decided here, not by the model: a note the owner put in view (the open
note, a [[link]], a skill's input), or a new note in a folder that already exists. A recorded
note's frontmatter, its `slim-meeting` block and its transcript are `record.py`'s and the raw
source, so a block that reaches into them is refused.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .chunk import TRANSCRIPT_MARKER, parse_frontmatter
from .config import EXCLUDED_DIRS

_FILE = re.compile(r"^\s*(?:\*\*)?file(?:\*\*)?\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SEARCH = re.compile(r"^<{5,}\s*SEARCH\s*$")
_DIVIDER = re.compile(r"^={5,}\s*$")
_REPLACE = re.compile(r"^>{5,}\s*REPLACE\s*$")
_FENCE = re.compile(r"^\s*```[\w-]*\s*$")
_MEETING_OPEN = re.compile(r"^\s*(`{3,})slim-meeting")
# Binaries and SLIM's own derived output are never a copilot edit target.
_NEVER_TOP = {"Attachments", "_Reflections"}

PROTECTED = "part of the recording; SLIM never edits it"


@dataclass
class Block:
    path: str | None
    search: str
    replace: str
    complete: bool = True


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_path(raw: str) -> str:
    path = (raw or "").strip().strip("`*\"'").strip()
    return path[2:] if path.startswith("./") else path


def parse(answer: str) -> tuple[str, list[Block]]:
    """Split an answer into (prose, blocks). Prose is what the chat shows; blocks become the
    proposal. A code fence wrapped around the blocks goes with them."""
    lines = (answer or "").splitlines(keepends=True)
    prose: list[str] = []
    blocks: list[Block] = []
    path = None
    state, search, replace = "prose", [], []
    after_block = False
    # A fence that OPENS a FILE/SEARCH group: the next non-blank line starts the group.
    opening = set()
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            nxt = next((later for later in lines[i + 1:] if later.strip()), "")
            if _FILE.match(nxt.rstrip("\n")) or _SEARCH.match(nxt.rstrip("\n")):
                opening.add(i)
    for i, line in enumerate(lines):
        bare = line.rstrip("\n")
        if state == "prose" and i in opening:
            continue
        if state == "search":
            if _DIVIDER.match(bare):
                state = "replace"
            else:
                search.append(line)
            continue
        if state == "replace":
            if _REPLACE.match(bare):
                blocks.append(Block(path, "".join(search), "".join(replace)))
                state, search, replace, after_block = "prose", [], [], True
            else:
                replace.append(line)
            continue
        if _SEARCH.match(bare):
            state, search, replace = "search", [], []
            continue
        match = _FILE.match(bare)
        if match:
            path = _clean_path(match.group(1))
            after_block = True
            continue
        if _FENCE.match(bare) and after_block:
            continue                     # the fence around a FILE/SEARCH/REPLACE group
        if bare.strip():
            after_block = False
        prose.append(line)
    if state != "prose":
        blocks.append(Block(path, "".join(search), "".join(replace), complete=False))
    text = re.sub(r"\n{3,}", "\n\n", "".join(prose)).strip()
    return text, blocks


def _line_starts(text: str) -> list[int]:
    starts, offset = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    starts.append(offset)
    return starts


def protected_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges no edit may touch. Only a RECORDED note has any: one carrying a
    `slim-meeting` block or a transcript. On every other note the owner may change anything."""
    lines = text.splitlines(keepends=True)
    starts = _line_starts(text)
    ranges = []
    fence_at = next((i for i, line in enumerate(lines) if _MEETING_OPEN.match(line)), None)
    marker_at = next((i for i, line in enumerate(lines) if line.strip() == TRANSCRIPT_MARKER), None)
    if fence_at is None and marker_at is None:
        return []
    _fm, skip = parse_frontmatter(text)
    if skip:
        ranges.append((0, starts[skip]))
    if fence_at is not None:
        fence = _MEETING_OPEN.match(lines[fence_at]).group(1)
        close = next((i for i in range(fence_at + 1, len(lines)) if lines[i].strip() == fence),
                     len(lines) - 1)
        ranges.append((starts[fence_at], starts[close + 1]))
    if marker_at is not None and fence_at is None:
        # A memo-lane note: no block, and its transcript runs to the end of the file.
        ranges.append((starts[marker_at], len(text)))
    return ranges


def _find(text: str, search: str) -> tuple[list[tuple[int, int]], bool]:
    """Every place `search` occurs: exactly, else line by line ignoring trailing whitespace."""
    spans, start = [], text.find(search)
    while start != -1:
        spans.append((start, start + len(search)))
        start = text.find(search, start + 1)
    if spans:
        return spans, True
    want = [line.rstrip() for line in search.splitlines()]
    lines = text.splitlines(keepends=True)
    starts = _line_starts(text)
    have = [line.rstrip() for line in lines]
    for i in range(len(have) - len(want) + 1):
        if have[i:i + len(want)] == want:
            spans.append((starts[i], starts[i + len(want)]))
    return spans, False


def apply_blocks(text: str, blocks: list[Block]) -> tuple[str, list[dict]]:
    """Apply blocks in order to one file's text. A failed block changes nothing."""
    results = []
    for item in blocks:
        if not item.complete:
            results.append({"ok": False, "reason": "cut off before the change ended"})
            continue
        if not item.search.strip():
            text = (text.rstrip("\n") + "\n\n" if text.strip() else "") + item.replace.strip("\n") + "\n"
            results.append({"ok": True})
            continue
        spans, exact = _find(text, item.search)
        if not spans:
            results.append({"ok": False, "reason": "text not found in the note"})
            continue
        if len(spans) > 1:
            results.append({"ok": False, "reason": f"matches {len(spans)} places; include more lines"})
            continue
        start, end = spans[0]
        if any(start < stop and end > begin for begin, stop in protected_ranges(text)):
            results.append({"ok": False, "reason": PROTECTED})
            continue
        replacement = item.replace
        if not exact and not replacement.endswith("\n") and text[start:end].endswith("\n"):
            replacement += "\n"
        text = text[:start] + replacement + text[end:]
        results.append({"ok": True})
    return text, results


def note_path(raw: str | None) -> str | None:
    if not raw:
        return None
    path = PurePosixPath(raw)
    if (path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".md"
            or any(part.startswith(".") or part in EXCLUDED_DIRS for part in path.parts)
            or path.parts[0] in _NEVER_TOP):
        return None
    return path.as_posix()


def propose(vault: Path, blocks: list[Block], *, in_view: set[str]) -> dict:
    """Group blocks by file and compute each file's new text. Returns
    {"files": [{path, kind, base_hash, after, blocks}], "dropped": [{path, reason}]}."""
    by_path: dict[str, list[Block]] = {}
    dropped = []
    for item in blocks:
        if not item.path:
            dropped.append({"path": "", "reason": "no FILE line before the change"})
            continue
        path = note_path(item.path)
        if path is None:
            dropped.append({"path": item.path, "reason": "not a note path in this vault"})
            continue
        by_path.setdefault(path, []).append(item)
    files = []
    for path, group in by_path.items():
        target = Path(vault) / path
        if target.exists():
            if path not in in_view:
                stem = PurePosixPath(path).stem
                dropped.append({"path": path,
                                "reason": f"not in view; link it as [[{stem}]] to let SLIM edit it"})
                continue
            before = target.read_text(encoding="utf-8")
            kind, base = "edit", digest(before)
        elif not target.parent.is_dir():
            dropped.append({"path": path, "reason": "folder does not exist"})
            continue
        else:
            before, kind, base = "", "create", None
        if kind == "create" and any(item.search.strip() for item in group):
            results = [{"ok": False, "reason": "note does not exist"} for _ in group]
            after = before
        else:
            after, results = apply_blocks(before, group)
        changed = any(result["ok"] for result in results) and after != before
        files.append({"path": path, "kind": kind, "base_hash": base,
                      "after": after if changed else None, "blocks": results})
    return {"files": files, "dropped": dropped}
