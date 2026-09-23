"""EDITS — turn the copilot's proposed changes into exact file contents the owner can accept.

The model writes SEARCH/REPLACE blocks as plain text, never JSON: a JSON grammar loops on a
long string field and LaTeX backslashes do not survive it (CLAUDE.md, Ollama traps). Code
parses the blocks, applies them to the file ON DISK, and hands the plugin each file's new text
with the hash it was computed from. Nothing here writes: the plugin writes only what the owner
accepts, and only if the file still hashes the same.

What may be touched is decided here, not by the model: a note the owner put in view (the open
note, a [[link]], a skill's input), or a new note in a folder that already exists. A recorded
note's frontmatter, its `slim-meeting` block and its transcript are `record.py`'s and the raw
source; they must come out of a proposal byte for byte, or the whole file is refused.

A file's changes apply ALL OR NOTHING (review, 2026-09-23): a "move this section" is a delete
plus an insert, and applying the delete alone lost the section.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .chunk import TRANSCRIPT_MARKER, parse_frontmatter, set_frontmatter
from .config import EXCLUDED_DIRS

_FILE = re.compile(r"^\s*(?:\*\*)?file(?:\*\*)?\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SEARCH = re.compile(r"^<{5,}\s*SEARCH\s*$")
_DIVIDER = re.compile(r"^={5,}\s*$")
_REPLACE = re.compile(r"^>{5,}\s*REPLACE\s*$")
_FENCE = re.compile(r"^\s*```[\w-]*\s*$")
_MEETING_OPEN = re.compile(r"^\s*(`{3,})slim-meeting")
# Binaries, SLIM's own derived output and Obsidian's config are never a copilot edit target.
_NEVER_TOP = {"attachments", "_reflections"}
_EXCLUDED_CF = {name.casefold() for name in EXCLUDED_DIRS}
# Past this a note is edited in Obsidian, not from the sidebar: every proposal carries the
# file's whole new text, and a thread holds at most 4 MB (threads.MAX_THREAD_BYTES).
MAX_EDIT_CHARS = 400_000
# A note the copilot creates says so, like every other autonomous write (CLAUDE.md).
CREATED_ORIGIN = "copilot"

PROTECTED = "part of the recording; SLIM never edits it"


@dataclass
class Block:
    path: str | None
    search: str
    replace: str
    complete: bool = True
    problem: str = ""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_path(raw: str) -> str:
    path = (raw or "").strip().strip("`*\"'").strip()
    return path[2:] if path.startswith("./") else path


def _split_body(lines: list[str]) -> tuple[str, str] | None:
    """SEARCH and REPLACE from the lines between the markers. A setext heading's `=====`
    underline looks like the divider, so with several candidates only a lone 7-`=` line splits."""
    dividers = [i for i, line in enumerate(lines) if _DIVIDER.match(line.rstrip("\n"))]
    if len(dividers) > 1:
        dividers = [i for i in dividers if lines[i].strip() == "======="]
    if len(dividers) != 1:
        return None
    at = dividers[0]
    return "".join(lines[:at]), "".join(lines[at + 1:])


def parse(answer: str) -> tuple[str, list[Block]]:
    """Split an answer into (prose, blocks). Prose is what the chat shows; blocks become the
    proposal. A code fence wrapped around the blocks goes with them."""
    lines = (answer or "").splitlines(keepends=True)
    prose: list[str] = []
    blocks: list[Block] = []
    path = None
    body: list[str] | None = None
    after_block = False

    def is_file(line: str) -> str | None:
        match = _FILE.match(line.rstrip("\n"))
        value = _clean_path(match.group(1)) if match else ""
        return value if value.lower().endswith(".md") else None

    # A fence that OPENS a FILE/SEARCH group: the next non-blank line starts the group.
    opening = set()
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            nxt = next((later for later in lines[i + 1:] if later.strip()), "")
            if is_file(nxt) or _SEARCH.match(nxt.rstrip("\n")):
                opening.add(i)
    for i, line in enumerate(lines):
        bare = line.rstrip("\n")
        if body is not None:
            if _REPLACE.match(bare):
                split = _split_body(body)
                if split is None:
                    blocks.append(Block(path, "", "", problem="the change was malformed"))
                else:
                    blocks.append(Block(path, *split))
                body, after_block = None, True
            else:
                body.append(line)
            continue
        if i in opening:
            continue
        if _SEARCH.match(bare):
            body = []
            continue
        found = is_file(line)
        if found:
            path, after_block = found, True
            continue
        if _FENCE.match(bare) and after_block:
            continue                     # the fence closing a FILE/SEARCH/REPLACE group
        if bare.strip():
            # Prose between blocks ends the FILE: a later block must name its own.
            after_block, path = False, None if blocks else path
        prose.append(line)
    if body is not None:
        blocks.append(Block(path, "".join(body), "", complete=False))
    # Greedy decoding repeats itself; a repeated append must not append twice.
    unique, seen = [], set()
    for item in blocks:
        key = (item.path, item.search, item.replace, item.complete, item.problem)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    text = re.sub(r"\n{3,}", "\n\n", "".join(prose)).strip()
    return text, unique


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
        # A memo-lane or imported note: no block, and its transcript runs to the end of the file.
        ranges.append((starts[marker_at], len(text)))
    return ranges


def _protected_slices(text: str) -> list[str]:
    return [text[begin:end] for begin, end in protected_ranges(text)]


def _find(text: str, search: str) -> tuple[list[tuple[int, int]], bool]:
    """Every place `search` occurs as whole lines: exactly, else line by line ignoring trailing
    whitespace. A match must start at a line start: `Total: 5` is not inside `Grand Total: 5`."""
    spans, start = [], text.find(search)
    while start != -1:
        if start == 0 or text[start - 1] == "\n":
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


def _adds_markers(replace: str) -> bool:
    return any(_MEETING_OPEN.match(line) or line.strip() == TRANSCRIPT_MARKER
               for line in replace.splitlines())


def apply_blocks(text: str, blocks: list[Block]) -> tuple[str, list[dict]]:
    """Apply blocks in order to one file's text, all or nothing: if any block fails, the text
    comes back unchanged and every result says why the file was refused."""
    original, before = text, _protected_slices(text)
    results = []
    for item in blocks:
        if item.problem:
            results.append({"ok": False, "reason": item.problem})
            continue
        if not item.complete:
            results.append({"ok": False, "reason": "cut off before the change ended"})
            continue
        if _adds_markers(item.replace):
            results.append({"ok": False, "reason": "adds a recording's block or transcript heading"})
            continue
        if not item.search.strip():
            # An append lands before a transcript that runs to the end of the note, never in it.
            tail = next((begin for begin, end in protected_ranges(text) if end >= len(text)), None)
            if tail == 0:
                results.append({"ok": False, "reason": PROTECTED})
                continue
            head, rest = (text, "") if tail is None else (text[:tail], text[tail:])
            added = item.replace.strip("\n") + "\n"
            text = (head.rstrip("\n") + "\n\n" if head.strip() else "") + added + ("\n" + rest if rest else "")
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
        if not exact and replacement and not replacement.endswith("\n") and text[start:end].endswith("\n"):
            replacement += "\n"
        if end == len(text) and not text.endswith("\n") and replacement.endswith("\n"):
            replacement = replacement[:-1]          # the note had no final newline; keep it so
        text = text[:start] + replacement + text[end:]
        results.append({"ok": True})
    if all(result["ok"] for result in results) and _protected_slices(text) != before:
        results = [{"ok": False, "reason": "would change the recording's protected parts"}
                   for _ in results]
    if not all(result["ok"] for result in results):
        failed = next(result["reason"] for result in results if not result["ok"])
        results = [result if not result["ok"] else
                   {"ok": False, "reason": f"held back: another change to this note failed ({failed})"}
                   for result in results]
        return original, results
    return text, results


def note_path(raw: str | None) -> str | None:
    """A vault-relative Markdown path, or None. Obsidian's `normalizePath` turns `\\` into `/`,
    so a backslash could smuggle `..` past a POSIX check; such a path is refused outright."""
    if not raw or any(ch in raw for ch in "\\:\0") or any(ord(ch) < 32 for ch in raw):
        return None
    if unicodedata.normalize("NFC", raw) != raw:
        return None
    path = PurePosixPath(raw)
    parts = [part.casefold() for part in path.parts]
    if (path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".md"
            or any(part.startswith(".") or part in _EXCLUDED_CF for part in parts)
            or parts[0] in _NEVER_TOP):
        return None
    return path.as_posix()


def _exact(vault: Path, path: str) -> bool:
    """True when every part of `path` exists with exactly this spelling. APFS matches names
    case-insensitively, so `is_file()` alone calls `notes/a.md` the existing `Notes/A.md`."""
    here = Path(vault)
    for part in PurePosixPath(path).parts:
        try:
            if part not in os.listdir(here):
                return False
        except OSError:
            return False
        here = here / part
    return True


def _read(target: Path) -> str:
    with open(target, encoding="utf-8", newline="") as handle:   # as Obsidian reads it: no \r\n folding
        return handle.read()


def _with_origin(text: str) -> str:
    _fm, skip = parse_frontmatter(text)
    if skip:
        return set_frontmatter(text, {"origin": CREATED_ORIGIN})
    return f"---\norigin: {CREATED_ORIGIN}\n---\n\n{text.lstrip()}"


def propose(vault: Path, blocks: list[Block], *, in_view: set[str]) -> dict:
    """Group blocks by file and compute each file's new text. Returns
    {"files": [{path, kind, base_hash, after, blocks}], "dropped": [{path, reason}]}."""
    by_path: dict[str, list[Block]] = {}
    dropped = []
    # `Journal/` is the floor (CLAUDE.md): a journal's words never leave it through a proposal.
    journal = any(path.casefold().startswith("journal/") for path in in_view)
    for item in blocks:
        if not item.path:
            dropped.append({"path": "", "reason": "no FILE line before the change"})
            continue
        path = note_path(item.path)
        if path is None:
            dropped.append({"path": item.path, "reason": "not a note path in this vault"})
            continue
        if journal and not path.casefold().startswith("journal/"):
            dropped.append({"path": path, "reason": "a Journal note's text stays in Journal/"})
            continue
        by_path.setdefault(path, []).append(item)
    files = []
    for path, group in by_path.items():
        target = Path(vault) / path
        if target.exists():
            if not _exact(vault, path):
                dropped.append({"path": path, "reason": "a note with this name exists in different letter case"})
                continue
            if path not in in_view:
                stem = PurePosixPath(path).stem
                dropped.append({"path": path,
                                "reason": f"not in view; link it as [[{stem}]] to let SLIM edit it"})
                continue
            try:
                before = _read(target)
            except (OSError, UnicodeDecodeError) as exc:
                dropped.append({"path": path, "reason": f"could not be read ({type(exc).__name__})"})
                continue
            if "\r" in before or before.startswith("\ufeff"):
                dropped.append({"path": path, "reason": "uses Windows line endings or a byte-order mark; edit it in Obsidian"})
                continue
            if len(before) > MAX_EDIT_CHARS:
                dropped.append({"path": path, "reason": "too large to edit from the sidebar"})
                continue
            kind, base = "edit", digest(before)
        elif not _exact(vault, PurePosixPath(path).parent.as_posix()):
            dropped.append({"path": path, "reason": "folder does not exist"})
            continue
        else:
            before, kind, base = "", "create", None
        if kind == "create" and any(item.search.strip() for item in group):
            results = [{"ok": False, "reason": "note does not exist"} for _ in group]
            after = before
        else:
            after, results = apply_blocks(before, group)
        if all(result["ok"] for result in results) and not after.strip():
            results = [{"ok": False, "reason": "would leave the note empty"} for _ in group]
        changed = all(result["ok"] for result in results) and after != before
        if changed and kind == "create":
            after = _with_origin(after)
        files.append({"path": path, "kind": kind, "base_hash": base,
                      "after": after if changed else None, "blocks": results})
    return {"files": files, "dropped": dropped}
