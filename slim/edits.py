"""EDITS — turn the copilot's proposed changes into exact file contents the owner can accept.

The model writes SEARCH/REPLACE blocks as plain text, never JSON: a JSON grammar loops on a
long string field and LaTeX backslashes do not survive it (CLAUDE.md, Ollama traps). To rewrite
most of a note it writes a NOTE block, the note's whole new text, instead of copying it all into
a SEARCH first; a recording's own parts stay as they are whatever it writes for them. Code
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

import difflib
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .chunk import TRANSCRIPT_MARKER, parse_frontmatter, set_frontmatter
from .config import EXCLUDED_DIRS

_FILE = re.compile(r"^\s*(?:\*\*)?file(?:\*\*)?\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SEARCH = re.compile(r"^<{5,}\s*SEARCH\s*$")
_DIVIDER = re.compile(r"^={5,}\s*$")
_REPLACE = re.compile(r"^>{5,}\s*REPLACE\s*$")
_NOTE = re.compile(r"^<{5,}\s*NOTE\s*$")
# The model closes a NOTE with REPLACE's marker, its own opening one, or a bare `NOTE` (2026-10-12).
_NOTE_END = re.compile(r"^(?:>{5,}\s*(?:NOTE|REPLACE)|<{5,}\s*NOTE|NOTE)\s*$")
_FENCE = re.compile(r"^\s*```[\w-]*\s*$")
_MEETING_OPEN = re.compile(r"^\s*(`{3,})slim-meeting")
# A lone `...` in new text is the model's "the rest as before", and it would delete the rest.
_ELISION = re.compile(r"^[ \t]*(?:\.\.\.|…)[ \t]*$", re.MULTILINE)
# A SEARCH copied from a long passage carries mistakes: 6 of 20 whole-note rewrites were refused
# for a mistyped line, each copy with the note's line count and 95-99.7% of its lines exact
# (2026-10-12). Past this many lines it matches by its first and last lines, if nearly every
# line between them is the note's.
MIN_CLOSE_LINES = 5
CLOSE_RATIO = 0.9
# A SEARCH copying a note from its top is a rewrite of the note. Past this many lines, or a
# quarter of the note's, the copilot cuts it off and the model writes the note once instead,
# as a NOTE block: the copy cost as much as the new text and was where the mistakes were.
REWRITE_AFTER_LINES = 8
# Binaries, SLIM's own derived output and Obsidian's config are never a copilot edit target.
_NEVER_TOP = {"attachments", "_reflections"}
_EXCLUDED_CF = {name.casefold() for name in EXCLUDED_DIRS}
# Past this a note is edited in Obsidian, not from the sidebar: every proposal carries the
# file's whole new text, and a thread holds at most 4 MB (threads.MAX_THREAD_BYTES).
MAX_EDIT_CHARS = 400_000
# A note the copilot creates says so, like every other autonomous write (CLAUDE.md).
CREATED_ORIGIN = "copilot"

PROTECTED = "part of the recording; SLIM never edits it"
SKIPS = "the new text skips lines with `...`"


@dataclass
class Block:
    path: str | None
    search: str
    replace: str
    complete: bool = True
    problem: str = ""
    whole: bool = False          # a NOTE block: `replace` is the note's whole new text


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_path(raw: str) -> str:
    # `· the open note` is the pack's label after the path, which the model copies.
    path = (raw or "").split(" · ")[0].strip().strip("`*\"'=").strip()
    return path[2:] if path.startswith("./") else path


def _split_body(lines: list[str]) -> tuple[str, str] | None:
    """SEARCH and REPLACE from the lines between the markers. A setext heading's `=====`
    underline looks like the divider, so with several candidates only a lone 7-`=` line splits."""
    dividers = [i for i, line in enumerate(lines) if _DIVIDER.match(line.rstrip("\n"))]
    if len(dividers) > 1:
        dividers = [i for i in dividers if lines[i].strip() == "======="]
    if len(dividers) == 2 and dividers[1] == len(lines) - 1:
        # The model echoes the divider just before `>>>>>>> REPLACE` (3 rewrites in 20).
        lines, dividers = lines[:-1], dividers[:1]
    if len(dividers) != 1:
        return None
    at = dividers[0]
    return "".join(lines[:at]), "".join(lines[at + 1:])


def _opens(line: str) -> bool:
    bare = line.rstrip("\n")
    return bool(_SEARCH.match(bare) or _NOTE.match(bare))


def parse(answer: str) -> tuple[str, list[Block]]:
    """Split an answer into (prose, blocks). Prose is what the chat shows; blocks become the
    proposal. A code fence wrapped around the blocks goes with them."""
    lines = (answer or "").splitlines(keepends=True)
    prose: list[str] = []
    blocks: list[Block] = []
    path = None
    body: list[str] | None = None
    whole = False
    after_block = False

    def is_file(line: str) -> str | None:
        match = _FILE.match(line.rstrip("\n"))
        value = _clean_path(match.group(1)) if match else ""
        return value if value.lower().endswith(".md") else None

    def names_file(i: int) -> str | None:
        """A FILE line, or a bare note path right above a block: on a long path the model
        drops the `FILE:` (1 run in 4, 2026-10-12)."""
        found = is_file(lines[i])
        if found or not _opens(next((later for later in lines[i + 1:] if later.strip()), "")):
            return found
        value = _clean_path(lines[i])
        return value if value.lower().endswith(".md") else None

    # A fence that OPENS a FILE/SEARCH group: the next non-blank line starts the group.
    opening = set()
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            nxt = next((later for later in lines[i + 1:] if later.strip()), "")
            if is_file(nxt) or _opens(nxt):
                opening.add(i)
    for i, line in enumerate(lines):
        bare = line.rstrip("\n")
        if body is not None:
            if whole and _NOTE_END.match(bare):
                blocks.append(Block(path, "", "".join(body), whole=True))
                body, after_block = None, True
            elif not whole and _REPLACE.match(bare):
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
        if _opens(line):
            body, whole = [], bool(_NOTE.match(bare))
            continue
        found = names_file(i)
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
        # A second divider where `>>>>>>> REPLACE` belongs closes the block (1 run in 8); one
        # divider alone at the end is a block cut off before its new text.
        closed = (not whole and body[-1:] and body[-1].strip() == "======="
                  and sum(line.strip() == "=======" for line in body) == 2)
        if closed:
            blocks.append(Block(path, *_split_body(body)))
        elif whole:
            # A NOTE the model closed with a divider ends with one (1 run in 8).
            blocks.append(Block(path, "", "".join(body), whole=True,
                                complete=bool(body) and body[-1].strip() == "======="))
        else:
            blocks.append(Block(path, "".join(body), "", complete=False))
    # Greedy decoding repeats itself; a repeated append must not append twice.
    unique, seen = [], set()
    for item in blocks:
        key = (item.path, item.search, item.replace, item.complete, item.problem, item.whole)
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


# A SEARCH that starts mid-line must be this specific to be trusted: `Total: 5` inside
# `Grand Total: 5` is a different line, `$\eta$ is the learnign rate.` inside a paragraph is not.
MIN_MIDLINE_CHARS = 20


def _find(text: str, search: str) -> tuple[list[tuple[int, int]], bool]:
    """Where `search` occurs: exactly at a line start; else line by line ignoring trailing
    whitespace; else, for a long fragment only, exactly once anywhere in a line."""
    spans, anywhere, start = [], [], text.find(search)
    while start != -1:
        anywhere.append((start, start + len(search)))
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
    if not spans and len(anywhere) == 1 and len(search.strip()) >= MIN_MIDLINE_CHARS:
        return anywhere, True
    return spans, False


def _close(text: str, search: str) -> list[tuple[int, int]]:
    """A long SEARCH copied with a few mistakes: its first and last lines match one place each,
    in that order, and nearly every line between them is the note's."""
    want = [line.rstrip() for line in search.strip("\n").splitlines()]
    if sum(bool(line.strip()) for line in want) < MIN_CLOSE_LINES:
        return []
    have = [line.rstrip() for line in text.splitlines()]
    firsts = [i for i, line in enumerate(have) if line == want[0]]
    if len(firsts) != 1:
        return []
    lasts = [j for j in range(firsts[0] + 1, len(have)) if have[j] == want[-1]]
    if len(lasts) != 1:
        return []
    begin, end = firsts[0], lasts[0] + 1
    if difflib.SequenceMatcher(None, have[begin:end], want, autojunk=False).ratio() < CLOSE_RATIO:
        return []
    starts = _line_starts(text)
    return [(starts[begin], starts[end])]


def _undivided(text: str, item: Block) -> Block:
    """A NOTE written like a SEARCH/REPLACE, as text, a divider and more text (3 rewrites in 8,
    2026-10-12): a copy of the note before the divider makes it one, else the text after the
    divider is the note, and a divider at the end closed it. A note with a 7-`=` line of its
    own (a setext heading) keeps its NOTE as written."""
    lines = item.replace.splitlines(keepends=True)
    cuts = [i for i, line in enumerate(lines) if line.strip() == "======="]
    if not item.whole or len(cuts) != 1 or any(line.strip() == "=======" for line in text.splitlines()):
        return item
    before, after = "".join(lines[:cuts[0]]), "".join(lines[cuts[0] + 1:])
    if not after.strip():
        return replace(item, replace=before)
    if before.strip() and (_find(text, before)[0] or _close(text, before)):
        return Block(item.path, before, after)
    return replace(item, replace=after)


def _append(text: str, added: str) -> str | None:
    """`added` at the end of the note: before a transcript that runs to the end, never in it."""
    tail = next((begin for begin, end in protected_ranges(text) if end >= len(text)), None)
    if tail == 0:
        return None
    head, rest = (text, "") if tail is None else (text[:tail], text[tail:])
    added = added.strip("\n") + "\n"
    return (head.rstrip("\n") + "\n\n" if head.strip() else "") + added + ("\n" + rest if rest else "")


def _kept(text: str, new: str = "") -> list[tuple[int, int]]:
    """What a NOTE block leaves as it is: a recording's parts, or the frontmatter of a note
    whose new text has none of its own."""
    keep = protected_ranges(text)
    _fm, skip = parse_frontmatter(text)
    if not keep and skip and not parse_frontmatter(new.lstrip("\n"))[1]:
        keep = [(0, _line_starts(text)[skip])]
    return keep


def _rewrite(text: str, new: str) -> str | None:
    """The note after a NOTE block: `new` takes the place of the owner's text, and what the
    model wrote for a recording's frontmatter, block or transcript is dropped, never applied.
    None when a recording holds the owner's text in more than one place."""
    keep = _kept(text, new)
    if protected_ranges(text):
        for begin, end in reversed(protected_ranges(new)):
            new = new[:begin] + new[end:]
        new = new.lstrip("\n")
        _fm, own = parse_frontmatter(new)
        new = new[_line_starts(new)[own]:] if own else new
    gaps, at = [], 0
    for begin, end in sorted(keep):
        gaps.append((at, begin))
        at = end
    gaps.append((at, len(text)))
    owned = [(begin, end) for begin, end in gaps if text[begin:end].strip()]
    if len(owned) > 1:
        return None
    if not owned:
        return _append(text, new)
    begin, end = owned[0]
    gap, body = text[begin:end], new.strip("\n")
    lead, trail = gap[:len(gap) - len(gap.lstrip("\n"))], gap[len(gap.rstrip("\n")):]
    return text[:begin] + lead + (body + trail if body else "") + text[end:]


def _tops(text: str) -> dict[str, int]:
    """Where a copy of the whole note starts, by its first line: the note's first line of text,
    after any frontmatter, and the first after each part of a recording."""
    starts = _line_starts(text)
    _fm, skip = parse_frontmatter(text)
    tops = {}
    for at in sorted({starts[skip]} | {end for _begin, end in protected_ranges(text)}):
        line = next((line.rstrip() for line in text[at:].splitlines() if line.strip()), "")
        tops.setdefault(line, at)
    tops.pop("", None)
    return tops


def _file_above(lines: list[str], at: int) -> tuple[int | None, str]:
    """The line naming a block's file, above the block opening at `at`, and its path."""
    above = next((i for i in range(at - 1, -1, -1) if lines[i].strip()), None)
    if above is None:
        return None, ""
    match = _FILE.match(lines[above])
    named = _clean_path(match.group(1) if match else lines[above])
    return (above, named) if named.lower().endswith(".md") else (None, "")


def rewrite_prefill(answer: str, notes: dict[str, str]) -> tuple[bool, str | None]:
    """Is the unfinished answer's first SEARCH copying one of `notes` from its top?
    (False, None) until enough of it has arrived to tell; (True, None) when it is not; and
    (True, prefill) when it is: the answer up to that block with a NOTE block for that note
    opened in its place, holding the note as it is up to where the copy began (its frontmatter,
    a recording's block), for the model to continue."""
    lines = answer.splitlines(keepends=True)
    at = next((i for i, line in enumerate(lines) if _SEARCH.match(line.rstrip("\n"))), None)
    if at is None:
        return False, None
    if any(_DIVIDER.match(line.rstrip("\n")) for line in lines[at + 1:]):
        return True, None
    copied = [line.rstrip() for line in lines[at + 1:] if line.endswith("\n")]
    if copied[:1] == ["---"]:                           # a copied frontmatter is not the text
        close = next((i for i in range(1, len(copied)) if copied[i] == "---"), None)
        copied = [] if close is None else copied[close + 1:]
    while copied and not copied[0].strip():
        copied.pop(0)
    if not copied:
        return False, None
    above, named = _file_above(lines, at)
    candidates = {named: notes[named]} if named in notes else notes
    path = next((path for path, text in candidates.items() if copied[0] in _tops(text)), None)
    if path is None:
        return True, None
    text = notes[path]
    own = text.count("\n") - sum(text[begin:end].count("\n") for begin, end in _kept(text))
    if len(copied) < max(REWRITE_AFTER_LINES, own // 4):
        return False, None
    head = "".join(lines[:at if above is None else above])
    return True, head + f"FILE: {path}\n<<<<<<< NOTE\n" + text[:_tops(text)[copied[0]]]


def recording_prefill(answer: str, notes: dict[str, str]) -> tuple[str, bool] | None:
    """An open NOTE block for a recording whose last line begins one of the recording's own
    parts: the answer with that part as it is on disk in place of the model's copy, and True
    when the block ends there instead (at a transcript, which runs to the end). The model wrote
    a whole meeting block and transcript into a NOTE, 32k characters, and ran out (2026-10-12)."""
    lines = answer.splitlines(keepends=True)
    last = lines[-1].rstrip("\n") if lines and lines[-1].endswith("\n") else ""
    fence, transcript = _MEETING_OPEN.match(last), last.strip() == TRANSCRIPT_MARKER
    if not (fence or transcript):
        return None
    at = next((i for i in range(len(lines) - 2, -1, -1)
               if _opens(lines[i]) or _NOTE_END.match(lines[i].rstrip("\n"))
               or _REPLACE.match(lines[i].rstrip("\n"))), None)
    if at is None or not _NOTE.match(lines[at].rstrip("\n")):
        return None
    text = notes.get(_file_above(lines, at)[1])
    ranges = protected_ranges(text) if text else []
    if not ranges:
        return None
    head = "".join(lines[:-1])
    if transcript:
        return head + ">>>>>>> NOTE\n", True
    block = next((text[begin:end] for begin, end in ranges if _MEETING_OPEN.match(text[begin:end])), None)
    return (head + block, False) if block else None


def _markers(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    return (sum(bool(_MEETING_OPEN.match(line)) for line in lines),
            sum(line.strip() == TRANSCRIPT_MARKER for line in lines))


def _adds_markers(search: str, replace: str) -> bool:
    """A marker the SEARCH already held may be kept (text inserted above `## Transcript`)."""
    return any(after > before for before, after in zip(_markers(search), _markers(replace)))


def apply_blocks(text: str, blocks: list[Block]) -> tuple[str, list[dict]]:
    """Apply blocks in order to one file's text, all or nothing: if any block fails, the text
    comes back unchanged and every result says why the file was refused."""
    original, before = text, _protected_slices(text)
    results = []
    for item in blocks:
        item = _undivided(text, item)
        if item.problem:
            results.append({"ok": False, "reason": item.problem})
            continue
        if not item.complete:
            results.append({"ok": False, "reason": "cut off before the change ended"})
            continue
        if item.whole:
            after = _rewrite(text, item.replace)
            if after is None:
                results.append({"ok": False, "reason": "this recording has your text in more than "
                                                       "one place; change one part at a time"})
            elif _adds_markers(text, after):
                results.append({"ok": False, "reason": "adds a recording's block or transcript heading"})
            elif len(_ELISION.findall(after)) > len(_ELISION.findall(text)):
                results.append({"ok": False, "reason": SKIPS})
            else:
                text = after
                results.append({"ok": True})
            continue
        if _adds_markers(item.search, item.replace):
            results.append({"ok": False, "reason": "adds a recording's block or transcript heading"})
            continue
        if not item.search.strip():
            after = _append(text, item.replace)
            if after is None:
                results.append({"ok": False, "reason": PROTECTED})
            else:
                text = after
                results.append({"ok": True})
            continue
        spans, exact = _find(text, item.search)
        spans = spans or _close(text, item.search)
        if not spans:
            results.append({"ok": False, "reason": "text not found in the note"})
            continue
        if len(spans) > 1:
            results.append({"ok": False, "reason": f"matches {len(spans)} places; include more lines"})
            continue
        start, end = spans[0]
        if len(_ELISION.findall(item.replace)) > len(_ELISION.findall(text[start:end])):
            results.append({"ok": False, "reason": SKIPS})
            continue
        replacement = item.replace
        if not exact:
            # The match ignored trailing whitespace; an unchanged line keeps its own, or the
            # two-space Markdown hard breaks in 56 notes would vanish on every nearby edit.
            kept = {}
            for line in text[start:end].splitlines(keepends=True):
                kept.setdefault(line.rstrip(), line.rstrip("\n"))
            out = []
            for line in replacement.splitlines(keepends=True):
                body = line.rstrip("\n")
                out.append(kept.get(body, body) + line[len(body):] if body == body.rstrip() else line)
            replacement = "".join(out)
        if not exact and replacement and not replacement.endswith("\n") and text[start:end].endswith("\n"):
            replacement += "\n"
        if end == len(text) and not text.endswith("\n") and replacement.endswith("\n"):
            replacement = replacement[:-1]          # the note had no final newline; keep it so
        text = text[:start] + replacement + text[end:]
        results.append({"ok": True})
    if all(result["ok"] for result in results) and _protected_slices(text) != before:
        # Which block did it is not worth guessing: the file is refused whole.
        results = [{"ok": False, "reason": PROTECTED} for _ in results]
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


def _on_floor(path: str) -> bool:
    return path.casefold().startswith(("journal/", "inbox/journal/"))


def _read(target: Path) -> str:
    with open(target, encoding="utf-8", newline="") as handle:   # as Obsidian reads it: no \r\n folding
        return handle.read()


def _with_origin(text: str) -> str:
    _fm, skip = parse_frontmatter(text)
    if skip:
        return set_frontmatter(text, {"origin": CREATED_ORIGIN})
    return f"---\norigin: {CREATED_ORIGIN}\n---\n\n{text.lstrip()}"


def _inside(search: str, selection: str) -> bool:
    lines = lambda text: "\n".join(line.rstrip() for line in text.strip("\n").splitlines())
    return bool(search.strip()) and lines(search) in lines(selection)


def propose(vault: Path, blocks: list[Block], *, in_view: set[str], creates_only: bool = False,
            within: tuple[str, str] | None = None) -> dict:
    """Group blocks by file and compute each file's new text. Returns
    {"files": [{path, kind, base_hash, after, blocks}], "dropped": [{path, reason}]}.

    `creates_only`: a new-note skill proposes new notes and nothing else. `within`: a selection
    skill changes only the selected text of the open note."""
    by_path: dict[str, list[Block]] = {}
    dropped = []
    # `Journal/` is the floor (CLAUDE.md): a journal's words never leave it through a proposal.
    # The memo sweep drops journals in `Inbox/Journal/` first; that is the floor too.
    journal = any(_on_floor(path) for path in in_view)
    for item in blocks:
        if not item.path:
            dropped.append({"path": "", "reason": "no FILE line before the change"})
            continue
        path = note_path(item.path)
        if path is None:
            dropped.append({"path": item.path, "reason": "not a note path in this vault"})
            continue
        if journal and not _on_floor(path):
            dropped.append({"path": path, "reason": "a Journal note's text stays in Journal/"})
            continue
        if creates_only and (Path(vault) / path).exists():
            dropped.append({"path": path, "reason": "this skill only creates new notes"})
            continue
        if within and (path != within[0] or not _inside(item.search, within[1])):
            dropped.append({"path": path, "reason": "outside the text you selected"})
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
                dropped.append({"path": path, "reason": "not part of this request"})
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
