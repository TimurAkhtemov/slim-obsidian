"""EDITS — the open note as the copilot's editing tools see it, and what those tools may change.

Code cuts the note into blocks (a heading, a paragraph, a list item, a table, a quote, fenced code
or `$$` math) and gives each a handle the model points at: [b12] is that block, s7 is the section
under the heading [b7], and b11-b15 is those blocks and everything between them. Handles never
change during a run; blocks an edit makes get new ones. The model calls the tools in `TOOLS`
with handles; code applies each call to a draft held here (`Note`), refuses what it must not do
with a reason the model reads back, and hands each change on as splices — {from, to, insert} in
UTF-16 units of the text just before it, which the plugin applies through the editor. Nothing
here writes a note.

What may be touched is decided here, not by the model. A recorded note's frontmatter, its
`slim-meeting` block and its transcript are `record.py`'s and the raw source: their blocks are
[locked] and every call that touches them is refused. A new note goes in a folder that already
exists, never into `Journal/`, and a Journal note's text never leaves it (`create_note`).
"""
from __future__ import annotations

import difflib
import hashlib
import os
import re
import textwrap
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .chunk import TRANSCRIPT_MARKER, parse_frontmatter, set_frontmatter
from .config import EXCLUDED_DIRS

_MEETING_OPEN = re.compile(r"^\s*(`{3,})slim-meeting")
# A lone `...` in new text is the model's "the rest as before", and it would delete the rest.
_ELISION = re.compile(r"^[ \t]*(?:\.\.\.|…)[ \t]*$", re.MULTILINE)
# Obsidian renders no Markdown or math inside these, in Live Preview, and an unclosed one leaves
# everything after it raw (live, 2026-10-13: a `<details>` whose `</details>` a later step dropped).
# A new `<details>` is written as the callout that folds the same way (`fold`); the rest are refused.
_HTML_BLOCK = re.compile(r"<(/?(?:details|summary|div))\b", re.IGNORECASE)
_DETAILS = re.compile(r"<(/?)details\b([^>]*)>", re.IGNORECASE)
_SUMMARY = re.compile(r"\s*<summary\b[^>]*>", re.IGNORECASE)
_SUMMARY_END = re.compile(r"</summary\s*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
# Code is shown, not rendered: a fence (closed by a run as long or longer, or the end of the text;
# a backtick fence's info string has no backtick) and a code span (closed by a backtick run of
# the same length, before a blank line).
_CODE = re.compile(r"^ {0,3}(?P<ticks>`{3,})[^`\n]*\n[\s\S]*?(?:^ {0,3}(?P=ticks)`*[ \t]*$|\Z)"
                   r"|^ {0,3}(?P<tildes>~{3,})[^\n]*\n[\s\S]*?(?:^ {0,3}(?P=tildes)~*[ \t]*$|\Z)"
                   r"|(?<!`)(?P<run>`+)(?!`)(?:[^\n]|\n(?![ \t]*\n))*?(?<!`)(?P=run)(?!`)", re.MULTILINE)
# Binaries, SLIM's own derived output and Obsidian's config are never a copilot edit target.
_NEVER_TOP = {"attachments", "_reflections"}
_EXCLUDED_CF = {name.casefold() for name in EXCLUDED_DIRS}
# Past this a note is edited in Obsidian, not from the sidebar: the whole note rides in the
# request and in every call of a run, and a new note's text rides in the thread.
MAX_EDIT_CHARS = 400_000
# A note the copilot creates says so, like every other autonomous write (CLAUDE.md).
CREATED_ORIGIN = "copilot"
# A heading the note already has, in new text, is a copy of its section when the new text also
# carries at least this share of that section's lines; below it, it is a heading that repeats
# on purpose ("### Example" under each section).
COPY_SHARE = 0.6

PROTECTED = "part of the recording; SLIM never edits it"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_path(raw: str) -> str:
    # `· the open note` is the pack's label after the path, which the model copies.
    path = (raw or "").split(" · ")[0].strip().strip("`*\"'=").strip()
    return path[2:] if path.startswith("./") else path


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


def same_recording(before: str, now: str) -> bool:
    """Are a recording's own parts (frontmatter, meeting block, transcript) the same in both?
    Restoring `before` over `now` would otherwise undo a resume or a card edit. Two plain notes
    have none, so they are always the same."""
    take = lambda text: [text[start:stop] for start, stop in protected_ranges(text)]
    return take(before) == take(now)


def _markers(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    return (sum(bool(_MEETING_OPEN.match(line)) for line in lines),
            sum(line.strip() == TRANSCRIPT_MARKER for line in lines))


# --- blocks -----------------------------------------------------------------------------------

_HEADING = re.compile(r"(#{1,6})\s")
_LIST_ITEM = re.compile(r"(?:[-*+]|\d+[.)])\s")


def _kind(line: str) -> str:
    text = line.strip()
    if _HEADING.match(text):
        return "heading"
    if text.startswith(("```", "~~~")):
        return "fence"
    if text.startswith("$$"):
        return "math"
    if text.startswith("|"):
        return "table"
    if _LIST_ITEM.match(text):
        return "list"
    if text.startswith(">"):
        return "quote"
    return "para"


def _blocks(lines: list[str]) -> list[tuple[str, int, int]]:
    """(kind, start, end) blocks covering `lines` in order. Blank lines belong to the block
    before them, so a heading renamed is the heading alone; leading ones to the first block."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        start = i
        while i < n and not lines[i].strip():
            i += 1
        if i == n:
            out.append(("blank", start, n))
            break
        kind, first = _kind(lines[i]), lines[i].strip()
        i += 1
        if kind == "fence" or (kind == "math" and (first == "$$" or not first.endswith("$$"))):
            close = first[:3] if kind == "fence" else "$$"
            while i < n and not lines[i].strip().startswith(close):
                i += 1
            i = min(i + 1, n)
        elif kind == "list":
            while i < n and lines[i].strip() and (lines[i][:1] in " \t" or _kind(lines[i]) == "list"):
                i += 1
        elif kind != "heading":
            while i < n and lines[i].strip() and _kind(lines[i]) == kind:
                i += 1
        while i < n and not lines[i].strip():
            i += 1
        out.append((kind, start, i))
    return out


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _spans(lines: list[str]) -> list[tuple[str, int, int]]:
    """_blocks, with each list cut at its top-level items, so fixing one item never means
    retyping the list: a list marker at the list's own indentation starts an item; nested and
    continuation lines stay with theirs, and the blank lines after the list with the last one."""
    out = []
    for kind, start, end in _blocks(lines):
        first = next((k for k in range(start, end) if lines[k].strip()), start)
        if kind != "list":
            out.append((kind, start, end))
            continue
        cuts = [k for k in range(first + 1, end) if lines[k].strip() and _indent(lines[k]) <= _indent(lines[first])
                and _LIST_ITEM.match(lines[k].strip())]
        bounds = [start, *cuts, end]
        out += [("list", a, b) for a, b in zip(bounds, bounds[1:])]
    return out


# --- handles ----------------------------------------------------------------------------------

class ToolError(ValueError):
    """A tool call code refuses; the message goes back to the model as the call's result, and
    `code`, a short word, goes to the trace (the message can quote the note)."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


_LABEL = re.compile(r"^\[(?:b\d+(?: s\d+)?|locked)\] ?", re.MULTILINE)
_HANDLE = re.compile(r"\b[bs]\d+\b")
# What joins two handles into a range: b11-b15, b11 – s14, [b11]-[b15], b11 through b15.
_RANGE_MARK = re.compile(r"\]?\s*(?:[-–—…]|\.{2,3}|(?i:to|through|thru|until))\s*\[?")
# A handle in words meant for the owner (a plan's steps, its summary), with the words around it
# that only make sense with a handle: "after block b6", "[b11-b15]", "the section s4".
_REFERENCE = re.compile(r"(?:\b(?P<prep>(?i:after|before|below|above|under|into|in|at|of|from|to|with|on))\s+)?"
                        r"(?:(?i:the)\s+)?(?P<word>(?i:blocks?|sections?|range|heading|paragraph)\s+)?"
                        r"(?P<open>\[)?\b(?P<kind>[bs])(?P<n>\d+)(?P<range>\s*[-–]\s*[bs]?\d+)?\b\]?")
_EMBED = re.compile(r"!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_IMAGE = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".avif")
_MARKUP = re.compile(r"^(?:\s*>)*\s*(?:\[![^\]]*\][-+]?\s*)?(?:#{1,6}\s+|(?:[-*+]|\d+[.)])\s+(?:\[.\]\s+)?)?")
QUOTE_WORDS = 8
QUOTE_CHARS = 60


def quote(text: str) -> str:
    """Some text as the owner would point at it: its first line, without Markdown's markers,
    cut at a word to QUOTE_WORDS words and QUOTE_CHARS characters. An image is "the image"."""
    line = next((line for line in text.splitlines() if line.strip()), "")
    embed = _EMBED.fullmatch(line.strip())
    if embed:
        return "the image" if embed.group(1).strip().lower().endswith(_IMAGE) else "the embed"
    words = _MARKUP.sub("", line).replace("**", "").replace("__", "").split()
    kept = words[:QUOTE_WORDS]
    while len(kept) > 1 and len(" ".join(kept)) > QUOTE_CHARS:
        kept.pop()
    cut = " ".join(kept)
    if len(cut) > QUOTE_CHARS:
        return f"“{cut[:QUOTE_CHARS]}…”"
    return f"“{cut.rstrip(',;:') + '…' if len(kept) < len(words) else cut}”"


@dataclass
class Block:
    handle: str
    kind: str
    text: str
    locked: bool = False
    owner: str | None = None          # the change that last made or replaced this block
    fresh: bool = False               # all of it written by this run: none of it the owner's

    @property
    def level(self) -> int:
        if self.kind != "heading":
            return 0
        return len(_HEADING.match(self.text.lstrip()).group(1))


def strip_labels(markdown: str) -> str:
    """The model sometimes copies the [b7] labels into its text; they are never content."""
    return _LABEL.sub("", markdown.replace("\r\n", "\n")).strip("\n")


def handles_in(target) -> list[str]:
    """The handles in a target as the model wrote it. Only its first line counts, so a label
    with the block's text pasted after it still points at the block."""
    return _HANDLE.findall(str(target).split("\n", 1)[0])


def handle_name(target) -> str:
    """A target as code names it back: b9, s7, b11-b15, or its first line if it holds no handle."""
    found = handles_in(target)
    if not found:
        return str(target).strip().split("\n", 1)[0][:60]
    return found[0] if len(found) == 1 else f"{found[0]}-{found[-1]}"


def headings(text: str) -> list[str]:
    return [m.group(0).strip() for m in re.finditer(r"^#{1,6} .+$", text, re.M)]


# A trailing note the model writes after a heading, in its outline, the note, or both.
ANNOTATION = re.compile(r"\s*\((?:new|added|moved|renamed|updated|expanded|merged|revised)(?: section)?\)\s*$", re.I)


def heading_key(heading: str) -> tuple[int, str]:
    """A heading as compared: its level (the count of #s) and its words, casefolded, whitespace
    collapsed, without a trailing note like "(New Section)"."""
    m = re.match(r"\s*(#*)\s*(.*)", heading)
    return len(m.group(1)), " ".join(ANNOTATION.sub("", m.group(2)).casefold().split())


def heading_level(text: str) -> int:
    """The highest rank (fewest #s) of any heading in markdown, or 0 if it has none."""
    lines = text.splitlines(keepends=True)
    levels = [Block("", kind, "".join(lines[start:end])).level for kind, start, end in _spans(lines)]
    return min((level for level in levels if level), default=0)


def _end(blocks: list[Block], i: int) -> int:
    """Where the section of the heading at i ends: at the next heading of its level or higher."""
    j = i + 1
    while j < len(blocks) and not 0 < blocks[j].level <= blocks[i].level:
        j += 1
    return j


def cut_refused(cut: tuple[str, int, str]) -> ToolError:
    s, _, first = cut
    return ToolError(f"that would cut {s}: its text from {first} on would end up under your new heading. "
                     f"Put it after {s} (after that whole section) or before {s}.", "cut")


def placed(cut: tuple[str, int, str]) -> str:
    return f"; placed after {cut[0]} (the end of that section) so it doesn't cut it"


def u16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def splices(before: str, after: str) -> list[dict]:
    """Line-level edits turning `before` into `after`: {from, to, insert} in UTF-16 units of
    `before` (JavaScript string indices, as the plugin's editor counts), applied last-first. Only
    the lines between what both share at their start and end are diffed: difflib is quadratic in
    repeated lines, and over a whole long note (thousands of blank ones) a change took a second."""
    a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
    top = 0
    while top < min(len(a), len(b)) and a[top] == b[top]:
        top += 1
    end = 0
    while end < min(len(a), len(b)) - top and a[-1 - end] == b[-1 - end]:
        end += 1
    starts = [0]
    for line in a:
        starts.append(starts[-1] + len(line))
    out = []
    same = difflib.SequenceMatcher(None, a[top:len(a) - end], b[top:len(b) - end], autojunk=False)
    for tag, i1, i2, j1, j2 in same.get_opcodes():
        if tag != "equal":
            out.append({"from": u16(before[:starts[top + i1]]), "to": u16(before[:starts[top + i2]]),
                        "insert": "".join(b[top + j1:top + j2])})
    return out


def apply_splices(text: str, edits_: list[dict]) -> str:
    units = text.encode("utf-16-le")
    for item in sorted(edits_, key=lambda s: s["from"], reverse=True):
        units = units[:2 * item["from"]] + item["insert"].encode("utf-16-le") + units[2 * item["to"]:]
    return units.decode("utf-16-le")


class Note:
    """The draft a run edits: the note as blocks with handles."""

    def __init__(self, text: str):
        self._next = 1
        self.blocks = self._fresh(text, owner=None)
        self.made: dict[str, int] = {}      # each change of the run: the step that made it
        self.roots: dict[str, str] = {}     # each change: the change it is part of, or itself
        ranges = protected_ranges(text)
        pos = 0
        for block in self.blocks:
            end = pos + len(block.text)
            block.locked = any(start < end and pos < stop for start, stop in ranges)
            pos = end
        # A transcript with no meeting block runs to the end of the file: nothing may follow it.
        self.tail_locked = bool(ranges) and ranges[-1][1] == len(text) and not any(
            _MEETING_OPEN.match(line) for line in text.splitlines())

    def _fresh(self, text: str, owner: str | None) -> list[Block]:
        lines = text.splitlines(keepends=True)
        out = []
        for kind, start, end in _spans(lines):
            out.append(Block(f"b{self._next}", kind, "".join(lines[start:end]), owner=owner))
            self._next += 1
        return out

    def text(self) -> str:
        return "".join(block.text for block in self.blocks)

    def _range(self) -> str:
        live = [b for b in self.blocks if not b.locked and b.kind != "blank"]
        return f"{live[0].handle}–{live[-1].handle}, and s<n> for a heading's section" if live else "none"

    def _plain(self, handle: str) -> str:
        """s<n> where b<n> is not a heading means the block itself."""
        if handle[0] == "s":
            block = next((b for b in self.blocks if b.handle == "b" + handle[1:]), None)
            if block is not None and not block.level:
                return "b" + handle[1:]
        return handle

    def _handles(self, target) -> list[str]:
        """The target's handles, with s<n> on a plain block read as b<n>, and a range's last handle
        past the end of the note (a number no block was ever given) read as the note's last block.
        Handles listed rather than joined as a range ("b3, b7") are one target only when nothing
        lies between them."""
        line = str(target).split("\n", 1)[0]
        marks = list(_HANDLE.finditer(line))
        found = [self._plain(m.group()) for m in marks]
        joined = [bool(_RANGE_MARK.fullmatch(line[a.end():b.start()])) for a, b in zip(marks, marks[1:])]
        live = [b for b in self.blocks if b.kind != "blank"]
        if joined and joined[-1] and live and int(found[-1][1:]) >= self._next:
            found[-1] = live[-1].handle
        if not all(joined):
            self._listed(found, joined)
        return found

    def _listed(self, found: list[str], joined: list[bool]) -> None:
        """A list must cover the range it would make: "b4, b5" is b4-b5, while "b3, b7" would take in
        everything between and is refused. Handles joined as a range cover all between them."""
        spans = [self._span(h) for h in found]
        covered, start = set(), spans[0][0]
        for n, (i, j) in enumerate(spans):
            if n and not joined[n - 1]:
                start = i
            covered.update(range(start, j))
        if any(k not in covered for k in range(spans[0][0], spans[-1][1]) if self.blocks[k].kind != "blank"):
            raise ToolError(f"{', '.join(found)} are not next to each other: make one call for each (the range "
                            f"{found[0]}-{found[-1]} would take in everything between them)", "list")

    def name(self, target) -> str:
        """A target as code names it back: b9, s7, b11-b15, or its first line if it holds no handle."""
        found = self._handles(target)
        if not found:
            return handle_name(target)
        return found[0] if len(found) == 1 else f"{found[0]}-{found[-1]}"

    def span(self, target) -> tuple[int, int]:
        """A block, a section, or a range from the first handle's start to the last one's end."""
        found = self._handles(target)
        if not found:
            raise ToolError(f"'{handle_name(target)}' is not a handle; use one shown in brackets "
                            f"({self._range()})", "handle")
        if len(found) == 1:
            return self._span(found[0])
        first, last = found[0], found[-1]
        if len(found) == 2 and first[0] != last[0] and first[1:] == last[1:]:
            n = first[1:]
            raise ToolError(f"b{n} is the heading line alone and s{n} is the heading with everything under it: "
                            "use one of them", "ambiguous")
        i, _ = self._span(first)
        k, l = self._span(last)
        if k < i:
            raise ToolError(f"{first}-{last}: a range goes from the earlier handle to the later one", "range")
        return i, l

    def _span(self, handle: str) -> tuple[int, int]:
        i = next((k for k, b in enumerate(self.blocks) if b.handle == "b" + handle[1:]), None)
        if i is None:
            raise ToolError(f"there is no {handle}; use a handle shown in brackets ({self._range()})", "handle")
        if handle[0] == "b" or not self.blocks[i].level:
            return i, i + 1
        return i, _end(self.blocks, i)

    def plain(self, words: str, bare: bool = True) -> str:
        """Words for the owner with every handle turned into the text it points at ("after block
        b6" → "after the image"). A handle the note does not have goes, with its words, when it is
        plainly one ("block b99", "[b9]", "b2-b4"); a bare "s3" or "b12" it does not have is a word.
        With `bare` off (an answer, about anything) only a plain handle is one, never a bare word."""
        def put(match: re.Match) -> str:
            block = next((b for b in self.blocks if b.handle == "b" + match["n"]), None)
            if not bare and not (match["word"] or match["open"] or match["range"]):
                return match[0]
            if block is None:
                return "" if match["word"] or match["open"] or match["range"] else match[0]
            return f"{match['prep'] + ' ' if match['prep'] else ''}{quote(block.text)}"
        said = _REFERENCE.sub(put, words)
        return re.sub(r" +([,.;:)])", r"\1", re.sub(r" {2,}", " ", said)).strip()

    def target(self, handle, only: set[str] | None = None) -> tuple[int, int]:
        """A span a tool may change: nothing in it is part of a recording or outside `only`."""
        i, j = self.span(handle)
        self._check(handle, i, j, only)
        return i, j

    def pick(self, target, only: set[str] | None = None) -> tuple[int, int, str, str]:
        """A tool's target: its span, trimmed, checked, and named back, with a note on any trim."""
        i, j = self.span(target)
        found = self._handles(target)
        name, dropped = self.name(target), []
        while len(found) > 1 and j - i > 1:     # a range never ends on the next section's heading
            last = self.blocks[j - 1]
            if not last.level or _end(self.blocks, j - 1) <= j:
                break
            dropped.append(last.handle)
            j -= 1
        said = ""
        if dropped:
            name = self.blocks[i].handle if j - i == 1 else f"{found[0]}-{self.blocks[j - 1].handle}"
            one = len(dropped) == 1
            said = (f"{', '.join(reversed(dropped))} {'is the heading' if one else 'are the headings'} of the next "
                    f"section; left {'it' if one else 'them'} out")
        self._check(name, i, j, only)
        return i, j, name, said

    def _check(self, handle: str, i: int, j: int, only: set[str] | None) -> None:
        if any(b.locked for b in self.blocks[i:j]):
            raise ToolError(f"{handle} is {PROTECTED}", "locked")
        if only is not None and any(b.handle not in only for b in self.blocks[i:j] if b.kind != "blank"):
            allowed = ", ".join(sorted(only, key=lambda h: int(h[1:])))
            raise ToolError(f"{handle} is outside the selected text; change only {allowed}", "selection")

    def offset(self, index: int) -> int:
        return sum(len(b.text) for b in self.blocks[:index])

    def check_insert(self, pos: int) -> None:
        """New text may not land inside a recording's protected parts, above its frontmatter,
        or after a transcript that runs to the end of the file."""
        text, at = self.text(), self.offset(pos)
        for start, stop in protected_ranges(text):
            if start < at < stop or (at == 0 and start == 0):
                raise ToolError(f"that place is inside the recording, which is {PROTECTED}", "locked")
        if self.tail_locked and pos == len(self.blocks):
            raise ToolError("nothing can go after the transcript; put it before the transcript", "locked")

    def below_properties(self, pos: int) -> tuple[int, str]:
        """New text never goes above a note's properties (its frontmatter) or inside them: Obsidian
        reads them only at the very top. A place there is right after them instead, with what the
        result says of it. (A recording's are its own: `check_insert` refuses that place first.)"""
        text = self.text()
        _fm, skip = parse_frontmatter(text)
        if not skip:
            return pos, ""
        end, top, at = _line_starts(text)[skip], 0, 0
        while top < len(self.blocks) and at < end:
            at += len(self.blocks[top].text)
            top += 1
        if pos >= top:
            return pos, ""
        return top, f"; placed after {self.blocks[top - 1].handle} (the end of the note's properties, which stay at its top)"

    def cut(self, pos: int, level: int, blocks: list[Block] | None = None) -> tuple[str, int, str] | None:
        """Text holding a heading of `level` may not land inside a section of that level or lower
        rank: the rest of the section would end up under the new heading. Every enclosing
        section is checked, and the outermost one cut is returned (its handle, its end, and the
        first block that would end up under the new heading), so the place it offers is safe."""
        blocks = self.blocks if blocks is None else blocks
        if not level:
            return None
        for h in range(pos):
            if not 0 < blocks[h].level or level > blocks[h].level:
                continue
            end = _end(blocks, h)
            rest = [b for b in blocks[pos:end] if b.kind != "blank"]
            if rest:
                return "s" + blocks[h].handle[1:], end, rest[0].handle
        return None

    def sections(self) -> str:
        """The note's map: a line per heading a step may change, indented by level, with its
        section's handle and first and last block."""
        heads = [(i, b) for i, b in enumerate(self.blocks) if b.level and not b.locked]
        top = min((b.level for _, b in heads), default=1)
        lines = []
        for i, b in heads:
            last = next(c for c in reversed(self.blocks[i:_end(self.blocks, i)]) if c.kind != "blank")
            span = b.handle if last is b else f"{b.handle}–{last.handle}"
            lines.append(f"{'  ' * (b.level - top)}s{b.handle[1:]}  {span}  {b.text.strip().splitlines()[0]}")
        return "\n".join(lines)

    def then(self, index: int) -> str:
        """What follows a change, so the model sees any leftovers below it."""
        following = next((b for b in self.blocks[index:] if b.kind != "blank"), None)
        if following is None:
            return "then: (end of the note)"
        return f"then: {self._label(following)}{following.text.split(chr(10), 1)[0][:100]}"

    def owners(self, i: int, j: int) -> set[str]:
        return {b.owner for b in self.blocks[i:j] if b.owner}

    def _separate(self, pos: int) -> None:
        """The block before `pos` ends with a blank line, so whatever comes next starts a block."""
        if 0 < pos <= len(self.blocks):
            prev = self.blocks[pos - 1]
            if not prev.text.endswith("\n"):
                prev.text += "\n"
            if not prev.text.endswith("\n\n"):
                prev.text += "\n"

    def _spacing(self, pos: int) -> str:
        """How the list at `pos` ends an item before the next: with a blank line when it is loose,
        a bare newline when it is tight, as the two items nearest `pos` show (one item is tight)."""
        for k in (pos - 1, pos, pos - 2):
            if 0 <= k < len(self.blocks) - 1 and self.blocks[k].kind == self.blocks[k + 1].kind == "list":
                return "\n\n" if self.blocks[k].text.endswith("\n\n") else "\n"
        return "\n"

    def _repeats(self, i: int, j: int, texts: list[str]) -> dict[int, Block]:
        """Which of blocks i..j each new block's text repeats word for word, by the new one's index."""
        same = difflib.SequenceMatcher(None, [b.text.rstrip("\n") for b in self.blocks[i:j]],
                                       [text.rstrip("\n") for text in texts], autojunk=False)
        return {n + k: self.blocks[i + o + k] for o, n, size in same.get_matching_blocks() for k in range(size)}

    def _new_text(self, i: int, j: int, markdown: str) -> str:
        old = "".join(b.text for b in self.blocks[i:j])
        return markdown + old[len(old.rstrip("\n")):] if markdown else ""

    def changed(self, i: int, j: int, markdown: str) -> list[Block]:
        """The blocks among i..j that `markdown` in their place does not repeat word for word."""
        lines = self._new_text(i, j, markdown).splitlines(keepends=True)
        repeated = {id(b) for b in self._repeats(i, j, ["".join(lines[a:z]) for _kind, a, z in _spans(lines)]).values()}
        return [b for b in self.blocks[i:j] if id(b) not in repeated and b.kind != "blank"]

    def replace(self, i: int, j: int, markdown: str, owner: str, keep: str | None = None) -> list[Block]:
        gone = self.changed(i, j, markdown)
        new = self._fresh(self._new_text(i, j, markdown), owner) if markdown else []
        kept = self._repeats(i, j, [b.text for b in new])
        for n, b in enumerate(new):         # a block retyped as it was is still whoever's it was
            b.owner, b.fresh = (kept[n].owner, kept[n].fresh) if n in kept else (owner, all(g.fresh for g in gone))
        if keep and len(new) == 1:
            new[0].handle = keep
        self.blocks[i:j] = new
        if not new and 0 < i < len(self.blocks) and not self.blocks[i - 1].kind == self.blocks[i].kind == "list":
            self._separate(i)           # (an item gone from the middle of a list leaves it tight)
        return new

    def insert(self, pos: int, markdown: str, owner: str) -> list[Block]:
        prev = self.blocks[pos - 1] if pos > 0 else None
        following = self.blocks[pos] if pos < len(self.blocks) else None
        new = self._fresh(markdown + "\n", owner)
        for b in new:
            b.fresh = True
        ending = "\n\n" if following else "\n"
        if prev and prev.kind == new[0].kind == "list":    # new items join prev's list, tight
            loose = prev.text.endswith("\n\n")
            prev.text = prev.text.rstrip("\n") + "\n"
            if not loose and following and following.kind == new[-1].kind == "list":
                ending = "\n"
        else:
            self._separate(pos)
        new[-1].text = new[-1].text.rstrip("\n") + ending
        self.blocks[pos:pos] = new
        return new

    def move(self, i: int, j: int, pos: int) -> list[Block]:
        moved = self.blocks[i:j]
        del self.blocks[i:j]
        if 0 < i < len(self.blocks) and not self.blocks[i - 1].kind == self.blocks[i].kind == "list":
            self._separate(i)           # (an item gone from the middle of a list leaves it tight)
        if pos > i:
            pos -= j - i
        prev = self.blocks[pos - 1] if pos > 0 else None
        following = self.blocks[pos] if pos < len(self.blocks) else None
        first, last = moved[0], moved[-1]
        # Items moved among a list's items take its spacing: a list reordered reads as it did.
        after_item = prev is not None and prev.kind == first.kind == "list"
        before_item = following is not None and following.kind == last.kind == "list"
        gap = self._spacing(pos)
        ending = gap if before_item else None
        if after_item and not before_item:  # after the list's last item: what ended the list ends it now
            if last.kind == "list":
                ending = prev.text[len(prev.text.rstrip("\n")):]
            prev.text = prev.text.rstrip("\n") + gap
        if ending is None and following and not last.text.endswith("\n\n"):
            ending = "\n\n"
        if ending is not None:
            last.text = last.text.rstrip("\n") + ending
        if not after_item and not (before_item and first.kind == "list"):
            self._separate(pos)
        self.blocks[pos:pos] = moved
        return moved

    def _label(self, block: Block) -> str:
        if block.locked:
            return "[locked] "
        return "" if block.kind == "blank" else f"[{block.handle}] "

    def render(self, blocks: list[Block] | None = None) -> str:
        return "".join(self._label(b) + b.text for b in (self.blocks if blocks is None else blocks))


# --- the tools --------------------------------------------------------------------------------

@dataclass
class Change:
    id: str
    step: int
    op: str
    label: str
    splices: list[dict]
    needs: list[str] = field(default_factory=list)
    # The earlier change this one belongs to, when it only reworks text this run wrote: to the
    # owner it is that change, never a removal of their text.
    part_of: str | None = None


def _text(args: dict, name: str, required: bool = True) -> str | None:
    """An argument's text. markdown and new may be empty on purpose (a deletion); missing, they
    are refused like any other, naming what the call gave instead (a misnamed parameter)."""
    value = args.get(name)
    if value is None or (isinstance(value, str) and not value.strip() and name not in ("markdown", "new")):
        if required:
            given = ", ".join(key for key, v in args.items() if v is not None and key != name)
            raise ToolError(f"{name} is missing; this call gave {given or 'nothing'}", "args")
        return None
    if not isinstance(value, str):
        raise ToolError(f"{name} must be text", "args")
    return value


def _handle(note: Note, args: dict, name: str, required: bool = True) -> str | None:
    # A place written null is no place: the reader keeps a text parameter's "null" as text.
    if str(args.get(name)).strip().lower() == "null":
        args = {**args, name: None}
    value = _text(args, name, required)
    return note.name(value) if value else None


def _where(note: Note, args: dict) -> tuple[str, bool]:
    after, before = _handle(note, args, "after", False), _handle(note, args, "before", False)
    if bool(after) == bool(before):
        raise ToolError("give exactly one of after or before", "args")
    return (after, True) if after else (before, False)


def _written(markdown: str, old: str = "", shown: str | None = None) -> None:
    """New text may not add a recording's block or transcript heading, which would lock the
    owner's text below it, nor a lone `...` for "the rest as before", which would delete the rest:
    both read in the text as the model wrote it, before `fold` puts any of it inside a callout. Nor
    an HTML block, read in the text as it goes in (`shown`)."""
    if any(new > was for was, new in zip(_markers(old), _markers(markdown))):
        raise ToolError("that adds a recording's block or transcript heading; SLIM never writes one", "marker")
    if len(_ELISION.findall(markdown)) > len(_ELISION.findall(old)):
        raise ToolError("the new text skips lines with `...`; write out what stays, or change only what "
                        "changes", "elision")
    if _html_lines(markdown if shown is None else shown) - _html_lines(old):    # each line's tags, kept from the old text or none
        raise ToolError("Obsidian renders no Markdown or math inside HTML tags; write Markdown. A heading already "
                        "folds in Obsidian; for a section that starts folded, write a callout: "
                        "> [!note]- Title, with each line under it starting with > .", "html")


def _html_lines(text: str) -> Counter:
    """The HTML block tags of each line outside code, in order: a line keeps its tags only where
    the old text had a line with the same ones, never a tag counted elsewhere."""
    lines = (_line_tags(line) for line in _CODE.sub("", text).splitlines())
    return Counter(tags for tags in lines if tags)


def _line_tags(line: str) -> tuple[str, ...]:
    return tuple(tag.lower() for tag in _HTML_BLOCK.findall(line))


def _line_end(text: str, pos: int) -> int:
    end = text.find("\n", pos)
    return len(text) if end < 0 else end


def fold(markdown: str, old: str = "") -> tuple[str, list[str]]:
    """`markdown` with each <details> the old text did not have written as the callout Obsidian
    folds the same way, with Markdown rendered inside: `> [!note]- Title`, `+` for <details open>,
    its <summary> the title. One never closed runs to the end of the text; one in code is code; a
    <details> of the old text (a line with the same tags, as `_html_lines` counts) stays as it was.
    Returns the text and the titles of the callouts it made, outer first."""
    code = [m.span() for m in _CODE.finditer(markdown)]
    tags = [m for m in _DETAILS.finditer(markdown) if not any(a <= m.start() < b for a, b in code)]
    had = _html_lines(old)
    out, titles, at, k = [], [], 0, 0
    while k < len(tags):
        tag = tags[k]
        if tag.group(1) or tag.start() < at:        # a </details> with none open, or text already written
            k += 1
            continue
        depth, close = 0, None
        for n in range(k, len(tags)):
            depth += -1 if tags[n].group(1) else 1
            if depth == 0:
                close = n
                break
        start = markdown.rfind("\n", 0, tag.start()) + 1
        last = close + 1 if close is not None else len(tags)
        key = _line_tags(_CODE.sub("", markdown[start:_line_end(markdown, tag.start())]))
        if had[key]:
            had[key] -= 1
            k = last
            continue
        inner_end, close_end = (tags[close].start(), tags[close].end()) if close is not None else (len(markdown),) * 2
        title, body = "", markdown[tag.end():inner_end]
        summary = _SUMMARY.match(body)
        if summary:
            stop = _SUMMARY_END.search(body, summary.end())
            cut = stop.start() if stop else _line_end(body, summary.end())
            title, body = body[summary.end():cut], body[stop.end() if stop else cut:]
        name = " ".join(_TAG.sub("", title).split()).lstrip("#").strip() or "Details"
        body, inner = fold(body)
        titles += [name, *inner]
        lines = textwrap.dedent(body).strip("\n").split("\n") if body.strip() else []
        sign = "+" if re.search(r"\bopen\b", tag.group(2), re.IGNORECASE) else "-"
        lead = markdown[start:tag.start()]
        indent, before = lead[:len(lead) - len(lead.lstrip(" \t"))], lead.strip()
        before = "" if re.fullmatch(r"#{1,6}", before) else before     # "### <details>": the callout is the heading
        end = _line_end(markdown, close_end)
        after, more = fold(markdown[close_end:end].strip())     # another <details> may follow on the line
        titles += more
        piece = "\n".join(indent + line for line in [f"> [!note]{sign} {name}",
                                                     *(f"> {line}" if line.strip() else ">" for line in lines)])
        done = "".join(out) + markdown[at:start]
        # Set apart, so a line around it is never read as part of the callout.
        if before:
            piece = f"{indent}{before}\n\n{piece}"
        elif done.strip() and not re.search(r"\n[ \t]*\n\Z", done):
            piece = "\n" + piece
        if after:
            piece += "\n\n" + "\n".join(indent + line for line in after.split("\n"))
        elif markdown[end + 1:_line_end(markdown, end + 1)].strip():
            piece += "\n"
        out.append(markdown[at:start] + piece)
        at, k = end, last
    out.append(markdown[at:])
    return "".join(out), titles


def _callouts(titles: list[str]) -> str:
    """What a tool's result says of the <details> it wrote as callouts."""
    if not titles:
        return ""
    return (f" (written as {'a callout' if len(titles) == 1 else 'callouts'}: Obsidian folds a callout, and shows "
            "no Markdown inside HTML)")


def _own(note: Note, changed: list[Block], target: str, ctx: dict) -> None:
    """A later step may not rewrite or delete what an earlier step of the run wrote: steps that
    reworked their own text left a short section where a long one had landed (live, 2026-10-13).
    Only the blocks a call really changes count, and only those the run wrote whole: a block of the
    owner's that a step edited stays open to the next (two typos in one paragraph). One such block
    among others is enough, and is named, so the model can leave it as it is."""
    earlier = [(b, note.made[b.owner]) for b in changed
               if b.fresh and note.made.get(b.owner) is not None and note.made[b.owner] < ctx["step"]]
    if earlier:
        block, step = earlier[0]
        name = target if len(changed) == 1 else block.handle
        raise ToolError(f"{name} is what you wrote in step {step}; it stays as written. Make only what "
                        "this step still needs; if the note already has it, reply without a tool call.", "own")


def _lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _copied(note: Note, markdown: str, mine: range, place: str) -> None:
    """New text may not copy a section the note has outside `mine`: the model copies a section
    it meant to move, and the note ends up with it twice. A copy is a heading the note has
    (by `heading_key`) brought with most of its section's lines; the heading alone, over text of
    its own, is a repeat on purpose."""
    written = set(_lines(markdown))
    for heading in headings(markdown):
        for k, b in enumerate(note.blocks):
            if k in mine or not b.level or not any(heading_key(h) == heading_key(heading) for h in headings(b.text)):
                continue
            body = _lines("".join(c.text for c in note.blocks[k + 1:_end(note.blocks, k)]))
            if (1 + sum(line in written for line in body)) / (1 + len(body)) < COPY_SHARE:
                continue
            s = "s" + b.handle[1:]
            raise ToolError(f"{heading} is already in the note as {s}; to put that section here, use move {s} "
                            f"{place}, and replace it afterwards if it needs new text.", "copy")


def _renames(note: Note, i: int, j: int, markdown: str) -> bool:
    """A whole section replaced by a heading alone: the model meant a rename, not to drop the text."""
    one = _spans(markdown.splitlines(keepends=True))
    return (j - i > 1 and note.blocks[i].level > 0 and (i, j) == note.span("s" + note.blocks[i].handle[1:])
            and len(one) == 1 and one[0][0] == "heading")


def lands(note: Note, tool: str, args: dict, markdown: str,
          only: set[str] | None = None) -> tuple[int, int, str, str, bool]:
    """Where a replace or an insert puts `markdown` (as the model wrote it, labels stripped): for
    the tool, and for its draft while the text is still coming (`copilot._Drafts`). The blocks
    i..j it changes, i == j where an insert goes in; its target or anchor as named back; what the
    result adds about the place; and whether a replace renames: a whole section replaced by one
    heading line changes that heading alone, block i, and the section's text stays. Raises
    ToolError for a place the text may not go."""
    if tool == "insert":
        anchor, after = _where(note, args)
        i, j = note.span(anchor)
        note.check_insert(j if after else i)
        pos, said = note.below_properties(j if after else i)
        cut = note.cut(pos, heading_level(fold(markdown)[0]))
        if cut and not after:
            raise cut_refused(cut)
        if cut:                         # after a heading or inside a section: after that whole section instead
            pos, said = cut[1], placed(cut)
            note.check_insert(pos)
        return pos, pos, anchor, said, False
    i, j, target, trimmed = note.pick(_handle(note, args, "target"), only)
    old = "".join(b.text for b in note.blocks[i:j])
    return i, j, target, trimmed, _renames(note, i, j, fold(markdown, old)[0])


def _replace(note: Note, args: dict, ctx: dict):
    markdown = strip_labels(_text(args, "markdown"))
    i, j, target, trimmed, renames = lands(note, "replace", args, markdown, ctx["only"])
    old = "".join(b.text for b in note.blocks[i:j])
    raw, (markdown, folded) = markdown, fold(markdown, old)
    ctx["targets"] = note.changed(i, i + 1 if renames else j, markdown)
    _own(note, ctx["targets"], target, ctx)
    _written(raw, old, markdown)
    renamed = f"Renamed {quote(note.blocks[i].text)} to {quote(markdown)}"
    if renames:
        _copied(note, markdown, range(i, i + 1), "to where you want it")
        heading = note.blocks[i].handle
        needs = note.owners(i, i + 1)
        new = note.replace(i, i + 1, markdown, ctx["id"], keep=heading)
        return needs, renamed, (f"renamed the heading; the section's text is kept:\n"
                                f"{note.render(new).rstrip()}\n{note.then(i + 1)}")
    _copied(note, markdown, range(i, j), "to where you want it")
    one = _spans(markdown.splitlines(keepends=True))
    label = (renamed if j - i == 1 and note.blocks[i].level and len(one) == 1 and one[0][0] == "heading"
             else f"Rewrote {quote(old)}" if markdown else f"Removed {quote(old)}")
    needs = note.owners(i, j)
    new = note.replace(i, j, markdown, ctx["id"], keep=note.blocks[i].handle if j - i == 1 else None)
    then, done = note.then(i + len(new)), f"{target} ({trimmed})" if trimmed else target
    if not markdown:
        return needs, label, f"deleted {done}\n{then}"
    return needs, label, f"replaced {done}{_callouts(folded)}; it now reads:\n{note.render(new).rstrip()}\n{then}"


def _edit_text(note: Note, args: dict, ctx: dict):
    """A wording fix inside a block or range: the model quotes what changes, never retypes the rest."""
    i, j, target, _ = note.pick(_handle(note, args, "target"), ctx["only"])
    old = args.get("old")
    if not isinstance(old, str) or not old:
        raise ToolError("old is empty; quote the exact text to change", "args")
    new = _text(args, "new")
    text = "".join(b.text for b in note.blocks[i:j])
    found = text.count(old)
    if not found:
        raise ToolError(f"'{old}' is not in {target}; it reads: {text.rstrip()[:300]}", "quote")
    if found > 1:
        raise ToolError(f"'{old}' appears {found} times in {target}; quote a little more around it", "quote")
    raw = text.replace(old, new, 1)
    changed, folded = fold(raw, text)
    ctx["targets"] = note.changed(i, j, changed.rstrip("\n"))
    _own(note, ctx["targets"], target, ctx)
    _written(raw, text, changed)
    _copied(note, changed, range(i, j), "to where you want it")
    needs = note.owners(i, j)
    made = note.replace(i, j, changed.rstrip("\n"), ctx["id"], keep=note.blocks[i].handle if j - i == 1 else None)
    then = note.then(i + len(made))
    shown = fold(new)[0] if folded else new
    label = f"Changed {quote(old)} to {quote(shown)}" if new.strip() else f"Removed {quote(old)}"
    return needs, label, f"edited {target}{_callouts(folded)}; it now reads:\n{note.render(made).rstrip()}\n{then}"


def _insert(note: Note, args: dict, ctx: dict):
    markdown = strip_labels(_text(args, "markdown", False) or "")
    if not markdown.strip():
        raise ToolError("markdown is empty; write the new text", "empty")
    raw, (markdown, folded) = markdown, fold(markdown)
    anchor, after = _where(note, args)
    i, j = note.span(anchor)
    where = "after" if after else "before"
    _written(raw, "", markdown)
    _copied(note, markdown, range(0), f"{where} {anchor}")
    pos, _, _, said, _ = lands(note, "insert", args, raw)
    needs = note.owners(i, j)
    new = note.insert(pos, markdown, ctx["id"])
    made = " ".join(b.handle for b in new if b.kind != "blank")
    then = note.then(pos + len(new))
    return needs, f"Added {quote(markdown)}", (f"inserted {made} {where} {anchor}{said}{_callouts(folded)}:\n"
                                               f"{note.render(new).rstrip()}\n{then}")


def _move(note: Note, args: dict, ctx: dict):
    i, j, target, trimmed = note.pick(_handle(note, args, "target"), ctx["only"])
    anchor, after = _where(note, args)
    k, l = note.span(anchor)
    pos = l if after else k
    if i < pos < j:
        raise ToolError(f"{anchor} is inside {target}; choose a place outside it", "place")
    if pos in (i, j):
        raise ToolError(f"{target} is already there", "place")
    note.check_insert(pos)
    pos, said = note.below_properties(pos)
    if said and pos in (i, j):
        raise ToolError(f"{target} is already right after the note's properties, which stay at its top", "place")
    level = min((b.level for b in note.blocks[i:j] if b.level), default=0)
    cut = note.cut(pos - (j - i) if pos > i else pos, level, note.blocks[:i] + note.blocks[j:])
    if cut and not after:
        raise cut_refused(cut)
    if cut:                             # after the whole section it would cut, counted without the moved blocks
        pos, said = (cut[1] if cut[1] <= i else cut[1] + (j - i)), placed(cut)
        if pos in (i, j):
            raise ToolError(f"{target} is already right after {cut[0]}", "place")
        note.check_insert(pos)
    needs = note.owners(i, j) | note.owners(k, l)
    ctx["targets"] = [b for b in note.blocks[i:j] if b.kind != "blank"]
    label = f"Moved {quote(''.join(b.text for b in note.blocks[i:j]))}"
    moved = ctx["moved"] = note.move(i, j, pos)
    then = note.then(next(n for n, b in enumerate(note.blocks) if b is moved[-1]) + 1)
    where = "after" if after else "before"
    done = f" ({trimmed})" if trimmed else ""
    return needs, label, f"moved {target} {where} {anchor}{done}{said}\n{then}"


# One table: each tool's schema, and the function that applies it to the draft. `plan` and
# `done` are the run's own and never touch the note; `create_note` makes a new note
# (`create_note` below), not a change to the open one.
_HANDLE_ARG = {"type": "string", "description": "a block (b12), a section (s7), or a range (b11-b15)"}
TOOLS: dict[str, dict] = {
    "plan": {
        "description": "Start changing the note: one sentence for the owner saying what you will do, and the steps, in order.",
        "parameters": {"type": "object", "properties": {
            "intro": {"type": "string", "description": "one sentence for the owner"},
            "outline": {"type": "array", "items": {"type": "string"},
                        "description": "the note's headings, in order, as they should read when you're done, "
                                       "each with its # marks"},
            "steps": {"type": "array", "items": {"type": "string"}, "description": "1 to 8 short steps"}},
            "required": ["intro", "outline", "steps"]},
        "apply": None},
    "replace": {
        "description": "Replace a block or a whole section with new markdown. Empty markdown deletes it.",
        "parameters": {"type": "object", "properties": {"target": _HANDLE_ARG, "markdown": {"type": "string"}},
                       "required": ["target", "markdown"]},
        "apply": _replace},
    "edit_text": {
        "description": "Change a few words inside one block: quote the exact text to change (old) and its "
                       "replacement (new). Use this for wording fixes; it keeps the rest of the block as it is.",
        "parameters": {"type": "object", "properties": {"target": _HANDLE_ARG, "old": {"type": "string"},
                                                       "new": {"type": "string"}},
                       "required": ["target", "old", "new"]},
        "apply": _edit_text},
    "insert": {
        "description": "Insert new markdown after or before a block or section. Give exactly one of after or before.",
        # The place before the text: the model writes them in this order, so the text can show
        # where it goes as it is written (`copilot._Drafts`).
        "parameters": {"type": "object", "properties": {"after": _HANDLE_ARG, "before": _HANDLE_ARG,
                                                       "markdown": {"type": "string"}}, "required": ["markdown"]},
        "apply": _insert},
    "move": {
        "description": "Move a block or section to after or before another one. Give exactly one of after or before.",
        "parameters": {"type": "object", "properties": {"target": _HANDLE_ARG, "after": _HANDLE_ARG,
                                                       "before": _HANDLE_ARG}, "required": ["target"]},
        "apply": _move},
    "create_note": {
        "description": "Create a new note in a folder that exists.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "the new note's path in the vault, ending in .md"},
            "markdown": {"type": "string", "description": "the new note's whole text"}},
            "required": ["path", "markdown"]},
        "apply": None},
    "done": {
        "description": "Finish: one sentence for the owner saying what changed.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
        "apply": None},
}


def _move_splices(before: str, old: list[tuple[Block, str]], blocks: list[Block], moved: list[Block]) -> list[dict]:
    """A move as what it is: the moved blocks taken out, and put in at their new place with the
    blank lines either side gained or lost. Never a line diff of the whole note, which keeps the
    longest run in place and can "move" a short neighbour round it — a recording's meeting block."""
    chunk, spans, at = {id(b) for b in moved}, {}, 0
    for b, text in old:
        spans[id(b)] = (at, at + len(text))
        at += len(text)
    was = {id(b): text for b, text in old}
    edits_at: dict[int, tuple[int, str]] = {}

    def put(start: int, stop: int, text: str) -> None:
        to, said = edits_at.get(start, (start, ""))
        edits_at[start] = (max(to, stop), said + text)

    put(spans[id(moved[0])][0], spans[id(moved[-1])][1], "")
    if id(blocks[0]) in chunk:
        put(0, 0, "".join(b.text for b in moved))
    for k, b in enumerate(blocks):
        if id(b) in chunk:
            continue
        start, end = spans[id(b)]
        old = was[id(b)]
        # A blank line it gained, or lost beside a list's items: only a block's end ever changes.
        kept = start + (len(old) if b.text == old else len(os.path.commonprefix([old, b.text])))
        tail = b.text[kept - start:]
        if k + 1 < len(blocks) and id(blocks[k + 1]) in chunk:
            tail += "".join(m.text for m in moved)
        put(kept, end, tail)
    return [{"from": u16(before[:start]), "to": u16(before[:stop]), "insert": text}
            for start, (stop, text) in sorted(edits_at.items()) if text or stop > start]


def _touches(made: list[dict], before: str) -> bool:
    """Does a splice reach into a recording's own parts? A fail-safe behind every tool's checks."""
    for start, stop in protected_ranges(before):
        lo, hi = u16(before[:start]), u16(before[:stop])
        for splice in made:
            f, t = splice["from"], splice["to"]
            if (f < t and f < hi and lo < t) or (f == t and (lo < f < hi or f == lo == 0)):
                return True
    return False


def apply(note: Note, name: str, args: dict, *, change_id: str, step: int,
          only: set[str] | None = None) -> tuple[Change, str]:
    """One tool call applied to the draft: the change it made, and the result the model reads.
    Raises ToolError, with the note as it was, for a call code refuses — and for any change whose
    splices would reach into a recording's own parts, whatever the tools' own checks said."""
    tool = TOOLS.get(name)
    if not tool or not tool["apply"]:
        raise ToolError(f"{name} is not an editing tool", "tool")
    if not isinstance(args, dict):
        raise ToolError("the arguments must be an object", "args")
    old, before = [(b, b.text) for b in note.blocks], note.text()

    def undo() -> None:
        note.blocks = [b for b, _ in old]
        for b, text in old:
            b.text = text

    ctx = {"id": change_id, "only": only, "step": step}
    needs, label, result = tool["apply"](note, args, ctx)
    after = note.text()
    if after == before:
        undo()
        raise ToolError("that changes nothing", "noop")
    made = _move_splices(before, old, note.blocks, ctx["moved"]) if "moved" in ctx else []
    if apply_splices(before, made) != after:
        made = splices(before, after)
    if _touches(made, before):
        undo()
        raise ToolError(f"that change would reach text that is {PROTECTED}", "locked")
    needs.discard(change_id)
    # It changed only text this run wrote whole, none of the owner's: part of the change it grew from.
    reworked = ctx.get("targets", [])
    part_of = min((note.roots.get(b.owner, b.owner) for b in reworked), key=lambda c: int(c[1:])) \
        if reworked and all(b.fresh and b.owner for b in reworked) else None
    note.made[change_id], note.roots[change_id] = step, part_of or change_id
    return Change(change_id, step, name, label, made, sorted(needs, key=lambda c: int(c[1:])), part_of), result


def ollama_tools(names: list[str]) -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": TOOLS[n]["description"],
                                              "parameters": TOOLS[n]["parameters"]}} for n in names]


# --- a new note -------------------------------------------------------------------------------

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


def create_note(vault: Path, raw_path: str, markdown: str, *, folder: str) -> dict:
    """A new note a skill makes: {path, text}, or ToolError. A bare name goes in `folder`, the
    open note's folder. `Journal/` is the floor (CLAUDE.md): nothing is created in it, and a
    Journal note's words never leave it. Nothing is written here; the plugin creates the note."""
    if not isinstance(raw_path, str) or not isinstance(markdown, str):
        raise ToolError("path and markdown must be text", "args")
    raw = _clean_path(raw_path)
    if raw and "/" not in raw and folder not in ("", "."):
        raw = f"{folder}/{raw}"
    path = note_path(raw)
    if path is None:
        raise ToolError(f"'{raw_path}' is not a note path in this vault; give a path ending in .md",
                        "path")
    if _on_floor(path):
        raise ToolError("SLIM never creates a note in Journal/", "journal")
    if _on_floor(folder.rstrip("/") + "/"):
        raise ToolError("a Journal note's text stays in Journal/", "journal")
    parent = PurePosixPath(path).parent.as_posix()
    if not _exact(vault, parent):
        raise ToolError(f"the folder {parent}/ does not exist; use a folder that does, such as {folder}/",
                        "folder")
    if (Path(vault) / path).exists():
        raise ToolError(f"{path} already exists; give the new note a name of its own", "exists")
    text = strip_labels(markdown)
    if not text.strip():
        raise ToolError("markdown is empty; write the new note", "empty")
    if len(text) > MAX_EDIT_CHARS:
        raise ToolError("that note is too large to create from the sidebar", "size")
    return {"path": path, "text": _with_origin(text.strip("\n") + "\n")}
