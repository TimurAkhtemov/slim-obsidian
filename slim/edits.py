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
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .chunk import TRANSCRIPT_MARKER, parse_frontmatter, set_frontmatter
from .config import EXCLUDED_DIRS

_MEETING_OPEN = re.compile(r"^\s*(`{3,})slim-meeting")
# A lone `...` in new text is the model's "the rest as before", and it would delete the rest.
_ELISION = re.compile(r"^[ \t]*(?:\.\.\.|…)[ \t]*$", re.MULTILINE)
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


@dataclass
class Block:
    handle: str
    kind: str
    text: str
    locked: bool = False
    owner: str | None = None          # the change that last made or replaced this block

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
    `before` (JavaScript string indices, as the plugin's editor counts), applied last-first."""
    a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
    starts = [0]
    for line in a:
        starts.append(starts[-1] + len(line))
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append({"from": u16(before[:starts[i1]]), "to": u16(before[:starts[i2]]),
                        "insert": "".join(b[j1:j2])})
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
        past the end of the note read as the note's last block."""
        found = [self._plain(h) for h in handles_in(target)]
        live = [b for b in self.blocks if b.kind != "blank"]
        if len(found) > 1 and live and not any(b.handle == "b" + found[-1][1:] for b in self.blocks) \
                and int(found[-1][1:]) > int(live[-1].handle[1:]):
            found[-1] = live[-1].handle
        return found

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

    def replace(self, i: int, j: int, markdown: str, owner: str, keep: str | None = None) -> list[Block]:
        old = "".join(b.text for b in self.blocks[i:j])
        ending = old[len(old.rstrip("\n")):]
        new = self._fresh(markdown + ending, owner) if markdown else []
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
        if 0 < i < len(self.blocks):
            self._separate(i)
        if pos > i:
            pos -= j - i
        if pos < len(self.blocks) and not moved[-1].text.endswith("\n\n"):
            moved[-1].text = moved[-1].text.rstrip("\n") + "\n\n"
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


def _text(args: dict, name: str, required: bool = True) -> str | None:
    value = args.get(name)
    if value is None or (isinstance(value, str) and not value.strip() and name != "markdown"):
        if required:
            raise ToolError(f"{name} is missing", "args")
        return None
    if not isinstance(value, str):
        raise ToolError(f"{name} must be text", "args")
    return value


def _handle(note: Note, args: dict, name: str, required: bool = True) -> str | None:
    value = _text(args, name, required)
    return note.name(value) if value else None


def _where(note: Note, args: dict) -> tuple[str, bool]:
    after, before = _handle(note, args, "after", False), _handle(note, args, "before", False)
    if bool(after) == bool(before):
        raise ToolError("give exactly one of after or before", "args")
    return (after, True) if after else (before, False)


def _written(markdown: str, old: str = "") -> None:
    """New text may not add a recording's block or transcript heading, which would lock the
    owner's text below it, nor a lone `...` for "the rest as before", which would delete the rest."""
    if any(new > was for was, new in zip(_markers(old), _markers(markdown))):
        raise ToolError("that adds a recording's block or transcript heading; SLIM never writes one", "marker")
    if len(_ELISION.findall(markdown)) > len(_ELISION.findall(old)):
        raise ToolError("the new text skips lines with `...`; write out what stays, or change only what "
                        "changes", "elision")


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


def _replace(note: Note, args: dict, ctx: dict):
    markdown = strip_labels(_text(args, "markdown", False) or "")
    i, j, target, trimmed = note.pick(_handle(note, args, "target"), ctx["only"])
    _written(markdown, "".join(b.text for b in note.blocks[i:j]))
    if _renames(note, i, j, markdown):
        _copied(note, markdown, range(i, i + 1), "to where you want it")
        heading = note.blocks[i].handle
        needs = note.owners(i, i + 1)
        new = note.replace(i, i + 1, markdown, ctx["id"], keep=heading)
        return needs, f"Renamed {heading}", (f"renamed the heading; the section's text is kept:\n"
                                             f"{note.render(new).rstrip()}\n{note.then(i + 1)}")
    _copied(note, markdown, range(i, j), "to where you want it")
    needs = note.owners(i, j)
    new = note.replace(i, j, markdown, ctx["id"], keep=note.blocks[i].handle if j - i == 1 else None)
    then, done = note.then(i + len(new)), f"{target} ({trimmed})" if trimmed else target
    if not markdown:
        return needs, f"Deleted {target}", f"deleted {done}\n{then}"
    return needs, f"Replaced {target}", f"replaced {done}; it now reads:\n{note.render(new).rstrip()}\n{then}"


def _edit_text(note: Note, args: dict, ctx: dict):
    """A wording fix inside a block or range: the model quotes what changes, never retypes the rest."""
    i, j, target, _ = note.pick(_handle(note, args, "target"), ctx["only"])
    old, new = args.get("old"), args.get("new") or ""
    if not isinstance(old, str) or not old:
        raise ToolError("old is empty; quote the exact text to change", "args")
    if not isinstance(new, str):
        raise ToolError("new must be text", "args")
    text = "".join(b.text for b in note.blocks[i:j])
    found = text.count(old)
    if not found:
        raise ToolError(f"'{old}' is not in {target}; it reads: {text.rstrip()[:300]}", "quote")
    if found > 1:
        raise ToolError(f"'{old}' appears {found} times in {target}; quote a little more around it", "quote")
    changed = text.replace(old, new, 1)
    _written(changed, text)
    _copied(note, changed, range(i, j), "to where you want it")
    needs = note.owners(i, j)
    made = note.replace(i, j, changed.rstrip("\n"), ctx["id"], keep=note.blocks[i].handle if j - i == 1 else None)
    then = note.then(i + len(made))
    return needs, f"Edited {target}", f"edited {target}; it now reads:\n{note.render(made).rstrip()}\n{then}"


def _insert(note: Note, args: dict, ctx: dict):
    markdown = strip_labels(_text(args, "markdown", False) or "")
    if not markdown.strip():
        raise ToolError("markdown is empty; write the new text", "empty")
    anchor, after = _where(note, args)
    i, j = note.span(anchor)
    pos = j if after else i
    _written(markdown)
    _copied(note, markdown, range(0), f"{'after' if after else 'before'} {anchor}")
    note.check_insert(pos)
    cut, said, where = note.cut(pos, heading_level(markdown)), "", "after" if after else "before"
    if cut and not after:
        raise cut_refused(cut)
    label = f"Inserted {where} {anchor}"
    if cut:                             # after a heading or inside a section: after that whole section instead
        pos, said, label = cut[1], placed(cut), f"Inserted after {cut[0]}"
        note.check_insert(pos)
    needs = note.owners(i, j)
    new = note.insert(pos, markdown, ctx["id"])
    made = " ".join(b.handle for b in new if b.kind != "blank")
    then = note.then(pos + len(new))
    return needs, label, f"inserted {made} {where} {anchor}{said}:\n{note.render(new).rstrip()}\n{then}"


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
    level = min((b.level for b in note.blocks[i:j] if b.level), default=0)
    cut, said = note.cut(pos - (j - i) if pos > i else pos, level, note.blocks[:i] + note.blocks[j:]), ""
    if cut and not after:
        raise cut_refused(cut)
    if cut:                             # after the whole section it would cut, counted without the moved blocks
        pos, said = (cut[1] if cut[1] <= i else cut[1] + (j - i)), placed(cut)
        if pos in (i, j):
            raise ToolError(f"{target} is already right after {cut[0]}", "place")
        note.check_insert(pos)
    needs = note.owners(i, j) | note.owners(k, l)
    moved = ctx["moved"] = note.move(i, j, pos)
    then = note.then(next(n for n, b in enumerate(note.blocks) if b is moved[-1]) + 1)
    where = "after" if after else "before"
    done = f" ({trimmed})" if trimmed else ""
    label = f"Moved {target} after {cut[0]}" if cut else f"Moved {target} {where} {anchor}"
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
        "parameters": {"type": "object", "properties": {"markdown": {"type": "string"}, "after": _HANDLE_ARG,
                                                       "before": _HANDLE_ARG}, "required": ["markdown"]},
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
    blank lines either side gained. Never a line diff of the whole note, which keeps the longest
    run in place and can "move" a short neighbour round it — a recording's meeting block."""
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
        end = spans[id(b)][1]
        put(end, end, b.text[len(was[id(b)]):])         # a blank line it gained
        if k + 1 < len(blocks) and id(blocks[k + 1]) in chunk:
            put(end, end, "".join(m.text for m in moved))
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

    ctx = {"id": change_id, "only": only}
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
    return Change(change_id, step, name, label, made, sorted(needs, key=lambda c: int(c[1:]))), result


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
