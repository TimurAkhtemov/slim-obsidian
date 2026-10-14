"""EDITS_NOTE — the open note as blocks with handles, the draft the copilot's editing tools change.

Code cuts the note into blocks (a heading, a paragraph, a list item, a table, a quote, fenced code
or `$$` math) and gives each a handle the model points at: [b12] is that block, s7 is the section
under the heading [b7], and b11-b15 is those blocks and everything between them. Handles never
change during a run; blocks an edit makes get new ones. A recorded note's frontmatter, its
`slim-meeting` block and its transcript are [locked] (`protected_ranges`). The tools that change
the draft are `edits.py`'s.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .chunk import TRANSCRIPT_MARKER, parse_frontmatter

_MEETING_OPEN = re.compile(r"^\s*(`{3,})slim-meeting")

PROTECTED = "part of the recording; SLIM never edits it"


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
