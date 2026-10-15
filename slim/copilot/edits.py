"""EDITS — the open note as the copilot's editing tools see it, and what those tools may change.

Code cuts the note into blocks (a heading, a paragraph, a list item, a table, a quote, fenced code
or `$$` math) and gives each a handle the model points at: [b12] is that block, s7 is the section
under the heading [b7], and b11-b15 is those blocks and everything between them. Handles never
change during a run; blocks an edit makes get new ones. The model calls the tools in `TOOLS`
with handles; code applies each call to a draft (`copilot.blocks.Note`), refuses what it must not do
with a reason the model reads back, and hands each change on as splices — {from, to, insert} in
UTF-16 units of the text just before it, which the plugin applies through the editor. Nothing
here writes a note.

What may be touched is decided here, not by the model. A recorded note's frontmatter, its
`slim-meeting` block and its transcript are `recorder/note.py`'s and the raw source: their blocks are
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

from . import blocks
from ..index.chunk import TRANSCRIPT_MARKER
from ..config import EXCLUDED_DIRS
from .blocks import Block, Note, ToolError
from ..vault.frontmatter import parse_frontmatter, set_frontmatter

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


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_path(raw: str) -> str:
    # `· the open note` is the pack's label after the path, which the model copies.
    path = (raw or "").split(" · ")[0].strip().strip("`*\"'=").strip()
    return path[2:] if path.startswith("./") else path


def _markers(text: str) -> tuple[int, int]:
    lines = text.splitlines()
    return (sum(bool(blocks._MEETING_OPEN.match(line)) for line in lines),
            sum(line.strip() == TRANSCRIPT_MARKER for line in lines))


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
    for heading in blocks.headings(markdown):
        for k, b in enumerate(note.blocks):
            if k in mine or not b.level or not any(blocks.heading_key(h) == blocks.heading_key(heading) for h in blocks.headings(b.text)):
                continue
            body = _lines("".join(c.text for c in note.blocks[k + 1:blocks._end(note.blocks, k)]))
            if (1 + sum(line in written for line in body)) / (1 + len(body)) < COPY_SHARE:
                continue
            s = "s" + b.handle[1:]
            raise ToolError(f"{heading} is already in the note as {s}; to put that section here, use move {s} "
                            f"{place}, and replace it afterwards if it needs new text.", "copy")


def _renames(note: Note, i: int, j: int, markdown: str) -> bool:
    """A whole section replaced by a heading alone: the model meant a rename, not to drop the text."""
    one = blocks._spans(markdown.splitlines(keepends=True))
    return (j - i > 1 and note.blocks[i].level > 0 and (i, j) == note.span("s" + note.blocks[i].handle[1:])
            and len(one) == 1 and one[0][0] == "heading")


def lands(note: Note, tool: str, args: dict, markdown: str,
          only: set[str] | None = None) -> tuple[int, int, str, str, bool]:
    """Where a replace or an insert puts `markdown` (as the model wrote it, labels stripped): for
    the tool, and for its draft while the text is still coming (`copilot.run._Drafts`). The blocks
    i..j it changes, i == j where an insert goes in; its target or anchor as named back; what the
    result adds about the place; and whether a replace renames: a whole section replaced by one
    heading line changes that heading alone, block i, and the section's text stays. Raises
    ToolError for a place the text may not go."""
    if tool == "insert":
        anchor, after = _where(note, args)
        i, j = note.span(anchor)
        note.check_insert(j if after else i)
        pos, said = note.below_properties(j if after else i)
        cut = note.cut(pos, blocks.heading_level(fold(markdown)[0]))
        if cut and not after:
            raise blocks.cut_refused(cut)
        if cut:                         # after a heading or inside a section: after that whole section instead
            pos, said = cut[1], blocks.placed(cut)
            note.check_insert(pos)
        return pos, pos, anchor, said, False
    i, j, target, trimmed = note.pick(_handle(note, args, "target"), only)
    old = "".join(b.text for b in note.blocks[i:j])
    return i, j, target, trimmed, _renames(note, i, j, fold(markdown, old)[0])


def _replace(note: Note, args: dict, ctx: dict):
    markdown = blocks.strip_labels(_text(args, "markdown"))
    i, j, target, trimmed, renames = lands(note, "replace", args, markdown, ctx["only"])
    old = "".join(b.text for b in note.blocks[i:j])
    raw, (markdown, folded) = markdown, fold(markdown, old)
    ctx["targets"] = note.changed(i, i + 1 if renames else j, markdown)
    _own(note, ctx["targets"], target, ctx)
    _written(raw, old, markdown)
    renamed = f"Renamed {blocks.quote(note.blocks[i].text)} to {blocks.quote(markdown)}"
    if renames:
        _copied(note, markdown, range(i, i + 1), "to where you want it")
        heading = note.blocks[i].handle
        needs = note.owners(i, i + 1)
        new = note.replace(i, i + 1, markdown, ctx["id"], keep=heading)
        return needs, renamed, (f"renamed the heading; the section's text is kept:\n"
                                f"{note.render(new).rstrip()}\n{note.then(i + 1)}")
    _copied(note, markdown, range(i, j), "to where you want it")
    one = blocks._spans(markdown.splitlines(keepends=True))
    label = (renamed if j - i == 1 and note.blocks[i].level and len(one) == 1 and one[0][0] == "heading"
             else f"Rewrote {blocks.quote(old)}" if markdown else f"Removed {blocks.quote(old)}")
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
    label = f"Changed {blocks.quote(old)} to {blocks.quote(shown)}" if new.strip() else f"Removed {blocks.quote(old)}"
    return needs, label, f"edited {target}{_callouts(folded)}; it now reads:\n{note.render(made).rstrip()}\n{then}"


def _insert(note: Note, args: dict, ctx: dict):
    markdown = blocks.strip_labels(_text(args, "markdown", False) or "")
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
    return needs, f"Added {blocks.quote(markdown)}", (f"inserted {made} {where} {anchor}{said}{_callouts(folded)}:\n"
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
        raise blocks.cut_refused(cut)
    if cut:                             # after the whole section it would cut, counted without the moved blocks
        pos, said = (cut[1] if cut[1] <= i else cut[1] + (j - i)), blocks.placed(cut)
        if pos in (i, j):
            raise ToolError(f"{target} is already right after {cut[0]}", "place")
        note.check_insert(pos)
    needs = note.owners(i, j) | note.owners(k, l)
    ctx["targets"] = [b for b in note.blocks[i:j] if b.kind != "blank"]
    label = f"Moved {blocks.quote(''.join(b.text for b in note.blocks[i:j]))}"
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
        # where it goes as it is written (`copilot.run._Drafts`).
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


def _move_splices(before: str, old: list[tuple[Block, str]], after: list[Block], moved: list[Block]) -> list[dict]:
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
    if id(after[0]) in chunk:
        put(0, 0, "".join(b.text for b in moved))
    for k, b in enumerate(after):
        if id(b) in chunk:
            continue
        start, end = spans[id(b)]
        old = was[id(b)]
        # A blank line it gained, or lost beside a list's items: only a block's end ever changes.
        kept = start + (len(old) if b.text == old else len(os.path.commonprefix([old, b.text])))
        tail = b.text[kept - start:]
        if k + 1 < len(after) and id(after[k + 1]) in chunk:
            tail += "".join(m.text for m in moved)
        put(kept, end, tail)
    return [{"from": u16(before[:start]), "to": u16(before[:stop]), "insert": text}
            for start, (stop, text) in sorted(edits_at.items()) if text or stop > start]


def _touches(made: list[dict], before: str) -> bool:
    """Does a splice reach into a recording's own parts? A fail-safe behind every tool's checks."""
    for start, stop in blocks.protected_ranges(before):
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
        raise ToolError(f"that change would reach text that is {blocks.PROTECTED}", "locked")
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
    text = blocks.strip_labels(markdown)
    if not text.strip():
        raise ToolError("markdown is empty; write the new note", "empty")
    if len(text) > MAX_EDIT_CHARS:
        raise ToolError("that note is too large to create from the sidebar", "size")
    return {"path": path, "text": _with_origin(text.strip("\n") + "\n")}
