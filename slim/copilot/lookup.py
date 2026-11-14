"""LOOKUP — three read-only tools that let the model look in the vault: `search` (the index),
`read_note` (a note or one section of it) and `list_folder`. They see only what the index sees
(`config.EXCLUDED_TOP` is closed to them), except the open note, which is read as the editor
holds it, with its block handles. Nothing here writes. `Reader.call` never raises: bad input
comes back as "refused: <reason>" for the model to read."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path, PurePosixPath

from . import blocks, edits
from ..config import EXCLUDED_DIRS, EXCLUDED_TOP
from ..index import search as search_mod

SEARCH_HITS = 8
PASSAGE_CHARS = 600
READ_MAX_CHARS = 40_000
LIST_MAX = 200

_TOP_CF = {name.casefold() for name in EXCLUDED_TOP}
_DIRS_CF = {name.casefold() for name in EXCLUDED_DIRS}
_HANDLE = re.compile(r"[bs]\d+(?:\s*[-–]\s*[bs]\d+)?")     # a block, a section, or a range
_PAGE = re.compile(r"(b\d+)(?:#(\d+))?")                    # a locked block, or one page of it
_WORD = re.compile(r"\w+")

TOOLS: dict[str, dict] = {
    "search": {
        "description": "Search the vault's notes for a topic. Returns the best matching passages "
                       "with their note paths; read_note one to see more.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for, in words"},
            "folder": {"type": "string",
                       "description": "optional: only notes under this folder, or only this one note; "
                                      "\"\" searches the whole vault"}},
            "required": ["query"]},
        "apply": None},
    "read_note": {
        "description": "Read a note from the vault, whole or one section of it.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "the note's vault path, ending in .md"},
            "section": {"type": "string",
                        "description": "optional: a heading's text, or for the open note a handle like s7"}},
            "required": ["path"]},
        "apply": None},
    "list_folder": {
        "description": "List a vault folder: its folders, then its notes. An empty path is the vault's root.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "the folder's vault path, or \"\" for the root"}},
            "required": ["path"]},
        "apply": None},
}
NAMES = ("search", "read_note", "list_folder")


class _Refused(Exception):
    pass


def searchable(path: str) -> bool:
    """Whether search can look under this folder: a pinned folder in an excluded one (`Skills/`)
    is never a search's default, or every default search would be refused."""
    return path.split("/")[0].casefold() not in _TOP_CF


def _clean(raw) -> str:
    if not isinstance(raw, str):
        raise _Refused("path must be text")
    path = edits._clean_path(raw)
    return path.rstrip("/") if path else ""


def _check(path: str, *, open_path: str | None = None) -> PurePosixPath:
    """The same rules as `edits.note_path`, for a note or a folder: inside the vault, no hidden
    or excluded parts."""
    if (any(ch in path for ch in "\\:\0") or any(ord(ch) < 32 for ch in path)
            or unicodedata.normalize("NFC", path) != path):
        raise _Refused(f"'{path}' is not a vault path")
    pure = PurePosixPath(path) if path else PurePosixPath(".")
    if pure.is_absolute() or ".." in pure.parts:
        raise _Refused(f"'{path}' is not a vault path; use a path relative to the vault")
    parts = [p.casefold() for p in pure.parts]
    if any(p.startswith(".") or p in _DIRS_CF for p in parts):
        raise _Refused(f"'{path}' is not in the vault's notes")
    if parts and parts[0] in _TOP_CF and path != open_path:
        raise _Refused(f"'{path}' is not in the vault's notes")
    return pure


class Reader:
    def __init__(self, con, vault: Path, *, default_folder: str | None = None,
                 open_note: tuple[str, blocks.Note] | None = None):
        self.con, self.vault = con, Path(vault)
        self.default_folder = default_folder
        self.open_note = open_note
        self.read: list[str] = []
        self.searched: list[str] = []
        self.seen: list[str] = []          # every note read or found, in the order it was looked at
        self.shown: set[str] = set()

    def call(self, name: str, args: dict) -> str:
        try:
            if name not in NAMES:
                raise _Refused(f"there is no tool '{name}'")
            if not isinstance(args, dict):
                raise _Refused("arguments must be an object")
            return getattr(self, "_" + name)(args)
        except _Refused as exc:
            return f"refused: {exc}"
        except (OSError, UnicodeDecodeError) as exc:
            return f"refused: could not read that ({type(exc).__name__})"
        except Exception as exc:       # the embedder, sqlite: nothing escapes into the agent loop
            return f"refused: {name} failed ({type(exc).__name__})"

    # --- search ---------------------------------------------------------------------------

    def _folder(self, raw) -> PurePosixPath:
        """Where a search looks: a folder, or one note (a note's path is its own prefix)."""
        path = _clean(raw)
        pure = _check(path, open_path=self.open_note[0] if self.open_note else None)
        target = self.vault / pure
        found = target.is_dir() or (pure.suffix.lower() == ".md" and target.is_file())
        if path and not (edits._exact(self.vault, pure.as_posix()) and found):
            raise _Refused(f"there is no folder or note '{path}'; list_folder shows what exists")
        return pure

    def _search(self, args: dict) -> str:
        if self.con is None:
            raise _Refused("search is not available")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise _Refused("query must be words to look for")
        # No folder is the default one (a pinned folder); "" or "/" is the whole vault, said outright.
        folder = args.get("folder")
        if folder is None:
            folder = self.default_folder
        elif isinstance(folder, str) and folder.strip() in ("", "/"):
            folder = None
        prefix = self._folder(folder).as_posix() if folder else ""
        prefixes = (prefix,) if prefix not in ("", ".") else None
        hits = search_mod.hybrid(self.con, query, k=SEARCH_HITS, prefixes=prefixes)
        out = []
        for hit in hits:
            if edits.note_path(hit.path) is None or hit.path.split("/")[0].casefold() in _TOP_CF:
                continue
            head = f"{hit.path} · {hit.title or PurePosixPath(hit.path).stem}"
            if hit.heading_path:
                head += f" § {hit.heading_path}"
            passage = _passage(hit.text, query, f"{hit.path} {hit.title or ''}")
            out.append(f"{head}\n{passage}")
            if hit.path not in self.searched:
                self.searched.append(hit.path)
            self.seen.append(hit.path)
        return "\n\n".join(out) if out else "no notes matched"

    # --- read_note ------------------------------------------------------------------------

    def _is_open(self, path: str) -> bool:
        return self.open_note is not None and self.open_note[0] == path

    def _read_note(self, args: dict) -> str:
        raw = _clean(args.get("path"))
        pure = _check(raw, open_path=self.open_note[0] if self.open_note else None)
        path = pure.as_posix()
        if pure.suffix.lower() != ".md":
            raise _Refused(f"'{raw}' is not a note path; give a path ending in .md")
        if not (edits._exact(self.vault, path) and (self.vault / path).is_file()):
            path = self._resolve(path, raw)
        section = args.get("section")
        if section is not None and not isinstance(section, str):
            raise _Refused("section must be text")
        section = (section or "").strip()
        if self._is_open(path):
            body = self._open(section)
        else:
            text = edits._read(self.vault / path)
            if section:
                from . import context          # lazy: context imports edits, which lists these tools
                found = context.section_of(text, section)
                if found is None:
                    heads = "\n".join(blocks.headings(text)) or "(none)"
                    raise _Refused(f"'{path}' has no section '{section}'; its headings:\n{heads}")
                text = found
            body = self._cap(text, blocks.headings(text) if not section else [])
        if path not in self.read:
            self.read.append(path)
        self.seen.append(path)
        return f"=== {path} ===\n{body}"

    def _resolve(self, path: str, raw: str) -> str:
        """The one note whose path ends with what the model wrote: it drops a top folder
        (`job-search/…` for `Notes/job-search/…`, seen live 2026-11-14) or gives a bare file name.
        Several matches are named back; none is the plain refusal."""
        tail = "/" + path.casefold()
        pool = list(self.seen) + ([self.open_note[0]] if self.open_note else [])
        if self.con is not None:
            pool += [row[0] for row in self.con.execute(
                "SELECT path FROM sources WHERE lower(path) LIKE ?", ("%" + path.casefold(),))]
        found = [p for p in dict.fromkeys(pool)
                 if ("/" + p.casefold()).endswith(tail) and (searchable(p) or self._is_open(p))
                 and edits._exact(self.vault, p) and (self.vault / p).is_file()]
        if len(found) == 1:
            return found[0]
        if found:
            raise _Refused(f"several notes end with '{raw}': " + ", ".join(found[:8]) + "; give the full path")
        raise _Refused(f"there is no note '{raw}'; search or list_folder to find it")

    @staticmethod
    def _cap(text: str, heads: list[str]) -> str:
        if len(text) <= READ_MAX_CHARS:
            return text
        outline = "\n".join(heads)
        return (f"{outline}\n\n" if outline else "") + text[:READ_MAX_CHARS] + \
            "\n[… the rest: read_note(path, section) for one section]"

    def _open(self, section: str) -> str:
        """The open note as the editor holds it, with handles; what it renders is `.shown`."""
        path, note = self.open_note
        page = _PAGE.fullmatch(section)
        if page:
            block = next((b for b in note.blocks if b.handle == page[1]), None)
            # A recording's part, or a block too long to read whole (one pasted 60k paragraph):
            # a page at a time, read-only.
            if block is not None and (block.locked or len(note.render([block])) > READ_MAX_CHARS):
                if not block.locked:
                    note.paged.add(block.handle)
                return self._pages(path, block, int(page[2] or 1))
            if page[2]:
                raise _Refused(f"{page[1]} is short enough to read whole; read it without a page")
        if section:
            target = section
            if not _HANDLE.fullmatch(section):
                want = blocks.heading_key(section)[1] if section.startswith("#") else \
                    " ".join(section.casefold().split())
                head = next((b for b in note.blocks if b.level and not b.locked
                             and blocks.heading_key(b.text.strip().splitlines()[0])[1] == want), None)
                if head is None:
                    raise _Refused(f"the open note has no section '{section}'; its sections:\n{note.sections()}")
                target = "s" + head.handle[1:]
            try:
                i, j = note.span(target)
            except blocks.ToolError as exc:
                raise _Refused(str(exc)) from exc
            chosen = note.blocks[i:j]
        else:
            chosen = list(note.blocks)
        text = note.render(chosen)
        if len(text) > READ_MAX_CHARS:
            # Whole blocks only, so what it shows and what it may then change (`.shown`) agree.
            text, chosen, _reached = note.head(chosen, READ_MAX_CHARS, path)
            sections = "" if section else note.sections()
            text = (f"Sections:\n{sections}\n\n" if sections else "") + text + \
                "[… the rest: read_note(path, section) for one section]"
        self.shown |= {b.handle for b in chosen if b.kind != "blank" and not b.locked}
        return text

    @staticmethod
    def _pages(path: str, block: blocks.Block, n: int) -> str:
        """A recording's locked block (a long transcript), or a block too long to read whole,
        read-only, a page of READ_MAX_CHARS at a time, each cut at a line where one is near. It never
        joins `.shown`: no tool may change it."""
        pages, rest = [], block.text
        while len(rest) > READ_MAX_CHARS:
            cut = rest.rfind("\n", 0, READ_MAX_CHARS) + 1
            if cut <= READ_MAX_CHARS // 2:
                cut = READ_MAX_CHARS
            pages.append(rest[:cut])
            rest = rest[cut:]
        pages.append(rest)
        h = block.handle
        if not 1 <= n <= len(pages):
            raise _Refused(f"{h} has {len(pages)} page{'s' if len(pages) > 1 else ''}")
        what = "part of the recording" if block.locked else "too long to show whole"
        said = f"[{'locked ' if block.locked else ''}{h}, page {n} of {len(pages)}: {what}, read-only]\n{pages[n - 1].rstrip()}"
        if n < len(pages):
            said += f'\n[Next page: read_note("{path}", "{h}#{n + 1}")]'
        return said

    # --- list_folder ----------------------------------------------------------------------

    def _list_folder(self, args: dict) -> str:
        raw = _clean(args.get("path"))
        pure = _check(raw)
        path = "" if pure.as_posix() == "." else pure.as_posix()
        if path and not (edits._exact(self.vault, path) and (self.vault / path).is_dir()):
            raise _Refused(f"there is no folder '{raw}'")
        return listing(self.vault / path, root=not path)


def _passage(text: str, query: str, named: str) -> str:
    """What a search shows of a hit: the PASSAGE_CHARS of it where the query's words are, as grep
    shows the matching line, not the section's head (live 2026-11-14 the figures asked about sat
    1,300 characters in). Words the hit's path or title already name don't steer the window."""
    flat = " ".join(text.split())
    if len(flat) <= PASSAGE_CHARS:
        return flat
    skip = {w.casefold() for w in _WORD.findall(named)}
    terms = {w.casefold() for w in _WORD.findall(query) if len(w) >= 3} - skip

    def matches(word: str) -> set[str]:
        word = word.casefold()
        return {t for t in terms if word.startswith(t) or (len(word) >= 4 and t.startswith(word))}

    hits = [(m.start(), matches(m.group())) for m in _WORD.finditer(flat)]
    hits = [(at, found) for at, found in hits if found]
    start, best = 0, 0
    for at, _ in hits:
        begin = max(0, at - PASSAGE_CHARS // 5)
        covered = set().union(*(f for a, f in hits if begin <= a < begin + PASSAGE_CHARS))
        if len(covered) > best:
            start, best = begin, len(covered)
    if start:
        start = min(flat.find(" ", start) + 1 or start, start + 40)   # begin on a word
    start = min(start, len(flat) - PASSAGE_CHARS)
    window = flat[start:start + PASSAGE_CHARS]
    return ("… " if start else "") + window + (" …" if start + PASSAGE_CHARS < len(flat) else "")


def listing(folder: Path, *, root: bool) -> str:
    """A folder one level deep: its folders, then its notes, without hidden ones, and at the
    vault's `root` without the excluded ones. A pinned folder is listed here too, whatever it is."""
    folders, notes = [], []
    for entry in os.scandir(folder):
        low = entry.name.casefold()
        if entry.name.startswith(".") or low in _DIRS_CF or (root and low in _TOP_CF):
            continue
        if entry.is_dir():
            folders.append(entry.name + "/")
        elif entry.name.lower().endswith(".md"):
            notes.append(entry.name)
    lines = sorted(folders) + sorted(notes)
    if not lines:
        return "(empty)"
    more = len(lines) - LIST_MAX
    out = lines[:LIST_MAX]
    if more > 0:
        out.append(f"[… {more} more]")
    return "\n".join(out)
