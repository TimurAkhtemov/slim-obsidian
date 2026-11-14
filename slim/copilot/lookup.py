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
PASSAGE_CHARS = 400
READ_MAX_CHARS = 40_000
LIST_MAX = 200

_TOP_CF = {name.casefold() for name in EXCLUDED_TOP}
_DIRS_CF = {name.casefold() for name in EXCLUDED_DIRS}
_HANDLE = re.compile(r"[bs]\d+")

TOOLS: dict[str, dict] = {
    "search": {
        "description": "Search the vault's notes for a topic. Returns the best matching passages "
                       "with their note paths; read_note one to see more.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for, in words"},
            "folder": {"type": "string", "description": "optional: only notes under this folder"}},
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
        path = _clean(raw)
        pure = _check(path)
        if path and not (edits._exact(self.vault, pure.as_posix()) and (self.vault / pure).is_dir()):
            raise _Refused(f"there is no folder '{path}'; list_folder shows what exists")
        return pure

    def _search(self, args: dict) -> str:
        if self.con is None:
            raise _Refused("search is not available")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise _Refused("query must be words to look for")
        folder = args.get("folder") or self.default_folder
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
            passage = " ".join(hit.text.split())[:PASSAGE_CHARS]
            out.append(f"{head}\n{passage}")
            if hit.path not in self.searched:
                self.searched.append(hit.path)
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
            raise _Refused(f"there is no note '{raw}'; search or list_folder to find it")
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
        return f"=== {path} ===\n{body}"

    @staticmethod
    def _cap(text: str, heads: list[str]) -> str:
        if len(text) <= READ_MAX_CHARS:
            return text
        outline = "\n".join(heads)
        return (f"{outline}\n\n" if outline else "") + text[:READ_MAX_CHARS] + \
            "\n[… the rest: read_note(path, section) for one section]"

    def _open(self, section: str) -> str:
        """The open note as the editor holds it, with handles; what it renders is `.shown`."""
        note = self.open_note[1]
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
            text, chosen = note.head(chosen, READ_MAX_CHARS)
            sections = "" if section else note.sections()
            text = (f"Sections:\n{sections}\n\n" if sections else "") + text + \
                "[… the rest: read_note(path, section) for one section]"
        self.shown |= {b.handle for b in chosen if b.kind != "blank" and not b.locked}
        return text

    # --- list_folder ----------------------------------------------------------------------

    def _list_folder(self, args: dict) -> str:
        raw = _clean(args.get("path"))
        pure = _check(raw)
        path = "" if pure.as_posix() == "." else pure.as_posix()
        if path and not (edits._exact(self.vault, path) and (self.vault / path).is_dir()):
            raise _Refused(f"there is no folder '{raw}'")
        return listing(self.vault / path, root=not path)


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
