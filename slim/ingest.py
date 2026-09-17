"""Vault ingestion: source identity, chunking, FTS indexing.

Identity rules:
  - content hash is identity; path is an attribute
  - rename  = a new path whose content hash matches a source whose old path is gone
  - edit    = same path, new hash -> fragments rebuilt
  - excluded dirs -> never ingested (and evicted if present)
Idempotent: re-running over an unchanged vault writes nothing.
"""

import hashlib
import os
import sqlite3
import uuid
from pathlib import Path

from .chunk import chunk_markdown, parse_frontmatter, strip_derived_blocks, strip_meeting_wrapper
from .config import EXCLUDED_DIRS, EXCLUDED_TOP, VAULT


class IngestError(RuntimeError):
    """A note cannot be indexed."""


# Casefolded: on case-insensitive APFS, a folder created as `profile/` is the same folder as
# `Profile/`, so exclusion must match the way the filesystem does.
_EXCLUDED_DIRS_CF = frozenset(name.casefold() for name in EXCLUDED_DIRS)
_EXCLUDED_TOP_CF = frozenset(name.casefold() for name in EXCLUDED_TOP)


# The recorder's working copy: `<recording id>.slim-draft.md` directly under Journal/ or
# Capture/_unfiled/ (obsidian/main.js DRAFT_SUFFIX). `record.write_recording` files it as a
# real note, so indexing the draft gives the finished note a same-content rival. A
# `.slim-draft.md` anywhere else is the user's own file and is indexed like any other.
DRAFT_SUFFIX = ".slim-draft.md"
_DRAFT_PARENTS = {("journal",), ("capture", "_unfiled")}


def _managed_draft(rel: Path) -> bool:
    parent = tuple(part.casefold() for part in rel.parts[:-1])
    return rel.name.endswith(DRAFT_SUFFIX) and parent in _DRAFT_PARENTS


def indexable(rel: Path) -> bool:
    """One rule for the sweep and the single-note path: never index what the other skips."""
    if not rel.parts or rel.parts[0].casefold() in _EXCLUDED_TOP_CF:
        return False
    if any(part.casefold() in _EXCLUDED_DIRS_CF for part in rel.parts):
        return False
    return not _managed_draft(rel)


def eligible_files(vault: Path):
    for f in sorted(vault.rglob("*.md")):
        if indexable(f.relative_to(vault)):
            yield f


def _confirmed_gone(path: str, vault: Path) -> bool:
    """True only when a source's absence is positively confirmed by a direct stat.

    Fail closed in every ambiguous direction: an unresolvable namespace, a permission
    or I/O error, or an unexpected stat result all mean "cannot prove it is gone", and
    a source that cannot be proven gone is never swept.
    """
    try:
        target = source_file(path, vault=vault)
    except IngestError:
        return False
    try:
        os.stat(target)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def source_file(path: str, *, vault: Path = VAULT) -> Path:
    """Resolve one indexed path back to its vault file."""
    rel = str(path).replace("\\", "/").strip("/")
    target = Path(vault) / rel
    try:
        target.resolve().relative_to(Path(vault).resolve())
    except (ValueError, OSError) as exc:
        raise IngestError(f"vault source resolves outside the vault: {path!r}") from exc
    return target


def source_fields(fm: dict) -> dict:
    """The three frontmatter facts the index keeps. Everything else a note declares is for
    Obsidian, not retrieval — `reflect` needs `type` and `authored_at`, the copilot needs
    `title`, and nothing reads any of the rest."""
    return {
        "title": fm.get("title") or None,
        "type": fm.get("type") or None,
        "authored_at": fm.get("date") or fm.get("created_time") or None,
    }


def write_fragments(con: sqlite3.Connection, source_id: str, content_hash: str, text: str):
    # A recorded note carries a DERIVED summary above its transcript. Strip that fenced block
    # before chunking so SLIM never indexes its own opinion as evidence.
    text = strip_meeting_wrapper(strip_derived_blocks(text))
    con.execute("DELETE FROM fragments WHERE source_id = ?", (source_id,))
    for frag in chunk_markdown(text):
        con.execute(
            "INSERT INTO fragments (source_id, content_hash, seq, heading_path, text) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, content_hash, frag.seq, frag.heading_path, frag.text))


def ingest_note(con: sqlite3.Connection, vault: Path, rel_path: str) -> dict:
    """Synchronize one saved vault Markdown note without sweeping any other source.

    Same per-file body as the sweep (`_sync_file`), so the two make identical identity
    decisions; only the rename rule differs, because a single-note sync sees one file.
    """
    vault = Path(vault).resolve()
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix.casefold() != ".md":
        raise IngestError("note path must be a vault-relative Markdown path")
    if not indexable(rel):
        raise IngestError("note is not indexable")
    target = (vault / rel).resolve()
    try:
        target.relative_to(vault)
    except ValueError as exc:
        raise IngestError("note path must be vault-relative") from exc
    if not target.is_file():
        raise IngestError(f"saved note does not exist: {rel.as_posix()}")

    def claimable(candidate: str) -> bool:
        # The sweep's rule is "not on disk": a same-content source whose file is gone is
        # this note's previous path, so the rename keeps its id.
        return not (vault / candidate).exists()

    stats = dict.fromkeys(("unchanged", "new", "edited", "renamed"), 0)
    source_id = _sync_file(con, target, rel.as_posix(), stats, claimable=claimable)
    con.commit()
    return dict(con.execute("SELECT id, path, title, type, current_hash FROM sources WHERE id = ?", (source_id,)).fetchone())


def _sync_file(con: sqlite3.Connection, f: Path, rel: str, stats: dict, *,
               claimable, verbose: bool = False) -> str:
    """One file into sources/fragments. Returns its source id.

    `claimable(path)` decides whether an existing same-content source may be treated as
    this file's previous path (a rename) — the caller knows what it can see.
    """
    raw = f.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    fm, _ = parse_frontmatter(text)
    content_hash = hashlib.sha256(raw).hexdigest()
    fields = source_fields(fm)

    row = con.execute("SELECT * FROM sources WHERE path = ?", (rel,)).fetchone()
    if row and row["current_hash"] == content_hash and not row["deleted"]:
        stats["unchanged"] += 1
        return row["id"]

    if row:  # same path, new content (or resurrected) -> edit
        source_id = row["id"]
        con.execute(
            "UPDATE sources SET title=:title, type=:type, authored_at=:authored_at, "
            "current_hash=:h, deleted=0 WHERE id=:id",
            {**fields, "h": content_hash, "id": source_id})
        stats["edited"] += 1
    else:
        # rename check: same content, old path no longer visible to the caller
        match = None
        for cand in con.execute(
                "SELECT * FROM sources WHERE current_hash = ? AND deleted = 0", (content_hash,)):
            if claimable(cand["path"]):
                match = cand
                break
        if match:
            source_id = match["id"]
            con.execute(
                "UPDATE sources SET path=:path, title=:title, type=:type, "
                "authored_at=:authored_at, deleted=0 WHERE id=:id",
                {**fields, "path": rel, "id": source_id})
            stats["renamed"] += 1
            if verbose:
                print(f"  renamed: {match['path']} -> {rel}")
            return source_id  # content unchanged: fragments still valid
        source_id = str(uuid.uuid4())
        con.execute(
            "INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
            "VALUES (:id, :path, :title, :type, :authored_at, :h)",
            {**fields, "id": source_id, "path": rel, "h": content_hash})
        stats["new"] += 1

    write_fragments(con, source_id, content_hash, text)
    if verbose:
        print(f"  {'edited' if row else 'new'}: {rel}")
    return source_id


def ingest(con: sqlite3.Connection, vault: Path = None, verbose: bool = False) -> dict:
    vault = Path(vault or VAULT)
    stats = {"unchanged": 0, "new": 0, "edited": 0, "renamed": 0, "removed": 0}
    seen_paths = set()

    files = [(f, str(f.relative_to(vault))) for f in eligible_files(vault)]
    files.sort(key=lambda item: item[1].casefold())
    disk_paths = {rel for _f, rel in files}

    def claimable(path: str) -> bool:
        return path not in disk_paths

    for f, rel in files:
        _sync_file(con, f, rel, stats, claimable=claimable, verbose=verbose)
        seen_paths.add(rel)

    # Removal sweep: sources whose file is gone and not claimed by a rename, AND sources whose
    # file is still there but whose path is no longer `indexable()` — a note moved into
    # `.trash/` or an excluded folder is evicted the same way a deleted one is.
    #
    # `seen_paths` alone is NOT proof of absence: both walkers swallow per-directory errors
    # (os.walk's default onerror, rglob's suppressed scandir failures), so one transient
    # permission glitch on a subtree would silently drop every source under it. Deletion
    # requires POSITIVE confirmation from a direct stat — found-by-walk is evidence of
    # presence, but missed-by-walk is not evidence of absence (2026-07-16).
    for row in con.execute("SELECT id, path FROM sources WHERE deleted = 0"):
        if row["path"] in seen_paths:
            continue
        if not indexable(Path(row["path"])) or _confirmed_gone(row["path"], vault):
            con.execute("DELETE FROM fragments WHERE source_id = ?", (row["id"],))
            con.execute("UPDATE sources SET deleted = 1 WHERE id = ?", (row["id"],))
            stats["removed"] += 1

    con.commit()
    return stats
