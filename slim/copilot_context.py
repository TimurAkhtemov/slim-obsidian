"""What the copilot reads: the open note synced into the index, the notes around it by scope, the
passages a question retrieves, the open note's images, and the whole notes a skill reads
(`gather`). `copilot.py` runs the turn."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from . import config as config_mod, edits as edits_mod, embed as embed_mod, ingest as ingest_mod
from . import llm, search as search_mod, skills as skills_mod
from .vault import note_images
from .chunk import HEADING, source_text
from .vault.frontmatter import body_after_frontmatter, parse_frontmatter

# Skills read WHOLE notes, chosen by code. Half the resident window in chars/4, so
# the answer and the thinking keep room and every call still uses num_ctx=RESIDENT_CTX (a
# smaller bespoke window is a second runner). What does not fit is named, never dropped silently.
INPUT_BUDGET_CHARS = llm.RESIDENT_CTX * 4 // 2
MAX_SELECTION_CHARS = 20_000
_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
# The open note's images, those beside the passages the pack carries first. Each costs ~450
# prompt tokens and ~1.3 s before the first word (measured 2026-09-22 on the resident), and a
# lecture note embeds up to 60, so a turn sends at most this; the median note embeds 4.
MAX_NOTE_IMAGES = 6


class CopilotError(RuntimeError):
    """A user-facing copilot request refusal."""


def sync_note(con, vault: Path, path: str, *, embed: bool = True) -> dict:
    try:
        source = ingest_mod.ingest_note(con, vault, path)
    except ingest_mod.IngestError as exc:
        raise CopilotError(str(exc)) from exc
    if embed:
        embed_mod.embed_source(con, source["id"])
    return source


def _matches(path: str, prefix: str) -> bool:
    value, root = path.casefold(), prefix.rstrip("/").casefold()
    return value == root or value.startswith(root + "/")


def scope_levels(path: str) -> list[tuple[str, tuple[str, ...]]]:
    note = PurePosixPath(path)
    parent = note.parent.as_posix()
    if note.is_absolute() or ".." in note.parts or parent == ".":
        return []
    if note.parts[0].casefold() == "journal":
        return [("folder", (parent,))]
    subject_prefixes = ()
    for project_id in config_mod.registry():
        prefixes = config_mod.project_prefixes(project_id)
        if any(_matches(path, prefix) for prefix in prefixes):
            subject_prefixes = prefixes
            break
    if not subject_prefixes:
        return [("folder", (parent,))]
    matched_root = next(prefix for prefix in subject_prefixes if _matches(path, prefix))
    # Sliced, not `relative_to`: `_matches` casefolds and APFS is case-insensitive, so
    # `notes/School/x.md` under prefix `Notes/school` must not raise here.
    relative_parts = note.parts[len(PurePosixPath(matched_root).parts):]
    course = relative_parts[0] if len(relative_parts) > 1 else None
    levels = [("folder", (parent,))]
    if course:
        levels.append(("course", tuple(f"{prefix}/{course}" for prefix in subject_prefixes)))
    levels.append(("subject", subject_prefixes))
    out = []
    for label, prefixes in levels:
        normalized = tuple(dict.fromkeys(prefix.rstrip("/") for prefix in prefixes))
        if not out or normalized != out[-1][1]:
            out.append((label, normalized))
    return out


# The "what is this note about" query for Nearby notes: title, headings and the opening
# of the first fragments. Bounded because it is embedded once per open and sent to FTS as
# one OR-query — at 5,000 chars that scan was the slowest thing in opening a note.
RELATED_QUERY_CHARS = 1500


def _source_query(con, source: dict) -> str:
    rows = con.execute("SELECT heading_path, text FROM fragments WHERE source_id = ? ORDER BY seq LIMIT 4", (source["id"],)).fetchall()
    pieces = [source.get("title") or PurePosixPath(source["path"]).stem]
    pieces.extend(row["heading_path"] or "" for row in rows)
    pieces.extend(row["text"][:400] for row in rows)
    return "\n".join(piece for piece in pieces if piece)[:RELATED_QUERY_CHARS]


def related_notes(con, vault: Path, source: dict, *, limit: int = 5) -> list[dict]:
    """Nearby notes, nearest scope first: ONE search over the widest scope, labelled in code."""
    levels = scope_levels(source["path"])
    if not levels:
        return []
    roots = tuple(dict.fromkeys(prefix for _label, prefixes in levels for prefix in prefixes))
    hits = search_mod.hybrid(con, _source_query(con, source), k=max(20, limit * 4), prefixes=roots)
    found, seen = [], {source["id"]}
    for label, prefixes in levels:
        for hit in hits:
            if hit.source_id in seen or not any(_matches(hit.path, prefix) for prefix in prefixes):
                continue
            seen.add(hit.source_id)
            if not (Path(vault) / hit.path).is_file():
                continue
            found.append({"source_id": hit.source_id, "path": hit.path, "title": hit.title or PurePosixPath(hit.path).stem, "scope": label})
            if len(found) >= limit:
                return found
    return found


def context_payload(con, vault: Path, path: str, *, related: bool = True) -> dict:
    # Always embed: `related=False` is the plugin's pre-question refresh, and a fragment
    # without a vector is invisible to the question's retrieval. Unchanged note = no rows.
    source = sync_note(con, vault, path, embed=True)
    return {"source": source, "related": related_notes(con, vault, source) if related else []}


def collect_evidence(con, source: dict, question: str, *, limit: int) -> tuple[list[dict], list[str]]:
    """Fill a bounded pack from the active note outward, one path level at a time."""
    hits = []
    seen = set()
    levels_used = ["note"]

    def add(candidates):
        for hit in candidates:
            if hit.fragment_id in seen:
                continue
            seen.add(hit.fragment_id)
            hits.append(hit)
            if len(hits) >= limit:
                break

    # cap=limit: PER_SOURCE_CAP stops one note flooding a vault-wide pack, but the open note
    # IS the subject here — capped at 3, a 12-slot pack was 9 fragments of sibling notes.
    # A note path is its own prefix, so the first pass scopes to the open note alone.
    add(search_mod.hybrid(con, question, k=limit, cap=limit, prefixes=(source["path"],)))
    if not hits:
        rows = con.execute(
            "SELECT f.id AS fragment_id, s.id AS source_id, s.path, s.title, "
            "f.heading_path, f.text FROM fragments f "
            "JOIN sources s ON s.id=f.source_id WHERE s.id=? ORDER BY f.seq LIMIT ?",
            (source["id"], limit),).fetchall()
        add(search_mod._rows_to_hits(rows))
    for label, prefixes in scope_levels(source["path"]):
        if len(hits) >= limit:
            break
        before = len(hits)
        add(search_mod.hybrid(con, question, k=limit, prefixes=prefixes))
        if len(hits) > before:
            levels_used.append(label)

    evidence = [{
        "n": index + 1, "source_id": hit.source_id, "path": hit.path,
        "title": hit.title or PurePosixPath(hit.path).stem,
        "heading": hit.heading_path, "text": hit.text,
    } for index, hit in enumerate(hits)]
    return evidence, levels_used


@dataclass
class Pack:
    """What a skill reads: whole notes, in order, and what did not make it in."""
    docs: list[dict] = field(default_factory=list)      # {path, title, text, role, date}
    dropped: list[str] = field(default_factory=list)    # did not fit the budget
    skipped: list[str] = field(default_factory=list)    # had no such section
    missing: list[str] = field(default_factory=list)    # [[links]] that resolve to nothing
    cut: list[str] = field(default_factory=list)        # included, but truncated to fit

    @property
    def paths(self) -> set[str]:
        return {doc["path"] for doc in self.docs if doc["role"] != "selection"}


def _readable(text: str) -> str:
    """The note as a reader sees it — no frontmatter, no derived summary, no meeting fence."""
    body = source_text(body_after_frontmatter(text))
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def section_of(text: str, heading: str) -> str | None:
    """The body under the first heading named `heading`, up to the next heading at its level
    or above. None when the note has no such section, or it is empty."""
    want = heading.strip().lstrip("#").strip().casefold()
    # SLIM's own summary is blanked first (it may carry a "Notes" heading of its own) and the
    # `slim-meeting` fence around a recording's sections comes off, as ingest does; then a
    # `# comment` inside a code fence is code, not a heading — as in `chunk.chunk_markdown`.
    lines = source_text(text).splitlines(keepends=True)
    fenced, in_code = [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
            fenced.append(True)
        else:
            fenced.append(in_code)
    for i, line in enumerate(lines):
        match = None if fenced[i] else HEADING.match(line.rstrip("\n"))
        if not match or match.group(2).strip().casefold() != want:
            continue
        level, out = len(match.group(1)), []
        for j in range(i + 1, len(lines)):
            nxt = None if fenced[j] else HEADING.match(lines[j].rstrip("\n"))
            if nxt and len(nxt.group(1)) <= level:
                break
            out.append(lines[j])
        body = "".join(out).strip("\n")
        return body if body.strip() else None
    return None


def _note_date(path: Path, fm: dict) -> str:
    for key in ("recorded_at", "date", "created_time"):
        if fm.get(key):
            return str(fm[key])
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _passes(fm: dict, wanted: dict[str, list[str]]) -> bool:
    for key, options in wanted.items():
        have = fm.get(key)
        values = {str(v).casefold() for v in (have if isinstance(have, list) else [have]) if v is not None}
        if not values & {option.casefold() for option in options}:
            return False
    return True


def resolve_link(con, vault: Path, name: str, near: str) -> str | None:
    """`[[name]]` to a vault path the way Obsidian reads it: an exact path first, else a note
    with that file name, the open note's folder first, then the shortest path."""
    target = name.strip()
    if not target:
        return None
    if not target.lower().endswith(".md"):
        target += ".md"
    if edits_mod.note_path(target) and (Path(vault) / target).is_file():
        return target
    stem = PurePosixPath(target).name.casefold()
    rows = con.execute("SELECT path FROM sources WHERE deleted=0 AND path LIKE ?",
                       (f"%{PurePosixPath(target).name}",)).fetchall()
    want = target.casefold()
    # `[[work/budget]]` must not match `homework/budget.md`: a partial path ends at a `/`.
    candidates = [row["path"] for row in rows
                  if PurePosixPath(row["path"]).name.casefold() == stem
                  and (row["path"].casefold() == want or row["path"].casefold().endswith("/" + want))]
    parent = PurePosixPath(near).parent.as_posix()
    candidates.sort(key=lambda path: (PurePosixPath(path).parent.as_posix() != parent, len(path), path))
    return next((path for path in candidates if (Path(vault) / path).is_file()), None)


def gather(con, vault: Path, source: dict, *, skill: skills_mod.Skill | None = None,
           links_from: str = "", selection: str = "") -> Pack:
    """Build the input for a skill, in code. Priority: the selection, the open note,
    each [[link]] in `links_from`, then the folder's notes newest first while the budget lasts."""
    vault = Path(vault)
    pack, budget = Pack(), INPUT_BUDGET_CHARS
    open_path = source["path"]
    wants = skill.input if skill else "note"
    section = skill.section if skill else ""
    selection = (selection or "").strip()[:MAX_SELECTION_CHARS]

    def add(path: str, text: str, role: str, date: str = "", *, may_cut: bool = False) -> bool:
        nonlocal budget
        if len(text) > budget:
            if not may_cut or budget < 1000:
                pack.dropped.append(path)
                return False
            text = text[:budget] + "\n[… cut to fit]"
            pack.cut.append(path)
        budget -= len(text)
        pack.docs.append({"path": path, "title": PurePosixPath(path).stem, "text": text,
                          "role": role, "date": date})
        return True

    def read(path: str) -> str | None:
        try:
            return (vault / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    if wants == "selection" and not selection:
        raise CopilotError(f"/{skill.name} works on selected text: select some in the note first")
    if selection:
        add(open_path, selection, "selection")
    if wants != "folder":
        text = read(open_path)
        if text is None:
            raise CopilotError("the open note could not be read; refresh the sidebar")
        body = section_of(text, section) if section else text
        if body is None:
            raise CopilotError(f"this note has no \"{section}\" section")
        add(open_path, _readable(body) if not section else body, "open", may_cut=True)

    for name in dict.fromkeys(match.group(1).strip() for match in _LINK.finditer(links_from)):
        path = resolve_link(con, vault, name, open_path)
        if path is None:
            pack.missing.append(name)
        elif path not in pack.paths:
            text = read(path)
            if text is None:
                pack.missing.append(name)
            else:
                add(path, _readable(text), "linked")

    if wants == "folder":
        folder = PurePosixPath(open_path).parent
        if folder.as_posix() == ".":
            raise CopilotError("the open note is not in a folder")
        found = []
        for file in sorted((vault / folder).glob("*.md")):
            text = read((folder / file.name).as_posix())
            if text is None:
                continue
            fm, _skip = parse_frontmatter(text)
            if not _passes(fm, skill.filter):
                continue
            path = (folder / file.name).as_posix()
            body = section_of(text, section) if section else _readable(text)
            if body is None:
                pack.skipped.append(path)
                continue
            found.append((_note_date(file, fm), path, body))
        if not found and not pack.skipped:
            wanted = "; ".join(f"{k}: {' or '.join(v)}" for k, v in skill.filter.items()) or "any"
            raise CopilotError(f"no notes in {folder.as_posix()}/ match ({wanted})")
        kept = []
        for date, path, body in sorted(found, reverse=True):          # newest first
            if path in pack.paths:
                continue
            if len(body) <= budget:
                budget -= len(body)
                kept.append((date, path, body))
            else:
                pack.dropped.append(path)
        for date, path, body in sorted(kept):                         # read oldest first
            pack.docs.append({"path": path, "title": PurePosixPath(path).stem, "text": body,
                              "role": "folder", "date": date[:10]})
        if not kept and any(path not in pack.paths for _date, path, _body in found):
            raise CopilotError("the matching notes are too long to read together")
        if not kept:
            raise CopilotError(f"none of the matching notes in {folder.as_posix()}/ has a "
                               f"\"{section}\" section")
    return pack



def pack_inputs(pack: Pack) -> dict:
    return {"used": [doc["path"] for doc in pack.docs if doc["role"] != "selection"],
            "selection": any(doc["role"] == "selection" for doc in pack.docs),
            "dropped": pack.dropped, "skipped": pack.skipped, "missing": pack.missing,
            "cut": pack.cut}


def open_note_images(vault: Path, source: dict, evidence: list[dict]) -> tuple[list[tuple[str, str]], list[dict]]:
    """The open note's images: those in its passages in the pack, then the rest in note order.

    ⚠ The pack ORDERS the images, never excludes them: an equation that exists only as a
    screenshot matches no question, so its passage is the one retrieval skips (measured
    2026-09-22). A sibling note's images stay out: they are context, and every image costs
    seconds of prefill."""
    packed = [item["text"] for item in evidence if item.get("source_id") == source["id"]]
    try:
        packed.append((vault / source["path"]).read_text(encoding="utf-8"))
    except OSError:
        pass
    return note_images.load("\n".join(packed), vault, note_rel=source["path"],
                            limit=MAX_NOTE_IMAGES)
