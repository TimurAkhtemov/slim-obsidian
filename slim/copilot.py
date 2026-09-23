"""The open-note copilot: gather the notes around the one they have open, run one turn against
them, and name which notes the answer used. Called by `chat.py`'s sidebar routes."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from . import config as config_mod, edits as edits_mod, embed as embed_mod, ingest as ingest_mod
from . import copilot_images, llm, search as search_mod, skills as skills_mod, trace
from .chunk import (HEADING, body_after_frontmatter, parse_frontmatter, strip_derived_blocks,
                    strip_meeting_wrapper)

# The persona is short on purpose: ~3,800 words of prompt read as lobotomized (2026-07-31), and
# an 84-word one that opened with "answer from the numbered evidence" read as a citation bot
# (2026-09-01). The notes pack rides in the SYSTEM message, so their turn is just their words — a
# pack inside the user turn made every question a retrieval task.
PERSONA = r"""You are SLIM, the assistant inside this Obsidian vault. You are talking with its owner about
the note they have open, and about anything else they bring up.

Talk like a sharp, friendly tutor and colleague. Explain from first principles at their level,
use the examples in their own notes when they fit, and be direct. Use your general knowledge
freely — the notes are context, not a limit. When you draw on a note, mention it by its title
in passing ("your Module 2 notes call this…"); never say "evidence" or "the provided notes".
If a note contradicts what you know, say so. When it helps, close with one short check
question or a next step; skip it for a quick lookup.

Write Markdown. Write math as $inline$ or $$display$$, never \( \) or \[ \]. Read attached
screenshots directly; for an equation, state what you see, name any symbol you are unsure of,
then explain."""

# One of these follows the persona. Ask is the default; Edit is the sidebar's toggle, or a skill
# whose output is a change. The block format is parsed by `edits.parse`, never trusted: code
# decides which files a block may touch and applies it itself.
ASK_RIDER = """In Ask mode you cannot change files. If they ask you to change a note, give them the text to
paste, and mention that Edit mode can apply it for them."""

EDIT_RIDER = """You can change notes. The owner reviews every change and applies it with one click; nothing
changes until they accept. First say in a sentence or two what you are changing. Then write one
block per change, exactly like this:

FILE: <the note's path, as shown after === below>
<<<<<<< SEARCH
<lines copied exactly from that note>
=======
<the lines that replace them>
>>>>>>> REPLACE

Copy SEARCH lines exactly, with enough of them to be unique, and change only what needs
changing. An empty SEARCH adds the text to the end of the note. To create a note, write a new
path in an existing folder (the open note's folder is `{folder}/`) with an empty SEARCH. A
recording's frontmatter, its slim-meeting block and its transcript cannot be changed."""

# Sources are decided AFTER the answer by a bounded structured call, not by a contract inside
# the prose. Measured 2026-09-01: [n] markers made a citation bot; a trailing `Notes used:` line
# arrived 4 times in 12 at temperature 0.7. The pack carries no numbers, so prose cannot cite;
# the follow-up shares the answer call's prefix, so the KV cache covers all but a few tokens.
SOURCES_QUESTION = """Which of these notes did your reply actually draw on? Answer with their numbers only, or an
empty list if you answered from general knowledge."""
SOURCES_NUM_PREDICT = 64


# 1400 clipped a worked example mid-sentence, and Quick is the mode nearly every turn uses.
QUICK_NUM_PREDICT = 2400
# A proposal can carry a whole new note or several rewritten sections: room for the text AND,
# in Deep, the thinking before it. A block cut off mid-way is refused, not half-applied.
PROPOSE_NUM_PREDICT = {"quick": 8192, "deep": 16384}
# Skills and edits read WHOLE notes, chosen by code. Half the resident window in chars/4, so
# the answer and the thinking keep room and every call still uses num_ctx=RESIDENT_CTX (a
# smaller bespoke window is a second runner). What does not fit is named, never dropped silently.
INPUT_BUDGET_CHARS = llm.RESIDENT_CTX * 4 // 2
MAX_SELECTION_CHARS = 20_000
_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
# Thinking tokens bill against num_predict but return in a separate field (llm.chat's
# docstring). Deep mode must budget for the reasoning AND the answer, or a long think returns
# content="" and reads as "the model can't answer": one benign sample spent ~3k of 4096.
DEEP_NUM_PREDICT = 4096 * 3
# While the model reasons no content reaches the client, so its Stop (a destroyed socket)
# goes unnoticed until content starts. Re-emitting the stage this often is the write that
# raises BrokenPipe inside the Ollama request and frees llm._CALL_LOCK.
HEARTBEAT_S = 1.0
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


def system_message(evidence: list[dict]) -> str:
    # ⚠ NO PROFILE (since 2026-09-03). `Profile/me.md` used to ride here, and measured
    # 2026-09-01 across three live samples the model bent every answer back to the profile —
    # a career-pivot flourish inside an EQUATION explanation, every time. The persona and the
    # notes are the whole system message.
    parts = [PERSONA, ASK_RIDER]
    rendered = "\n\n".join(
        f"### {item['title']}"
        + (f" § {item['heading']}" if item.get("heading") else "")
        + f"\n{item['text']}" for item in evidence)
    parts.append("Notes in view (the open note first):\n"
                 + (rendered or "(nothing indexed for this note yet)"))
    return "\n\n".join(parts)


@dataclass
class Pack:
    """What a skill or an edit reads: whole notes, in order, and what did not make it in."""
    docs: list[dict] = field(default_factory=list)      # {path, title, text, role, date}
    dropped: list[str] = field(default_factory=list)    # did not fit the budget
    skipped: list[str] = field(default_factory=list)    # had no such section
    missing: list[str] = field(default_factory=list)    # [[links]] that resolve to nothing
    cut: list[str] = field(default_factory=list)        # included, but truncated to fit

    @property
    def paths(self) -> set[str]:
        return {doc["path"] for doc in self.docs if doc["role"] != "selection"}


def _readable(text: str, *, raw: bool) -> str:
    """A proposal needs the file as it is, so SEARCH lines can match; otherwise the note as a
    reader sees it — no frontmatter, no derived summary, no meeting fence."""
    if raw:
        return text
    body = strip_meeting_wrapper(strip_derived_blocks(body_after_frontmatter(text)))
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def section_of(text: str, heading: str) -> str | None:
    """The body under the first heading named `heading`, up to the next heading at its level
    or above. None when the note has no such section, or it is empty."""
    want = heading.strip().lstrip("#").strip().casefold()
    # SLIM's own summary is blanked first (it may carry a "Notes" heading of its own) and the
    # `slim-meeting` fence around a recording's sections comes off, as ingest does; then a
    # `# comment` inside a code fence is code, not a heading — as in `chunk.chunk_markdown`.
    lines = strip_meeting_wrapper(strip_derived_blocks(text)).splitlines(keepends=True)
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
           links_from: str = "", selection: str = "", raw: bool = False) -> Pack:
    """Build the input for a skill or an edit, in code. Priority: the selection, the open note,
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
        add(open_path, _readable(body, raw=raw) if not section else body, "open", may_cut=True)

    for name in dict.fromkeys(match.group(1).strip() for match in _LINK.finditer(links_from)):
        path = resolve_link(con, vault, name, open_path)
        if path is None:
            pack.missing.append(name)
        elif path not in pack.paths:
            text = read(path)
            if text is None:
                pack.missing.append(name)
            else:
                add(path, _readable(text, raw=raw), "linked")

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
            body = section_of(text, section) if section else _readable(text, raw=raw)
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


def pack_message(pack: Pack, *, proposing: bool, folder: str) -> str:
    labels = {"open": "the open note", "selection": "the text they selected in this note",
              "linked": "a note they linked"}
    rider = EDIT_RIDER.format(folder=folder) if proposing else ASK_RIDER
    rendered = "\n\n".join(
        f"=== {doc['path']} · {labels.get(doc['role']) or doc['date'] or 'a note in this folder'} ===\n"
        f"{doc['text']}" for doc in pack.docs)
    return "\n\n".join([PERSONA, rider, "Notes in view:\n\n" + (rendered or "(none)")])


def pack_inputs(pack: Pack) -> dict:
    return {"used": [doc["path"] for doc in pack.docs if doc["role"] != "selection"],
            "selection": any(doc["role"] == "selection" for doc in pack.docs),
            "dropped": pack.dropped, "skipped": pack.skipped, "missing": pack.missing,
            "cut": pack.cut}


def note_choices(evidence: list[dict]) -> list[dict]:
    """One numbered choice per NOTE, in pack order: sources are shown per note, not per fragment."""
    choices, seen = [], {}
    for item in evidence:
        key = item.get("source_id") or item["path"]
        if key in seen:
            continue
        seen[key] = True
        choices.append({"n": len(choices) + 1, "key": key, "title": item["title"], "path": item["path"]})
    return choices


def notes_used(messages: list[dict], answer: str, evidence: list[dict]) -> tuple[list[dict], dict]:
    """Ask the same model, on the same prefix, which notes the answer drew on.

    Returns (citations, stats). A refused or malformed call yields no sources — the answer
    stands either way; sources decorate it.
    """
    choices = note_choices(evidence)
    if not choices or not answer:
        return [], {}
    listing = "\n".join(f"{choice['n']}. {choice['title']}" for choice in choices)
    schema = {
        "type": "object",
        "properties": {"notes": {"type": "array", "maxItems": len(choices),
                                 "items": {"type": "integer", "minimum": 1, "maximum": len(choices)}}},
        "required": ["notes"], "additionalProperties": False,
    }
    try:
        decision, stats = llm.chat_json(
            [*messages, {"role": "assistant", "content": answer},
             {"role": "user", "content": f"{SOURCES_QUESTION}\n{listing}"}],
            schema, model=llm.MODEL, num_ctx=llm.RESIDENT_CTX,
            num_predict=SOURCES_NUM_PREDICT, temperature=0.0, think=False, timeout=60)
        picked = {int(n) for n in decision.get("notes", []) if isinstance(n, int)}
    except (llm.LLMError, KeyError, TypeError, ValueError, AttributeError):
        return [], {}
    keys = {choice["key"] for choice in choices if choice["n"] in picked}
    cited = [item for item in evidence if (item.get("source_id") or item["path"]) in keys]
    return cited, stats


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
    return copilot_images.note_images("\n".join(packed), vault, note_rel=source["path"],
                                      limit=MAX_NOTE_IMAGES)


def generate_answer(question: str, history: list[dict], evidence: list[dict], mode: str,
                    *, images=None, note_images=None, on_stage=None, on_delta=None,
                    system: str | None = None, cited: list[dict] | None = None,
                    proposing: bool = False) -> dict:
    """Generate one answer; reasoning is private and only content deltas are surfaced.

    TWO depths, and THEY pick (2026-09-03). Auto spent a whole extra classifier call on a choice
    that is one click in the sidebar, and it resolved to Quick nearly every time.

    A skill or an edit passes its own `system` (whole notes chosen by code) and `cited`: code
    already knows which notes it read, so the sources call is skipped.
    """
    if mode not in {"quick", "deep"}:
        raise CopilotError("reasoning mode must be quick or deep")
    stage = on_stage or (lambda _name: None)
    delta = on_delta or (lambda _text: None)
    stage("Thinking deeply" if mode == "deep" else "Writing")
    transcript = []
    for turn in history[-12:]:
        if not isinstance(turn, dict) or turn.get("role") not in {"you", "slim"}:
            continue
        message = {
            "role": "user" if turn["role"] == "you" else "assistant",
            "content": str(turn.get("text") or "")[:4000],
        }
        if message["role"] == "user" and turn.get("images"):
            message["images"] = list(turn["images"])
        transcript.append(message)
    if note_images:
        # Their own message, so a screenshot they pasted is never mistaken for the note's. The
        # names are the embeds as the note spells them, which is how the model places them.
        names = ", ".join(f"`{name}`" for name, _data in note_images)
        transcript.append({"role": "user",
                           "content": "(These are the images embedded in the open note itself, "
                                      f"not ones I attached: {names}.)",
                           "images": [data for _name, data in note_images]})
    current = {"role": "user", "content": question}
    if images:
        current["images"] = list(images)
    messages = [{"role": "system", "content": system or system_message(evidence)},
                *transcript, current]
    started = time.time()
    last_beat = [started]

    def heartbeat(_n_chars):
        if time.time() - last_beat[0] >= HEARTBEAT_S:
            last_beat[0] = time.time()
            stage("Thinking deeply")

    text, stats = llm.chat_stream(
        messages, model=llm.MODEL, num_ctx=llm.RESIDENT_CTX,
        num_predict=(PROPOSE_NUM_PREDICT[mode] if proposing
                     else DEEP_NUM_PREDICT if mode == "deep" else QUICK_NUM_PREDICT),
        temperature=llm.REPLY_TEMPERATURE, think=mode == "deep",
        timeout=600, on_delta=delta, on_thinking=heartbeat)
    answer = (text or "").strip()
    if not answer and stats.get("thinking_ate_the_budget"):
        raise CopilotError("the model spent its whole output budget thinking and wrote no "
                           "answer — ask again in Quick mode, or ask a narrower question")
    if not answer:
        raise CopilotError("the local model returned no answer")
    if cited is not None:
        citations, sources_stats = cited, {}
    else:
        if evidence:
            stage("Noting sources")
        citations, sources_stats = notes_used(messages, answer, evidence)
    return {
        "answer": answer,
        "mode": mode,
        "citations": citations,
        "verification": {"truncated": bool(stats.get("output_truncated"))},
        "timing": {"total_ms": round((time.time() - started) * 1000),
                   "answer": stats, "sources": sources_stats},
    }


def run_turn(con, source_id: str, question: str, history: list[dict], mode: str,
             *, images=None, on_stage=None, on_delta=None, edit: bool = False,
             selection: str = "", vault: Path | None = None) -> dict:
    """One sidebar turn. A `/command` naming a skill, or Edit mode, reads whole notes chosen by
    code (`gather`); anything else is today's retrieval turn. Edit mode and a skill whose output
    is a change end with a proposal the plugin shows for Accept/Reject — nothing is written here."""
    vault = Path(vault or config_mod.VAULT)
    source_row = con.execute(
        "SELECT id, path, title, type, current_hash FROM sources "
        "WHERE id=? AND deleted=0", (source_id,)).fetchone()
    if source_row is None:
        raise CopilotError("the note is no longer indexed; refresh the sidebar")
    source = dict(source_row)
    loaded = skills_mod.load(vault)
    found = skills_mod.invocation(question, loaded) if question.startswith("/") else None
    skill, args = found or (None, "")
    if skill and skill.mode in ("quick", "deep"):
        mode = skill.mode                      # the skill's author chose its depth
    proposing = edit or bool(skill and skill.proposes)
    # A /command in an earlier turn is its whole prompt to the model: "/quiz" alone never told
    # the answering turn how to grade.
    history = [{**turn, "text": skills_mod.expand(turn.get("text") or "", loaded)}
               if turn.get("role") == "you" else turn for turn in history]
    # Machine facts only (no prose, no reasoning): the thread JSON keeps completed turns,
    # so this is the one record of a refused or stopped turn.
    entry = {"question": question, "source": source["path"], "source_id": source_id,
             "mode": mode, "history_turns": len(history),
             "images": len(images or []),
             "history_images": sum(len(turn.get("images") or []) for turn in history),
             "skill": skill.name if skill else None, "edit": proposing,
             "selection_chars": len(selection or "")}
    started = time.time()
    try:
        if skill and skill.error:
            raise CopilotError(f"/{skill.name} cannot run: {skill.error} "
                               f"({skills_mod.VAULT_DIR}/{skill.name}.md)")
        unknown = re.match(r"^/([a-z0-9][a-z0-9-]*)(?:\s|$)", question)
        if unknown and not skill:
            raise CopilotError(f"there is no /{unknown.group(1)} skill; type / to see them")
        if skill or proposing:
            pack = gather(con, vault, source, skill=skill, links_from=args if skill else question,
                          selection=selection, raw=proposing)
            folder = PurePosixPath(source["path"]).parent.as_posix()
            cited = [{"n": i + 1, "path": doc["path"], "title": doc["title"]}
                     for i, doc in enumerate(d for d in pack.docs if d["role"] != "selection")]
            # No retrieval orders a skill's images, so they come in note order.
            shown, skipped = open_note_images(vault, source, [])
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(
                skills_mod.render_prompt(skill, args) if skill else question, history, [], mode,
                images=images, note_images=shown, on_stage=on_stage, on_delta=on_delta,
                system=pack_message(pack, proposing=proposing, folder=folder),
                cited=cited, proposing=proposing)
            result["inputs"] = pack_inputs(pack)
            levels = []
            # An overflowing prompt loses its START — the notes and the edit instructions —
            # silently. Nothing built on that is offered for Accept.
            if proposing and result["timing"]["answer"].get("ctx_saturated"):
                raise CopilotError("these notes are too long for one request; ask about fewer notes")
            if proposing:
                prose, blocks = edits_mod.parse(result["answer"])
                # What a skill may touch is its frontmatter's word, enforced here: a note in the
                # pack can carry instructions the model obeys (measured live, 2026-09-23).
                result["proposal"] = edits_mod.propose(
                    vault, blocks, in_view=pack.paths,
                    creates_only=bool(skill and skill.output == "new-note"),
                    within=(source["path"], selection) if skill and skill.input == "selection" else None)
                if not blocks:
                    result["proposal"] = None
                result["answer"] = prose or "Proposed changes are below."
        else:
            limit = 12 if mode == "deep" else 6
            evidence, levels = collect_evidence(con, source, question, limit=limit)
            shown, skipped = open_note_images(vault, source, evidence)
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(
                question, history, evidence, mode, images=images, note_images=shown,
                on_stage=on_stage, on_delta=on_delta)
    except (CopilotError, llm.LLMError) as exc:
        trace.record("copilot", {**entry, "status": "failed", "error": str(exc),
                                 "duration_ms": round((time.time() - started) * 1000)})
        raise
    result.update({"source": source, "retrieval_levels": levels, "skill": entry["skill"],
                   "edit": proposing})
    record = {
        **entry, "status": "answered",
        "cited": sorted({item["path"] for item in result["citations"]}),
        "truncated": result["verification"]["truncated"], "timing": result["timing"],
    }
    if "inputs" in result:
        record["inputs"] = result["inputs"]
    else:
        record.update({"evidence": [item["path"] for item in evidence], "retrieval_levels": levels})
    if result.get("proposal"):
        record["proposal"] = {
            "files": [{"path": item["path"], "kind": item["kind"],
                       "applied": sum(block["ok"] for block in item["blocks"]),
                       "failed": [block["reason"] for block in item["blocks"] if not block["ok"]]}
                      for item in result["proposal"]["files"]],
            "dropped": result["proposal"]["dropped"]}
    trace.record("copilot", record)
    return result
