"""The open-note copilot: gather the notes around the one they have open, run one turn against
them, and name which notes the answer used — or, when the turn edits, carry out the edit one
model call at a time and hand each change to the plugin as it lands. Called by `chat.py`'s
sidebar routes."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
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

# One of these follows the persona. A turn answers or edits with EDIT_RIDER, the measured one
# (the prototype's, word for word). A skill that answers in the chat gets ASK_RIDER, and one
# that makes a new note NEW_NOTE_RIDER. What a call may touch is code's, never the rider's
# (`edits.py`).
ASK_RIDER = """In this reply you cannot change files. If they ask you to change a note, give them the text to
paste."""

EDIT_RIDER = """You can edit the open note shown below. Each block in it starts with a handle in brackets:
[b12] is that block. A heading's section (the heading and everything under it) is s plus the
heading's number: s7 is the section under [b7] ## …. A range like b11-b15 covers those blocks and
everything between them. Handles are labels, not text: never write them into the note. Parts marked
[locked] belong to a recording and cannot be changed.

To answer a question, just reply in text and change nothing.
To make one small change, make it right away, pointing at handles. To change more, first make a plan: a one-sentence intro for the owner saying what you will do,
the note's headings in their final order (each with its # marks), and the steps in order.
You will then be given one step at a time; make that step's changes, pointing at handles.
For a small wording fix, use edit_text instead of retyping the block.
Use the replace, edit_text, insert and move tools for a small change and for each step, and the plan tool to start a larger change."""

NEW_NOTE_RIDER = """You can create a new note with the create_note tool: its path, in a folder that already exists
(the open note's folder is `{folder}/`), and its whole text in Markdown. Then call done with one
sentence for the owner saying what you made."""

# Sources are decided AFTER the answer by a bounded structured call, not by a contract inside
# the prose. Measured 2026-09-01: [n] markers made a citation bot; a trailing `Notes used:` line
# arrived 4 times in 12 at temperature 0.7. The pack carries no numbers, so prose cannot cite;
# the follow-up shares the answer call's prefix, so the KV cache covers all but a few tokens.
SOURCES_QUESTION = """Which of these notes did your reply actually draw on? Answer with their numbers only, or an
empty list if you answered from general knowledge."""
SOURCES_NUM_PREDICT = 64


# 1400 clipped a worked example mid-sentence, and Quick is the mode nearly every turn uses.
# It is also the first call's cap in a turn that may edit: that call may be the answer.
QUICK_NUM_PREDICT = 2400
# THE EDITING RUN (`_turn`). Measured on the prototype (2026-10-12, 13 cases, 5 runs each):
# prompt instructions hold this model to no limit, so every limit is code.
MAX_STEPS = 8
MAX_CALLS = 30                   # tool calls in one run
MAX_REFUSALS = 3                 # refused calls in a row end the run
STEP_NUM_PREDICT = 2048          # a step's reply: room for one section's new text
THINK_BUDGET = 8192              # on top of the first call's cap in Deep; thinking counts against it
RUN_OUTPUT_TOKENS = 12_000       # the reply that crosses it is used, then the run stops (one ran to 25k)
STEP_TEMPERATURE = 0.0           # a step, and any call that makes a wanted change, writes at 0.0
STEP_TOOLS = ("replace", "edit_text", "insert", "move")
# THE INTENT GATE. Before a question, one small structured call decides whether the owner wants
# the note changed. It reads its own short prompt — SLIM's last answer and the question, never
# the note — so it costs a few hundred tokens, and the note's handles reach only the edit path.
# "No": today's Ask path (retrieval, no handles, a streamed answer, the sources call). Otherwise
# the edit path, with the tools at STEP_TEMPERATURE: retyping at 0.7 drifted (a plural dropped,
# quotes straightened), and a text-only reply is nudged once. A gate that fails or cannot tell
# counts as a change but is never nudged: a text reply then is the answer. Off, every question
# takes the edit path. FIRST_TEMPERATURE is the edit path's first call when no gate decided,
# and a new note's.
INTENT_GATE = True
GATE_SYSTEM = ("You decide whether the owner's last message asks to change the note they have open. "
               "Reply with JSON only.")
GATE_SCHEMA = {"type": "object", "properties": {"change_note": {"type": "boolean"}}, "required": ["change_note"]}
GATE_NUM_PREDICT = 20
NUDGE = "Use the tools to make the change."
FIRST_TEMPERATURE = llm.REPLY_TEMPERATURE
ONE_REPLY = "(done in one reply)"      # the one step of a run whose first reply made its change
# Skills read WHOLE notes, chosen by code. Half the resident window in chars/4, so
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
# THE WINDOW RULE (`fit_window`): the conversation goes to the model whole, and the resident
# window is its only limit. Ollama has no tokenize endpoint, so tokens are estimated from
# characters. Real copilot prompts rebuilt from the threads and divided by the prompt_tokens
# Ollama reported ran 3.8-4.1 characters per token (2026-10-12, three prompts, template
# included). At 3 the estimate counts more tokens than any of them used, so history goes early
# rather than overflowing; denser text (LaTeX, code) runs lower, and `ctx_saturated` is the
# backstop.
CHARS_PER_TOKEN = 3
# The most an image can cost: the model reads 32-pixel squares (patch 16, merge 2) and every
# image arrives at most 1600 pixels on a side (`copilot_images.NOTE_IMAGE_MAX_EDGE`, and the
# plugin's resize of a paste). A typical screenshot costs ~450 (MAX_NOTE_IMAGES' measurement).
IMAGE_TOKENS = (1600 // 32) ** 2


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
    return "\n\n".join([PERSONA, ASK_RIDER, "Notes in view (the open note first):\n"
                         + (_passages(evidence) or "(nothing indexed for this note yet)")])


def _passages(evidence: list[dict]) -> str:
    return "\n\n".join(
        f"### {item['title']}"
        + (f" § {item['heading']}" if item.get("heading") else "")
        + f"\n{item['text']}" for item in evidence)


def edit_message(path: str, note: edits_mod.Note) -> str:
    """The system message of the edit path, as measured: the persona and EDIT_RIDER, then the open
    note whole, with its handles and the map of its sections."""
    sections = note.sections()
    shown = f"Sections:\n{sections}\n\n" if sections else ""
    return "\n\n".join([PERSONA, EDIT_RIDER,
                         f"Notes in view:\n\n=== {path} · the open note ===\n{shown}{note.render()}"])


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


def pack_message(pack: Pack, rider: str) -> str:
    labels = {"open": "the open note", "selection": "the text they selected in this note",
              "linked": "a note they linked"}
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


def _tokens(message: dict) -> float:
    """A message's tokens as THE WINDOW RULE estimates them."""
    return len(message["content"]) / CHARS_PER_TOKEN + IMAGE_TOKENS * len(message.get("images") or ())


def fit_window(transcript: list[dict], others: list[dict], num_predict: int) -> list[dict]:
    """The newest messages of `transcript`, each whole, that fit the resident window beside
    `others` (the system message, the open note's images, the question) and the reply's
    `num_predict`, which in Deep mode holds the thinking too. When the conversation does not
    fit, its OLDEST messages go, whole: a cut inside one (1,000 characters, never measured)
    kept the start of an answer and dropped its end, where the plan or the offer sits."""
    room = llm.RESIDENT_CTX - num_predict - sum(map(_tokens, others))
    kept = []
    for message in reversed(transcript):
        room -= _tokens(message)
        if room < 0:
            break
        kept.append(message)
    return kept[::-1]


def _conversation(history: list[dict], question: str, images=None,
                  note_images=None) -> tuple[list[dict], list[dict], dict]:
    """The conversation as model messages: the earlier turns, the open note's images in a
    message of their own, and the question with any images they attached."""
    transcript = []
    for turn in history:
        if not isinstance(turn, dict) or turn.get("role") not in {"you", "slim"}:
            continue
        message = {
            "role": "user" if turn["role"] == "you" else "assistant",
            "content": str(turn.get("text") or ""),
        }
        if message["role"] == "user" and turn.get("images"):
            message["images"] = list(turn["images"])
        transcript.append(message)
    embedded = []
    if note_images:
        # Their own message, so a screenshot they pasted is never mistaken for the note's. The
        # names are the embeds as the note spells them, which is how the model places them.
        names = ", ".join(f"`{name}`" for name, _data in note_images)
        embedded.append({"role": "user",
                         "content": "(These are the images embedded in the open note itself, "
                                    f"not ones I attached: {names}.)",
                         "images": [data for _name, data in note_images]})
    current = {"role": "user", "content": question}
    if images:
        current["images"] = list(images)
    return transcript, embedded, current


def generate_answer(question: str, history: list[dict], evidence: list[dict], mode: str,
                    *, images=None, note_images=None, on_stage=None, on_delta=None,
                    system: str | None = None, cited: list[dict] | None = None) -> dict:
    """Generate one answer; reasoning is private and only content deltas are surfaced. A skill
    that answers in the chat runs here.

    TWO depths, and THEY pick (2026-09-03). Auto spent a whole extra classifier call on a choice
    that is one click in the sidebar, and it resolved to Quick nearly every time.

    A skill passes its own `system` (whole notes chosen by code) and `cited`: code already
    knows which notes it read, so the sources call is skipped.
    """
    if mode not in {"quick", "deep"}:
        raise CopilotError("reasoning mode must be quick or deep")
    stage = on_stage or (lambda _name: None)
    delta = on_delta or (lambda _text: None)
    stage("Thinking deeply" if mode == "deep" else "Writing")
    transcript, embedded, current = _conversation(history, question, images, note_images)
    num_predict = DEEP_NUM_PREDICT if mode == "deep" else QUICK_NUM_PREDICT
    system_turn = {"role": "system", "content": system or system_message(evidence)}
    recent = fit_window(transcript, [system_turn, *embedded, current], num_predict)
    messages = [system_turn, *recent, *embedded, current]
    started = time.time()
    last_beat = [started]

    def heartbeat(_n_chars):
        if time.time() - last_beat[0] >= HEARTBEAT_S:
            last_beat[0] = time.time()
            stage("Thinking deeply")

    text, stats = llm.chat_stream(
        messages, model=llm.MODEL, num_ctx=llm.RESIDENT_CTX, timeout=600,
        temperature=llm.REPLY_TEMPERATURE, num_predict=num_predict, think=mode == "deep",
        on_delta=delta, on_thinking=heartbeat)
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
        "history_dropped": len(transcript) - len(recent),
        "timing": {"total_ms": round((time.time() - started) * 1000),
                   "answer": stats, "sources": sources_stats},
    }


# --- one chat that answers or edits ---------------------------------------------------------

class _Stopped(Exception):
    """A model call failed twice, the run is out of budget, or the owner stopped it: it ends
    where it is, and what landed stays."""


@dataclass
class _Reply:
    content: str
    calls: list[dict]
    malformed: list[str]
    message: dict                       # the assistant message, appended to the conversation
    stats: dict


# A JSON decode turns `\frac \beta \text \rho` into a control character and the command's
# rest. In a tool call's text that is repaired only inside math, and only before a letter: a
# real tab indents a nested list beside a formula, and `summarize._repair` (every tab, once a
# `$` is anywhere) would write it out as `\t`.
_MATH = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+?\$")
_MANGLED = re.compile(r"[\t\f\b\r](?=[A-Za-z])")
_COMMAND = {"\t": "\\t", "\f": "\\f", "\b": "\\b", "\r": "\\r"}


def _unmangle(text: str) -> str:
    return _MATH.sub(lambda math: _MANGLED.sub(lambda c: _COMMAND[c.group()], math.group()), text)


def _repair_args(value):
    """_unmangle over a tool call's arguments, whatever their shape."""
    if isinstance(value, str):
        return _unmangle(value)
    if isinstance(value, list):
        return [_repair_args(v) for v in value]
    if isinstance(value, dict):
        return {k: _repair_args(v) for k, v in value.items()}
    return value


def _call(messages: list[dict], offered: list[str], *, think: bool, temperature: float,
          num_predict: int, on_part=None, should_stop=None) -> _Reply:
    """One model call with the `offered` tools; its arguments repaired by `_repair_args`. It
    streams: `on_part` hears its tool calls as they decode (`llm._Reader`)."""
    msg, stats = llm.chat_tools(messages, edits_mod.ollama_tools(offered), think=think,
                                temperature=temperature, num_predict=num_predict,
                                on_part=on_part, should_stop=should_stop)
    calls = [{"name": c["name"], "arguments": _repair_args(c.get("arguments") or {})}
             for c in msg["tool_calls"]]
    # Ollama's own parser can leave a tool call in the text: unreadable, and never an answer.
    malformed = ["a tool call came back as text"] if "<tool_call>" in msg["content"] else []
    message = {"role": "assistant", "content": msg["content"]}
    if msg.get("raw_tool_calls"):
        message["tool_calls"] = msg["raw_tool_calls"]
    return _Reply(msg["content"], calls, malformed, message, stats or {})


def _results(results: list[str], instruction: str) -> list[dict]:
    """What code says back after a reply: a tool message per call, the next instruction after
    the last; a user message when there were no calls."""
    if not results:
        return [{"role": "user", "content": instruction}]
    out = [{"role": "tool", "content": result} for result in results]
    out[-1]["content"] += "\n\n" + instruction
    return out


def _summary(call: dict) -> str:
    args = call.get("arguments")
    return str((args if isinstance(args, dict) else {}).get("summary") or "").strip()


def _unreadable(malformed: list[str]) -> str:
    return f"refused: your tool call could not be read ({'; '.join(malformed)}); write it again"


# What the measured outlines carried besides headings: a list sent as one quoted string with
# literal \n between the headings, notes like "(New Section)" after a heading, and the rider's
# placeholder.
_LITERAL_NEWLINE = re.compile(r"\\n(?=(?:\\n)*\s*#)")
_PLACEHOLDER = re.compile(r"#*\s*<.*>")


def _outline(args: dict) -> list[str] | None:
    raw = args.get("outline")
    if isinstance(raw, str):
        raw = [raw]
    found = []
    for item in raw if isinstance(raw, list) else []:
        for line in _LITERAL_NEWLINE.sub("\n", str(item)).split("\n"):
            heading = edits_mod.ANNOTATION.sub(
                "", edits_mod.strip_labels(line.strip().strip("\"'").strip()).strip())
            if heading.startswith("#") and not _PLACEHOLDER.fullmatch(heading):
                found.append(heading)
    return found or None


_MARKER = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def _steps(raw) -> list[str]:
    """A plan's steps as the model sent them: a list, or one string of lines (as an outline can
    come), each without its list marker; empty ones dropped."""
    items = raw if isinstance(raw, list) else str(raw).split("\n") if raw else []
    found = (_MARKER.sub("", str(item).strip(), count=1).strip() for item in items)
    return [step for step in found if step]


def _against(note: edits_mod.Note, outline: list[str]) -> tuple[list[str], list[str]]:
    """The note's headings and the outline, both without a recording's locked headings:
    no step can change those, so they are never asked for."""
    locked = {edits_mod.heading_key(h) for b in note.blocks if b.locked for h in edits_mod.headings(b.text)}
    now = [h for b in note.blocks if not b.locked for h in edits_mod.headings(b.text)]
    return now, [h for h in outline if edits_mod.heading_key(h) not in locked]


# An outline heading and a note heading this alike, by their sets of words, are the same heading.
SAME_HEADING_WORDS = 0.6


def _same(a: tuple[int, str], b: tuple[int, str]) -> bool:
    """Two headings (as `heading_key`s) are the same when their words are equal, one starts with
    the other's words ("## Historical Evolution" and "… & Drivers"), or the words overlap by
    SAME_HEADING_WORDS: the model shortens and reorders its own outline's headings, and writes
    them at its own levels. Level is never compared."""
    x, y = a[1].split(), b[1].split()
    if not x or not y:
        return False
    shorter, longer = sorted((x, y), key=len)
    return longer[:len(shorter)] == shorter or len(set(x) & set(y)) / len(set(x) | set(y)) >= SAME_HEADING_WORDS




def _match(now: list[str], want: list[str]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """The outline is met when its headings appear among the note's in the same order (`_same`);
    others between them are fine. Returns the matched (outline, note) index pairs, then the
    outline's misses: those the note lacks, and those it has out of order (outside the longest
    run of outline headings the note has in order)."""
    a, b = [edits_mod.heading_key(h) for h in want], [edits_mod.heading_key(h) for h in now]
    longest = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            longest[i][j] = (longest[i + 1][j + 1] + 1 if _same(a[i], b[j])
                             else max(longest[i + 1][j], longest[i][j + 1]))
    pairs, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if _same(a[i], b[j]):
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif longest[i + 1][j] >= longest[i][j + 1]:
            i += 1
        else:
            j += 1
    kept = {k for k, _ in pairs}
    missed = [k for k in range(len(a)) if k not in kept]
    lacks = [k for k in missed if not any(_same(a[k], key) for key in b)]
    return pairs, lacks, [k for k in missed if k not in lacks]


def _outline_details(now: list[str], want: list[str]) -> dict:
    """How the note's headings stand against the outline, for the trace: outline headings it
    lacks, has out of order, and has at another level."""
    pairs, missing, misplaced = _match(now, want)
    level = lambda h: edits_mod.heading_key(h)[0]
    return {"missing": len(missing), "out_of_order": len(misplaced),
            "other_level": sum(level(want[i]) != level(now[j]) for i, j in pairs)}


_ITEM = re.compile(r"^(?:[-*]|\d+[.)])\s+(.+)$")


def _listed_plan(content: str) -> tuple[str, list[str]] | None:
    """A plan the model wrote as prose instead of calling `plan`, beside its first change:
    (intro, steps), from the last run of two or more list items and the reply's first sentence
    that is not one. A line indented under an item belongs to it; a reply may list its idea and
    then its steps, and the steps come last. None when there is no such list."""
    runs, run, intro = [], [], ""
    for line in content.splitlines():
        item = _ITEM.match(line)
        if item:
            run.append(item.group(1).strip())
        elif line.strip() and not (run and line[:1] in " \t"):
            if run:
                runs, run = runs + [run], []
            intro = intro or line.strip()
    runs = [r for r in runs + [run] if len(r) >= 2]
    if not runs:
        return None
    first = re.match(r"(.+?[.!?])(?:\s|$)", intro)
    return (first.group(1) if first else intro), runs[-1][:MAX_STEPS]


class _Drafts:
    """An insert's or replace's text in the note while the model writes it: a draft at the place
    the call names, from the moment that place is known, with what was written before it. Code
    only places it, in UTF-16 units of the run's draft as a change's splices are; it is never the
    note's text. The change that lands names the draft it replaces; a call refused or never
    applied ends its draft."""
    PLACES = {"insert": ("after", "before"), "replace": ("target",)}

    def __init__(self, note: edits_mod.Note, only: set[str] | None, on_edit):
        self.note, self.only, self.on_edit = note, only, on_edit
        self.reply, self.calls, self.open, self.failed = 0, {}, [], 0

    def next_reply(self) -> None:
        self.end()
        self.reply, self.calls = self.reply + 1, {}

    def part(self, kind: str, index: int, name, value) -> None:
        """`on_part` for a model call (`llm._Reader`). A draft is only ever shown: one that fails is
        counted for the trace, and the call goes on."""
        try:
            self._part(kind, index, name, value)
        except Exception:                   # noqa: BLE001
            self.failed += 1

    def _part(self, kind: str, index: int, name, value) -> None:
        call = self.calls.setdefault(index, {"tool": None, "text": [], "id": None})
        if kind == "call":
            call["tool"] = name
        elif call["tool"] not in self.PLACES:
            return
        elif kind == "value" and name in self.PLACES[call["tool"]] and call["id"] is None:
            self._start(index, call, name, value)
        elif kind == "text" and name == "markdown":
            call["text"].append(value)
            if call["id"]:
                self.on_edit("draft_text", {"id": call["id"], "text": value})

    def _start(self, index: int, call: dict, name: str, value) -> None:
        note = self.note
        try:
            if not isinstance(value, str) or not value.strip():
                return
            if call["tool"] == "insert":
                i, j = note.span(note.name(value))
                note.check_insert(j if name == "after" else i)
                start = end = note.offset(j if name == "after" else i)
            else:
                i, j, _target, _said = note.pick(note.name(value), self.only)
                start, end = note.offset(i), note.offset(j)
        except edits_mod.ToolError:
            return                              # no such place: the call will say so when applied
        text = note.text()
        call["id"] = f"d{self.reply}.{index}"
        self.open.append(call["id"])
        self.on_edit("draft", {"id": call["id"], "from": edits_mod.u16(text[:start]),
                               "to": edits_mod.u16(text[:end]), "text": "".join(call["text"])})

    def of(self, index: int) -> str | None:
        return (self.calls.get(index) or {}).get("id")

    def landed(self, draft: str) -> None:
        if draft in self.open:
            self.open.remove(draft)

    def end(self) -> None:
        for draft in self.open:
            self.on_edit("draft_end", {"id": draft})
        self.open = []


def _intent(question: str, history: list[dict]) -> tuple[bool | None, dict]:
    """Does the owner's last message ask to change the open note? (True/False, or None when the
    gate fails or cannot tell; and the call's stats.) Its prompt is its own: GATE_SYSTEM, SLIM's
    last answer whole (a plan to "make" lives there), and the question; no note, no tools."""
    messages = [{"role": "system", "content": GATE_SYSTEM}]
    last = next((t for t in reversed(history) if t.get("role") == "slim" and t.get("text")), None)
    if last:
        messages.append({"role": "assistant", "content": str(last["text"])})
    messages.append({"role": "user", "content": question})
    try:
        msg, stats = llm.chat_tools(messages, None, think=False, temperature=0.0,
                                    num_predict=GATE_NUM_PREDICT, fmt=GATE_SCHEMA)
    except llm.LLMError:
        return None, {}
    try:
        said = json.loads(msg["content"])["change_note"]
    except (ValueError, KeyError, TypeError):
        return None, stats or {}
    return (said if isinstance(said, bool) else None), stats or {}


def _turn(messages: list[dict], *, think: bool, tools: list[str], step_tools: tuple[str, ...],
          note: edits_mod.Note | None, before_text: str, only: set[str] | None = None, create=None,
          temperature: float = FIRST_TEMPERATURE, nudge: bool = False, on_edit, should_stop,
          on_stage=lambda _name: None) -> dict:
    """One editing turn. Every call offers the same `tools`, in the same order: they sit at the
    top of the prompt, and a different list re-reads the whole note. `step_tools` are those that
    change something. The first call answers in text, makes a small change at once, or plans a
    larger one, at `temperature`; with `nudge` (the gate wanted a change) a text-only reply is
    asked once to use the tools. A plan is carried out one call per step; its outline is only
    measured against the note, for the trace. Code holds the plan, applies every call to the draft
    `note` (or makes a new note through `create`), enforces every limit, and hands each part to
    `on_edit` as it lands: "start", "plan", "step" and "change". `should_stop()` is asked before
    every call, and `on_stage` hears each call start, so the chat can count its seconds."""
    out = {"kind": "answer", "answer": "", "plan": None, "changes": [], "steps_done": 0, "calls": 0,
           "refusals": [], "malformed": [], "empty_steps": 0, "stopped": None, "summary": "",
           "finished_early": False, "replanned": False, "outline": None, "outline_met": None,
           "retries": 0, "first_call_edits": 0, "timing": [], "nudged": False,
           "outline_details": None,
           "first_temperature": temperature}
    spent, failed = 0, []
    # Deep's thinking has its own allowance: it would otherwise eat the steps' budget.
    budget = RUN_OUTPUT_TOKENS + (THINK_BUDGET if think else 0)
    current = {"k": None, "count": 1, "landed": 0, "refused": False}      # the step(s) being worked
    steps: list[str] = []
    # What the owner reads names text, never a handle; the model keeps its own words.
    plain = note.plain if note is not None else (lambda words: words)
    drafts = _Drafts(note, only, on_edit) if note is not None else None

    def attempt(offered, think, temperature, num_predict):
        if not out["stopped"] and should_stop():
            out["stopped"] = "stopped"
        if out["stopped"]:
            raise _Stopped
        on_stage("Thinking deeply" if think else "Writing")
        if drafts:
            drafts.next_reply()
        reply = _call(messages, offered, think=think, temperature=temperature, num_predict=num_predict,
                      on_part=drafts.part if drafts else None, should_stop=should_stop)
        for index, call in enumerate(reply.calls):
            if drafts and drafts.of(index):
                call["draft"] = drafts.of(index)
        return reply

    def ask(offered, think, temperature, num_predict) -> _Reply:
        nonlocal spent
        try:
            try:
                reply = attempt(offered, think, temperature, num_predict)
            except llm.LLMError:        # a failed call (once Ollama's parser, now the transport): once more
                out["retries"] += 1
                try:
                    reply = attempt(offered, think, temperature, num_predict)
                except llm.LLMError as exc:
                    failed.append(exc)
                    out["stopped"] = f"error: {(str(exc) or type(exc).__name__).splitlines()[0][:200]}"
                    raise _Stopped from exc
        except llm.Stopped:             # Stop while the call streamed: it ends here, what landed stays
            out["stopped"] = "stopped"
            raise _Stopped from None
        spent += reply.stats.get("output_tokens") or 0
        if spent > budget:
            out["stopped"] = "token budget"
        out["timing"].append(reply.stats)
        if reply.stats.get("ctx_saturated"):
            # The prompt lost its start — the note and the rider — so this reply's calls are
            # guesses: none of them lands.
            if out["kind"] == "answer":
                raise CopilotError("this note is too long for one request; edit it in Obsidian")
            out["stopped"] = "error: the note no longer fits the model's window"
            raise _Stopped
        out["malformed"] += reply.malformed
        messages.append(reply.message)
        return reply

    def open_step(k: int, count: int = 1) -> None:
        current.update(k=k, count=count, landed=0, refused=False)
        for n in range(k, k + count):
            if steps[n] != ONE_REPLY:
                on_edit("step", {"index": n + 1, "state": "doing"})

    def close_step() -> None:
        k = current["k"]
        if k is None:
            return
        current["k"] = None
        state = "skipped" if not current["landed"] else "partial" if current["refused"] else "done"
        for n in range(k, k + current["count"]):
            if steps[n] != ONE_REPLY:
                on_edit("step", {"index": n + 1, "state": state})

    def finish() -> dict:
        close_step()
        if drafts:
            drafts.end()
            out["draft_failures"] = drafts.failed
        if out["outline"]:
            out["outline_details"] = _outline_details(*_against(note, out["outline"]))
            out["outline_met"] = not out["outline_details"]["missing"] and not out["outline_details"]["out_of_order"]
        return out

    def split(reply: _Reply):
        """A first reply's plan, and its other calls if any of them changes something."""
        plans = [c for c in reply.calls if c["name"] == "plan"]
        others = [c for c in reply.calls if c["name"] != "plan"]
        return plans, others if any(c["name"] in step_tools for c in others) else []

    try:
        first_predict = QUICK_NUM_PREDICT + (THINK_BUDGET if think else 0)
        first = ask(tools, think, temperature, first_predict)
        plans, edits = split(first)
        unread = first.malformed and not first.calls
        if not plans and not edits and not out["stopped"]:
            if unread or first.calls and "plan" in tools:
                # Calls that change nothing, or a call code could not read: the plan asked for once
                # more (a new note, where there is no plan, once more).
                out["replanned"] = True
                said = _unreadable(first.malformed) if unread else "refused: start with a plan"
                messages.extend(_results([said], "Make the plan first: one sentence for the owner, and the "
                                                 "steps in order." if "plan" in tools else
                                                 "Make the new note with the create_note tool."))
            elif nudge and not first.calls:     # the gate said change, and the reply is only text
                out["nudged"] = True
                messages.extend(_results([], NUDGE))
            if out["replanned"] or out["nudged"]:
                first = ask(tools, think, temperature, first_predict)
                plans, edits = split(first)
        if not plans and not edits:
            answer = first.content.strip() or next((_summary(c) for c in first.calls if c["name"] == "done"), "")
            out["answer"] = note.plain(answer, bare=False) if note is not None else answer
            if drafts:
                drafts.end()
            return out
        args = plans[0]["arguments"] if plans and isinstance(plans[0]["arguments"], dict) else {}
        steps += _steps(args.get("steps"))[:MAX_STEPS]
        intro = str(args.get("intro") or "").strip() if plans else first.content.strip()
        # A plan written as prose beside the first changes: a list naming more than the reply did
        # (a list of as many is a recap). The first reply's edit calls are its first steps.
        made = [c for c in edits if c["name"] in step_tools]
        listed = _listed_plan(first.content) if edits and not plans and note is not None else None
        covered = 1
        if listed and len(listed[1]) > len(made):
            intro, written = listed
            steps += written
            covered = len(made)
        elif edits and not steps:       # edits with no plan, or with an empty one: a one-step run
            steps.append(ONE_REPLY)
        intro, shown = plain(intro), [] if steps == [ONE_REPLY] else [plain(step) for step in steps]
        out.update(kind="edit", plan={"intro": intro, "steps": shown},
                   outline=_outline(args) if note is not None else None)
        on_edit("start", {"before_text": before_text})
        on_edit("plan", {"intro": intro, "steps": shown, "outline": out["outline"]})
        if not steps:
            out["stopped"] = "empty plan"
            return finish()

        results, in_row = ["ok: plan accepted"], 0

        def step(instruction: str, number: int):
            """One model call for one step, its tool calls applied in order, with the
            note's map as it is now."""
            sections = note.sections() if note is not None else ""
            now = f"\n\nSections now:\n{sections}" if sections else ""
            messages.extend(_results(results, instruction + now))
            reply = ask(tools, False, STEP_TEMPERATURE, STEP_NUM_PREDICT)
            return reply, apply_calls(reply.calls, number)

        def land(name: str, args, number: int) -> tuple[dict, str]:
            change_id = f"c{len(out['changes']) + 1}"
            if name == "create_note":
                if not isinstance(args, dict):
                    raise edits_mod.ToolError("the arguments must be an object", "args")
                made = create(args.get("path"), args.get("markdown"))
                return ({"id": change_id, "step": number, "op": "create", "label": f"Created {made['path']}",
                         "needs": [], "path": made["path"], "text": made["text"]}, f"created {made['path']}")
            change, said = edits_mod.apply(note, name, args, change_id=change_id, step=number, only=only)
            return asdict(change), said

        def apply_calls(calls: list[dict], number) -> int:
            """`number`: the step each call belongs to — one for all, or one per call."""
            nonlocal results, in_row
            results, landed, refused = [], 0, False
            for index, call in enumerate(calls):
                out["calls"] += 1
                name = call["name"]
                at = number[index] if isinstance(number, list) else number
                if name == "done":          # the work is finished: this step's calls stand, no more steps
                    out.update(summary=plain(_summary(call)), finished_early=True)
                    break
                if name == "plan":          # the plan restated: answered, and neither a refusal nor a change
                    results.append("ok: the plan is already set; make this step's changes")
                    continue
                if out["calls"] > MAX_CALLS:
                    out["stopped"] = out["stopped"] or "call limit"
                    refused = True
                    break
                try:
                    if name not in step_tools:
                        raise edits_mod.ToolError(f"{name} is not available now", "tool")
                    change, said = land(name, call.get("arguments") or {}, at)
                except edits_mod.ToolError as exc:
                    out["refusals"].append({"step": at, "tool": name, "code": exc.code})
                    results.append(f"refused: {exc}")
                    refused, in_row = True, in_row + 1
                    if in_row >= MAX_REFUSALS:
                        out["stopped"] = out["stopped"] or "refusals"
                        break
                    continue
                out["changes"].append(change)
                draft = call.get("draft")
                if draft:                   # the change takes its draft's place
                    drafts.landed(draft)
                on_edit("change", {**change, "draft": draft} if draft else change)
                results.append(f"ok: {said}")
                landed, in_row = landed + 1, 0
            if drafts:                      # refused, or never reached
                drafts.end()
            if current["k"] is not None:
                current["landed"] += landed
                current["refused"] = refused
            return landed

        def instruction(k: int) -> str:
            if steps[k] == ONE_REPLY:   # only ever a retry: the first reply's change was refused
                return "Make your change again, fixing what was refused."
            return f"Step {k + 1} of {len(steps)}: {steps[k]}\nMake this step's changes now."

        k, retried, pending = 0, False, first if edits else None
        while k < len(steps) and not out["stopped"]:
            if not retried:
                open_step(k, covered if pending else 1)
            if pending:                 # the first reply already holds the first steps' calls
                reply, pending = pending, None
                numbers, m = [], 0
                for call in edits:
                    numbers.append(min(m, covered - 1) + 1)
                    m += call["name"] in step_tools
                landed = out["first_call_edits"] = apply_calls(edits, numbers)
                results = (["ok: plan accepted"] if plans else []) + results
            else:
                reply, landed = step(instruction(k), k + 1)
            if out["stopped"]:
                break
            if out["finished_early"]:
                k = out["steps_done"] = k + current["count"]
                break
            acted = [c for c in reply.calls if c["name"] != "plan"]
            if reply.malformed and not acted:
                results = [_unreadable(reply.malformed)]
            elif not acted:
                out["empty_steps"] += 1
            if (acted or reply.malformed) and not landed and not retried:
                retried = True              # every call refused or unreadable: the same step once more, with why
                current["count"] = 1        # (the steps after it run as their own)
                continue
            span = current["count"]
            close_step()
            k = out["steps_done"] = k + span
            retried = False
        close_step()

        if not out["stopped"] and not out["finished_early"]:
            n = len(out["changes"])
            out["summary"] = intro or f"Done: {n} change{'' if n == 1 else 's'}"
    except _Stopped:
        if out["kind"] == "answer":     # no run started, and no answer either
            if failed:
                raise failed[-1]
            raise CopilotError("the turn was stopped before it was answered")
    return finish()


def _note_text(vault: Path, path: str) -> str:
    """The open note from disk, when the editor's text was not sent."""
    try:
        text = edits_mod._read(Path(vault) / path)
    except (OSError, UnicodeDecodeError) as exc:
        raise CopilotError("the open note could not be read; refresh the sidebar") from exc
    if "\r" in text or text.startswith("﻿"):
        raise CopilotError("this note uses Windows line endings or a byte-order mark; edit it in Obsidian")
    return text


def _selected(note: edits_mod.Note, text: str, selection: str, skill: skills_mod.Skill) -> set[str]:
    """The blocks a selection touches: the only ones a selection skill may change."""
    chosen = (selection or "").strip()
    if not chosen:
        raise CopilotError(f"/{skill.name} works on selected text: select some in the note first")
    start = text.find(chosen)
    if start < 0:
        raise CopilotError("the selected text is not in the note as it is now; select it again")
    end, pos, found = start + len(chosen), 0, set()
    for block in note.blocks:
        if pos < end and start < pos + len(block.text):
            found.add(block.handle)
        pos += len(block.text)
    return found


def _facts(out: dict, run: dict) -> dict:
    """An editing run as the trace keeps it: machine facts, never the note's words."""
    return {"steps_planned": len(run["plan"]["steps"]), "steps_done": run["steps_done"],
            "calls": out["calls"], "model_calls": len(out["timing"]),
            "refusals": [item["code"] for item in out["refusals"]], "malformed": len(out["malformed"]),
            "empty_steps": out["empty_steps"], "changes": len(out["changes"]),
            "first_call_edits": out["first_call_edits"], "replanned": out["replanned"],
            "retries": out["retries"], "stopped": out["stopped"], "finished_early": out["finished_early"],
            "outline_met": out["outline_met"], "outline_details": out["outline_details"],
            "nudged": out["nudged"], "draft_failures": out.get("draft_failures", 0),
            "first_temperature": out["first_temperature"], "timing": out["timing"]}


def _answer_or_edit(con, vault: Path, source: dict, question: str, history: list[dict], mode: str,
                    *, skill, args: str, images, selection: str, doc: str | None, on_stage,
                    on_delta, on_edit, should_stop, entry: dict) -> tuple[dict, list[str], list[dict]]:
    """A turn that may edit: (result, retrieval levels, evidence). A plain question meets the
    intent gate first; "no" is today's Ask path (`generate_answer`, with retrieval and the
    sources call), anything else the edit path: the open note whole, with handles, and the
    tools. A selection skill may change only the blocks the selection touches, with `replace`
    and `edit_text`; a new-note skill reads its notes as a chat skill does and may only create
    notes."""
    stage = on_stage or (lambda _name: None)
    folder = PurePosixPath(source["path"]).parent.as_posix()
    started = time.time()
    # The model lock is held for the whole turn: a request slipping in between two steps would
    # evict the prefix every later step reads.
    with llm.hold():
        intent, gate = None, {}
        if not skill and INTENT_GATE:
            if should_stop():
                raise CopilotError("the turn was stopped before it was answered")
            intent, gate = _intent(question, history)
            entry.update(intent=intent, gate={"prompt_tokens": gate.get("prompt_tokens"),
                                              "duration_s": gate.get("duration_s")})
        if intent is False:             # no change wanted: today's Ask path
            evidence, levels = collect_evidence(con, source, question, limit=12 if mode == "deep" else 6)
            shown, skipped = open_note_images(vault, source, evidence)
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(question, history, evidence, mode, images=images, note_images=shown,
                                     on_stage=on_stage, on_delta=on_delta)
            result["timing"]["gate"] = gate
            result["edit_run"] = None
            return result, levels, evidence

        text = doc if doc is not None else _note_text(vault, source["path"])
        # A note too big to edit here is still asked about (the Ask path above never reads it).
        if len(text) > edits_mod.MAX_EDIT_CHARS and not (skill and skill.output == "new-note"):
            raise CopilotError(f"this note is too long to edit here "
                               f"(over {edits_mod.MAX_EDIT_CHARS:,} characters)")
        prompt = skills_mod.render_prompt(skill, args) if skill else question
        note, only, create, inputs, cited = edits_mod.Note(text), None, None, None, None
        tools, step_tools = ["plan", *STEP_TOOLS], STEP_TOOLS
        # A change the gate wanted, or an edit skill's, is written at STEP_TEMPERATURE.
        temperature = STEP_TEMPERATURE if not skill and INTENT_GATE else FIRST_TEMPERATURE
        if skill and skill.output == "new-note":
            pack = gather(con, vault, source, skill=skill, links_from=args, selection=selection)
            system = pack_message(pack, NEW_NOTE_RIDER.format(folder=folder))
            cited = [{"n": i + 1, "path": d["path"], "title": d["title"]}
                     for i, d in enumerate(d for d in pack.docs if d["role"] != "selection")]
            inputs, note = pack_inputs(pack), None
            tools, step_tools = ["create_note", "done"], ("create_note",)
            made: set[str] = set()

            def create(path, markdown):
                new = edits_mod.create_note(vault, path, markdown, folder=folder)
                if new["path"].casefold() in made:
                    raise edits_mod.ToolError(f"{new['path']} is already made in this turn; give the next "
                                              "note a name of its own", "exists")
                made.add(new["path"].casefold())
                return new
        else:
            if skill:
                # An edit skill: a wanted change by its frontmatter. Code knows what it read (the
                # open note), so it makes no sources call.
                cited, temperature = [{"n": 1, "path": source["path"],
                                       "title": PurePosixPath(source["path"]).stem}], STEP_TEMPERATURE
                if skill.input == "selection":
                    only = _selected(note, text, selection, skill)
                    prompt += f"\n\nSelected text:\n{selection.strip()}"
                    tools, step_tools = ["plan", "replace", "edit_text"], ("replace", "edit_text")
            system = edit_message(source["path"], note)
        # No retrieval orders the edit path's images, so they come in note order.
        shown, skipped = open_note_images(vault, source, [])
        entry.update(note_images=len(shown), note_images_skipped=len(skipped))

        transcript, embedded, current = _conversation(history, prompt, images, shown)
        system_turn = {"role": "system", "content": system}
        room = QUICK_NUM_PREDICT + (THINK_BUDGET if mode == "deep" else 0)
        if sum(map(_tokens, [system_turn, *embedded, current])) + room > llm.RESIDENT_CTX:
            raise CopilotError("this note is too long for one request; edit it in Obsidian")
        recent = fit_window(transcript, [system_turn, *embedded, current], room)
        messages = [system_turn, *recent, *embedded, current]
        out = _turn(list(messages), think=mode == "deep", tools=tools, step_tools=step_tools,
                    note=note, before_text=text, only=only, create=create, temperature=temperature,
                    nudge=intent is True, on_edit=on_edit or (lambda _event, _payload: None),
                    should_stop=should_stop, on_stage=stage)
    last = out["timing"][-1] if out["timing"] else {}
    run = None
    if out["kind"] == "answer":     # a text reply on the edit path: said once, whole
        answer = out["answer"]
        if not answer and last.get("thinking_ate_the_budget"):
            raise CopilotError("the model spent its whole output budget thinking and wrote no "
                               "answer — ask again in Quick mode, or ask a narrower question")
        if not answer:
            raise CopilotError("the local model returned no answer")
        (on_delta or (lambda _text: None))(answer)
    else:
        run = {"plan": out["plan"], "steps_done": min(out["steps_done"], len(out["plan"]["steps"])),
               "changes": out["changes"], "stopped": out["stopped"], "outline_met": out["outline_met"],
               "hash_before": edits_mod.digest(text),
               "hash_after": edits_mod.digest(note.text() if note is not None else text),
               "summary": out["summary"]}
        answer = out["summary"] or out["plan"]["intro"]
        entry["edit_run"] = _facts(out, run)
    result = {
        "answer": answer,
        "mode": mode,
        # A skill names what code read; the edit path makes no sources call.
        "citations": cited or [],
        "verification": {"truncated": bool(last.get("output_truncated"))},
        "history_dropped": len(transcript) - len(recent),
        "timing": {"total_ms": round((time.time() - started) * 1000), "gate": gate,
                   "answer": out["timing"][0] if out["timing"] else {}, "sources": {}},
        "edit_run": run,
    }
    if inputs is not None:
        result["inputs"] = inputs
    return result, [], []


def run_turn(con, source_id: str, question: str, history: list[dict], mode: str,
             *, images=None, on_stage=None, on_delta=None, on_edit=None, selection: str = "",
             vault: Path | None = None, doc: str | None = None, doc_hash: str | None = None,
             should_stop=lambda: False) -> dict:
    """One sidebar turn. A `/command` whose output is chat reads whole notes chosen by code
    (`gather`) and answers. Every other turn is one chat that answers or edits: it may answer in
    text, or change the open note, which it reads from `doc` — the editor's text, whose
    `doc_hash` must match — or else from disk. `on_edit(event, payload)` gets each part of an
    edit as it lands; nothing is written here: the plugin applies each change through the editor."""
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
             "skill": skill.name if skill else None,
             "selection_chars": len(selection or "")}
    started = time.time()
    evidence, levels = [], []
    try:
        if mode not in {"quick", "deep"}:
            raise CopilotError("reasoning mode must be quick or deep")
        if skill and skill.error:
            raise CopilotError(f"/{skill.name} cannot run: {skill.error} "
                               f"({skills_mod.VAULT_DIR}/{skill.name}.md)")
        unknown = re.match(r"^/([a-z0-9][a-z0-9-]*)(?:\s|$)", question)
        if unknown and not skill:
            raise CopilotError(f"there is no /{unknown.group(1)} skill; type / to see them")
        if doc is not None and edits_mod.digest(doc) != doc_hash:
            raise CopilotError("the note changed while sending; try again")
        if skill and skill.output == "chat":
            pack = gather(con, vault, source, skill=skill, links_from=args, selection=selection)
            cited = [{"n": i + 1, "path": doc["path"], "title": doc["title"]}
                     for i, doc in enumerate(d for d in pack.docs if d["role"] != "selection")]
            # No retrieval orders a skill's images, so they come in note order.
            shown, skipped = open_note_images(vault, source, [])
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(
                skills_mod.render_prompt(skill, args), history, [], mode,
                images=images, note_images=shown, on_stage=on_stage, on_delta=on_delta,
                system=pack_message(pack, ASK_RIDER), cited=cited)
            result.update(inputs=pack_inputs(pack), edit_run=None)
        else:
            result, levels, evidence = _answer_or_edit(
                con, vault, source, question, history, mode, skill=skill, args=args, images=images,
                selection=selection, doc=doc, on_stage=on_stage, on_delta=on_delta, on_edit=on_edit,
                should_stop=should_stop, entry=entry)
    except Exception as exc:
        # Any failure leaves its record, a closed chat's BrokenPipe included; the message only
        # for refusals and model errors, whose words are SLIM's own.
        failure = {**entry, "status": "failed", "error_type": type(exc).__name__,
                   "duration_ms": round((time.time() - started) * 1000)}
        if isinstance(exc, (CopilotError, llm.LLMError)):
            failure["error"] = str(exc)
        trace.record("copilot", failure)
        raise
    result.update({"source": source, "retrieval_levels": levels, "skill": entry["skill"]})
    record = {
        **entry, "status": "edited" if result["edit_run"] else "answered",
        "cited": sorted({item["path"] for item in result["citations"]}),
        "truncated": result["verification"]["truncated"],
        "history_dropped": result["history_dropped"], "timing": result["timing"],
    }
    if "inputs" in result:
        record["inputs"] = result["inputs"]
    else:
        record.update({"evidence": [item["path"] for item in evidence], "retrieval_levels": levels})
    trace.record("copilot", record)
    return result
