"""The open-note copilot: gather the notes around the one they have open, run one turn against
them, and name which notes the answer used. Called by `chat.py`'s sidebar routes."""
from __future__ import annotations

import time
from pathlib import Path, PurePosixPath

from . import config as config_mod, embed as embed_mod, ingest as ingest_mod
from . import llm, search as search_mod, trace

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
then explain.

You cannot edit files yet. If they ask you to change a note, say so and give them the text to
paste."""

# Sources are decided AFTER the answer by a bounded structured call, not by a contract inside
# the prose. Measured 2026-09-01: [n] markers made a citation bot; a trailing `Notes used:` line
# arrived 4 times in 12 at temperature 0.7. The pack carries no numbers, so prose cannot cite;
# the follow-up shares the answer call's prefix, so the KV cache covers all but a few tokens.
SOURCES_QUESTION = """Which of these notes did your reply actually draw on? Answer with their numbers only, or an
empty list if you answered from general knowledge."""
SOURCES_NUM_PREDICT = 64


# 1400 clipped a worked example mid-sentence, and Quick is the mode nearly every turn uses.
QUICK_NUM_PREDICT = 2400
# Thinking tokens bill against num_predict but return in a separate field (llm.chat's
# docstring). Deep mode must budget for the reasoning AND the answer, or a long think returns
# content="" and reads as "the model can't answer": one benign sample spent ~3k of 4096.
DEEP_NUM_PREDICT = 4096 * 3
# While the model reasons no content reaches the client, so its Stop (a destroyed socket)
# goes unnoticed until content starts. Re-emitting the stage this often is the write that
# raises BrokenPipe inside the Ollama request and frees llm._CALL_LOCK.
HEARTBEAT_S = 1.0


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
    parts = [PERSONA]
    rendered = "\n\n".join(
        f"### {item['title']}"
        + (f" § {item['heading']}" if item.get("heading") else "")
        + f"\n{item['text']}" for item in evidence)
    parts.append("Notes in view (the open note first):\n"
                 + (rendered or "(nothing indexed for this note yet)"))
    return "\n\n".join(parts)


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


def generate_answer(question: str, history: list[dict], evidence: list[dict], mode: str,
                    *, images=None, on_stage=None, on_delta=None) -> dict:
    """Generate one answer; reasoning is private and only content deltas are surfaced.

    TWO depths, and THEY pick (2026-09-03). Auto spent a whole extra classifier call on a choice
    that is one click in the sidebar, and it resolved to Quick nearly every time.
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
    current = {"role": "user", "content": question}
    if images:
        current["images"] = list(images)
    messages = [{"role": "system", "content": system_message(evidence)},
                *transcript, current]
    started = time.time()
    last_beat = [started]

    def heartbeat(_n_chars):
        if time.time() - last_beat[0] >= HEARTBEAT_S:
            last_beat[0] = time.time()
            stage("Thinking deeply")

    text, stats = llm.chat_stream(
        messages, model=llm.MODEL, num_ctx=llm.RESIDENT_CTX,
        num_predict=DEEP_NUM_PREDICT if mode == "deep" else QUICK_NUM_PREDICT,
        temperature=llm.REPLY_TEMPERATURE, think=mode == "deep",
        timeout=600, on_delta=delta, on_thinking=heartbeat)
    answer = (text or "").strip()
    if not answer and stats.get("thinking_ate_the_budget"):
        raise CopilotError("the model spent its whole output budget thinking and wrote no "
                           "answer — ask again in Quick mode, or ask a narrower question")
    if not answer:
        raise CopilotError("the local model returned no answer")
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
             *, images=None, on_stage=None, on_delta=None) -> dict:
    source_row = con.execute(
        "SELECT id, path, title, type, current_hash FROM sources "
        "WHERE id=? AND deleted=0", (source_id,)).fetchone()
    if source_row is None:
        raise CopilotError("the note is no longer indexed; refresh the sidebar")
    source = dict(source_row)
    # Machine facts only (no prose, no reasoning): the thread JSON keeps completed turns,
    # so this is the one record of a refused or stopped turn.
    entry = {"question": question, "source": source["path"], "source_id": source_id,
             "mode": mode, "history_turns": len(history),
             "images": len(images or []),
             "history_images": sum(len(turn.get("images") or []) for turn in history)}
    started = time.time()
    try:
        limit = 12 if mode == "deep" else 6
        evidence, levels = collect_evidence(con, source, question, limit=limit)
        result = generate_answer(
            question, history, evidence, mode, images=images,
            on_stage=on_stage, on_delta=on_delta)
    except (CopilotError, llm.LLMError) as exc:
        trace.record("copilot", {**entry, "status": "failed", "error": str(exc),
                                 "duration_ms": round((time.time() - started) * 1000)})
        raise
    result.update({"source": source, "retrieval_levels": levels})
    trace.record("copilot", {
        **entry, "status": "answered",
        "evidence": [item["path"] for item in evidence], "retrieval_levels": levels,
        "cited": sorted({item["path"] for item in result["citations"]}),
        "truncated": result["verification"]["truncated"], "timing": result["timing"],
    })
    return result
