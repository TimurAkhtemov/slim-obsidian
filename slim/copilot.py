"""The open-note copilot: gather the notes around the one they have open, run one turn against
them, and name which notes the answer used — or, when the turn edits, carry out the edit one
model call at a time and hand each change to the plugin as it lands. Called by `chat.py`'s
sidebar routes. What it reads is `copilot_context.py`'s; the editing run is `copilot_edit.py`'s."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path, PurePosixPath

from . import config as config_mod, copilot_context, copilot_edit, edits as edits_mod, edits_note
from . import llm, skills as skills_mod, trace
from .copilot_context import CopilotError

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
# Thinking tokens bill against num_predict but return in a separate field (llm.chat's
# docstring). Deep mode must budget for the reasoning AND the answer, or a long think returns
# content="" and reads as "the model can't answer": one benign sample spent ~3k of 4096.
DEEP_NUM_PREDICT = 4096 * 3
# While the model reasons no content reaches the client, so its Stop (a destroyed socket)
# goes unnoticed until content starts. Re-emitting the stage this often is the write that
# raises BrokenPipe inside the Ollama request and frees llm._CALL_LOCK.
HEARTBEAT_S = 1.0
# THE WINDOW RULE (`fit_window`): the conversation goes to the model whole, and the resident
# window is its only limit. Ollama has no tokenize endpoint, so tokens are estimated from
# characters. Real copilot prompts rebuilt from the threads and divided by the prompt_tokens
# Ollama reported ran 3.8-4.1 characters per token (2026-10-12, three prompts, template
# included). At 3 the estimate counts more tokens than any of them used, so history goes early
# rather than overflowing; denser text (LaTeX, code) runs lower, and `ctx_saturated` is the
# backstop.
CHARS_PER_TOKEN = 3
# The most an image can cost: the model reads 32-pixel squares (patch 16, merge 2) and every
# image arrives at most 1600 pixels on a side (`note_images.NOTE_IMAGE_MAX_EDGE`, and the
# plugin's resize of a paste). A typical screenshot costs ~450 (`copilot_context.MAX_NOTE_IMAGES`).
IMAGE_TOKENS = (1600 // 32) ** 2



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


def edit_message(path: str, note: edits_note.Note) -> str:
    """The system message of the edit path, as measured: the persona and EDIT_RIDER, then the open
    note whole, with its handles and the map of its sections."""
    sections = note.sections()
    shown = f"Sections:\n{sections}\n\n" if sections else ""
    return "\n\n".join([PERSONA, EDIT_RIDER,
                         f"Notes in view:\n\n=== {path} · the open note ===\n{shown}{note.render()}"])



def pack_message(pack: copilot_context.Pack, rider: str) -> str:
    labels = {"open": "the open note", "selection": "the text they selected in this note",
              "linked": "a note they linked"}
    rendered = "\n\n".join(
        f"=== {doc['path']} · {labels.get(doc['role']) or doc['date'] or 'a note in this folder'} ===\n"
        f"{doc['text']}" for doc in pack.docs)
    return "\n\n".join([PERSONA, rider, "Notes in view:\n\n" + (rendered or "(none)")])



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




def _answer_or_edit(con, vault: Path, source: dict, question: str, history: list[dict], mode: str,
                    *, skill, args: str, images, selection: str, selection_from: int | None,
                    doc: str | None, on_stage, on_delta, on_edit, should_stop,
                    entry: dict) -> tuple[dict, list[str], list[dict]]:
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
            evidence, levels = copilot_context.collect_evidence(con, source, question, limit=12 if mode == "deep" else 6)
            shown, skipped = copilot_context.open_note_images(vault, source, evidence)
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(question, history, evidence, mode, images=images, note_images=shown,
                                     on_stage=on_stage, on_delta=on_delta)
            result["timing"]["gate"] = gate
            result["edit_run"] = None
            return result, levels, evidence

        text = doc if doc is not None else copilot_edit._note_text(vault, source["path"])
        # A note too big to edit here is still asked about (the Ask path above never reads it).
        if len(text) > edits_mod.MAX_EDIT_CHARS and not (skill and skill.output == "new-note"):
            raise CopilotError(f"this note is too long to edit here "
                               f"(over {edits_mod.MAX_EDIT_CHARS:,} characters)")
        prompt = skills_mod.render_prompt(skill, args) if skill else question
        note, only, create, inputs, cited = edits_note.Note(text), None, None, None, None
        tools, step_tools = ["plan", *copilot_edit.STEP_TOOLS], copilot_edit.STEP_TOOLS
        # A change the gate wanted, or an edit skill's, is written at STEP_TEMPERATURE.
        temperature = (copilot_edit.STEP_TEMPERATURE if not skill and INTENT_GATE
                       else copilot_edit.FIRST_TEMPERATURE)
        if skill and skill.output == "new-note":
            pack = copilot_context.gather(con, vault, source, skill=skill, links_from=args, selection=selection)
            system = pack_message(pack, NEW_NOTE_RIDER.format(folder=folder))
            cited = [{"n": i + 1, "path": d["path"], "title": d["title"]}
                     for i, d in enumerate(d for d in pack.docs if d["role"] != "selection")]
            inputs, note = copilot_context.pack_inputs(pack), None
            tools, step_tools = ["create_note", "done"], ("create_note",)
            made: set[str] = set()

            def create(path, markdown):
                new = edits_mod.create_note(vault, path, markdown, folder=folder)
                if new["path"].casefold() in made:
                    raise edits_note.ToolError(f"{new['path']} is already made in this turn; give the next "
                                               "note a name of its own", "exists")
                made.add(new["path"].casefold())
                return new
        else:
            if skill:
                # An edit skill: a wanted change by its frontmatter. Code knows what it read (the
                # open note), so it makes no sources call.
                cited, temperature = ([{"n": 1, "path": source["path"],
                                        "title": PurePosixPath(source["path"]).stem}],
                                      copilot_edit.STEP_TEMPERATURE)
                if skill.input == "selection":
                    only = copilot_edit._selected(note, text, selection, skill, selection_from)
                    prompt += f"\n\nSelected text:\n{selection.strip()}"
                    tools, step_tools = ["plan", "replace", "edit_text"], ("replace", "edit_text")
            system = edit_message(source["path"], note)
        # No retrieval orders the edit path's images, so they come in note order.
        shown, skipped = copilot_context.open_note_images(vault, source, [])
        entry.update(note_images=len(shown), note_images_skipped=len(skipped))

        transcript, embedded, current = _conversation(history, prompt, images, shown)
        system_turn = {"role": "system", "content": system}
        room = QUICK_NUM_PREDICT + (copilot_edit.THINK_BUDGET if mode == "deep" else 0)
        if sum(map(_tokens, [system_turn, *embedded, current])) + room > llm.RESIDENT_CTX:
            raise CopilotError("this note is too long for one request; edit it in Obsidian")
        recent = fit_window(transcript, [system_turn, *embedded, current], room)
        messages = [system_turn, *recent, *embedded, current]
        out = copilot_edit._turn(list(messages), think=mode == "deep", tools=tools, step_tools=step_tools,
                                 note=note, before_text=text, only=only, create=create,
                                 temperature=temperature, first_predict=room, nudge=intent is True,
                                 on_edit=on_edit or (lambda _event, _payload: None),
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
        entry["edit_run"] = copilot_edit._facts(out, run)
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
             selection_from: int | None = None, vault: Path | None = None, doc: str | None = None,
             doc_hash: str | None = None, should_stop=lambda: False) -> dict:
    """One sidebar turn. A `/command` whose output is chat reads whole notes chosen by code
    (`gather`) and answers. Every other turn is one chat that answers or edits: it may answer in
    text, or change the open note, which it reads from `doc` — the editor's text, whose
    `doc_hash` must match — or else from disk; `selection_from` is where the selection starts
    in it, in UTF-16 units. `on_edit(event, payload)` gets each part of an edit as it lands;
    nothing is written here: the plugin applies each change through the editor."""
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
            pack = copilot_context.gather(con, vault, source, skill=skill, links_from=args, selection=selection)
            cited = [{"n": i + 1, "path": doc["path"], "title": doc["title"]}
                     for i, doc in enumerate(d for d in pack.docs if d["role"] != "selection")]
            # No retrieval orders a skill's images, so they come in note order.
            shown, skipped = copilot_context.open_note_images(vault, source, [])
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(
                skills_mod.render_prompt(skill, args), history, [], mode,
                images=images, note_images=shown, on_stage=on_stage, on_delta=on_delta,
                system=pack_message(pack, ASK_RIDER), cited=cited)
            result.update(inputs=copilot_context.pack_inputs(pack), edit_run=None)
        else:
            result, levels, evidence = _answer_or_edit(
                con, vault, source, question, history, mode, skill=skill, args=args, images=images,
                selection=selection, selection_from=selection_from, doc=doc, on_stage=on_stage,
                on_delta=on_delta, on_edit=on_edit, should_stop=should_stop, entry=entry)
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
