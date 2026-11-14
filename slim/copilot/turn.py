"""The sidebar's copilot: one turn of THE LOOP, which answers, looks things up in the vault
(`copilot/lookup.py`) and edits the open note one model call at a time, handing each change to the
plugin as it lands. Behind `LOOP = False`, the turn as it was: the intent gate, then the Ask path
(the notes around the open one, and which the answer used) or the editing run. Called by
`server.py`'s sidebar routes. What it reads is `copilot/context.py`'s; the run is `copilot/run.py`'s."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path, PurePosixPath

from .. import config as config_mod, llm, trace
from . import blocks, context, edits as edits_mod, lookup, run as edit_run, skills as skills_mod
from .context import CopilotError

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

# THE LOOP (2026-11-13, the owner's call): one loop answers, looks things up and edits; the model
# decides, with no gate. The open note rides with the question (`open_note_message`), so the
# system message holds no note and stays a cached prefix (`loop_system`). LOOP_RIDER is EDIT_RIDER's
# measured handle and plan text, the note now with their message; the lookups are the prompt's rules.
# Nothing in it asks for citations: sources are what code saw the turn look at. Off, a turn takes the gate
# and the Ask path, as before, until the owner has compared the two.
#
# ONE system message and ONE tool list (LOOP_TOOLS, in this order) for every turn that is not a
# skill's: the tools render into the system prompt, so a rider or a list that followed the open
# note (none, plain, too long) changed the prompt from its first token and re-read the pins and
# the history. With no note it may change, a call to an edit tool is refused (`run._turn`'s
# `no_edit`), with a reason the model reads.
LOOP = True
LOOKING = """When the question needs a note you have not been shown, look for it with search, read_note and
list_folder. The open note is already shown: never look it up, except a part it says is not shown.
A plain question about the open note needs no tool."""
CREATING = "create_note makes a new note, in a folder that already exists."
CANNOT_EDIT = ("When no note is shown, or a note is shown without handles, it can't be edited here: if they ask\n"
               "for a change, give them the text to paste.")
LOOP_RIDER = ("You can edit the open note shown with their message."
              + EDIT_RIDER.removeprefix("You can edit the open note shown below.")
              + f"\n\n{CREATING}\n{CANNOT_EDIT}")
LOOP_TOOLS = [*lookup.NAMES, "plan", *edit_run.STEP_TOOLS, "create_note"]
# Qwen's recommended sampling with thinking on, the Pi coding agent's on the same server: every loop
# call but a step's, which stays at STEP_TEMPERATURE with none of these (measured, 2026-10-12).
LOOP_TEMPERATURE = 1.0
LOOP_SAMPLING = {"top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}

# THE LOOP'S PROMPT is shaped like the Pi coding agent's (2026-11-14, the owner's call): who it is
# and where, one line per tool, short rules, and the edit rider. Pi, on the same model and server,
# looked things up; the tutor persona's "use your general knowledge freely" made a question about
# the owner's own numbers a guess, with no lookup (live, 2026-11-13). Skills keep the persona.
LOOP_TOOL_LINES = {
    "search": "Search the vault's notes for a topic; returns the best passages with their note paths",
    "read_note": "Read a note, whole or one section of it",
    "list_folder": "List a folder's folders and notes",
    "plan": "Start a larger change to the open note: an intro, its headings in order, the steps",
    "replace": "Replace a block, a section or a range of the open note with new Markdown",
    "edit_text": "Change a few words inside one block of the open note",
    "insert": "Add new Markdown after or before a block or section of the open note",
    "move": "Move a block or section of the open note to another place in it",
    "create_note": "Make a new note, in a folder that already exists",
}
MARKDOWN_RULE = PERSONA[PERSONA.index("Write Markdown."):]       # the persona's measured math line


def loop_system() -> str:
    """The one system message of every loop turn that is not a skill's."""
    owner = config_mod.OWNER
    tools = "\n".join(f"- {name}: {LOOP_TOOL_LINES[name]}" for name in LOOP_TOOLS)
    rules = "\n".join(f"- {rule}" for rule in (
        f"Questions about {owner}'s own work, people, plans, meetings or numbers are answered from the "
        "vault: look them up before answering, with search and read_note",
        "If the vault doesn't have it, say so; only then add general knowledge, labelled as such",
        f"Never present a guess about {owner}'s situation as fact",
        "The open note is shown with their message: never look it up, except a part it says is not "
        "shown. A question only about the open note needs no tool",
        "Be concise in your responses",
        "Name notes by title when you use them",
        " ".join(MARKDOWN_RULE.split()),
    ))
    return (f"You are SLIM, an assistant operating inside {owner}'s Obsidian vault. You help {owner} by "
            "searching and reading their notes, answering questions, and editing the note they have open."
            f"\n\n<tools>\n{tools}\n</tools>\n\n<rules>\n{rules}\n</rules>\n\n<editing>\n{LOOP_RIDER}\n</editing>")
NO_NOTE = "no note is open to change; give the text to paste"
PLAIN_NOTE = "this note can't be changed here; give the text to paste"
SKILL_RIDER = f"{ASK_RIDER}\n\n{LOOKING}"

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
# THE LOOP's sources: the notes a turn looked at, a chip each under the answer.
SOURCES_MAX = 8


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
# plugin's resize of a paste). A typical screenshot costs ~450 (`copilot.context.MAX_NOTE_IMAGES`).
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


def edit_message(path: str, note: blocks.Note) -> str:
    """The system message of the edit path, as measured: the persona and EDIT_RIDER, then the open
    note whole, with its handles and the map of its sections."""
    sections = note.sections()
    shown = f"Sections:\n{sections}\n\n" if sections else ""
    return "\n\n".join([PERSONA, EDIT_RIDER,
                         f"Notes in view:\n\n=== {path} · the open note ===\n{shown}{note.render()}"])



def pack_docs(pack: context.Pack) -> str:
    labels = {"open": "the open note", "selection": "the text they selected in this note",
              "linked": "a note they linked"}
    return "\n\n".join(
        f"=== {doc['path']} · {labels.get(doc['role']) or doc['date'] or 'a note in this folder'} ===\n"
        f"{doc['text']}" for doc in pack.docs)


def pack_message(pack: context.Pack, rider: str) -> str:
    return "\n\n".join([PERSONA, rider, "Notes in view:\n\n" + (pack_docs(pack) or "(none)")])


def open_note_message(path: str, note: blocks.Note) -> str:
    """The open note as it rides with the question: its sections map, then its blocks with their
    handles, whole up to ATTACH_MAX_CHARS. A longer note shows the whole blocks that fit, and those
    are all a tool may change (`note.shown`) until it reads another section of it."""
    sections = note.sections()
    shown = f"Sections:\n{sections}\n\n" if sections else ""
    if len(note.text()) <= context.ATTACH_MAX_CHARS:
        rendered = note.render()
    else:
        rendered, kept = note.head(note.blocks, context.ATTACH_MAX_CHARS, path)
        note.shown = {b.handle for b in kept if b.kind != "blank" and not b.locked}
        rest = [b for b in note.blocks if b.kind != "blank" and not b.locked and b.handle not in note.shown]
        if rest:
            span = f"{rest[0].handle}-{rest[-1].handle}" if len(rest) > 1 else rest[0].handle
            heads = [b for b in rest if b.level]
            rendered = rendered.rstrip("\n") + (
                f'\n\n[Not shown: from {rest[0].handle} on. read_note("{path}", "s{heads[0].handle[1:]}") shows '
                f'the first section not shown, with its handles; any s<N>, or a range like "{span}", works too.]'
                if heads else
                f'\n\n[Not shown: from {rest[0].handle} on. read_note("{path}", "{span}") shows those blocks '
                "with their handles.]")
    return f"=== {path} · the open note ===\n{shown}{rendered.rstrip()}"


def plain_note_message(path: str, text: str) -> str:
    """A note the editing tools cannot take, as it rides with the question: its text, no handles."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > context.ATTACH_MAX_CHARS:
        text = text[:context.ATTACH_MAX_CHARS] + "\n[… the rest: read_note(path, section) for one section]"
    return f"=== {path} · the open note ===\n{text.rstrip()}"



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
            evidence, levels = context.collect_evidence(con, source, question, limit=12 if mode == "deep" else 6)
            shown, skipped = context.open_note_images(vault, source, evidence)
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(question, history, evidence, mode, images=images, note_images=shown,
                                     on_stage=on_stage, on_delta=on_delta)
            result["timing"]["gate"] = gate
            result["edit_run"] = None
            return result, levels, evidence

        text = doc if doc is not None else edit_run._note_text(vault, source["path"])
        # A note too big to edit here is still asked about (the Ask path above never reads it).
        if len(text) > edits_mod.MAX_EDIT_CHARS and not (skill and skill.output == "new-note"):
            raise CopilotError(f"this note is too long to edit here "
                               f"(over {edits_mod.MAX_EDIT_CHARS:,} characters)")
        prompt = skills_mod.render_prompt(skill, args) if skill else question
        note, only, create, inputs, cited = blocks.Note(text), None, None, None, None
        tools, step_tools = ["plan", *edit_run.STEP_TOOLS], edit_run.STEP_TOOLS
        # A change the gate wanted, or an edit skill's, is written at STEP_TEMPERATURE.
        temperature = (edit_run.STEP_TEMPERATURE if not skill and INTENT_GATE
                       else edit_run.FIRST_TEMPERATURE)
        if skill and skill.output == "new-note":
            pack = context.gather(con, vault, source, skill=skill, links_from=args, selection=selection)
            system = pack_message(pack, NEW_NOTE_RIDER.format(folder=folder))
            cited = [{"n": i + 1, "path": d["path"], "title": d["title"]}
                     for i, d in enumerate(d for d in pack.docs if d["role"] != "selection")]
            inputs, note = context.pack_inputs(pack), None
            tools, step_tools = ["create_note", "done"], ("create_note",)
            made: set[str] = set()

            def create(path, markdown):
                new = edits_mod.create_note(vault, path, markdown, folder=folder)
                if new["path"].casefold() in made:
                    raise blocks.ToolError(f"{new['path']} is already made in this turn; give the next "
                                           "note a name of its own", "exists")
                made.add(new["path"].casefold())
                return new
        else:
            if skill:
                # An edit skill: a wanted change by its frontmatter. Code knows what it read (the
                # open note), so it makes no sources call.
                cited, temperature = ([{"n": 1, "path": source["path"],
                                        "title": PurePosixPath(source["path"]).stem}],
                                      edit_run.STEP_TEMPERATURE)
                if skill.input == "selection":
                    only = edit_run._selected(note, text, selection, skill, selection_from)
                    prompt += f"\n\nSelected text:\n{selection.strip()}"
                    tools, step_tools = ["plan", "replace", "edit_text"], ("replace", "edit_text")
            system = edit_message(source["path"], note)
        # No retrieval orders the edit path's images, so they come in note order.
        shown, skipped = context.open_note_images(vault, source, [])
        entry.update(note_images=len(shown), note_images_skipped=len(skipped))

        transcript, embedded, current = _conversation(history, prompt, images, shown)
        system_turn = {"role": "system", "content": system}
        room = QUICK_NUM_PREDICT + (edit_run.THINK_BUDGET if mode == "deep" else 0)
        if sum(map(_tokens, [system_turn, *embedded, current])) + room > llm.RESIDENT_CTX:
            raise CopilotError("this note is too long for one request; edit it in Obsidian")
        recent = fit_window(transcript, [system_turn, *embedded, current], room)
        messages = [system_turn, *recent, *embedded, current]
        out = edit_run._turn(list(messages), think=mode == "deep", tools=tools, step_tools=step_tools,
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
        entry["edit_run"] = edit_run._facts(out, run)
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


def _loop(con, vault: Path, source: dict | None, question: str, history: list[dict], mode: str, *,
          pins, skill, args: str, images, selection: str, doc: str | None, on_stage, on_delta,
          on_edit, should_stop, entry: dict, on_reset=None, on_tool=None) -> dict:
    """One turn of THE LOOP. Every call offers LOOP_TOOLS under one system message, whatever is
    open; with no note it may change (none, the chip's ✕, a note the tools cannot take, or one
    sent without the editor's text for being over MAX_EDIT_CHARS), the edit tools refuse. A chat
    skill offers the lookups alone, its notes (`gather`) in the open note's place. The messages:
    the system message (`loop_system`; a skill's is the persona and SKILL_RIDER), the pins
    and an assistant's "Noted." (roles alternate), the history that fits, the open note's images,
    then the open note and the question. Reply text streams through `on_delta`, and `on_reset()`
    says a new model call started, so the text streamed before it was not the answer."""
    stage = on_stage or (lambda _name: None)
    path = source["path"] if source else None
    folder = PurePosixPath(path).parent.as_posix() if path else ""
    started = time.time()
    pinned, held, left = context.pinned_message(con, vault, list(pins))
    folders = {str(pin.get("path") or "").rstrip("/") for pin in pins if pin.get("kind") == "folder"}
    scope = next((held_path for held_path in held if held_path in folders and lookup.searchable(held_path)), None)
    entry.update(pins=len(held), pins_left_out=len(left))
    with llm.hold():
        note, text, cited, inputs, no_edit = None, None, [], None, None
        tools, step_tools = LOOP_TOOLS, (*edit_run.STEP_TOOLS, "create_note")
        if skill:                               # a chat skill: its notes, chosen by code
            pack = context.gather(con, vault, source, skill=skill, links_from=args, selection=selection)
            cited = [doc_["path"] for doc_ in pack.docs if doc_["role"] not in ("open", "selection")]
            inputs = context.pack_inputs(pack)
            tools, step_tools = list(lookup.NAMES), ()
            asked = (f"Notes in view:\n\n{pack_docs(pack) or '(none)'}\n\n---\n\n"
                     f"{skills_mod.render_prompt(skill, args)}")
        elif source is not None:
            try:
                text = doc if doc is not None else edits_mod._read(vault / path)
            except (OSError, UnicodeDecodeError) as exc:
                raise CopilotError("the open note could not be read; refresh the sidebar") from exc
            # A note the editing tools cannot take (Windows line endings, a byte-order mark), or one
            # the plugin sent no editor text for (over MAX_EDIT_CHARS), is still asked about: it
            # rides as plain text, and the edit tools refuse.
            if "\r" in text or text.startswith("\ufeff") or (doc is None and len(text) > edits_mod.MAX_EDIT_CHARS):
                asked, no_edit = f"{plain_note_message(path, text)}\n\n---\n\n{question}", PLAIN_NOTE
                entry["plain_note"] = True
            else:
                note = blocks.Note(text)
                asked = f"{open_note_message(path, note)}\n\n---\n\n{question}"
        else:
            asked, no_edit = question, NO_NOTE
        reader = lookup.Reader(con, vault, default_folder=scope,
                               open_note=(path, note) if note is not None else None)
        made: set[str] = set()

        def create(new_path, markdown):
            new = edits_mod.create_note(vault, new_path, markdown, folder=folder)
            if new["path"].casefold() in made:
                raise blocks.ToolError(f"{new['path']} is already made in this turn; give the next "
                                       "note a name of its own", "exists")
            made.add(new["path"].casefold())
            return new

        shown, skipped = context.open_note_images(vault, source, []) if source else ([], [])
        entry.update(note_images=len(shown), note_images_skipped=len(skipped))
        transcript, embedded, current = _conversation(history, asked, images, shown)
        system_turn = {"role": "system", "content": f"{PERSONA}\n\n{SKILL_RIDER}" if skill else loop_system()}
        pinned_turns = ([{"role": "user", "content": pinned}, {"role": "assistant", "content": "Noted."}]
                        if pinned else [])
        # Pi's setup: every call but a step's thinks, a skill's as before (Deep only).
        think = mode == "deep" or not skill
        think_budget = edit_run.DEEP_THINK_BUDGET if mode == "deep" and not skill else edit_run.THINK_BUDGET
        room = QUICK_NUM_PREDICT + (think_budget if think else 0)
        fixed = [system_turn, *pinned_turns, *embedded, current]
        if sum(map(_tokens, fixed)) + room > llm.RESIDENT_CTX:
            raise CopilotError("the open note and the pins are too long for one request; unpin something")
        recent = fit_window(transcript, fixed, room)
        messages = [system_turn, *pinned_turns, *recent, *embedded, current]

        def fits(sent: list[dict], results: list[str]) -> bool:
            """THE WINDOW RULE for what a lookup brings back: the prompt with these results and
            the reply's room still inside the resident window."""
            used = sum(map(_tokens, sent)) + sum(map(len, results)) / CHARS_PER_TOKEN
            return used + room <= llm.RESIDENT_CTX

        out = edit_run._turn(list(messages), think=think, tools=tools, step_tools=step_tools,
                             note=note, before_text=text or "", create=create, think_budget=think_budget,
                             think_stage="Thinking deeply" if mode == "deep" else "Thinking",
                             temperature=edit_run.FIRST_TEMPERATURE if skill else LOOP_TEMPERATURE,
                             sampling=None if skill else LOOP_SAMPLING, first_predict=room,
                             on_edit=on_edit or (lambda _event, _payload: None), should_stop=should_stop,
                             on_stage=stage, lookups=reader, on_reply=on_delta, no_edit=no_edit,
                             # filled as the turn runs: traced even when it fails or is refused
                             looked_up=entry.setdefault("looked_up", []), on_reset=on_reset, fits=fits,
                             on_tool=on_tool,
                             # pins, lookups and history fill it too, not only the note
                             too_long="this conversation no longer fits the model's window; "
                                      "start a new chat or unpin something")
    if out["kind"] == "answer" and out["refusals"]:
        entry["refusals"] = [item["code"] for item in out["refusals"]]
    last = out["timing"][-1] if out["timing"] else {}
    run = None
    if out["kind"] == "answer":     # it streamed as it decoded (on_delta)
        answer = out["answer"]
        if not answer and last.get("thinking_ate_the_budget"):
            raise CopilotError("the model spent its whole output budget thinking and wrote no answer — "
                               + ("ask again in Quick mode, or " if mode == "deep" else "") + "ask a narrower question")
        if not answer and out["stopped"]:
            raise CopilotError(f"the turn ended before an answer ({out['stopped']})")
        if not answer:
            raise CopilotError("the local model returned no answer")
    else:
        run = {"plan": out["plan"], "steps_done": min(out["steps_done"], len(out["plan"]["steps"])),
               "changes": out["changes"], "stopped": out["stopped"], "outline_met": out["outline_met"],
               "hash_before": edits_mod.digest(text or ""),
               "hash_after": edits_mod.digest(note.text() if note is not None else text or ""),
               "summary": out["summary"]}
        answer = out["summary"] or out["plan"]["intro"]
        entry["edit_run"] = edit_run._facts(out, run)
    # Sources are the notes code saw the turn look at, with no call to decide them: a skill's other
    # notes, then those it read or its searches found (in order, at most SOURCES_MAX), then the
    # notes pinned. The open note only when a tool read it: it rides with every question.
    notes = {str(pin.get("path") or "") for pin in pins if pin.get("kind") != "folder"}
    pinned_notes = [p for p in held if p in notes]
    looked = [p for p in dict.fromkeys(reader.seen) if p != path or p in reader.read][:SOURCES_MAX]
    paths = [p for p in dict.fromkeys(cited + looked + pinned_notes) if p != path or p in reader.read]
    result = {
        "answer": answer,
        "mode": mode,
        "citations": [{"n": i + 1, "path": p, "title": PurePosixPath(p).stem} for i, p in enumerate(paths)],
        "verification": {"truncated": bool(last.get("output_truncated"))},
        "history_dropped": len(transcript) - len(recent),
        "timing": {"total_ms": round((time.time() - started) * 1000), "gate": {},
                   "answer": out["timing"][0] if out["timing"] else {}, "sources": {}},
        "edit_run": run,
        "looked_up": out["looked_up"],
        "tools": out["tools"],
    }
    if inputs is not None:
        result["inputs"] = inputs
    return result


# The plugin starts an earlier question with the note it was asked on: "(on Notes/a.md) /quiz".
_ASKED_ON = re.compile(r"\(on [^\n]*?\) (?=/)")


def _expanded(text: str, loaded: dict) -> str:
    """An earlier question as the model reads it: a /command is its skill's whole prompt, after
    the note it was asked on."""
    on = _ASKED_ON.match(text)
    head = on.group() if on else ""
    return head + skills_mod.expand(text[len(head):], loaded)


def _source(con, vault: Path, open_path: str | None) -> dict | None:
    """The open note's source row, found by its path. In THE LOOP a note the index leaves out
    (`Profile/`, `_Reflections/`…) is still the open note, read from disk; a file that is not a
    note, or none, is no note open. Off, the gate's path needs an indexed note."""
    if open_path is None:
        if not LOOP:
            raise CopilotError("open a note to ask about it")
        return None
    pure = PurePosixPath(open_path) if isinstance(open_path, str) else None
    if pure is None or not open_path.strip() or pure.is_absolute() or ".." in pure.parts:
        raise CopilotError("the open note's path is not a vault path")
    row = con.execute("SELECT id, path, title, type, current_hash FROM sources "
                      "WHERE path=? AND deleted=0", (open_path,)).fetchone()
    if row is not None:
        return dict(row)
    if not LOOP:
        raise CopilotError("the note is no longer indexed; refresh the sidebar")
    if pure.suffix.lower() != ".md":
        return None
    if not (Path(vault) / open_path).is_file():
        raise CopilotError("the open note could not be read; refresh the sidebar")
    return {"id": None, "path": open_path, "title": pure.stem}


def run_turn(con, open_path: str | None, question: str, history: list[dict], mode: str,
             *, pins=(), images=None, on_stage=None, on_delta=None, on_edit=None, selection: str = "",
             selection_from: int | None = None, vault: Path | None = None, doc: str | None = None,
             doc_hash: str | None = None, should_stop=lambda: False, on_reset=None, on_tool=None) -> dict:
    """One sidebar turn, on the note open at `open_path` (None: no note open), with the chat's
    `@` pins. With LOOP it is one loop that answers, looks things up and edits (`_loop`); an edit
    or new-note skill runs as before. Off, a `/command` whose output is chat reads whole notes
    chosen by code (`gather`) and answers, and every other turn meets the intent gate. A turn may
    change the open note, which it reads from `doc` — the editor's text, whose `doc_hash` must
    match — or else from disk; `selection_from` is where the selection starts in it, in UTF-16
    units. `on_edit(event, payload)` gets each part of an edit as it lands; nothing is written
    here: the plugin applies each change through the editor. In THE LOOP `on_reset()` hears each
    model call after a turn's first start: what `on_delta` streamed before it is not the answer, and
    `on_tool(tool)` each lookup once it ran, {name, label, refused}: the turn keeps them as "tools"."""
    vault = Path(vault or config_mod.VAULT)
    source = _source(con, vault, open_path)
    loaded = skills_mod.load(vault)
    found = skills_mod.invocation(question, loaded) if question.startswith("/") else None
    skill, args = found or (None, "")
    if skill and skill.mode in ("quick", "deep"):
        mode = skill.mode                      # the skill's author chose its depth
    # A /command in an earlier turn is its whole prompt to the model: "/quiz" alone never told
    # the answering turn how to grade.
    history = [{**turn, "text": _expanded(turn.get("text") or "", loaded)}
               if turn.get("role") == "you" else turn for turn in history]
    # Machine facts only (no prose, no reasoning): the thread JSON keeps completed turns,
    # so this is the one record of a refused or stopped turn.
    entry = {"question": question, "source": source["path"] if source else None,
             "source_id": source["id"] if source else None, "open_path": open_path,
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
        if LOOP and skill and source is None:
            raise CopilotError(f"/{skill.name} works on the open note: open a note first")
        if LOOP and not (skill and skill.output != "chat"):
            result = _loop(con, vault, source, question, history, mode, pins=pins, skill=skill, args=args,
                           images=images, selection=selection, doc=doc, on_stage=on_stage,
                           on_delta=on_delta, on_edit=on_edit, should_stop=should_stop, entry=entry,
                           on_reset=on_reset, on_tool=on_tool)
        elif skill and skill.output == "chat":
            pack = context.gather(con, vault, source, skill=skill, links_from=args, selection=selection)
            cited = [{"n": i + 1, "path": doc["path"], "title": doc["title"]}
                     for i, doc in enumerate(d for d in pack.docs if d["role"] != "selection")]
            # No retrieval orders a skill's images, so they come in note order.
            shown, skipped = context.open_note_images(vault, source, [])
            entry.update(note_images=len(shown), note_images_skipped=len(skipped))
            result = generate_answer(
                skills_mod.render_prompt(skill, args), history, [], mode,
                images=images, note_images=shown, on_stage=on_stage, on_delta=on_delta,
                system=pack_message(pack, ASK_RIDER), cited=cited)
            result.update(inputs=context.pack_inputs(pack), edit_run=None)
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
    result.setdefault("looked_up", [])
    result.setdefault("tools", [])
    if result["edit_run"]:
        result["edit_run"]["path"] = source["path"] if source else None
    record = {
        **entry, "status": "edited" if result["edit_run"] else "answered",
        "cited": sorted({item["path"] for item in result["citations"]}),
        "truncated": result["verification"]["truncated"],
        "history_dropped": result["history_dropped"], "timing": result["timing"],
        "looked_up": result["looked_up"],
    }
    if "inputs" in result:
        record["inputs"] = result["inputs"]
    else:
        record.update({"evidence": [item["path"] for item in evidence], "retrieval_levels": levels})
    trace.record("copilot", record)
    return result
