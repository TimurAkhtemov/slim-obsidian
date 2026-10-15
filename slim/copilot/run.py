"""The copilot's editing run: one model call at a time against the open note's handles, code
holding the plan and every limit, each change handed to the plugin as it lands. `copilot/turn.py`
decides a turn edits and builds its messages; `edits.py` applies each call."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import llm
from . import blocks, edits as edits_mod, skills as skills_mod
from .context import CopilotError

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
NUDGE = "Use the tools to make the change."
FIRST_TEMPERATURE = llm.REPLY_TEMPERATURE
ONE_REPLY = "(done in one reply)"      # the one step of a run whose first reply made its change


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


def _call(messages: list[dict], offered: list[str], *, think: bool, temperature: float,
          num_predict: int, on_part=None, should_stop=None) -> _Reply:
    """One model call with the `offered` tools. It streams: `on_part` hears its tool calls as they
    decode (`llm_render._Reader`). A call's text is never JSON: `\\frac` and a real tab arrive as written."""
    msg, stats = llm.chat_tools(messages, edits_mod.ollama_tools(offered), think=think,
                                temperature=temperature, num_predict=num_predict,
                                on_part=on_part, should_stop=should_stop)
    calls = [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in msg["tool_calls"]]
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
            heading = blocks.ANNOTATION.sub(
                "", blocks.strip_labels(line.strip().strip("\"'").strip()).strip())
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


def _against(note: blocks.Note, outline: list[str]) -> tuple[list[str], list[str]]:
    """The note's headings and the outline, both without a recording's locked headings:
    no step can change those, so they are never asked for."""
    locked = {blocks.heading_key(h) for b in note.blocks if b.locked for h in blocks.headings(b.text)}
    now = [h for b in note.blocks if not b.locked for h in blocks.headings(b.text)]
    return now, [h for h in outline if blocks.heading_key(h) not in locked]


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
    a, b = [blocks.heading_key(h) for h in want], [blocks.heading_key(h) for h in now]
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
    level = lambda h: blocks.heading_key(h)[0]
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
    only places it, in UTF-16 units of the run's draft as a change's splices are, where the tool
    will put the change (`edits.lands`, which the tool calls too); it is never the note's text. The
    change that lands names the draft it replaces; a call refused or never applied ends its draft,
    and one that turns into text ends it at once."""
    PLACES = {"insert": ("after", "before"), "replace": ("target",)}

    def __init__(self, note: blocks.Note, only: set[str] | None, on_edit):
        self.note, self.only, self.on_edit = note, only, on_edit
        self.reply, self.calls, self.open, self.failed = 0, {}, [], 0
        self.begun = 0                      # the reply's calls so far, dropped ones too: each draft id is its own

    def next_reply(self) -> None:
        self.end()
        self.reply, self.calls, self.begun = self.reply + 1, {}, 0

    def part(self, kind: str, index: int, name, value) -> None:
        """`on_part` for a model call (`llm_render._Reader`). A draft is only ever shown: one that fails is
        counted for the trace, and the call goes on."""
        try:
            self._part(kind, index, name, value)
        except Exception:                   # noqa: BLE001
            self.failed += 1

    def _part(self, kind: str, index: int, name, value) -> None:
        if kind == "drop":                  # it turned into text: its draft goes, and the next call has its index
            dead = self.calls.pop(index, None)
            if dead and dead["id"] in self.open:
                self.open.remove(dead["id"])
                self.on_edit("draft_end", {"id": dead["id"]})
            return
        call = self.calls.setdefault(index, {"tool": None, "place": None, "text": [], "id": None, "at": None,
                                             "line": True, "n": self.begun})
        if kind == "call":
            call["tool"], self.begun = name, self.begun + 1
        elif call["tool"] not in self.PLACES:
            return
        elif kind == "value" and name in self.PLACES[call["tool"]]:
            call["place"] = (name, value)   # given again, the last one is the call's: the draft moves
            self._show(index, call)
        elif kind == "parameter" and name == "markdown" and call["text"]:
            call["text"], call["line"] = [], True   # given again, the last one is the call's: it starts over
            self._show(index, call)
        elif kind == "text" and name == "markdown":
            # A heading can move an insert past a section: the text decides the place, so it is
            # placed again as each line starts and ends.
            placed = call["line"] or "\n" in value
            call["text"].append(value)
            call["line"], call["whole"] = value.endswith("\n"), False
            if call["id"] and placed and self._where(call) != call["at"]:
                self._show(index, call)
            elif call["id"]:
                self.on_edit("draft_text", {"id": call["id"], "text": value})
        elif kind == "value" and name == "markdown":
            call["whole"] = True
            if call["id"] and self._where(call) != call["at"]:
                self._show(index, call)

    def _where(self, call: dict) -> tuple[int, int] | None:
        """Where the call's change will go: where its tool puts it with the text so far
        (`edits.lands`). A rename dims its heading alone only once its text is whole: one heading
        line still being written is as likely the start of a section's new text."""
        if call["place"] is None:
            return None
        name, value = call["place"]
        try:
            i, j, _name, _said, renames = edits_mod.lands(self.note, call["tool"], {name: value},
                                                          blocks.strip_labels("".join(call["text"])), self.only)
        except blocks.ToolError:
            return None                     # no such place: the call will say so when applied
        return self.note.offset(i), self.note.offset(i + 1 if renames and call.get("whole") else j)

    def _show(self, index: int, call: dict) -> None:
        at = self._where(call)
        if at is None:
            return
        if call["id"] is None:
            call["id"] = f"d{self.reply}.{call['n']}"
            self.open.append(call["id"])
        call["at"] = at
        text = self.note.text()
        self.on_edit("draft", {"id": call["id"], "from": edits_mod.u16(text[:at[0]]),
                               "to": edits_mod.u16(text[:at[1]]), "text": "".join(call["text"])})

    def of(self, index: int) -> str | None:
        return (self.calls.get(index) or {}).get("id")

    def landed(self, draft: str) -> None:
        if draft in self.open:
            self.open.remove(draft)

    def end(self) -> None:
        drafts, self.open = self.open, []
        for draft in drafts:
            try:
                self.on_edit("draft_end", {"id": draft})
            except Exception:               # noqa: BLE001
                self.failed += 1


def _turn(messages: list[dict], *, think: bool, tools: list[str], step_tools: tuple[str, ...],
          note: blocks.Note | None, before_text: str, only: set[str] | None = None, create=None,
          temperature: float = FIRST_TEMPERATURE, first_predict: int, nudge: bool = False, on_edit,
          should_stop, on_stage=lambda _name: None) -> dict:
    """One editing turn. Every call offers the same `tools`, in the same order: they sit at the
    top of the prompt, and a different list re-reads the whole note. `step_tools` are those that
    change something. The first call answers in text, makes a small change at once, or plans a
    larger one, at `temperature` and within `first_predict` tokens; with `nudge` (the gate wanted a
    change) a text-only reply is asked once to use the tools. A plan is carried out one call per step; its outline is only
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
                    raise blocks.ToolError("the arguments must be an object", "args")
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
                # The work is finished: this step's calls stand, no more steps. Only where done is
                # offered (a new note); elsewhere it is refused like any tool not offered.
                if name == "done" and "done" in tools:
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
                        raise blocks.ToolError(f"{name} is not available now", "tool")
                    change, said = land(name, call.get("arguments") or {}, at)
                except blocks.ToolError as exc:
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


def _index_of_u16(text: str, units: int) -> int | None:
    """The index into `text` of an offset in UTF-16 units (the editor's), or None when no
    character of `text` starts there."""
    encoded = text.encode("utf-16-le")
    if not 0 <= 2 * units <= len(encoded):
        return None
    try:
        return len(encoded[:2 * units].decode("utf-16-le"))
    except UnicodeDecodeError:          # inside a character outside the BMP
        return None


def _selected(note: blocks.Note, text: str, selection: str, skill: skills_mod.Skill,
              selection_from: int | None = None) -> set[str]:
    """The blocks a selection touches: the only ones a selection skill may change. It is where
    the editor says it starts (`selection_from`): the same words can be in a note twice. Without
    that, or when the words are not there, they are found, if the note has them once."""
    chosen = (selection or "").strip()
    if not chosen:
        raise CopilotError(f"/{skill.name} works on selected text: select some in the note first")
    start = _index_of_u16(text, selection_from) if selection_from is not None else None
    if start is not None and text.startswith(selection, start):
        start += len(selection) - len(selection.lstrip())
    elif (count := text.count(chosen)) > 1:
        raise CopilotError(f"the selected text appears {count} times in the note; select a little more around it")
    elif not count:
        raise CopilotError("the selected text is not in the note as it is now; select it again")
    else:
        start = text.find(chosen)
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
