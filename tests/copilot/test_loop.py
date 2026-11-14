"""One agent loop: every call is offered the lookup tools and, with a note open, the edit tools;
the model decides. The open note rides with the question, `@` pins in a message of their own."""

from pathlib import Path

import pytest

from slim import db, llm
from slim.copilot import context, edits, run as edit_run, turn as copilot_turn
from slim.index import search

from .conftest import scripted

OPEN = "Notes/school/n.md"
NOTE = "# Title\n\nIntro para.\n\n## A\n\nText a.\n"
LOOKUPS = ["search", "read_note", "list_folder"]
RESET = None                    # a new model call: the plugin clears the text streamed so far
ALL_TOOLS = [*LOOKUPS, "plan", "replace", "edit_text", "insert", "move", "create_note"]


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "copilot.db")


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Notes/school/cs-201").mkdir(parents=True)
    (root / "Notes/school/cs-202").mkdir(parents=True)
    return root


@pytest.fixture(autouse=True)
def lexical_only(monkeypatch):
    """Search is the index's own (FTS), with no embedder."""
    monkeypatch.setattr(search, "vector", lambda *a, **k: [])


def put(con, vault, path: str, text: str, *, index: bool = True) -> str:
    target = vault / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    if index:
        context.sync_note(con, vault, path, embed=False)
    return path


@pytest.fixture
def ask(con, vault):
    """run_turn in loop mode on `open_path` (written with `text` when given). Returns the result,
    and its on_edit events, deltas and stages."""
    def go(question, *, open_path=OPEN, text=NOTE, pins=(), history=(), effort="medium", doc=True, tools=None):
        sent = {}
        if open_path is not None and text is not None:
            put(con, vault, open_path, text, index=not open_path.startswith("Profile/"))
            if doc:
                sent = {"doc": text, "doc_hash": edits.digest(text)}
        events, deltas, stages = [], [], []
        result = copilot_turn.run_turn(con, open_path, question, list(history), effort, pins=list(pins),
                                       vault=vault, on_edit=lambda e, p: events.append((e, p)),
                                       on_delta=deltas.append, on_reset=lambda: deltas.append(RESET),
                                       on_stage=stages.append, on_tool=tools.append if tools is not None else None,
                                       **sent)
        result["events"], result["deltas"], result["stages"] = events, deltas, stages
        return result
    return go


def test_question_about_open_note_needs_no_tools(ask, monkeypatch):
    seen = scripted(monkeypatch, ("It introduces A.", []))
    got = ask("what is this note about?")
    assert got["answer"] == "It introduces A." and got["edit_run"] is None
    assert got["deltas"] == ["It introduces A."]                # streamed as it decoded, said once
    assert got["looked_up"] == [] and got["citations"] == [] and len(seen) == 1
    assert seen[0]["opts"]["temperature"] == copilot_turn.LOOP_TEMPERATURE == 1.0


def test_lookup_then_answer(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    seen = scripted(monkeypatch,
                    ("", [("search", {"query": "zebras"})]),
                    ("", [("read_note", {"path": "Notes/school/cs-201/zebra.md"})]),
                    ("They have stripes.", []))
    got = ask("what did I write about zebras?")
    assert got["answer"] == "They have stripes." and len(seen) == 3
    assert [c["path"] for c in got["citations"]] == ["Notes/school/cs-201/zebra.md"]   # searched and read: once
    searched, read = seen[1]["messages"][-1], seen[2]["messages"][-1]
    assert searched["role"] == "tool" and "Notes/school/cs-201/zebra.md" in searched["content"]
    assert read["role"] == "tool" and read["content"].startswith("=== Notes/school/cs-201/zebra.md ===")
    assert "Zebras have stripes." in read["content"]
    assert got["looked_up"] == [{"name": "search", "refused": False},
                                {"name": "read_note", "path": "Notes/school/cs-201/zebra.md", "refused": False}]
    assert all(call["tools"] == seen[0]["tools"] for call in seen)       # one tool list, one cached prefix


def test_lookup_then_edit(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/b.md", "# B\n\nThe better intro.\n")
    seen = scripted(monkeypatch,
                    ("", [("read_note", {"path": "Notes/school/b.md"})]),
                    ("", [("edit_text", {"target": "b2", "old": "Intro para.", "new": "The better intro."})]))
    got = ask("use the intro from b.md")
    run = got["edit_run"]
    assert len(run["changes"]) == 1 and run["path"] == OPEN
    assert seen[1]["opts"]["temperature"] == copilot_turn.LOOP_TEMPERATURE   # no edit had happened yet
    final = edits.apply_splices(NOTE, run["changes"][0]["splices"])
    assert "The better intro." in final and (Path(vault) / OPEN).read_text() == NOTE


def test_lookup_cap(ask, monkeypatch):
    lookups = [("", [("list_folder", {"path": "Notes"})])] * 7
    seen = scripted(monkeypatch, *lookups, ("Done looking.", []))
    got = ask("look around")
    assert got["answer"] == "Done looking." and len(seen) == 8
    for n in range(1, 7):
        assert not seen[n]["messages"][-1]["content"].startswith("refused:")
    assert seen[7]["messages"][-1]["content"].startswith("refused: answer or act now")
    assert got["looked_up"][-1]["refused"] is True and len(got["looked_up"]) == 7


def test_no_open_note_offers_the_same_tools(ask, monkeypatch):
    seen = scripted(monkeypatch, ("Hello.", []))
    got = ask("hi", open_path=None)
    assert seen[0]["tools"] == ALL_TOOLS
    assert got["answer"] == "Hello." and got["source"] is None
    assert seen[0]["messages"][-1] == {"role": "user", "content": "hi"}


def test_with_an_open_note_the_tools_are_lookups_then_edits(ask, monkeypatch):
    seen = scripted(monkeypatch, ("ok", []))
    ask("q")
    assert seen[0]["tools"] == ALL_TOOLS


def test_note_skill_without_open_note_refuses(ask, monkeypatch):
    seen = scripted(monkeypatch)
    with pytest.raises(copilot_turn.CopilotError, match="open a note"):
        ask("/quiz", open_path=None)
    assert seen == []


def test_long_note_edit_needs_read_section(ask, monkeypatch):
    sections = [f"## Part {n}\n\n" + f"Paragraph {n} " + "word " * 1200 + "\n\n" for n in range(10)]
    text = "# Long\n\n" + "".join(sections)
    assert len(text) > 60_000
    from slim.copilot import blocks
    note = blocks.Note(text)
    last_head = [b for b in note.blocks if b.text.startswith("## Part 9")][0]
    section, para = "s" + last_head.handle[1:], "b" + str(int(last_head.handle[1:]) + 1)
    seen = scripted(monkeypatch,
                    ("", [("replace", {"target": para, "markdown": "Short now."})]),
                    ("", [("read_note", {"path": OPEN, "section": section})]),
                    ("", [("replace", {"target": para, "markdown": "Short now."})]))
    got = ask("shorten the last part", text=text)
    asked = seen[0]["messages"][-1]["content"]
    first_unread = next(b.handle for b in note.blocks if b.level and f"[{b.handle}]" not in asked)
    assert len(asked) < len(text) and f'read_note("{OPEN}", "s{first_unread[1:]}")' in asked
    assert "a range like" in asked
    assert f"{section}  " in asked                                # the sections map names every section
    refused = seen[1]["messages"][-1]["content"]
    assert "read_note it first" in refused
    run = got["edit_run"]
    assert len(run["changes"]) == 1 and len(seen) == 3
    assert "Short now." in edits.apply_splices(text, run["changes"][0]["splices"])


def test_open_note_rides_with_question(ask, monkeypatch):
    seen = scripted(monkeypatch, ("ok", []))
    ask("what is A?")
    system, current = seen[0]["messages"][0], seen[0]["messages"][-1]
    assert system["role"] == "system" and "Intro para." not in system["content"]
    assert system["content"] == copilot_turn.loop_system()
    assert current["role"] == "user"
    assert current["content"].startswith(f"=== {OPEN} · the open note ===\nSections:\n")
    assert "[b2] Intro para." in current["content"]
    assert current["content"].endswith("\n\n---\n\nwhat is A?")


def test_the_loop_prompt_is_pi_shaped_and_vault_first(ask, monkeypatch):
    """Like the Pi coding agent's: who it is and where, one line per tool, short rules. Questions
    about the owner's own life are looked up first; general knowledge only after, and labelled."""
    monkeypatch.setattr(copilot_turn.config_mod, "OWNER", "Sam")
    seen = scripted(monkeypatch, ("ok", []))
    ask("what is A?")
    system = seen[0]["messages"][0]["content"]
    assert system.startswith("You are SLIM, an assistant operating inside Sam's Obsidian vault. You help Sam by "
                             "searching and reading their notes, answering questions, and editing the note they have open.")
    for name in copilot_turn.LOOP_TOOLS:
        assert f"\n- {name}: " in system                               # every tool offered, one line each
    assert "general knowledge freely" not in system and "tutor" not in system
    assert "Questions about Sam's own work, people, plans, meetings or numbers are answered from the vault" in system
    assert "look them up before answering" in system
    assert "If the vault doesn't have it, say so" in system and "labelled as such" in system
    assert "Never present a guess about Sam's situation as fact" in system
    assert "Be concise" in system and "by title" in system
    assert "Write math as $inline$ or $$display$$, never \\( \\) or \\[ \\]." in system
    # The measured handle and plan sentences, word for word.
    assert copilot_turn.EDIT_RIDER.removeprefix("You can edit the open note shown below.") in system


def test_a_skill_keeps_its_own_prompt(ask, monkeypatch):
    seen = scripted(monkeypatch, ("Question 1?", []))
    ask("/quiz")
    assert seen[0]["messages"][0]["content"] == f"{copilot_turn.PERSONA}\n\n{copilot_turn.SKILL_RIDER}"


def test_pins_ride_before_history(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/b.md", "# Bee\n\nBuzzing text.\n")
    seen = scripted(monkeypatch, ("ok", []))
    history = [{"role": "you", "text": "earlier"}, {"role": "slim", "text": "before"}]
    ask("q", pins=[{"kind": "note", "path": "Notes/b.md"}], history=history)
    messages = seen[0]["messages"]
    assert messages[1]["role"] == "user" and messages[1]["content"].startswith("Context I've pinned:\n")
    assert "Buzzing text." in messages[1]["content"]
    assert messages[2] == {"role": "assistant", "content": "Noted."}
    assert messages[3] == {"role": "user", "content": "earlier"}


def test_pinned_folder_scopes_search(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/graphs.md", "# Graphs\n\nDijkstra shortest path.\n")
    put(con, vault, "Notes/school/cs-202/sorting.md", "# Sorting\n\nDijkstra is not a sort.\n")
    seen = scripted(monkeypatch, ("", [("search", {"query": "Dijkstra"})]), ("ok", []))
    ask("q", pins=[{"kind": "folder", "path": "Notes/school/cs-202"}])
    assert "Searches default to Notes/school/cs-202/;" in seen[0]["messages"][1]["content"]
    found = seen[1]["messages"][-1]["content"]
    assert "cs-202/sorting.md" in found and "cs-201" not in found


def test_note_in_excluded_folder_is_context(ask, monkeypatch):
    seen = scripted(monkeypatch, ("ok", []))
    got = ask("who am I?", open_path="Profile/me.md", text="# Me\n\nI like graphs.\n")
    assert "I like graphs." in seen[0]["messages"][-1]["content"]
    assert got["source"] == {"id": None, "path": "Profile/me.md", "title": "me"}


def test_chat_skill_can_look_up(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    seen = scripted(monkeypatch, ("", [("search", {"query": "zebras"})]),
                    ("", [("read_note", {"path": "Notes/school/cs-201/zebra.md"})]), ("Question 1: stripes?", []))
    got = ask("/quiz")
    assert seen[0]["tools"] == LOOKUPS
    assert got["answer"] == "Question 1: stripes?" and got["skill"] == "quiz"
    assert [c["path"] for c in got["citations"]] == ["Notes/school/cs-201/zebra.md"]
    current = seen[0]["messages"][-1]["content"]
    assert f"=== {OPEN} · the open note ===" in current and "Quiz me on this note" in current


def test_the_trace_names_what_was_looked_up_and_the_open_path(ask, monkeypatch):
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    scripted(monkeypatch, ("", [("read_note", {"path": "Notes/missing.md"})]), ("Not there.", []))
    ask("read missing")
    (record,) = records
    assert record["open_path"] == OPEN and record["status"] == "answered"
    (looked,) = record["looked_up"]
    assert looked["name"] == "read_note" and looked["refused"] is True
    assert looked["said"].startswith("refused: there is no note")


def test_a_lookup_inside_a_step_runs_and_the_step_goes_on(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/b.md", "# B\n\nBee facts.\n")
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Add bee facts.", "steps": ["Add the facts after b2"]})]),
                    ("", [("read_note", {"path": "Notes/school/b.md"})]),
                    ("", [("insert", {"after": "b2", "markdown": "Bee facts."})]))
    got = ask("add the bee facts")
    run = got["edit_run"]
    assert len(seen) == 3 and len(run["changes"]) == 1 and run["steps_done"] == 1
    assert "Bee facts." in seen[2]["messages"][-1]["content"]          # the read went back to the step
    assert [p for e, p in got["events"] if e == "step"] == [{"index": 1, "state": "doing"},
                                                            {"index": 1, "state": "done"}]


# --- fix round 1: a long transcript pages, failures trace their lookups, plain notes, sources --

TRANSCRIPT_NOTE = ("---\ntitle: Call\n---\n\n```slim-meeting\n## Transcript\n\n"
                   + "".join(f"Me: line {i} " + "word " * 20 + "\n" for i in range(1100))
                   + "```\n\nMy notes after the call.\n")


def _fence(text):
    from slim.copilot import blocks
    return next(b for b in blocks.Note(text).blocks if b.locked and "Me: line 0" in b.text).handle


def test_long_recording_transcript_pages_through_read_note(ask, monkeypatch):
    assert len(TRANSCRIPT_NOTE) > 120_000
    fence = _fence(TRANSCRIPT_NOTE)
    pages = [("", [("read_note", {"path": OPEN, "section": fence if n == 1 else f"{fence}#{n}"})]) for n in (1, 2, 3, 4)]
    seen = scripted(monkeypatch, *pages, ("Read it all.", []))
    got = ask("what did I say at the end?", text=TRANSCRIPT_NOTE)
    asked = seen[0]["messages"][-1]["content"]
    assert f"[locked {fence}]" in asked and f'read_note("{OPEN}", "{fence}")' in asked
    assert "My notes after the call." in asked and "Me: line 5 " not in asked
    first, second, last = (seen[n]["messages"][-1]["content"] for n in (1, 2, 4))
    assert first.startswith(f"=== {OPEN} ===\n[locked {fence}, page 1 of 4") and "Me: line 0 " in first
    assert first.endswith(f'[Next page: read_note("{OPEN}", "{fence}#2")]')
    assert f"page 2 of 4" in second and second.endswith(f'"{fence}#3")]')
    assert "page 4 of 4" in last and "Me: line 1099 " in last and "Next page" not in last
    assert all(not looked["refused"] for looked in got["looked_up"])


def test_locked_page_never_editable(ask, monkeypatch):
    fence = _fence(TRANSCRIPT_NOTE)
    seen = scripted(monkeypatch, ("", [("read_note", {"path": OPEN, "section": fence})]),
                    ("", [("replace", {"target": fence, "markdown": "gone"})]), ("", []))
    got = ask("delete the transcript", text=TRANSCRIPT_NOTE)
    assert "part of the recording; SLIM never edits it" in seen[2]["messages"][-1]["content"]
    assert got["edit_run"]["changes"] == []


def test_search_can_look_inside_one_note(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    put(con, vault, "Notes/school/cs-202/horse.md", "# Horses\n\nZebras are not horses.\n")
    seen = scripted(monkeypatch, ("", [("search", {"query": "zebras", "folder": "Notes/school/cs-202/horse.md"})]),
                    ("ok", []))
    got = ask("q")
    found = seen[1]["messages"][-1]["content"]
    assert not found.startswith("refused:") and "Zebras are not horses." in found and "zebra.md" not in found
    assert [c["path"] for c in got["citations"]] == ["Notes/school/cs-202/horse.md"]   # a search hit is looked at


def test_failed_turn_traces_lookups(ask, monkeypatch):
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    scripted(monkeypatch, ("", [("list_folder", {"path": "Notes"})]), ("", []))
    with pytest.raises(copilot_turn.CopilotError, match="no answer"):
        ask("q")
    (record,) = records
    assert record["status"] == "failed" and record["open_path"] == OPEN
    assert record["looked_up"] == [{"name": "list_folder", "path": "Notes", "refused": False}]


def test_crlf_note_is_asked_about_not_refused(ask, monkeypatch):
    seen = scripted(monkeypatch, ("It has two lines.", []))
    got = ask("what is in it?", text="# Win\r\n\r\nLine one.\r\nLine two.\r\n", doc=False)
    assert got["answer"] == "It has two lines." and got["edit_run"] is None
    assert seen[0]["tools"] == ALL_TOOLS
    current = seen[0]["messages"][-1]["content"]
    assert current.startswith(f"=== {OPEN} · the open note ===\n# Win\n\nLine one.\nLine two.")
    assert "[b1]" not in current and "Sections:" not in current and "\r" not in current


def test_pinned_notes_are_sources(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/b.md", "# Bee\n\nBuzzing text.\n")
    scripted(monkeypatch, ("ok", []))
    got = ask("q", pins=[{"kind": "note", "path": "Notes/b.md"}, {"kind": "folder", "path": "Notes/school"}])
    assert [c["path"] for c in got["citations"]] == ["Notes/b.md"]


# --- fix round 2: one prompt, one tool list; no room; stages; the failure trace ---------------

CRLF = "# Win\r\n\r\nLine one.\r\nLine two.\r\n"


def test_one_system_message_and_tool_list_whatever_is_open(ask, monkeypatch):
    """The system message and the tools sit at the top of the prompt: a note, no note, a note the
    tools cannot take, and one too long to edit all read the same prefix."""
    monkeypatch.setattr(edits, "MAX_EDIT_CHARS", 20)
    seen = scripted(monkeypatch, *[("ok", [])] * 5)
    ask("q", text="# T\n\nx\n")                                 # a Markdown note, small enough
    ask("q", open_path=None)                                       # no note, or the chip's ✕
    ask("q", text=CRLF, doc=False)                                 # Windows line endings
    ask("q", doc=False)                                            # over the plugin's limit: no doc
    ask("/quiz")                                                   # a skill keeps its own
    assert len({c["messages"][0]["content"] for c in seen[:4]}) == 1
    assert all(c["tools"] == ALL_TOOLS for c in seen[:4])
    assert seen[4]["tools"] == LOOKUPS
    # Over the edit limit, the model is shown the note's text whole, with no handles to point at.
    assert seen[3]["messages"][-1]["content"] == f"=== {OPEN} · the open note ===\n{NOTE.rstrip()}\n\n---\n\nq"


@pytest.mark.parametrize("open_path, text, doc, said, limit", [
    (None, None, True, "refused: no note is open to change", None),
    (OPEN, CRLF, False, "refused: this note can't be changed here; give the text to paste", None),
    # over the plugin's edit limit, sent without the editor's text
    (OPEN, NOTE, False, "refused: this note can't be changed here; give the text to paste", 20),
])
def test_an_edit_call_with_no_editable_note_is_refused_and_the_turn_answers(
        ask, monkeypatch, open_path, text, doc, said, limit):
    if limit:
        monkeypatch.setattr(edits, "MAX_EDIT_CHARS", limit)
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b2", "markdown": "New."})]),
                    ("Paste this: New.", []))
    got = ask("change it", open_path=open_path, text=text, doc=doc)
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert seen[1]["messages"][-1]["content"].startswith(said)
    assert got["answer"] == "Paste this: New." and got["edit_run"] is None
    assert got["events"] == []
    assert records[0]["refusals"] == ["no_note"]


def test_a_plan_with_no_note_is_refused_too(ask, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["a", "b"]})]),
                    ("", [("plan", {"intro": "x", "steps": ["a", "b"]})]),
                    ("", [("insert", {"after": "b1", "markdown": "y"})]))
    with pytest.raises(copilot_turn.CopilotError, match="before an answer"):
        ask("plan it", open_path=None)
    assert len(seen) == 3                                          # MAX_REFUSALS in a row stop it


def test_a_pinned_excluded_folder_never_scopes_search(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/graphs.md", "# Graphs\n\nDijkstra shortest path.\n")
    put(con, vault, "Skills/s.md", "# S\n\nx\n", index=False)
    seen = scripted(monkeypatch, ("", [("search", {"query": "Dijkstra"})]), ("ok", []))
    ask("q", pins=[{"kind": "folder", "path": "Skills"}])
    found = seen[1]["messages"][-1]["content"]
    assert not found.startswith("refused:") and "cs-201/graphs.md" in found


def test_a_failure_inside_the_turn_still_traces_its_lookups(ask, monkeypatch):
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    scripted(monkeypatch, ("", [("list_folder", {"path": "Notes"})]),
             llm.LLMError("the server failed"), llm.LLMError("the server failed"))
    with pytest.raises(llm.LLMError):
        ask("q")
    (record,) = records
    assert record["status"] == "failed" and record["error_type"] == "LLMError"
    assert record["looked_up"] == [{"name": "list_folder", "path": "Notes", "refused": False}]


def test_each_new_call_in_a_turn_resets_the_streamed_text(ask, monkeypatch):
    scripted(monkeypatch, ("Let me look.", [("list_folder", {"path": "Notes"})]), ("Here it is.", []))
    got = ask("what is in Notes?")
    assert got["deltas"] == ["Let me look.", RESET, "Here it is."]


def test_the_stage_line_names_each_lookup(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    scripted(monkeypatch, ("", [("search", {"query": "zebras"})]),
             ("", [("read_note", {"path": "Notes/school/cs-201/zebra.md"})]),
             ("", [("list_folder", {"path": "Notes/school"}), ("list_folder", {"path": ""})]), ("Found them.", []))
    got = ask("zebras?")
    assert got["stages"] == ["Thinking", "Searching", "Thinking", "Reading zebra", "Thinking",
                             "Listing Notes/school", "Listing the vault", "Thinking"]


def test_a_lookup_that_would_overflow_the_window_is_refused(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/small.md", "# Small\n\nA short note.\n")
    put(con, vault, "Notes/big.md", "# Big\n\n" + "A long sentence of words. " * 400)
    seen = scripted(monkeypatch, ("ok", []))
    ask("q")
    prompt = sum(map(copilot_turn._tokens, seen[0]["messages"]))
    # Room for the prompt, the reply, and a little more: the small note fits, the big one not.
    monkeypatch.setattr(llm, "RESIDENT_CTX", int(prompt + copilot_turn.ANSWER_NUM_PREDICT + edit_run.THINK_BUDGET + 200))
    seen = scripted(monkeypatch, ("", [("read_note", {"path": "Notes/small.md"})]),
                    ("", [("read_note", {"path": "Notes/big.md"})]), ("With what I have.", []))
    got = ask("q")
    assert seen[1]["messages"][-1]["content"].startswith("=== Notes/small.md ===")
    assert seen[2]["messages"][-1]["content"] == ("refused: no room left in this conversation; "
                                                  "answer with what you have")
    assert got["answer"] == "With what I have."
    assert [looked["refused"] for looked in got["looked_up"]] == [False, True]
    assert [c["path"] for c in got["citations"]] == ["Notes/small.md"]   # what it never saw is no source


def test_a_saturated_window_names_the_conversation_not_the_note(ask, monkeypatch):
    seen = scripted(monkeypatch, ("", [("list_folder", {"path": "Notes"})]), ("cut", []))
    real = copilot_turn.llm.chat_tools

    def saturated(messages, tools=None, **opts):
        msg, stats = real(messages, tools, **opts)
        return msg, {**stats, "ctx_saturated": len(seen) == 2}
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", saturated)
    with pytest.raises(copilot_turn.CopilotError,
                       match="this conversation no longer fits the model's window; start a new chat or unpin something"):
        ask("q")


# --- Pi's setup: thinking on every loop call, a step as measured --------------------------------

@pytest.mark.parametrize("effort, budget, stage", [("low", 4096, "Thinking"), ("medium", 8192, "Thinking"),
                                                   ("high", 16384, "Thinking deeply")])
def test_every_loop_call_thinks_within_its_efforts_budget(ask, con, vault, monkeypatch, effort, budget, stage):
    """Pi's thinking levels. Thinking tokens count against the cap, so a call's cap is the
    answer's and the budget."""
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    seen = scripted(monkeypatch, ("", [("search", {"query": "zebras"})]), ("Stripes.", []))
    got = ask("zebras?", effort=effort)
    assert edit_run.THINK_BUDGETS == {"off": 0, "low": 4096, "medium": 8192, "high": 16384}
    for call in seen:
        assert call["opts"]["think"] is True
        assert call["opts"]["num_predict"] == copilot_turn.ANSWER_NUM_PREDICT + budget
        opts = call["opts"]
        assert (opts["temperature"], opts["top_p"], opts["top_k"], opts["presence_penalty"]) == (1.0, 0.95, 20, 1.5)
    assert got["stages"][0] == stage and got["effort"] == effort


def test_effort_off_thinks_on_no_call_and_samples_as_qwen_recommends_without(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    seen = scripted(monkeypatch, ("", [("search", {"query": "zebras"})]), ("Stripes.", []))
    got = ask("zebras?", effort="off")
    for call in seen:
        opts = call["opts"]
        assert opts["think"] is False and opts["num_predict"] == copilot_turn.ANSWER_NUM_PREDICT
        assert (opts["temperature"], opts["top_p"], opts["top_k"], opts["presence_penalty"]) == (0.7, 0.8, 20, 1.5)
    assert got["stages"][0] == "Writing" and got["answer"] == "Stripes."


def test_effort_off_keeps_a_plan_step_as_measured(ask, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Add a line.", "steps": ["Add a line after b2"]})]),
                    ("", [("insert", {"after": "b2", "markdown": "A line."})]))
    ask("add a line", effort="off")
    step = seen[1]["opts"]
    assert step["think"] is False and step["temperature"] == edit_run.STEP_TEMPERATURE
    assert not {"top_p", "top_k", "presence_penalty"} & set(step)


def test_a_plan_step_stays_as_measured(ask, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Add a line.", "steps": ["Add a line after b2"]})]),
                    ("", [("insert", {"after": "b2", "markdown": "A line."})]))
    got = ask("add a line")
    assert len(got["edit_run"]["changes"]) == 1
    first, step = seen[0]["opts"], seen[1]["opts"]
    assert first["think"] is True
    assert step["think"] is False and step["temperature"] == edit_run.STEP_TEMPERATURE == 0.0
    assert step["num_predict"] == edit_run.STEP_NUM_PREDICT
    assert not {"top_p", "top_k", "presence_penalty"} & set(step)        # a step's decoding as measured


def test_loop_calls_sample_as_qwen_recommends_for_thinking(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    seen = scripted(monkeypatch, ("", [("search", {"query": "zebras"})]), ("Stripes.", []))
    ask("zebras?")
    for call in seen:
        opts = call["opts"]
        assert (opts["temperature"], opts["top_p"], opts["top_k"], opts["presence_penalty"]) == (1.0, 0.95, 20, 1.5)


@pytest.mark.parametrize("effort", ["off", "low", "medium"])
def test_a_chat_skill_thinks_as_measured_below_high(ask, monkeypatch, effort):
    seen = scripted(monkeypatch, ("Question 1?", []))
    ask("/quiz", effort=effort)
    assert seen[0]["opts"]["think"] is False
    assert seen[0]["opts"]["num_predict"] == copilot_turn.ANSWER_NUM_PREDICT
    assert seen[0]["opts"]["temperature"] == llm.REPLY_TEMPERATURE and "top_p" not in seen[0]["opts"]


def test_a_chat_skill_thinks_at_high_with_its_measured_budget(ask, monkeypatch):
    seen = scripted(monkeypatch, ("Question 1?", []))
    ask("/quiz", effort="high")
    assert seen[0]["opts"]["think"] is True
    assert seen[0]["opts"]["num_predict"] == copilot_turn.ANSWER_NUM_PREDICT + edit_run.THINK_BUDGET == 2400 + 8192
    assert seen[0]["opts"]["temperature"] == llm.REPLY_TEMPERATURE and "top_p" not in seen[0]["opts"]


def test_each_thinking_call_has_its_own_allowance(ask, monkeypatch):
    """Thinking is output: lookups that each think long must not spend the turn's run budget and
    leave it with no answer."""
    many = edit_run.RUN_OUTPUT_TOKENS // 2
    seen = scripted(monkeypatch, *[("", [("list_folder", {"path": "Notes"})], many)] * 4, ("Found it.", []))
    got = ask("look around")
    assert got["answer"] == "Found it." and len(seen) == 5


# --- every lookup shown in the chat, as Pi shows each tool call --------------------------------

def test_each_lookup_is_shown_as_it_runs_and_kept_with_the_turn(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n\n## Food\n\nGrass.\n")
    scripted(monkeypatch,
             ("", [("search", {"query": "zebras"}), ("search", {"query": "grass", "folder": "Notes/school/cs-201/"})]),
             ("", [("read_note", {"path": "Notes/school/cs-201/zebra.md", "section": "Food"})]),
             ("", [("list_folder", {"path": "Notes/school"}), ("list_folder", {"path": ""})]),
             ("", [("read_note", {"path": "Notes/nope.md"})]),
             ("Grass.", []))
    heard = []
    got = ask("what do zebras eat?", tools=heard)
    assert [(t["label"], t["refused"]) for t in got["tools"]] == [
        ('Searched the vault for "zebras"', False),
        ('Searched Notes/school/cs-201 for "grass"', False),
        ("Read zebra § Food", False),
        ("Listed Notes/school/", False),
        ("Listed the vault", False),
        ("Read nope — refused", True)]
    assert [t["name"] for t in got["tools"]] == ["search", "search", "read_note", "list_folder", "list_folder",
                                                 "read_note"]
    assert heard == got["tools"]                                  # heard one by one, as each ran


def test_a_section_of_the_open_note_is_named_by_its_text_never_a_handle(ask, monkeypatch):
    scripted(monkeypatch, ("", [("read_note", {"path": OPEN, "section": "s3"})]), ("ok", []))
    got = ask("q")
    assert got["tools"][0]["label"] == "Read n § “A”"


def test_a_lookup_past_the_cap_and_one_with_no_room_are_shown_refused(ask, monkeypatch):
    scripted(monkeypatch, *[("", [("list_folder", {"path": "Notes"})])] * 7, ("Found it.", []))
    got = ask("look around")
    assert got["tools"][-1] == {"name": "list_folder", "label": "Listed Notes/ — refused", "refused": True}
    assert [t["refused"] for t in got["tools"]] == [False] * 6 + [True]


def test_a_pinned_folder_names_where_a_search_looked(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-202/sorting.md", "# Sorting\n\nDijkstra is not a sort.\n")
    scripted(monkeypatch, ("", [("search", {"query": "Dijkstra"})]), ("ok", []))
    got = ask("q", pins=[{"kind": "folder", "path": "Notes/school/cs-202"}])
    assert got["tools"][0]["label"] == 'Searched Notes/school/cs-202 for "Dijkstra"'


def test_a_turn_with_no_lookups_has_no_tools(ask, monkeypatch):
    scripted(monkeypatch, ("Hi.", []))
    assert ask("hi")["tools"] == []


# --- sources: the notes the turn looked at -------------------------------------------------------

def test_sources_are_the_notes_read_and_found_in_order_then_the_pins(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    put(con, vault, "Notes/school/cs-202/horse.md", "# Horses\n\nZebras are not horses.\n")
    put(con, vault, "Notes/b.md", "# Bee\n\nBuzzing text.\n")
    seen = scripted(monkeypatch, ("", [("read_note", {"path": "Notes/b.md"})]),
                    ("", [("search", {"query": "zebras"})]), ("Stripes.", []))
    got = ask("zebras?", pins=[{"kind": "note", "path": "Notes/b.md"}])
    paths = [c["path"] for c in got["citations"]]
    assert paths[0] == "Notes/b.md" and paths.count("Notes/b.md") == 1      # read first; pinned too: once
    assert set(paths[1:]) == {"Notes/school/cs-201/zebra.md", "Notes/school/cs-202/horse.md"}
    assert [c["n"] for c in got["citations"]] == [1, 2, 3]
    assert got["citations"][1]["title"] == Path(paths[1]).stem
    assert len(seen) == 3                                                    # no call to decide them


def test_the_open_note_is_a_source_only_when_read_with_a_tool(ask, con, vault, monkeypatch):
    scripted(monkeypatch, ("", [("search", {"query": "Intro para"})]), ("It is the intro.", []))
    got = ask("where is the intro?")
    assert OPEN not in [c["path"] for c in got["citations"]]                # a search hit of the open note
    scripted(monkeypatch, ("", [("read_note", {"path": OPEN, "section": "s3"})]), ("Text a.", []))
    got = ask("what is under A?")
    assert [c["path"] for c in got["citations"]] == [OPEN]


def test_at_most_eight_looked_at_notes_are_sources(ask, con, vault, monkeypatch):
    for n in range(10):
        put(con, vault, f"Notes/many/n{n}.md", f"# N{n}\n\nZebra fact {n}.\n")
    put(con, vault, "Notes/other/late.md", "# Late\n\nNothing here.\n")
    scripted(monkeypatch, ("", [("search", {"query": "zebra fact"})]),
             ("", [("read_note", {"path": "Notes/other/late.md"})]), ("Many.", []))
    got = ask("all of them?")
    paths = [c["path"] for c in got["citations"]]
    assert len(paths) == 8 and all(p.startswith("Notes/many/") for p in paths)   # the ninth looked at is not listed


# --- a turn's thinking is bounded -----------------------------------------------------------------

def test_a_model_that_keeps_calling_a_tool_not_offered_ends_within_the_bound(ask, monkeypatch):
    """A tool the loop never offers (one the model made up): a reply of it and a lookup that finds
    nothing is all refused, and counts toward MAX_REFUSALS as a lookup past the cap does."""
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    stuck = ("", [("web_search", {"query": "comp"}), ("read_note", {"path": "Notes/missing.md"})])
    seen = scripted(monkeypatch, *[stuck] * 40)
    with pytest.raises(copilot_turn.CopilotError, match="before an answer"):
        ask("q")
    assert len(seen) == edit_run.MAX_REFUSALS
    (record,) = records
    assert record["status"] == "failed" and len(record["looked_up"]) == edit_run.MAX_REFUSALS
    assert all(looked["refused"] for looked in record["looked_up"])


def test_a_turn_thinks_on_at_most_eight_calls(ask, monkeypatch):
    looks = [("", [("list_folder", {"path": "Notes"})])] * 8
    seen = scripted(monkeypatch, *looks, ("Answer.", []))
    got = ask("look around")
    assert got["answer"] == "Answer." and len(seen) == 9
    assert [call["opts"]["think"] for call in seen] == [True] * edit_run.MAX_THINKING_CALLS + [False]
    assert edit_run.MAX_THINKING_CALLS == 8


# --- a reply that claims an edit with no edit made ---------------------------------------------

CLAIMS = [
    # the live case, 2026-11-14: the new text in the chat, then the claim
    "Here is a simpler version:\n\n**Softmax** turns scores into probabilities.\n\n"
    "I've replaced “Softmax Function” with this condensed version.",
    "I have updated the note.",
    "I replaced the intro with a shorter one.",
    "I’ve condensed the explanation.",
    "Sure! I've shortened the section on softmax.",
    "OK, I rewrote the definition.",
    "- I've removed the long example.",
    "**I've simplified the note.**",
    "I'll simplify it now.",
    "Let me rewrite the section now:",
    "I will now update the heading.",
    "Done: 1 change",
    "Done — the note is shorter.",
    "**Done** — the note is shorter.",
    "Done.",
]
NOT_CLAIMS = [
    "The note has been updated by your instructor since last week.",
    "You could replace X with Y.",
    "I can rewrite it if you want.",
    "I'll rewrite it if you want.",
    "I added nothing; the note already covers it.",
    "I haven't changed the note. Here is a shorter version to paste:",
    "If I replaced the heading, the links to it would break.",
    "Let me know if you want me to shorten it.",
    'Your note says: "I\'ve updated the figures for Q3."',
    "“I have edited this section” is the note's last line.",
    "> I've replaced the old intro.\n\nThat line is from your note.",
    "```\nI've updated the note.\n```\nThat is the example's first line.",
    "Done right, softmax is stable.",
    "Softmax turns scores into probabilities.",
    "I added up the numbers: the total is 42.",
    "I've added a short summary below:",
    "Let me add some context now.",
    "I'll add one more point: softmax is now the default.",
    "I'll update the heading if you want; the note is now long.",
]


@pytest.mark.parametrize("reply", CLAIMS)
def test_a_first_person_edit_is_a_claim(reply):
    assert edit_run.claims_edit(reply)


@pytest.mark.parametrize("reply", NOT_CLAIMS)
def test_an_ordinary_answer_is_no_claim(reply):
    assert not edit_run.claims_edit(reply)


@pytest.fixture
def records(monkeypatch):
    found = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: found.append(payload))
    return found


def test_a_claimed_edit_that_made_none_is_corrected_once_and_then_lands(ask, monkeypatch, records):
    seen = scripted(monkeypatch, ("Simpler: Intro.\n\nI've replaced the intro with this.", []),
                    ("", [("edit_text", {"target": "b2", "old": "Intro para.", "new": "Intro."})]))
    got = ask("make the intro simpler")
    assert len(seen) == 2
    assert seen[1]["messages"][-1] == {"role": "user", "content": edit_run.CLAIMED_EDIT}
    assert seen[1]["messages"][-2]["content"].endswith("I've replaced the intro with this.")
    assert len(got["edit_run"]["changes"]) == 1 and edit_run.FLAGGED not in got["answer"]
    assert RESET in got["deltas"]                       # the false claim is not left as the answer
    assert records[0]["claimed_edit"] == "corrected"


def test_a_corrected_reply_that_says_it_changed_nothing_is_the_answer(ask, monkeypatch, records):
    seen = scripted(monkeypatch, ("I've shortened the note.", []),
                    ("I did not change the note. Paste this: Intro.", []))
    got = ask("make it shorter")
    assert len(seen) == 2 and got["answer"] == "I did not change the note. Paste this: Intro."
    assert got["deltas"] == ["I've shortened the note.", RESET, "I did not change the note. Paste this: Intro."]
    assert records[0]["claimed_edit"] == "corrected"


def test_a_claim_made_again_keeps_the_reply_and_says_no_change_was_made(ask, monkeypatch, records):
    seen = scripted(monkeypatch, ("I've shortened the note.", []), ("I have updated the note now.", []))
    got = ask("make it shorter")
    assert len(seen) == 2 and got["edit_run"] is None
    assert got["answer"] == "I have updated the note now.\n\nNo change was made to the note."
    assert got["answer"].count(edit_run.FLAGGED) == 1
    assert "".join(d for d in got["deltas"][got["deltas"].index(RESET) + 1:]) == got["answer"]
    assert records[0]["claimed_edit"] == "flagged"


def test_a_claim_whose_correction_fails_is_kept_flagged(ask, monkeypatch, records):
    scripted(monkeypatch, ("I've shortened the note.", []),
             llm.LLMError("the server failed"), llm.LLMError("the server failed"))
    got = ask("make it shorter")
    assert got["answer"] == "I've shortened the note.\n\nNo change was made to the note."
    assert records[0]["claimed_edit"] == "flagged"


def test_a_claim_whose_correction_ends_with_no_answer_is_kept_flagged(ask, monkeypatch, records):
    stuck = ("", [("web_search", {"query": "x"}), ("read_note", {"path": "Notes/missing.md"})])
    seen = scripted(monkeypatch, ("I've shortened the note.", []), *[stuck] * edit_run.MAX_REFUSALS)
    got = ask("make it shorter")
    assert len(seen) == 1 + edit_run.MAX_REFUSALS
    assert got["answer"] == "I've shortened the note.\n\nNo change was made to the note."
    assert records[0]["claimed_edit"] == "flagged"


EARLIER_EDIT = [{"role": "you", "text": "shorten the intro"},
                {"role": "slim", "text": "[Changed the note: Shortened the intro.; 1 change, 0 undone]"}]


def test_a_recap_of_an_earlier_edit_in_the_chat_is_not_corrected(ask, monkeypatch, records):
    seen = scripted(monkeypatch, ("I replaced the intro with a shorter version.", []))
    got = ask("What did you change?", history=EARLIER_EDIT)
    assert len(seen) == 1 and got["answer"] == "I replaced the intro with a shorter version."
    assert records[0]["claimed_edit"] is None


def test_a_claim_after_earlier_answers_that_edited_nothing_is_corrected(ask, monkeypatch, records):
    history = [{"role": "you", "text": "what is softmax?"}, {"role": "slim", "text": "It turns scores into probabilities."}]
    seen = scripted(monkeypatch, ("Simpler: Intro.\n\nI've replaced “Softmax Function” with this condensed version.", []),
                    ("I did not change the note.", []))
    ask("Can you change this so that it is a bit simpler?", history=history)
    assert len(seen) == 2 and records[0]["claimed_edit"] == "corrected"
    assert copilot_turn.edited_before(EARLIER_EDIT) and not copilot_turn.edited_before(history)


def test_a_claim_whose_correction_overflows_the_window_is_kept_flagged(ask, monkeypatch, records):
    seen = scripted(monkeypatch, ("I've shortened the note.", []), ("cut", []))
    real = copilot_turn.llm.chat_tools

    def saturated(messages, tools=None, **opts):
        msg, stats = real(messages, tools, **opts)
        return msg, {**stats, "ctx_saturated": len(seen) == 2}
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", saturated)
    got = ask("make it shorter")
    assert got["answer"] == "I've shortened the note.\n\nNo change was made to the note."
    assert records[0]["claimed_edit"] == "flagged"


def test_an_answer_with_no_claim_is_one_call(ask, monkeypatch, records):
    seen = scripted(monkeypatch, ("It introduces A.", []))
    got = ask("what is this note about?")
    assert len(seen) == 1 and got["answer"] == "It introduces A."
    assert records[0]["claimed_edit"] is None


def test_a_claim_beside_a_change_that_landed_is_not_corrected(ask, monkeypatch, records):
    seen = scripted(monkeypatch, ("I've replaced the intro.",
                                  [("edit_text", {"target": "b2", "old": "Intro para.", "new": "Intro."})]))
    got = ask("make the intro simpler")
    assert len(seen) == 1 and len(got["edit_run"]["changes"]) == 1
    assert records[0]["claimed_edit"] is None


@pytest.mark.parametrize("open_path, text, doc", [(None, None, True), (OPEN, CRLF, False)])
def test_a_claim_with_no_note_it_may_change_is_not_corrected(ask, monkeypatch, records, open_path, text, doc):
    seen = scripted(monkeypatch, ("I've updated the note.", []))
    got = ask("update it", open_path=open_path, text=text, doc=doc)
    assert len(seen) == 1 and got["answer"] == "I've updated the note."
    assert records[0]["claimed_edit"] is None


def test_a_first_call_edit_on_an_overflowing_window_lands_nothing_and_says_why(con, vault, monkeypatch):
    """The prompt lost its start, so the call's edit is a guess: nothing lands, and the turn says
    the conversation no longer fits."""
    put(con, vault, OPEN, NOTE)
    seen = scripted(monkeypatch, ("", [("edit_text", {"target": "b2", "old": "Intro para.", "new": "Intro."})]))
    real = copilot_turn.llm.chat_tools

    def saturated(messages, tools=None, **opts):
        msg, stats = real(messages, tools, **opts)
        return msg, {**stats, "ctx_saturated": True}
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", saturated)
    events = []
    with pytest.raises(copilot_turn.CopilotError, match="this conversation no longer fits the model's window"):
        copilot_turn.run_turn(con, OPEN, "shorten the intro", [], "medium", vault=vault, doc=NOTE,
                              doc_hash=edits.digest(NOTE), on_edit=lambda e, p: events.append(e))
    assert len(seen) == 1 and events == []
