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
    def go(question, *, open_path=OPEN, text=NOTE, pins=(), history=(), mode="quick", doc=True):
        sent = {}
        if open_path is not None and text is not None:
            put(con, vault, open_path, text, index=not open_path.startswith("Profile/"))
            if doc:
                sent = {"doc": text, "doc_hash": edits.digest(text)}
        events, deltas, stages = [], [], []
        result = copilot_turn.run_turn(con, open_path, question, list(history), mode, pins=list(pins),
                                       vault=vault, on_edit=lambda e, p: events.append((e, p)),
                                       on_delta=deltas.append, on_reset=lambda: deltas.append(RESET),
                                       on_stage=stages.append, **sent)
        result["events"], result["deltas"], result["stages"] = events, deltas, stages
        return result
    return go


def test_loop_is_on():
    assert copilot_turn.LOOP is True


def test_question_about_open_note_needs_no_tools(ask, monkeypatch):
    seen = scripted(monkeypatch, ("It introduces A.", []))
    got = ask("what is this note about?")
    assert got["answer"] == "It introduces A." and got["edit_run"] is None
    assert got["deltas"] == ["It introduces A."]                # streamed as it decoded, said once
    assert got["looked_up"] == [] and got["citations"] == [] and len(seen) == 1
    assert seen[0]["opts"]["temperature"] == llm.REPLY_TEMPERATURE and not seen.gate


def test_lookup_then_answer(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    seen = scripted(monkeypatch,
                    ("", [("search", {"query": "zebras"})]),
                    ("", [("read_note", {"path": "Notes/school/cs-201/zebra.md"})]),
                    ("They have stripes.", []))
    got = ask("what did I write about zebras?")
    assert got["answer"] == "They have stripes." and len(seen) == 3
    assert [c["path"] for c in got["citations"]] == ["Notes/school/cs-201/zebra.md"]   # opened, never the open note
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
    assert seen[1]["opts"]["temperature"] == llm.REPLY_TEMPERATURE      # no edit had happened yet
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


def test_gate_path_when_loop_off(con, vault, monkeypatch):
    monkeypatch.setattr(copilot_turn, "LOOP", False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot_turn.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    put(con, vault, OPEN, NOTE)
    seen = scripted(monkeypatch, ("An answer.", []), intent=False)
    got = copilot_turn.run_turn(con, OPEN, "what?", [], "quick", vault=vault)
    assert len(seen.gate) == 1 and seen[0]["stream"] and got["answer"] == "An answer."
    with pytest.raises(copilot_turn.CopilotError):
        copilot_turn.run_turn(con, None, "what?", [], "quick", vault=vault)


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
    seen = scripted(monkeypatch, *pages, ("Done.", []))
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
    assert got["citations"] == []                                # a search hit alone is not a source


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
    assert "[b2]" not in seen[3]["messages"][-1]["content"]       # shown without handles


@pytest.mark.parametrize("open_path, text, doc, said", [
    (None, None, True, "refused: no note is open to change"),
    (OPEN, CRLF, False, "refused: this note can't be changed here; give the text to paste"),
])
def test_an_edit_call_with_no_editable_note_is_refused_and_the_turn_answers(
        ask, monkeypatch, open_path, text, doc, said):
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
             ("", [("list_folder", {"path": "Notes/school"}), ("list_folder", {"path": ""})]), ("Done.", []))
    got = ask("zebras?")
    assert got["stages"] == ["Writing", "Searching", "Writing", "Reading zebra", "Writing",
                             "Listing Notes/school", "Listing the vault", "Writing"]


def test_a_lookup_that_would_overflow_the_window_is_refused(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/small.md", "# Small\n\nA short note.\n")
    put(con, vault, "Notes/big.md", "# Big\n\n" + "A long sentence of words. " * 400)
    seen = scripted(monkeypatch, ("ok", []))
    ask("q")
    prompt = sum(map(copilot_turn._tokens, seen[0]["messages"]))
    # Room for the prompt, the reply, and a little more: the small note fits, the big one not.
    monkeypatch.setattr(llm, "RESIDENT_CTX", int(prompt + copilot_turn.QUICK_NUM_PREDICT + 200))
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
