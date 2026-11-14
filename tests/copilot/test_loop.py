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
                                       on_delta=deltas.append, on_stage=stages.append, **sent)
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
    assert [c["path"] for c in got["citations"]] == [OPEN, "Notes/school/cs-201/zebra.md"]
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


def test_no_open_note_offers_lookups_and_create_only(ask, monkeypatch):
    seen = scripted(monkeypatch, ("Hello.", []))
    got = ask("hi", open_path=None)
    assert seen[0]["tools"] == ["search", "read_note", "list_folder", "create_note"]
    assert got["answer"] == "Hello." and got["source"] is None
    assert seen[0]["messages"][-1] == {"role": "user", "content": "hi"}


def test_with_an_open_note_the_tools_are_lookups_then_edits(ask, monkeypatch):
    seen = scripted(monkeypatch, ("ok", []))
    ask("q")
    assert seen[0]["tools"] == ["search", "read_note", "list_folder", "plan", "replace", "edit_text",
                                "insert", "move", "create_note"]


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
    assert len(asked) < len(text) and f'read_note("{OPEN}", "s<N>")' in asked
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
    assert system["content"].startswith(copilot_turn.PERSONA)
    assert current["role"] == "user"
    assert current["content"].startswith(f"=== {OPEN} · the open note ===\nSections:\n")
    assert "[b2] Intro para." in current["content"]
    assert current["content"].endswith("\n\n---\n\nwhat is A?")


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
    assert "Searches default to Notes/school/cs-202/." in seen[0]["messages"][1]["content"]
    found = seen[1]["messages"][-1]["content"]
    assert "cs-202/sorting.md" in found and "cs-201" not in found


def test_note_in_excluded_folder_is_context(ask, monkeypatch):
    seen = scripted(monkeypatch, ("ok", []))
    got = ask("who am I?", open_path="Profile/me.md", text="# Me\n\nI like graphs.\n")
    assert "I like graphs." in seen[0]["messages"][-1]["content"]
    assert got["source"] == {"id": None, "path": "Profile/me.md", "title": "me"}


def test_chat_skill_can_look_up(ask, con, vault, monkeypatch):
    put(con, vault, "Notes/school/cs-201/zebra.md", "# Zebras\n\nZebras have stripes.\n")
    seen = scripted(monkeypatch, ("", [("search", {"query": "zebras"})]), ("Question 1: stripes?", []))
    got = ask("/quiz")
    assert seen[0]["tools"] == LOOKUPS
    assert got["answer"] == "Question 1: stripes?" and got["skill"] == "quiz"
    paths = [c["path"] for c in got["citations"]]
    assert OPEN in paths and "Notes/school/cs-201/zebra.md" in paths
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
