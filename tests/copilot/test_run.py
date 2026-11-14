"""The editing run, ported from the measured prototype's loop tests: each runs a whole turn through
run_turn against a note sent as the editor's text, with a scripted model. A plain turn is the loop's
(its first call thinks and samples as Pi does; a step as measured); an edit skill runs on its own."""

from pathlib import Path

import pytest

from slim import db, llm
from slim.copilot import blocks, context, edits, run as edit_run, turn as copilot_turn
from slim.index import chunk

from .conftest import scripted


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "copilot.db")


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Notes/school/cs-201/week-01").mkdir(parents=True)
    (root / "Notes/school/cs-201/week-02").mkdir(parents=True)
    (root / "Notes/school/cs-202").mkdir(parents=True)
    (root / "Capture/school/cs-201").mkdir(parents=True)
    (root / "Journal").mkdir()
    return root


def write(root: Path, path: str, body: str) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"---\ntitle: {target.stem}\n---\n\n{body}\n", encoding="utf-8")
    return target


NOTE = "# Title\n\nIntro para.\n\n## A\n\nText a.\n- one\n- two\n\n## B\n\nText b.\n"
NOTE3 = "# Title\n\n## A\n\na\n\n## B\n\nb\n\n## C\n\nc\n"
STEP_TOOLS = ["replace", "edit_text", "insert", "move"]
LOOP_TOOLS = ["search", "read_note", "list_folder", "plan", *STEP_TOOLS, "create_note"]


def flaky(monkeypatch, seen, *failing):
    """The scripted model, raising an Ollama error on the given call numbers (1-based)."""
    real, count = copilot_turn.llm.chat_tools, [0]

    def chat_tools(messages, tools=None, **opts):
        count[0] += 1
        if count[0] in failing:
            raise llm.LLMError("ollama rejected the call: HTTP 500 — XML syntax error on line 10")
        return real(messages, tools, **opts)
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    return seen


@pytest.fixture
def turn(con, vault, monkeypatch):
    """run_turn on a note holding `text`, sent as the editor's text and its hash. Returns the
    turn, and every on_edit event as (event, payload)."""
    def go(question, *, text=NOTE, path="Notes/school/n.md", mode="quick", history=(),
           should_stop=lambda: False, selection="", doc=True):
        target = vault / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        source = context.sync_note(con, vault, path, embed=False)
        events, deltas, stages = [], [], []
        sent = {"doc": text, "doc_hash": edits.digest(text)} if doc else {}
        result = copilot_turn.run_turn(con, source["path"], question, list(history), mode, vault=vault,
                                       on_edit=lambda event, payload: events.append((event, payload)),
                                       on_delta=deltas.append, on_stage=stages.append, should_stop=should_stop,
                                       selection=selection, **sent)
        result["deltas"], result["stages"] = deltas, stages
        result["final"] = text if result["edit_run"] is None else _final(text, result)
        return result, events
    return go


def _final(text, result):
    for change in result["edit_run"]["changes"]:
        text = edits.apply_splices(text, change["splices"])
    return text


def test_an_edit_skill_carries_the_rider_and_the_note_with_handles(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b5", "markdown": "- **one**"})]))
    turn("/tidy", selection="- one", history=[{"role": "you", "text": "hi"}, {"role": "slim", "text": "hello"}])
    system, *rest = seen[0]["messages"]
    assert system["content"] == copilot_turn.PERSONA + "\n\n" + copilot_turn.EDIT_RIDER + "\n\n" + (
        "Notes in view:\n\n=== Notes/school/n.md · the open note ===\nSections:\n"
        "s1  b1–b8  # Title\n  s3  b3–b6  ## A\n  s7  b7–b8  ## B\n\n[b1] # Title\n\n[b2] Intro para.\n\n"
        "[b3] ## A\n\n[b4] Text a.\n[b5] - one\n[b6] - two\n\n[b7] ## B\n\n[b8] Text b.\n")
    assert [m["role"] for m in rest] == ["user", "assistant", "user"]
    assert rest[-1]["content"].endswith("Selected text:\n- one")
    opts = seen[0]["opts"]
    assert opts["think"] is False and opts["temperature"] == edit_run.STEP_TEMPERATURE == 0.0
    assert opts["num_predict"] == copilot_turn.QUICK_NUM_PREDICT >= edit_run.STEP_NUM_PREDICT


def test_the_rider_is_the_measured_one():
    assert "To answer a question, just reply in text and change nothing." in copilot_turn.EDIT_RIDER
    assert "To make one small change, make it right away, pointing at handles." in copilot_turn.EDIT_RIDER
    assert "For a small wording fix, use edit_text instead of retyping the block." in copilot_turn.EDIT_RIDER
    assert "Use the replace, edit_text, insert and move tools" in copilot_turn.EDIT_RIDER
    assert "the note's headings in their final order (each with its # marks)" in copilot_turn.EDIT_RIDER


def test_deep_gives_an_edit_skills_first_call_room_to_think(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b5", "markdown": "- **one**"})]))
    turn("/tidy", selection="- one", mode="deep")
    assert seen[0]["opts"]["think"] is True
    assert seen[0]["opts"]["num_predict"] == copilot_turn.QUICK_NUM_PREDICT + edit_run.THINK_BUDGET
    assert edit_run.THINK_BUDGET == 8192
    assert copilot_turn.QUICK_NUM_PREDICT + edit_run.THINK_BUDGET < edit_run.RUN_OUTPUT_TOKENS


def test_a_plan_streams_its_events_in_order_and_returns_the_run(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Rename A and move B up.", "outline": ["# Title", "## B", "## A renamed"],
                                    "steps": ["Rename A", "Move B above A"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("move", {"target": "s7", "before": "s3"})]))
    got, events = turn("Rename A and put B first", mode="deep")
    assert [e for e, _ in events] == ["start", "plan", "step", "change", "step", "step", "change", "step"]
    assert events[0][1] == {"before_text": NOTE}
    assert events[1][1] == {"intro": "Rename A and move B up.", "steps": ["Rename A", "Move B above A"],
                            "outline": ["# Title", "## B", "## A renamed"]}
    assert [p for e, p in events if e == "step"] == [
        {"index": 1, "state": "doing"}, {"index": 1, "state": "done"},
        {"index": 2, "state": "doing"}, {"index": 2, "state": "done"}]
    change = events[3][1]
    assert set(change) == {"id", "step", "op", "label", "needs", "splices", "part_of"}
    assert (change["id"], change["step"], change["op"], change["label"]) == ("c1", 1, "replace", "Renamed “A” to “A renamed”")
    run = got["edit_run"]
    assert set(run) == {"plan", "steps_done", "changes", "stopped", "outline_met",
                        "hash_before", "hash_after", "summary", "path"}
    assert run["path"] == "Notes/school/n.md"
    assert run["steps_done"] == 2 and run["stopped"] is None and run["outline_met"] is True
    assert run["summary"] == "Rename A and move B up." and got["answer"] == run["summary"]
    assert got["final"].index("## B") < got["final"].index("## A renamed")
    assert run["hash_before"] == edits.digest(NOTE) and run["hash_after"] == edits.digest(got["final"])
    assert [c["tools"] for c in seen] == [LOOP_TOOLS] * 3     # one list; no summary call
    assert "Step 2 of 2: Move B above A" in seen[2]["messages"][-1]["content"]
    assert seen[0]["opts"]["think"] is True and seen[1]["opts"]["think"] is False
    assert seen[0]["opts"]["temperature"] == copilot_turn.LOOP_TEMPERATURE
    assert seen[1]["opts"]["temperature"] == edit_run.STEP_TEMPERATURE == 0.0
    assert seen[1]["opts"]["num_predict"] == edit_run.STEP_NUM_PREDICT == 2048
    assert got["citations"] == []                          # nothing looked up, nothing to cite


def test_the_run_holds_the_model_lock_across_its_calls(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    turn("rename A")
    assert [call["held"] for call in seen] == [True, True]
    assert not llm._CALL_LOCK.locked()


def test_a_step_whose_calls_are_all_refused_is_tried_once_more_then_left(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "x", "steps": ["Fix it"]})]),
                    ("", [("replace", {"target": "b99", "markdown": "x"})]),
                    ("", [("replace", {"target": "b98", "markdown": "x"})]))
    got, events = turn("fix")
    assert got["edit_run"]["steps_done"] == 1 and got["edit_run"]["changes"] == []
    assert "there is no b99" in seen[2]["messages"][-1]["content"]
    assert [p for e, p in events if e == "step"] == [{"index": 1, "state": "doing"}, {"index": 1, "state": "skipped"}]


def test_three_refusals_in_a_row_stop_the_run(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "x", "steps": ["a", "b"]})]),
                    ("", [("replace", {"target": "b99", "markdown": "x"}), ("replace", {"target": "b98", "markdown": "x"}),
                          ("replace", {"target": "b97", "markdown": "x"})]))
    got, _ = turn("fix")
    assert got["edit_run"]["stopped"] == "refusals" and got["edit_run"]["summary"] == "" and len(seen) == 2


def test_the_call_limit_stops_the_run(turn, monkeypatch):
    monkeypatch.setattr(edit_run, "MAX_CALLS", 3)
    many = [("insert", {"after": "b2", "markdown": f"p{i}"}) for i in range(5)]
    scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["a"]})]), ("", many))
    got, events = turn("go")
    assert got["edit_run"]["stopped"] == "call limit" and len(got["edit_run"]["changes"]) == 3
    assert edit_run.MAX_CALLS == 3 and events[-1] == ("step", {"index": 1, "state": "partial"})


def test_a_plan_longer_than_the_limit_is_cut_by_code(turn, monkeypatch):
    scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": [f"s{i}" for i in range(12)]})]), *[("", [])] * 8)
    got, _ = turn("go")
    assert len(got["edit_run"]["plan"]["steps"]) == 8 == edit_run.MAX_STEPS and got["edit_run"]["steps_done"] == 8


def test_done_where_the_run_does_not_offer_it_is_refused_and_the_steps_go_on(turn, monkeypatch):
    # An edit run offers no done: one called anyway never ends the run with steps still to do.
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Rename A and B.", "steps": ["Rename A", "Rename B", "Check"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"}),
                          ("done", {"summary": "Renamed A; B was fine."})]),
                    ("", [("replace", {"target": "b7", "markdown": "## B renamed"})]),
                    ("", []))
    got, _ = turn("rename")
    run = got["edit_run"]
    assert "done" not in seen[0]["tools"]
    assert seen[2]["messages"][-1]["content"].startswith("refused: done is not available now\n\nStep 2 of 3")
    assert "## A renamed" in got["final"] and "## B renamed" in got["final"] and len(run["changes"]) == 2
    assert run["steps_done"] == 3 and run["stopped"] is None and run["summary"] == "Rename A and B."
    assert len(seen) == 4


def test_a_lone_edit_in_the_first_reply_is_a_one_step_run_in_one_call(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, events = turn("rename A")
    run = got["edit_run"]
    assert got["final"] == NOTE.replace("## A\n", "## A renamed\n") and len(seen) == 1
    assert run["plan"] == {"intro": "", "steps": []} and run["summary"] == "Done: 1 change"
    assert [e for e, _ in events] == ["start", "plan", "change"]          # no step lines for one reply
    assert run["changes"][0]["step"] == 1


def test_a_plan_and_an_edit_in_the_first_reply_go_on_from_step_two(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Rename A, then move B up.", "steps": ["Rename A", "Move B up"]}),
                          ("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("move", {"target": "s7", "before": "s3"})]))
    got, events = turn("go")
    assert "Step 2 of 2: Move B up" in seen[1]["messages"][-1]["content"]
    assert seen[1]["messages"][-2]["content"] == "ok: plan accepted"
    assert got["edit_run"]["steps_done"] == 2 and len(got["edit_run"]["changes"]) == 2 and len(seen) == 2
    assert [p for e, p in events if e == "step"] == [
        {"index": 1, "state": "doing"}, {"index": 1, "state": "done"},
        {"index": 2, "state": "doing"}, {"index": 2, "state": "done"}]


def test_a_refused_first_edit_is_tried_once_more(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b99", "markdown": "## A renamed"})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, _ = turn("rename A")
    assert "refused: there is no b99" in seen[1]["messages"][-1]["content"]
    assert "Make your change again, fixing what was refused." in seen[1]["messages"][-1]["content"]
    assert len(got["edit_run"]["changes"]) == 1 and len(seen) == 2


def test_a_first_reply_with_only_done_is_asked_for_a_plan_once_then_answers(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("done", {"summary": "Nothing to do."})]), ("A is the first section.", []))
    got, events = turn("what is A?")
    assert seen[1]["messages"][-1] == {"role": "tool", "content": "refused: start with a plan\n\nMake the plan first: "
                                                                  "one sentence for the owner, and the steps in order."}
    assert seen[0]["opts"]["temperature"] == seen[1]["opts"]["temperature"] == copilot_turn.LOOP_TEMPERATURE
    assert got["edit_run"] is None and got["answer"] == "A is the first section." and events == []


def test_a_first_call_that_could_not_be_read_is_asked_for_a_plan_once_more(turn, monkeypatch):
    seen = scripted(monkeypatch, ("<tool_call>{broken", []),
                    ("", [("plan", {"intro": "Rename A.", "steps": ["Rename A"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, _ = turn("rename A")
    assert "refused: your tool call could not be read (a tool call came back as text); write it again" in \
        seen[1]["messages"][-1]["content"]
    assert "## A renamed" in got["final"]


def test_an_out_of_order_heading_triggers_no_fix_up(turn, monkeypatch):
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    plan = {"intro": "x", "outline": ["# Title", "## B", "## A"], "steps": ["Move B"]}
    seen = scripted(monkeypatch, ("", [("plan", plan)]), ("", []))
    got, _ = turn("go")
    assert len(seen) == 2 and got["edit_run"]["outline_met"] is False
    assert records[-1]["edit_run"]["outline_details"] == {"missing": 0, "out_of_order": 1, "other_level": 0}


def test_deeper_sections_against_a_shallower_outline_count_as_met(turn, monkeypatch):
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    text = "# Overview 2026-01-13\n\n#### Context & Demographics\n\nx\n\n#### Historical Evolution & Drivers\n\ny\n"
    plan = {"intro": "x", "outline": ["# Overview", "## Context & Demographics", "## Historical Evolution"],
            "steps": ["Tidy"]}
    seen = scripted(monkeypatch, ("", [("plan", plan)]), ("", []))
    got, _ = turn("tidy", text=text)
    assert len(seen) == 2 and got["edit_run"]["outline_met"] is True
    assert records[-1]["edit_run"]["outline_details"] == {"missing": 0, "out_of_order": 0, "other_level": 2}


def test_the_outline_is_met_by_its_headings_in_order_with_others_between(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "outline": ["# title", "##   b"], "steps": ["Tidy"]})]),
                    ("", []))
    got, _ = turn("tidy")
    assert got["edit_run"]["outline_met"] is True and len(seen) == 2


def test_the_outline_is_read_leniently_and_locked_headings_are_left_out(turn, monkeypatch):
    memo = f"# Memo\n\nSummary.\n\n{chunk.TRANSCRIPT_MARKER}\nwords words\n"
    for outline in ([" [b1] #   Memo "], ["#Memo", chunk.TRANSCRIPT_MARKER]):
        seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "outline": outline, "steps": ["Shorten"]})]),
                        ("", [("replace", {"target": "b2", "markdown": "Short."})]))
        got, _ = turn("shorten", text=memo, path="Capture/m.md")
        assert got["edit_run"]["outline_met"] is True and len(seen) == 2, outline


def test_a_plan_during_a_step_is_answered_and_is_neither_a_refusal_nor_a_retry(turn, monkeypatch):
    set_ = "ok: the plan is already set; make this step's changes"
    again = ("plan", {"intro": "x", "steps": ["again"]})
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A", "Rename B"]})]),
                    ("", [again, ("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [again]))                              # only a plan: an empty step, not retried
    got, _ = turn("rename")
    assert len(got["edit_run"]["changes"]) == 1 and got["edit_run"]["steps_done"] == 2 and len(seen) == 3
    assert seen[2]["messages"][-2]["content"] == set_


def test_an_ollama_error_is_retried_once(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    flaky(monkeypatch, seen, 2)
    got, _ = turn("rename")
    assert got["edit_run"]["stopped"] is None and "## A renamed" in got["final"] and len(seen) == 2


def test_a_second_ollama_error_stops_the_run_and_keeps_what_landed(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A", "Rename B"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    flaky(monkeypatch, seen, 3, 4)
    got, events = turn("rename")
    assert got["edit_run"]["stopped"] == ("error: ollama rejected the call: HTTP 500 — XML syntax error on line 10")
    assert got["final"] == NOTE.replace("## A\n", "## A renamed\n") and got["edit_run"]["summary"] == ""
    assert events[-1] == ("step", {"index": 2, "state": "skipped"})


def test_a_first_call_that_fails_twice_is_a_failed_turn(turn, monkeypatch, con):
    seen = scripted(monkeypatch, ("never", []))
    flaky(monkeypatch, seen, 1, 2)
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    with pytest.raises(llm.LLMError, match="HTTP 500"):
        turn("q")
    assert records[-1]["status"] == "failed"


def test_outline_entries_are_cleaned_of_what_the_model_wrote_around_them():
    assert edit_run._outline({"outline": ['"# Softmax Function\\n\\n## Formula\\n\\n## Example"']}) == [
        "# Softmax Function", "## Formula", "## Example"]                          # one quoted string, \n literal
    assert edit_run._outline({"outline": ["## Toolkit (New Section)", "## Tools (added)", "## Bag of Words (BoW)"]}) == [
        "## Toolkit", "## Tools", "## Bag of Words (BoW)"]                         # annotations, not words
    assert edit_run._outline({"outline": ["# <the note's headings, one per line, in their final order>", "## <…>",
                                             "## Formula"]}) == ["## Formula"]         # the rider's placeholder copied


def _misses(now, want):
    """The outline's misses as the trace counts them: those the note lacks, those out of order."""
    return edit_run._match(now, want)[1:]


def test_an_annotation_on_either_side_does_not_make_a_miss():
    assert _misses(["# T", "## Toolkit (New Section)"], ["# T", "## Toolkit"]) == ([], [])
    assert _misses(["# T", "## Toolkit"], ["# T", "## toolkit (added)"]) == ([], [])
    assert _misses(["# T", "## Pipeline (Classical vs. Neural)"], ["# T", "## Pipeline"]) == ([], [])  # words' prefix
    assert _misses(["# T", "## Classical vs. Neural"], ["# T", "## Pipeline"]) == ([1], [])


def test_a_heading_at_another_level_is_not_missing(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "outline": ["# Title", "# A", "## B"], "steps": ["Promote A"]})]),
                    ("", []))
    got, _ = turn("promote A")
    assert len(seen) == 2 and got["edit_run"]["outline_met"] is True


def test_the_token_budget_stops_the_run_after_the_reply_that_crosses_it(turn, monkeypatch):
    # The first call thinks, so the run's budget is RUN_OUTPUT_TOKENS and its allowance.
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "x", "steps": ["Rename A", "Move B", "Rename B"]})], 100),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})], 5000),
                    ("", [("move", {"target": "s7", "before": "s3"})], 8000 + edit_run.THINK_BUDGET))
    got, _ = turn("go")
    run = got["edit_run"]
    assert run["stopped"] == "token budget" and len(seen) == 3 and run["summary"] == ""
    assert len(run["changes"]) == 2 and got["final"].index("## B") < got["final"].index("## A renamed")
    assert edit_run.RUN_OUTPUT_TOKENS == 12000


def test_a_runaway_reply_with_no_calls_stops_the_run_at_once(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Simplify"]})], 300),
                    ("A very long explanation with no tool calls …", [], 12500 + edit_run.THINK_BUDGET))
    got, _ = turn("simplify")
    assert got["edit_run"]["stopped"] == "token budget" and len(seen) == 2 and got["final"] == NOTE


def test_the_sections_map_follows_every_step_as_it_is_then(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Add C", "Tidy"]})]),
                    ("", [("insert", {"after": "s7", "markdown": "## C\n\nText c."})]),
                    ("", []))
    turn("add C")
    assert seen[1]["messages"][-1]["content"].endswith("\n\nSections now:\ns1  b1–b8  # Title\n  s3  b3–b6  ## A\n"
                                                        "  s7  b7–b8  ## B")
    assert seen[2]["messages"][-1]["content"].endswith("Make this step's changes now.\n\nSections now:\n"
                                                        "s1  b1–b10  # Title\n  s3  b3–b6  ## A\n"
                                                        "  s7  b7–b8  ## B\n  s9  b9–b10  ## C")


def test_should_stop_ends_the_run_after_the_current_call(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A", "Rename B"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("replace", {"target": "b7", "markdown": "## B renamed"})]))
    got, events = turn("rename both", should_stop=lambda: len(seen) >= 2)
    run = got["edit_run"]
    assert run["stopped"] == "stopped" and len(seen) == 2 and len(run["changes"]) == 1
    assert run["steps_done"] == 1 and run["summary"] == ""
    assert events[-2:] == [("step", {"index": 2, "state": "doing"}), ("step", {"index": 2, "state": "skipped"})]


def test_the_note_changed_while_sending_is_refused(turn, monkeypatch, con, vault):
    scripted(monkeypatch, ("never", []))
    write(vault, "Notes/school/h.md", "x")
    source = context.sync_note(con, vault, "Notes/school/h.md", embed=False)
    with pytest.raises(copilot_turn.CopilotError, match="the note changed while sending; try again"):
        copilot_turn.run_turn(con, source["path"], "q", [], "quick", vault=vault, doc=NOTE, doc_hash=edits.digest("other"))


def test_without_the_editors_text_the_note_is_read_from_the_vault(turn, monkeypatch):
    scripted(monkeypatch, ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, events = turn("rename A", doc=False)
    assert events[0] == ("start", {"before_text": NOTE}) and "## A renamed" in got["final"]


def test_a_recordings_own_parts_are_refused(turn, monkeypatch):
    memo = "---\ntype: meeting-note\n---\n\n# Call\n\none\n\n## Transcript\n\nMe: hi\n"
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Shorten"]})]),
                    ("", [("replace", {"target": "b4", "markdown": "## Transcript\n\nMe: bye"})]),
                    ("", [("replace", {"target": "b3", "markdown": "One line."})]))
    got, _ = turn("shorten", text=memo, path="Capture/call.md")
    assert "[locked] ## Transcript" in seen[0]["messages"][-1]["content"]
    assert "b4 is part of the recording; SLIM never edits it" in seen[2]["messages"][-1]["content"]
    assert got["final"] == memo.replace("one\n", "One line.\n")


def test_an_edit_turn_writes_machine_facts_to_the_trace(turn, monkeypatch):
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append((kind, payload)))
    scripted(monkeypatch, ("", [("plan", {"intro": "Rename A.", "steps": ["Rename A", "Rename B"]})]),
             ("", [("replace", {"target": "b99", "markdown": "x"}), ("replace", {"target": "b3", "markdown": "## A2"})]),
             ("", [("replace", {"target": "'Text b.'", "markdown": "x"})]),
             ("", [("replace", {"target": "b7", "markdown": "## B2"})]))
    turn("rename both")
    (kind, payload), = records
    facts = payload["edit_run"]
    assert kind == "copilot" and payload["status"] == "edited"
    assert facts["steps_planned"] == 2 and facts["steps_done"] == 2 and facts["changes"] == 2
    assert facts["calls"] == 4 and facts["model_calls"] == 4 and facts["refusals"] == ["handle", "handle"]
    assert facts["stopped"] is None and facts["outline_met"] is None
    assert [t["output_tokens"] for t in facts["timing"]] == [10, 10, 10, 10]
    assert "Text b" not in str(payload) and "A2" not in str(payload)       # facts, never the note's words


# --- an edit skill, and a turn stopped before it began --------------------------------------------

def test_an_edit_skill_is_a_wanted_change_without_asking(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b5", "markdown": "- **one**"})]))
    turn("/tidy", selection="- one")
    assert len(seen) == 1 and seen[0]["tools"] == ["plan", "replace", "edit_text"]
    assert seen[0]["opts"]["temperature"] == 0.0


def test_a_turn_stopped_before_its_first_call_makes_none(turn, monkeypatch):
    seen = scripted(monkeypatch, ("never", []))
    with pytest.raises(copilot_turn.CopilotError, match="stopped"):
        turn("What is A?", should_stop=lambda: True)
    assert seen == []


# --- a plan written as prose, with the first change made at once --------------------------------

PROSE = ("I'll tidy this note. It is long.\n\nHere is the plan:\n\n1. Rename A\n2. Shorten B\n\n"
         "Here are the steps I will take:\n1) Rename A to A renamed\n   (its text stays)\n"
         "2) Shorten the text under B\n* Check the title\n\nStarting with step 1.")


def test_a_plan_written_as_prose_runs_its_remaining_steps(turn, monkeypatch):
    seen = scripted(monkeypatch, (PROSE, [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("replace", {"target": "b8", "markdown": "B."})]),
                    ("", []))
    got, events = turn("tidy this note")
    steps = ["Rename A to A renamed", "Shorten the text under B", "Check the title"]     # the last list
    assert events[1] == ("plan", {"intro": "I'll tidy this note.", "steps": steps, "outline": None})
    assert [p for e, p in events if e == "step"] == [
        {"index": 1, "state": "doing"}, {"index": 1, "state": "done"},
        {"index": 2, "state": "doing"}, {"index": 2, "state": "done"},
        {"index": 3, "state": "doing"}, {"index": 3, "state": "skipped"}]
    assert "Step 2 of 3: Shorten the text under B" in seen[1]["messages"][-1]["content"]
    run = got["edit_run"]
    assert run["plan"] == {"intro": "I'll tidy this note.", "steps": steps} and run["steps_done"] == 3
    assert run["summary"] == "I'll tidy this note." and len(seen) == 3
    assert got["final"] == NOTE.replace("## A\n", "## A renamed\n").replace("Text b.", "B.")


@pytest.mark.parametrize("content", ["", "Renamed it.", "Renamed it:\n\n- ## A renamed"])
def test_an_edit_with_short_or_no_prose_stays_a_one_reply_edit(turn, monkeypatch, content):
    seen = scripted(monkeypatch, (content, [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, events = turn("rename A")
    assert len(seen) == 1 and got["edit_run"]["plan"]["steps"] == [] and [e for e, _ in events] == [
        "start", "plan", "change"]


def test_list_items_in_any_marker_make_the_steps_and_the_first_sentence_the_intro():
    assert edit_run._listed_plan("Plan: do it. Now.\n- a\n* b\n3. c\n4) d") == ("Plan: do it.", ["a", "b", "c", "d"])
    assert edit_run._listed_plan("Only one:\n- a") is None and edit_run._listed_plan("no list") is None
    assert edit_run._listed_plan("\n".join(f"- s{i}" for i in range(12)))[1] == [f"s{i}" for i in range(8)]


# --- one tool list for the whole run, so the prompt's start stays cached ------------------------

def test_every_call_in_a_run_is_offered_the_same_tools_in_the_same_order(turn, monkeypatch):
    plan = {"intro": "x", "outline": ["# Title", "## A", "## B", "## C"], "steps": ["Rename A", "Add C"]}
    seen = scripted(monkeypatch, ("", [("plan", plan)]),
                    ("", [("replace", {"target": "b99", "markdown": "x"})]),               # refused: retried
                    ("", [("plan", plan), ("replace", {"target": "b3", "markdown": "## A"})]),  # plan restated
                    ("", [("insert", {"after": "s7", "markdown": "## C"})]))               # step 2
    got, _ = turn("reorder")
    assert len(seen) == 4 and [c["tools"] for c in seen] == [LOOP_TOOLS] * 4
    assert any(m.get("content", "").startswith("ok: the plan is already set") for m in seen[3]["messages"])
    assert got["edit_run"]["outline_met"] is True


def test_a_skill_keeps_its_own_list_for_its_whole_run(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b99", "markdown": "x"})]),
                    ("", [("replace", {"target": "b5", "markdown": "- **one**"})]))
    turn("/tidy", selection="- one")
    assert [c["tools"] for c in seen] == [["plan", "replace", "edit_text"]] * 2


# --- an outline heading matches the note's when it names the same heading, roughly -------------

@pytest.mark.parametrize("outline, note", [
    ("## Historical Evolution", "## Historical Evolution & Drivers"),          # a prefix, in words
    ("# US Healthcare System Overview", "# US Healthcare System Overview 2026-01-13"),
    ("## Costs & Payment", "## costs  &  payment (new)"),                       # equal once cleaned
    ("## Key Takeaways and Conclusions", "## Conclusions and Key Takeaways"),  # the same words
    ("## Historical Evolution", "#### Historical Evolution"),                   # level is ignored
])
def test_an_outline_heading_matches_the_same_heading_written_a_little_differently(outline, note):
    assert _misses(["# T", note], ["# T", outline]) == ([], [])


@pytest.mark.parametrize("outline, note", [
    ("## History", "## Costs"),
    ("## A", "## Applications"),                     # a prefix in letters is not one in words
    ("## Payment Models", "## Treatment Patterns"),
])
def test_unrelated_headings_still_do_not_match(outline, note):
    assert _misses(["# T", note], ["# T", outline]) == ([1], [])


def test_matching_stays_an_ordered_subsequence():
    now = ["# T", "## Historical Evolution & Drivers", "## Costs"]
    assert _misses(now, ["# T", "## Costs", "## Historical Evolution"]) == ([], [1])


# --- C1: a tool call's text lands as the model wrote it: its real tabs, its LaTeX, its "null" -----

TABBED = "## List\n\n- item with $x$\n\t- nested under it\n\n## After\n\nend\n"


def test_a_tab_indented_list_beside_math_keeps_its_real_tabs(turn, monkeypatch):
    scripted(monkeypatch, ("", [("replace", {"target": "b2", "markdown": "- item with $x^2$\n\t- nested, renamed"})]))
    got, _ = turn("square x", text=TABBED)
    assert "- item with $x^2$\n\t- nested, renamed\n" in got["final"] and "\\t" not in got["final"]
    scripted(monkeypatch, ("", [("edit_text", {"target": "b2", "old": "$x$\n\t- nested", "new": "$y$\n\t- nested"})]))
    got, _ = turn("rename x", text=TABBED)
    assert got["final"] == TABBED.replace("$x$", "$y$")


def test_a_tool_calls_latex_and_tabs_land_exactly_as_the_model_wrote_them(turn, monkeypatch):
    # The raw call's text is never JSON: `\frac` is `\frac`, and a tab before a letter in math is a
    # tab (a repair for JSON's damage wrote it out as `\t`; review, 2026-10-13).
    math = "$$\n\tx = \\frac{1}{2}\\theta\n$$"
    wired(monkeypatch, "<tool_call>\n<function=insert>\n<parameter=after>\nb8\n</parameter>\n"
                       f"<parameter=markdown>\n{math}\n</parameter>\n</function>\n</tool_call>")
    got, _ = turn("add the formula")
    assert got["final"] == NOTE + "\n" + math + "\n"


def test_changing_a_word_to_null_writes_null(turn, monkeypatch):
    wired(monkeypatch, "<tool_call>\n<function=edit_text>\n<parameter=target>\nb2\n</parameter>\n"
                       "<parameter=old>\npara\n</parameter>\n<parameter=new>\nnull\n</parameter>\n</function>\n</tool_call>")
    got, _ = turn("change para to null")
    assert got["final"] == NOTE.replace("Intro para.", "Intro null.")


# --- I1: steps sent as one string are its lines, without their list markers ----------------------

def test_steps_sent_as_one_string_are_read_as_its_lines():
    assert edit_run._steps("1. Rename A\n2) Move B up\n\n- Check\n*  Done") == ["Rename A", "Move B up", "Check", "Done"]
    assert edit_run._steps(["Rename A", " ", "2. Move B"]) == ["Rename A", "Move B"]
    assert edit_run._steps(None) == [] and edit_run._steps("") == []


def test_a_plan_whose_steps_came_as_one_string_runs_each_of_them(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": "1. Rename A\n2. Move B up"})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("move", {"target": "s7", "before": "s3"})]))
    got, _ = turn("go")
    assert got["edit_run"]["plan"]["steps"] == ["Rename A", "Move B up"] and len(seen) == 3
    assert "Step 2 of 2: Move B up" in seen[2]["messages"][-1]["content"]


# --- I4: a list is a plan only when it names more than the reply already did -------------------

def test_a_recap_of_the_fixes_just_made_ends_after_one_call(turn, monkeypatch):
    recap = "Fixed both:\n- renamed A\n- shortened B"
    seen = scripted(monkeypatch, (recap, [("replace", {"target": "b3", "markdown": "## A renamed"}),
                                          ("replace", {"target": "b8", "markdown": "B."})]))
    got, events = turn("rename A and shorten B")
    assert len(seen) == 1 and got["edit_run"]["plan"]["steps"] == [] and len(got["edit_run"]["changes"]) == 2


def test_a_five_item_plan_with_one_edit_continues_from_step_two(turn, monkeypatch):
    plan = "I'll do it in five.\n1. Rename A\n2. Rename B\n3. Shorten A\n4. Shorten B\n5. Check"
    seen = scripted(monkeypatch, (plan, [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    *[("", [])] * 4)
    got, _ = turn("tidy")
    assert len(seen) == 5 and "Step 2 of 5: Rename B" in seen[1]["messages"][-1]["content"]
    assert got["edit_run"]["steps_done"] == 5


def test_two_edits_beside_a_four_item_plan_count_as_its_first_two_steps(turn, monkeypatch):
    plan = "Four parts.\n- Rename A\n- Rename B\n- Shorten A\n- Shorten B"
    seen = scripted(monkeypatch, (plan, [("replace", {"target": "b3", "markdown": "## A2"}),
                                         ("replace", {"target": "b7", "markdown": "## B2"})]),
                    ("", [("replace", {"target": "b4", "markdown": "A."})]),
                    ("", [("replace", {"target": "b8", "markdown": "B."})]))
    got, events = turn("tidy")
    assert len(seen) == 3 and "Step 3 of 4: Shorten A" in seen[1]["messages"][-1]["content"]
    assert [c["step"] for c in got["edit_run"]["changes"]] == [1, 2, 3, 4]
    assert [p for e, p in events if e == "step"][:4] == [
        {"index": 1, "state": "doing"}, {"index": 2, "state": "doing"},
        {"index": 1, "state": "done"}, {"index": 2, "state": "done"}]
    assert got["edit_run"]["steps_done"] == 4


# --- M2: thinking has its own allowance, beside the run's budget ---------------------------------

@pytest.mark.parametrize("mode", ["quick", "deep"])
def test_the_first_calls_thinking_does_not_eat_the_steps_budget(turn, monkeypatch, mode):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A", "Rename B"]})], 9000),
                    ("", [("replace", {"target": "b3", "markdown": "## A2"})], 4000),
                    ("", [("replace", {"target": "b7", "markdown": "## B2"})], 500))
    got, _ = turn("rename both", mode=mode)
    assert got["edit_run"]["stopped"] is None and len(seen) == 3 and len(got["edit_run"]["changes"]) == 2
    assert edit_run.RUN_OUTPUT_TOKENS + edit_run.THINK_BUDGET == 20192


# --- M3: the window is watched on every call, and estimated before the first ---------------------

def test_a_step_whose_prompt_overflowed_stops_the_run_before_its_calls_land(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A", "Rename B", "Rename C"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A2"})]),
                    ("", [("replace", {"target": "b7", "markdown": "## B2"})]))
    scripted_call = copilot_turn.llm.chat_tools

    def chat_tools(messages, tools=None, **opts):
        msg, stats = scripted_call(messages, tools, **opts)
        return msg, {**stats, "ctx_saturated": len(seen) == 3}
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    got, _ = turn("rename all")
    run = got["edit_run"]
    assert run["stopped"] == "error: the conversation no longer fits the model's window" and len(seen) == 3
    assert [c["label"] for c in run["changes"]] == ["Renamed “A” to “A2”"]    # step 2's call never landed


def test_a_note_too_long_for_the_window_is_refused_before_its_edit_call(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b3", "markdown": "## A2"})]))
    monkeypatch.setattr(copilot_turn.llm, "RESIDENT_CTX", copilot_turn.QUICK_NUM_PREDICT + 50)
    with pytest.raises(copilot_turn.CopilotError, match="too long for one request"):
        turn("rename A")
    assert seen == []
# --- M5: a note too big to edit here can still be asked about (test_loop.py); a skill refuses it -----

def test_an_edit_skill_refuses_a_note_too_big_to_edit(turn, monkeypatch):
    monkeypatch.setattr(edits, "MAX_EDIT_CHARS", 40)
    seen = scripted(monkeypatch, ("never", []))
    with pytest.raises(copilot_turn.CopilotError, match=r"this note is too long to edit here \(over 40 characters\)"):
        turn("/tidy", selection="- one")
    assert seen == []


# --- what the owner reads never names a handle; no calls after the steps; steps keep their work ---

def test_no_call_follows_the_steps_when_the_outline_misses_a_heading(turn, monkeypatch):
    plan = {"intro": "x", "outline": ["# Title", "## A", "## B", "## Z"], "steps": ["Add Z"]}
    seen = scripted(monkeypatch, ("", [("plan", plan)]), ("", []))
    got, _ = turn("go")
    assert len(seen) == 2 and "fixups" not in got["edit_run"] and got["edit_run"]["outline_met"] is False
    assert not hasattr(copilot_turn, "MAX_FIXUPS")


def test_the_outline_holds_headings_only():
    assert edit_run._outline({"outline": ["## QF 6", "### <details><summary>Why</summary>",
                                             "The core principle is that a dollar today", "Formula: $PV$"]}) == [
        "## QF 6"]                                  # a <…> heading reads as the rider's placeholder: dropped too


def test_the_plan_and_the_summary_the_owner_reads_name_text_not_handles(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Rename b3 and tidy s7.",
                                    "steps": ["Rename the heading after block b2", "Tidy [b7-b8]"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", []))
    got, events = turn("rename A")
    shown = {"intro": "Rename “A” and tidy “B”.", "steps": ["Rename the heading after “Intro para.”", "Tidy “B”"]}
    assert events[1][1] == {**shown, "outline": None} and got["edit_run"]["plan"] == shown
    assert got["edit_run"]["summary"] == "Rename “A” and tidy “B”." == got["answer"]
    assert "Step 1 of 2: Rename the heading after block b2" in seen[1]["messages"][-1]["content"]   # the model's own words


def test_a_text_answer_on_the_edit_path_names_text_not_handles(turn, monkeypatch):
    scripted(monkeypatch, ("See [b3] for that.", []), ("See [b3] for that.", []))
    got, _ = turn("where is A?")
    assert got["answer"] == "See “A” for that."


def test_a_later_step_that_would_rewrite_the_first_steps_text_is_refused_and_the_run_keeps_it(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Add C.", "steps": ["Add the section", "Add its text", "Close it"]})]),
                    ("", [("insert", {"after": "s7", "markdown": "## C\n\nShort c."})]),
                    ("", [("replace", {"target": "b10", "markdown": "A much longer text about c."})]),
                    ("", []),
                    ("", []))
    got, _ = turn("add a short section C")
    run = got["edit_run"]
    assert "## C\n\nShort c.\n" in got["final"] and "longer" not in got["final"] and len(run["changes"]) == 1
    assert seen[3]["messages"][-1]["content"].startswith(
        "refused: b10 is what you wrote in step 1; it stays as written.")
    assert run["changes"][0]["part_of"] is None and run["steps_done"] == 3 and len(seen) == 5


def test_a_text_answer_on_the_edit_path_keeps_bare_words_that_look_like_handles(turn, monkeypatch):
    scripted(monkeypatch, ("Use an s3 bucket; see block b3.", []), ("Use an s3 bucket; see block b3.", []))
    got, _ = turn("how do I store files?")
    assert got["answer"] == "Use an s3 bucket; see “A”."


def test_every_model_call_of_an_edit_run_names_its_stage_as_it_starts(turn, monkeypatch):
    # The chat counts each call's seconds from its stage: steps write, whatever the first call did.
    scripted(monkeypatch, ("", [("plan", {"intro": "Capitalize both.", "outline": [], "steps": ["Fix a", "Fix b"]})]),
             ("", [("edit_text", {"target": "b4", "old": "Text a.", "new": "Text A."})]),
             ("", [("edit_text", {"target": "b8", "old": "Text b.", "new": "Text B."})]))
    got, _ = turn("Capitalize both", mode="deep")
    assert got["edit_run"]["steps_done"] == 2
    assert got["stages"] == ["Thinking deeply", "Writing", "Writing"]


# --- the text of a change streams into the note as a draft (2026-10-13) ---

def streamed(monkeypatch, *replies, order=None):
    """The scripted model, its tool calls arriving through on_part as the reader hands them on:
    each parameter's text in two pieces. `order` puts a call's parameters in another order."""
    seen = scripted(monkeypatch, *replies)
    script = copilot_turn.llm.chat_tools

    def chat_tools(messages, tools=None, **opts):
        if opts.get("fmt") is None and opts.get("on_part"):
            content, calls, *_ = replies[len(seen)]
            for k, (name, args) in enumerate(calls):
                opts["on_part"]("call", k, name, None)
                for key in (order or {}).get(name, list(args)):
                    value = args[key]
                    opts["on_part"]("parameter", k, key, None)
                    if isinstance(value, str):
                        half = len(value) // 2
                        opts["on_part"]("text", k, key, value[:half])
                        opts["on_part"]("text", k, key, value[half:])
                    opts["on_part"]("value", k, key, value)
                opts["on_part"]("end", k, None, None)
        return script(messages, tools, **opts)
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    return seen


def wired(monkeypatch, *replies, size=3):
    """The model as the raw call hears it: each reply's text read by the real llm.chat_tools (its
    reply read by llm_render._Reader; the server's prompt faked), a few characters at a time."""
    pending = list(replies)

    def generate(body, on_chunk, should_stop, timeout):
        text = pending.pop(0)
        for i in range(0, len(text), size):
            on_chunk(text[i:i + size])
        return {"stop": True, "stop_type": "eos", "timings": {"predicted_n": 10}}
    monkeypatch.setattr(copilot_turn.llm, "_post", lambda url, body, model, timeout: {"prompt": ""})
    monkeypatch.setattr(copilot_turn.llm, "_generate", generate)


def test_a_call_that_turns_into_text_ends_its_draft_and_the_next_call_drafts_afresh(turn, monkeypatch):
    # Its </tool_call> came before its </function>: no call, as Ollama reads it (review, 2026-10-13).
    dead = ("<tool_call>\n<function=replace>\n<parameter=target>\nb4\n</parameter>\n<parameter=markdown>\n"
            "Dead text.\n</parameter>\n</tool_call>\n")
    live = ("<tool_call>\n<function=insert>\n<parameter=after>\nb2\n</parameter>\n<parameter=markdown>\nLive.\n"
            "</parameter>\n</function>\n</tool_call>")
    wired(monkeypatch, dead + live)
    got, events = turn("add a line")
    drafts = [(e, p) for e, p in events if e.startswith("draft")]
    first = drafts[0][1]
    assert (first["from"], first["to"]) == (NOTE.index("Text a."), NOTE.index("- one"))
    ended = drafts.index(("draft_end", {"id": first["id"]}))
    shown = next(p for e, p in drafts[ended:] if e == "draft")
    assert shown["id"] != first["id"] and shown["text"] == ""
    assert shown["from"] == shown["to"] == NOTE.index("## A")
    assert [p.get("draft") for e, p in events if e == "change"] == [shown["id"]]
    assert "Live." in got["final"] and "Dead text." not in got["final"]


def test_an_inserts_text_is_a_draft_at_its_place_until_its_change_lands(turn, monkeypatch):
    streamed(monkeypatch, ("", [("insert", {"after": "b4", "markdown": "New line."})]))
    got, events = turn("add a line")
    kinds = [e for e, _ in events]
    assert kinds == ["draft", "draft_text", "draft_text", "start", "plan", "change"]
    draft = events[0][1]
    assert draft == {"id": "d1.0", "from": NOTE.index("- one"), "to": NOTE.index("- one"), "text": ""}
    assert "".join(p["text"] for e, p in events if e == "draft_text") == "New line."
    assert events[-1][1]["draft"] == "d1.0" and events[-1][1]["id"] == "c1"
    assert "draft" not in got["edit_run"]["changes"][0]            # the turn keeps no draft


def test_text_written_before_its_place_shows_once_the_place_is_known(turn, monkeypatch):
    streamed(monkeypatch, ("", [("insert", {"after": "b4", "markdown": "New line."})]),
             order={"insert": ["markdown", "after"]})
    _got, events = turn("add a line")
    assert [e for e, _ in events][:2] == ["draft", "start"]
    assert events[0][1]["text"] == "New line."


def test_a_replace_draft_covers_its_target_and_a_refused_calls_draft_ends(turn, monkeypatch):
    streamed(monkeypatch, ("", [("replace", {"target": "s7", "markdown": "## B\n\nText B."}),
                                ("insert", {"after": "b99", "markdown": "Lost."}),
                                ("insert", {"before": "b2", "markdown": "<div>raw</div>"})]))
    got, events = turn("fix B")
    drafts = {p["id"]: p for e, p in events if e == "draft"}
    assert drafts["d1.0"]["from"] == NOTE.index("## B") and drafts["d1.0"]["to"] == len(NOTE)
    assert "d1.1" not in drafts                                   # b99 is no place: no draft
    assert [p["id"] for e, p in events if e == "draft_end"] == ["d1.2"]     # HTML refused
    assert [p.get("draft") for e, p in events if e == "change"] == ["d1.0"]


def test_stop_while_a_call_streams_ends_the_run_where_it_is(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "Two.", "outline": [], "steps": ["one", "two"]})]),
                    ("", [("insert", {"after": "b4", "markdown": "One."})]))
    script = copilot_turn.llm.chat_tools

    def chat_tools(messages, tools=None, **opts):
        if opts.get("fmt") is None and len(seen) == 2:            # the second step's call
            raise copilot_turn.llm.Stopped
        return script(messages, tools, **opts)
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    got, _events = turn("do both")
    assert got["edit_run"]["stopped"] == "stopped" and len(got["edit_run"]["changes"]) == 1


def test_a_draft_that_fails_is_counted_and_the_change_still_lands(turn, monkeypatch, tmp_path):
    streamed(monkeypatch, ("", [("insert", {"after": "b4", "markdown": "New line."})]))

    def where(self, call):
        raise RuntimeError("a bug in the draft")
    monkeypatch.setattr(edit_run._Drafts, "_where", where)
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    got, events = turn("add a line")
    assert [e for e, _ in events] == ["start", "plan", "change"] and "New line." in got["final"]
    assert records[-1]["edit_run"]["draft_failures"] == 1


def test_a_parameter_given_twice_moves_or_restarts_its_draft(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("insert", {"after": "b8", "markdown": "B"})]))
    script = copilot_turn.llm.chat_tools

    def chat_tools(messages, tools=None, **opts):
        if opts.get("on_part"):
            for part in [("call", 0, "insert", None), ("parameter", 0, "after", None), ("value", 0, "after", "b2"),
                         ("parameter", 0, "markdown", None), ("text", 0, "markdown", "A"), ("value", 0, "markdown", "A"),
                         ("parameter", 0, "after", None), ("value", 0, "after", "b8"),
                         ("parameter", 0, "markdown", None), ("text", 0, "markdown", "B"), ("value", 0, "markdown", "B"),
                         ("end", 0, None, None)]:
                opts["on_part"](*part)
        return script(messages, tools, **opts)
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    _got, events = turn("add a line")
    drafts = [(e, p) for e, p in events if e.startswith("draft")]
    assert drafts == [("draft", {"id": "d1.0", "from": NOTE.index("## A"), "to": NOTE.index("## A"), "text": ""}),
                      ("draft_text", {"id": "d1.0", "text": "A"}),
                      ("draft", {"id": "d1.0", "from": len(NOTE), "to": len(NOTE), "text": "A"}),       # moved
                      ("draft", {"id": "d1.0", "from": len(NOTE), "to": len(NOTE), "text": ""}),        # restarted
                      ("draft_text", {"id": "d1.0", "text": "B"})]


def test_a_draft_goes_where_its_change_will_land_as_its_text_decides_it(turn, monkeypatch):
    streamed(monkeypatch, ("", [("insert", {"after": "b3", "markdown": "## A2\n\nNew."}),
                                ("replace", {"target": "s7", "markdown": "## Bee"})]))
    _got, events = turn("add a section and rename B")
    starts = [(p["id"], p["from"], p["to"]) for e, p in events if e == "draft"]
    # The heading would cut s3 after its heading: the draft moves to after the whole section, as the
    # insert does. A rename dims only the heading it renames.
    assert ("d1.0", NOTE.index("Text a."), NOTE.index("Text a.")) in starts
    assert [s for s in starts if s[0] == "d1.0"][-1][1:] == (NOTE.index("## B"), NOTE.index("## B"))
    assert starts[-1] == ("d1.1", NOTE.index("## B"), NOTE.index("Text b."))


def test_a_section_deleted_or_rewritten_is_dimmed_whole_while_its_text_streams(turn, monkeypatch):
    # Empty text deletes the section, and a heading line still being written starts its new text:
    # only a whole text of one heading line is a rename.
    streamed(monkeypatch, ("", [("replace", {"target": "s7", "markdown": ""}),
                                ("replace", {"target": "s3", "markdown": "## A\n\nNew a."})]))
    _got, events = turn("drop B and rewrite A")
    starts = {(p["id"], p["from"], p["to"]) for e, p in events if e == "draft"}
    assert starts == {("d1.0", NOTE.index("## B"), len(NOTE)), ("d1.1", NOTE.index("## A"), NOTE.index("## B"))}


def test_a_draft_reads_its_text_as_its_tool_does(turn, monkeypatch):
    # A copied label is stripped, and a new <details> is a callout, before the text decides the
    # place: the labelled heading moves the draft past s3, as it moves the insert; the folded one does not.
    streamed(monkeypatch, ("", [("insert", {"after": "b4", "markdown": "[b9] ## A2\n\nNew."}),
                                ("insert", {"after": "b3", "markdown": "<details><summary>More</summary>\n\n"
                                                                      "## Inside\n\nx\n</details>"})]))
    _got, events = turn("add two")
    last = {p["id"]: (p["from"], p["to"]) for e, p in events if e == "draft"}
    assert last == {"d1.0": (NOTE.index("## B"),) * 2, "d1.1": (NOTE.index("Text a."),) * 2}


def test_a_draft_put_before_a_notes_properties_shows_after_them(turn, monkeypatch):
    text = "---\ntags: [a]\n---\n\n# Title\n\nIntro.\n"
    streamed(monkeypatch, ("", [("insert", {"before": "b1", "markdown": "TL;DR."})]))
    got, events = turn("add a summary at the top", text=text)
    assert {(p["from"], p["to"]) for e, p in events if e == "draft"} == {(text.index("# Title"),) * 2}
    assert got["final"] == text.replace("# Title", "TL;DR.\n\n# Title")


def test_a_draft_end_that_cannot_be_sent_is_counted_and_the_run_goes_on(con, vault, monkeypatch):
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *a, **k: [])
    streamed(monkeypatch, ("", [("insert", {"after": "b2", "markdown": "<div>raw</div>"}),
                                ("insert", {"after": "b4", "markdown": "Fine."})]))
    write(vault, "Notes/school/n.md", "x")
    (vault / "Notes/school/n.md").write_text(NOTE, encoding="utf-8")
    source = context.sync_note(con, vault, "Notes/school/n.md", embed=False)
    records, events = [], []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))

    def on_edit(event, payload):
        if event == "draft_end":
            raise RuntimeError("the chat is gone")
        events.append(event)
    got = copilot_turn.run_turn(con, source["path"], "add", [], "quick", vault=vault, doc=NOTE,
                                doc_hash=edits.digest(NOTE), on_edit=on_edit)
    assert len(got["edit_run"]["changes"]) == 1 and records[-1]["edit_run"]["draft_failures"] == 1


# --- a step's target: the handles its plan step names, judged by code (2026-10-22) -----------

def test_a_step_target_is_the_blocks_its_handles_cover():
    note = blocks.Note(NOTE)
    assert edit_run._target(note, "Rename B (s7)") == {"b7", "b8"}
    assert edit_run._target(note, "Tidy the list b4-b6 and [b2]") == {"b2", "b4", "b5", "b6"}
    assert edit_run._target(note, "Move s7 to s3") == {"b3", "b4", "b5", "b6", "b7", "b8"}
    assert edit_run._target(note, "Tidy up") is None and edit_run._target(note, "Fix b99") is None
    assert edit_run._target(note, "Rename the heading after block b2") == {"b2", "b3"}     # b2 a landmark
    assert edit_run._target(note, "Add a line before [b7]") == {"b6", "b7"}


def test_a_step_whose_changes_miss_its_section_is_sent_back_there_once(turn, monkeypatch):
    # The 2026-10-22 run: each step did the step before's leftover, and every step read done.
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "x", "steps": ["Fix the intro (b2)", "Rename B (s7)"]})]),
                    ("", [("edit_text", {"target": "b2", "old": "Intro para.", "new": "Intro paragraph."})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("replace", {"target": "b7", "markdown": "## B renamed"})]))
    got, events = turn("tidy it")
    asked = seen[3]["messages"][-1]["content"]
    assert "Step 2 of 2: Rename B (s7)" in asked and "were not in s7" in asked
    assert "## A renamed" in got["final"] and "## B renamed" in got["final"]      # the stray change stays
    assert len(got["edit_run"]["changes"]) == 3 and len(seen) == 4
    assert [p for e, p in events if e == "step"] == [
        {"index": 1, "state": "doing"}, {"index": 1, "state": "done"},
        {"index": 2, "state": "doing"}, {"index": 2, "state": "done"}]


def test_a_step_that_misses_its_section_twice_is_skipped_and_its_changes_stay(turn, monkeypatch):
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append((kind, payload)))
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "x", "steps": ["Rename B (s7)"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("edit_text", {"target": "b2", "old": "Intro para.", "new": "Intro."})]))
    got, events = turn("rename B")
    assert len(seen) == 3 and len(got["edit_run"]["changes"]) == 2 and got["edit_run"]["steps_done"] == 1
    assert [p for e, p in events if e == "step"] == [{"index": 1, "state": "doing"}, {"index": 1, "state": "skipped"}]
    facts = records[0][1]["edit_run"]
    assert facts["corrections"] == 1 and facts["steps_missed"] == 1


def test_a_step_that_names_no_handle_is_done_wherever_it_changed(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Tidy up"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, events = turn("tidy")
    assert len(seen) == 2 and events[-1] == ("step", {"index": 1, "state": "done"})


def test_a_first_reply_edit_off_its_step_is_judged_but_not_sent_back(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "x", "steps": ["Rename B (s7)", "Fix the intro (b2)"]}),
                          ("replace", {"target": "b3", "markdown": "## A renamed"})]),
                    ("", [("edit_text", {"target": "b2", "old": "Intro para.", "new": "Intro."})]))
    got, events = turn("go")
    assert len(seen) == 2 and len(got["edit_run"]["changes"]) == 2
    assert [p for e, p in events if e == "step"] == [
        {"index": 1, "state": "doing"}, {"index": 1, "state": "skipped"},
        {"index": 2, "state": "doing"}, {"index": 2, "state": "done"}]


# --- the lookup phase: read-only tools before, and inside, the run (2026-11-13) ---------------

def _looking(vault, monkeypatch, *replies, note=NOTE):
    from slim.copilot import lookup
    seen = scripted(monkeypatch, *replies)
    draft = blocks.Note(note)
    reader = lookup.Reader(None, vault, open_note=("Notes/school/n.md", draft))
    write(vault, "Notes/school/b.md", "Bee text.")
    events, said = [], []
    tools = ["search", "read_note", "list_folder", "plan", *STEP_TOOLS]
    out = edit_run._turn([{"role": "user", "content": "q"}], think=False, tools=tools,
                         step_tools=edit_run.STEP_TOOLS, note=draft, before_text=note, first_predict=100,
                         on_edit=lambda e, p: events.append((e, p)), should_stop=lambda: False,
                         lookups=reader, on_reply=said.append)
    return out, seen, events, said


def test_lookups_run_before_the_first_answer_and_the_reply_streams(vault, monkeypatch):
    out, seen, events, said = _looking(vault, monkeypatch,
                                       ("Let me look.", [("read_note", {"path": "Notes/school/b.md"})]),
                                       ("Bee says hi.", []))
    assert out["kind"] == "answer" and out["answer"] == "Bee says hi." and events == []
    assert said == ["Let me look.", "Bee says hi."]
    assert seen[1]["messages"][-1]["role"] == "tool" and "Bee text." in seen[1]["messages"][-1]["content"]
    assert out["looked_up"] == [{"name": "read_note", "path": "Notes/school/b.md", "refused": False}]
    assert out["calls"] == 1


def test_a_reply_mixing_a_lookup_and_a_tool_not_offered_runs_the_lookup(vault, monkeypatch):
    out, seen, _events, _said = _looking(vault, monkeypatch,
                                         ("", [("list_folder", {"path": "Notes"}), ("done", {"summary": "x"})]),
                                         ("ok", []))
    results = [m["content"] for m in seen[1]["messages"] if m["role"] == "tool"]
    assert results[0].startswith("school/") and results[1] == "refused: done is not available now"
    assert out["answer"] == "ok"


def test_a_model_that_keeps_looking_past_the_cap_is_stopped(vault, monkeypatch):
    replies = [("", [("list_folder", {"path": "Notes"})])] * (edit_run.MAX_LOOKUPS + edit_run.MAX_REFUSALS + 2)
    out, seen, _events, _said = _looking(vault, monkeypatch, *replies)
    assert len(seen) == edit_run.MAX_LOOKUPS + edit_run.MAX_REFUSALS
    assert out["stopped"] == "lookups" and out["answer"] == ""


def test_a_plan_beside_a_lookup_reads_first_then_does_its_step(vault, monkeypatch):
    out, seen, events, _said = _looking(
        vault, monkeypatch,
        ("", [("plan", {"intro": "Add bee.", "steps": ["Add it"]}), ("read_note", {"path": "Notes/school/b.md"})]),
        ("", [("insert", {"after": "b2", "markdown": "Bee text."})]))
    assert out["kind"] == "edit" and len(out["changes"]) == 1 and len(seen) == 2
    told = [m["content"] for m in seen[1]["messages"] if m["role"] == "tool"]
    assert told[0] == "ok: plan accepted" and "Bee text." in told[1]
    assert seen[1]["opts"]["temperature"] == edit_run.STEP_TEMPERATURE


def test_drafts_ignore_reply_text():
    events = []
    drafts = edit_run._Drafts(blocks.Note(NOTE), None, lambda e, p: events.append(e))
    drafts.part("reply", -1, None, "hello")
    assert drafts.calls == {} and drafts.failed == 0 and events == []
