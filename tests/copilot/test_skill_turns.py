"""Copilot skills: whole notes chosen by code. A skill that changes the note or makes a new one
writes nothing: its changes go to the plugin as they land."""

from pathlib import Path

import pytest

from slim import db
from slim.copilot import context, edits, blocks, skills, run as edit_run, turn as copilot_turn


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "skills.db")


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Capture/work").mkdir(parents=True)
    (root / "Notes/work").mkdir(parents=True)
    return root


def meeting(root: Path, name: str, when: str, notes: str | None) -> str:
    path = f"Capture/work/{name}.md"
    section = f"## Notes\n\n{notes}\n\n" if notes is not None else ""
    (root / path).write_text(
        f"---\ntitle: {name}\ndate: {when[:10]}\ntype: meeting\nrecorded_at: {when}\n---\n\n"
        f"```slim-meeting\n{section}## Transcript\n\nMe: words from {name}\n```\n",
        encoding="utf-8")
    return path


def note(root: Path, path: str, body: str) -> str:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(body, encoding="utf-8")
    return path


def open_source(con, vault, path):
    return context.sync_note(con, vault, path, embed=False)


def test_folder_input_filters_cuts_to_the_section_and_fills_newest_first(con, vault, monkeypatch):
    first = meeting(vault, "Kickoff", "2026-09-01T09:00:00", "old point " * 30)
    second = meeting(vault, "Review", "2026-09-08T09:00:00", "middle point")
    third = meeting(vault, "Retro", "2026-09-15T09:00:00", "newest point")
    meeting(vault, "Silent", "2026-09-16T09:00:00", None)                 # no Notes section
    note(vault, "Capture/work/Index.md", "---\ntype: note\n---\n\nnot a meeting\n")
    source = open_source(con, vault, second)
    skill = skills.load(vault)["consolidate-my-notes"]
    budget = len("middle point") + len("newest point") + 5
    monkeypatch.setattr(context, "INPUT_BUDGET_CHARS", budget)
    pack = context.gather(con, vault, source, skill=skill)
    assert [doc["path"] for doc in pack.docs] == [second, third]           # oldest first to read
    assert [doc["text"] for doc in pack.docs] == ["middle point", "newest point"]
    assert pack.dropped == [first]                                         # the oldest did not fit
    assert pack.skipped == ["Capture/work/Silent.md"]


def test_links_resolve_by_name_nearest_folder_first_and_missing_ones_are_named(con, vault):
    open_path = note(vault, "Notes/work/plan.md", "the plan\n")
    near = note(vault, "Notes/work/budget.md", "near budget\n")
    note(vault, "Capture/work/deeper/budget.md", "far budget\n")
    for path in (open_path, near, "Capture/work/deeper/budget.md"):
        open_source(con, vault, path)
    source = open_source(con, vault, open_path)
    pack = context.gather(con, vault, source, links_from="use [[budget]] and [[Nope|x]]")
    assert [(doc["path"], doc["role"]) for doc in pack.docs] == [
        (open_path, "open"), (near, "linked")]
    assert pack.missing == ["Nope"]


def test_a_selection_skill_without_a_selection_is_refused(con, vault):
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    with pytest.raises(copilot_turn.CopilotError, match="select some"):
        context.gather(con, vault, source, skill=skills.load(vault)["explain"])


def tooled(monkeypatch, *replies):
    """A fake model: each call returns the next (content, [(tool, args)])."""
    seen = []

    def chat_tools(messages, tools=None, **opts):
        seen.append({"messages": list(messages), "tools": [t["function"]["name"] for t in tools or []],
                     "opts": opts})
        content, calls = replies[len(seen) - 1]
        if opts.get("on_part") and content:
            opts["on_part"]("reply", -1, None, content)
        return ({"content": content, "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                 "raw_tool_calls": []}, {"duration_s": 1})
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    return seen


def test_a_question_names_no_mode_in_the_loop_or_an_edit_skill(con, vault, monkeypatch):
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    seen = tooled(monkeypatch, ("ok", []))
    turn = copilot_turn.run_turn(con, source["path"], "what is /usr/bin for?", [], "medium", vault=vault)
    system = seen[0]["messages"][0]["content"]
    assert system == copilot_turn.loop_system()                    # not a skill: the loop's own prompt
    edit_system = copilot_turn.edit_message("Notes/work/a.md", blocks.Note("text\n"))
    for text in (system, edit_system):
        assert "Edit mode" not in text and "Ask mode" not in text
    assert turn["edit_run"] is None and turn["skill"] is None and turn["answer"] == "ok"


def test_an_unknown_command_is_refused_not_asked(con, vault, monkeypatch):
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    seen = tooled(monkeypatch)
    with pytest.raises(copilot_turn.CopilotError, match="no /quizz skill"):
        copilot_turn.run_turn(con, source["path"], "/quizz me", [], "medium", vault=vault)
    assert seen == []


def test_a_skill_runs_at_its_own_depth_and_its_prompt_rides_in_history(con, vault, monkeypatch):
    meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    source = open_source(con, vault, meeting(vault, "Review", "2026-09-08T09:00:00", "monday"))
    seen = tooled(monkeypatch, ("Nothing to merge yet.", []), ("Right: 4.", []), ("Right: 4.", []))
    turn = copilot_turn.run_turn(con, source["path"], "/consolidate-my-notes", [], "medium", vault=vault)
    assert turn["effort"] == "high" and seen[0]["opts"]["think"] is True
    copilot_turn.run_turn(con, source["path"], "my answer is 4", [
        {"role": "you", "text": "/quiz"}, {"role": "slim", "text": "What is 2+2?"}], "medium", vault=vault)
    assert seen[1]["messages"][1]["content"].startswith("Quiz me on this note. Ask ONE question now")
    # The plugin names the note an earlier question was asked on; the /command after it still expands.
    copilot_turn.run_turn(con, source["path"], "my answer is 4", [
        {"role": "you", "text": f"(on {source['path']}) /quiz"}, {"role": "slim", "text": "What is 2+2?"}],
        "medium", vault=vault)
    assert seen[2]["messages"][1]["content"].startswith(f"(on {source['path']}) Quiz me on this note.")


def test_only_a_note_path_is_read_as_where_an_earlier_question_was_asked(vault):
    loaded = skills.load(vault)
    assert copilot_turn._expanded("(on Notes/a b.md) /quiz", loaded).startswith(
        "(on Notes/a b.md) Quiz me on this note.")
    typed = "(on the bus) what does ) /quiz mean"                  # their words, not the plugin's prefix
    assert copilot_turn._expanded(typed, loaded) == typed


def test_a_skill_reads_the_folder_when_the_invocation_says_so(con, vault, monkeypatch):
    kickoff = meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    source = open_source(con, vault, meeting(vault, "Review", "2026-09-08T09:00:00", "monday"))
    seen = tooled(monkeypatch, ("- [ ] ship — Me", []))
    turn = copilot_turn.run_turn(con, source["path"], "/actions folder only mine", [], "medium", vault=vault)
    assert turn["inputs"]["used"] == [kickoff, source["path"]]
    # A chat skill runs in the loop: its notes ride with its prompt, under the persona.
    notes, prompt = seen[0]["messages"][-1]["content"].split("\n\n---\n\n")
    assert "=== Capture/work/Kickoff.md · 2026-09-01 ===" in notes
    assert prompt.endswith("only mine") and "folder" not in prompt
    assert seen[0]["messages"][0]["content"].startswith(copilot_turn.PERSONA)
    assert [c["path"] for c in turn["citations"]] == [kickoff]       # what code read, the open note aside


def test_an_overflowing_prompt_never_makes_a_note(con, vault, monkeypatch):
    meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    source = open_source(con, vault, meeting(vault, "Review", "2026-09-08T09:00:00", "monday"))
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", lambda messages, tools=None, **k: (
        {"content": "", "tool_calls": [{"name": "create_note", "arguments": {"path": "x.md", "markdown": "x"}}],
         "raw_tool_calls": []}, {"ctx_saturated": True}))
    events = []
    with pytest.raises(copilot_turn.CopilotError, match="too long for one request"):
        copilot_turn.run_turn(con, source["path"], "/consolidate-my-notes", [], "medium", vault=vault,
                              on_edit=lambda *event: events.append(event))
    assert events == []


def test_section_of_skips_code_fences_and_the_derived_summary():
    text = ("<!-- slim:summary -->\n## Notes\nmodel opinion\n<!-- /slim:summary -->\n"
            "## Notes\n\nsetup:\n```bash\n# install deps\n## Notes\n```\nafter\n## Transcript\nMe: x\n")
    assert context.section_of(text, "Notes") == "setup:\n```bash\n# install deps\n## Notes\n```\nafter"


def test_a_partial_link_path_ends_at_a_folder_boundary(con, vault):
    open_path = note(vault, "Notes/work/plan.md", "the plan\n")
    note(vault, "Notes/homework/budget.md", "wrong\n")
    right = note(vault, "Notes/work/sub/budget.md", "right\n")
    for path in ("Notes/homework/budget.md", right):
        open_source(con, vault, path)
    assert context.resolve_link(con, vault, "work/sub/budget", open_path) == right
    assert context.resolve_link(con, vault, "ork/sub/budget", open_path) is None


def test_a_broken_vault_skill_says_which_file_to_fix(con, vault, monkeypatch):
    (vault / "Skills").mkdir()
    (vault / "Skills/bad.md").write_text("---\ninput: everything\n---\nbody\n")
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    tooled(monkeypatch)
    with pytest.raises(copilot_turn.CopilotError, match=r"/bad cannot run: input must be .*Skills/bad.md"):
        copilot_turn.run_turn(con, source["path"], "/bad", [], "medium", vault=vault)


def test_a_question_never_sends_the_selection(con, vault, monkeypatch):
    """A selection left in the note rode along with every question, unseen (review 2026-09-23)."""
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    seen = tooled(monkeypatch, ("ok", []))
    copilot_turn.run_turn(con, source["path"], "what does this mean?", [], "medium",
                          selection="leftover", vault=vault)
    assert seen[0]["messages"][-1]["content"].endswith("---\n\nwhat does this mean?")
    assert "leftover" not in str(seen[0]["messages"])


def test_a_turn_sees_the_open_notes_images_in_note_order(con, vault, monkeypatch):
    # "Turn this screenshot into LaTeX" is an edit: the model must see the image it is asked to
    # transcribe. What an image says is not searchable, so every turn carries them (at most 6).
    from PIL import Image
    open_path = note(vault, "Notes/work/eq.md", "# Eq\n\n![[psi.png|300]]\n\n![[mle.png]]\n")
    for name in ("psi.png", "mle.png"):
        Image.new("RGB", (12, 8), color="red").save(str(vault / "Notes/work" / name))
    source = open_source(con, vault, open_path)
    seen = tooled(monkeypatch, ("Nothing to change.", []))
    copilot_turn.run_turn(con, source["path"], "write the equation under its screenshot", [], "medium",
                          vault=vault)
    assert "replace" in seen[0]["tools"]
    shown = seen[0]["messages"][-2]
    assert len(shown["images"]) == 2
    assert shown["content"].index("`psi.png`") < shown["content"].index("`mle.png`")


def test_consolidation_makes_a_new_note_as_a_change_and_writes_nothing(con, vault, monkeypatch):
    kickoff = meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    open_path = meeting(vault, "Review", "2026-09-08T09:00:00", "moved to monday")
    source = open_source(con, vault, open_path)
    body = "# Work\n\n- Ship moved from Friday to Monday [[Review]]"
    seen = tooled(monkeypatch, ("", [("create_note", {"path": "Consolidated.md", "markdown": body}),
                                     ("create_note", {"path": "Capture/work/Consolidated.md", "markdown": body}),
                                     ("done", {"summary": "Merged your two meetings."})]))
    events = []
    turn = copilot_turn.run_turn(con, source["path"], "/consolidate-my-notes focus on dates", [], "high",
                                 vault=vault, on_edit=lambda event, payload: events.append((event, payload)))
    system, user = seen[0]["messages"][0]["content"], seen[0]["messages"][-1]["content"]
    assert seen[0]["tools"] == ["create_note", "done"] and len(seen) == 1
    assert seen[0]["opts"]["temperature"] == edit_run.FIRST_TEMPERATURE     # a new note is prose they read
    assert copilot_turn.NEW_NOTE_RIDER.format(folder="Capture/work") in system and copilot_turn.EDIT_RIDER not in system
    assert "=== Capture/work/Kickoff.md · 2026-09-01 ===\nship friday" in system
    assert "words from" not in system                                 # the section, not the transcript
    assert user.endswith("focus on dates")
    assert [event for event, _ in events] == ["start", "plan", "change"]
    created = events[2][1]
    assert created == {"id": "c1", "step": 1, "op": "create", "label": "Created Capture/work/Consolidated.md",
                       "needs": [], "path": "Capture/work/Consolidated.md",
                       "text": "---\norigin: copilot\n---\n\n" + body + "\n"}
    assert len(turn["edit_run"]["changes"]) == 1             # the same new note twice is refused
    assert turn["edit_run"]["summary"] == "Merged your two meetings." == turn["answer"]
    assert not (vault / "Capture/work/Consolidated.md").exists()       # the plugin creates it
    assert [c["path"] for c in turn["citations"]] == [kickoff, open_path]
    assert turn["inputs"]["used"] == [kickoff, open_path]


LIST_NOTE = "# Why\n\n*   **Lexical**: bank\n+ __Pragmatics__: salt\n\n## Next\n\nUntouched.\n"


def test_a_selection_skill_changes_only_the_selected_blocks_and_only_with_replace(con, vault, monkeypatch):
    source = open_source(con, vault, note(vault, "Notes/work/why.md", LIST_NOTE))
    selected = "*   **Lexical**: bank\n+ __Pragmatics__: salt"
    seen = tooled(monkeypatch, ("", [("replace", {"target": "b5", "markdown": "Changed."}),
                                     ("replace", {"target": "b2", "markdown": "- **Lexical**: bank"}),
                                     ("replace", {"target": "b3", "markdown": "- **Pragmatics**: salt"})]))
    turn = copilot_turn.run_turn(con, source["path"], "/tidy", [], "medium", vault=vault, selection=selected,
                                 doc=LIST_NOTE, doc_hash=edits.digest(LIST_NOTE))
    assert seen[0]["tools"] == ["plan", "replace", "edit_text"] and seen[0]["opts"]["temperature"] == 0.0
    assert seen[0]["messages"][-1]["content"].endswith("Selected text:\n" + selected)
    run = turn["edit_run"]
    text = LIST_NOTE
    for change in run["changes"]:
        text = edits.apply_splices(text, change["splices"])
    assert text == "# Why\n\n- **Lexical**: bank\n- **Pragmatics**: salt\n\n## Next\n\nUntouched.\n"
    assert [c["label"] for c in run["changes"]] == ["Rewrote “Lexical: bank”", "Rewrote “Pragmatics: salt”"]
    assert turn["citations"] == [{"n": 1, "path": "Notes/work/why.md", "title": "why"}]
    with pytest.raises(copilot_turn.CopilotError, match="select some in the note first"):
        copilot_turn.run_turn(con, source["path"], "/tidy", [], "medium", vault=vault)
    with pytest.raises(copilot_turn.CopilotError, match="not in the note as it is now"):
        copilot_turn.run_turn(con, source["path"], "/tidy", [], "medium", vault=vault, selection="gone text")


TWICE = "# Todo 📚\n\n- [ ] review chapter\n\n## Later\n\n- [ ] review chapter\n"


def test_a_selection_is_the_one_made_when_its_words_are_in_the_note_twice(con, vault, monkeypatch):
    """The editor says where a selection starts, in its own UTF-16 units (📚 is two). Without that,
    or when it does not fit the text, words the note has twice are refused, never guessed."""
    source = open_source(con, vault, note(vault, "Notes/work/todo.md", TWICE))
    selected, second = "- [ ] review chapter", TWICE.rindex("- [ ] review chapter")
    tooled(monkeypatch, ("", [("replace", {"target": "b2", "markdown": "- [x] review chapter"}),
                              ("replace", {"target": "b4", "markdown": "- [x] review chapter"})]))
    turn = copilot_turn.run_turn(con, source["path"], "/tidy", [], "medium", vault=vault, selection=selected,
                                 selection_from=edits.u16(TWICE[:second]), doc=TWICE, doc_hash=edits.digest(TWICE))
    text = TWICE
    for change in turn["edit_run"]["changes"]:
        text = edits.apply_splices(text, change["splices"])
    assert text == TWICE[:second] + "- [x] review chapter\n"           # the first is outside the selection
    for where in (None, second):        # no place, or one in code points: one unit short of it
        with pytest.raises(copilot_turn.CopilotError, match="appears 2 times in the note"):
            copilot_turn.run_turn(con, source["path"], "/tidy", [], "medium", vault=vault, selection=selected,
                                  selection_from=where, doc=TWICE, doc_hash=edits.digest(TWICE))


def test_a_new_note_call_that_could_not_be_read_is_asked_for_once_more(con, vault, monkeypatch):
    meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    source = open_source(con, vault, meeting(vault, "Review", "2026-09-08T09:00:00", "monday"))
    seen = tooled(monkeypatch, ("<tool_call>{broken", []),
                  ("", [("create_note", {"path": "All.md", "markdown": "# All"})]))
    turn = copilot_turn.run_turn(con, source["path"], "/consolidate-my-notes", [], "high", vault=vault)
    assert seen[1]["messages"][-1] == {"role": "tool", "content": (
        "refused: your tool call could not be read (a tool call came back as text); write it again\n\n"
        "Make the new note with the create_note tool.")}
    assert [c["path"] for c in turn["edit_run"]["changes"]] == ["Capture/work/All.md"]


def test_the_built_in_skills_speak_of_changes_not_search_blocks(vault):
    loaded = skills.load(vault)
    for skill in loaded.values():
        assert "SEARCH" not in skill.prompt and "ONE change" not in skill.prompt, skill.name
    assert " ".join(loaded["tidy"].prompt.split()) == (
        "Clean up the formatting of the selected text: consistent list markers, headings, spacing and "
        "emphasis. Keep every word. {{args}}")
    assert (loaded["tidy"].input, loaded["tidy"].output) == ("selection", "edit")
    assert not hasattr(loaded["tidy"], "proposes")


def test_brief_makes_one_new_note_in_the_open_notes_folder(con, vault, monkeypatch):
    source = open_source(con, vault, meeting(vault, "Pricing call", "2026-09-08T09:00:00", "raise to 12"))
    seen = tooled(monkeypatch, ("", [("create_note", {"path": "Pricing brief.md", "markdown": "## Goal\n\nx"}),
                                     ("done", {"summary": "Wrote the brief."})]))
    events = []
    turn = copilot_turn.run_turn(con, source["path"], "/brief", [], "medium", vault=vault,
                                 on_edit=lambda event, payload: events.append((event, payload)))
    assert seen[0]["tools"] == ["create_note", "done"] and seen[0]["opts"]["think"] is True    # its own depth
    assert "ONE new note in this folder" in seen[0]["messages"][-1]["content"]
    assert "the open note's folder is `Capture/work/`" in seen[0]["messages"][0]["content"]
    assert [p["path"] for e, p in events if e == "change"] == ["Capture/work/Pricing brief.md"]
    assert turn["edit_run"]["summary"] == "Wrote the brief." and not (vault / "Capture/work/Pricing brief.md").exists()


def test_a_new_notes_prose_list_is_never_read_as_a_plan(con, vault, monkeypatch):
    meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    source = open_source(con, vault, meeting(vault, "Review", "2026-09-08T09:00:00", "monday"))
    prose = "Merged them.\n- Shipping\n- Dates\n- Owners"
    seen = tooled(monkeypatch, (prose, [("create_note", {"path": "All.md", "markdown": "# All"})]))
    turn = copilot_turn.run_turn(con, source["path"], "/consolidate-my-notes", [], "high", vault=vault)
    assert len(seen) == 1 and turn["edit_run"]["plan"]["steps"] == []
