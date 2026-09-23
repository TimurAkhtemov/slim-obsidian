"""Copilot skills and Edit mode: whole notes chosen by code, and proposals that write nothing."""

from pathlib import Path

import pytest

from slim import copilot, db, skills


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
    return copilot.sync_note(con, vault, path, embed=False)


def fake_model(monkeypatch, reply: str, seen: dict):
    def fake_stream(messages, **kwargs):
        seen["messages"] = messages
        seen["num_predict"] = kwargs["num_predict"]
        return reply, {}
    monkeypatch.setattr(copilot.llm, "chat_stream", fake_stream)
    monkeypatch.setattr(copilot.llm, "chat_json",
                        lambda *a, **k: pytest.fail("a skill must not ask the model for sources"))


def test_folder_input_filters_cuts_to_the_section_and_fills_newest_first(con, vault, monkeypatch):
    first = meeting(vault, "Kickoff", "2026-09-01T09:00:00", "old point " * 30)
    second = meeting(vault, "Review", "2026-09-08T09:00:00", "middle point")
    third = meeting(vault, "Retro", "2026-09-15T09:00:00", "newest point")
    meeting(vault, "Silent", "2026-09-16T09:00:00", None)                 # no Notes section
    note(vault, "Capture/work/Index.md", "---\ntype: note\n---\n\nnot a meeting\n")
    source = open_source(con, vault, second)
    skill = skills.load(vault)["consolidate-my-notes"]
    budget = len("middle point") + len("newest point") + 5
    monkeypatch.setattr(copilot, "INPUT_BUDGET_CHARS", budget)
    pack = copilot.gather(con, vault, source, skill=skill)
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
    pack = copilot.gather(con, vault, source, links_from="use [[budget]] and [[Nope|x]]")
    assert [(doc["path"], doc["role"]) for doc in pack.docs] == [
        (open_path, "open"), (near, "linked")]
    assert pack.missing == ["Nope"]


def test_a_selection_skill_without_a_selection_is_refused(con, vault):
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    with pytest.raises(copilot.CopilotError, match="select some"):
        copilot.gather(con, vault, source, skill=skills.load(vault)["explain"])


def test_consolidation_proposes_a_new_note_and_writes_nothing(con, vault, monkeypatch):
    meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    open_path = meeting(vault, "Review", "2026-09-08T09:00:00", "moved to monday")
    source = open_source(con, vault, open_path)
    seen = {}
    fake_model(monkeypatch, (
        "Merged your two meetings.\n\n"
        "FILE: Capture/work/Consolidated.md\n<<<<<<< SEARCH\n=======\n"
        "# Work\n\n- Ship moved from Friday to Monday [[Review]]\n>>>>>>> REPLACE\n"), seen)
    turn = copilot.run_turn(con, source["id"], "/consolidate-my-notes focus on dates", [], "deep",
                            vault=vault)
    system, user = seen["messages"][0]["content"], seen["messages"][-1]["content"]
    assert copilot.EDIT_RIDER.split("\n")[0] in system
    assert "=== Capture/work/Kickoff.md · 2026-09-01 ===\nship friday" in system
    assert "words from" not in system                                 # the section, not the transcript
    assert user.endswith("focus on dates")
    assert seen["num_predict"] == copilot.PROPOSE_NUM_PREDICT["deep"]
    assert turn["answer"] == "Merged your two meetings."
    assert turn["skill"] == "consolidate-my-notes" and turn["edit"] is True
    (created,) = turn["proposal"]["files"]
    assert created["kind"] == "create" and created["path"] == "Capture/work/Consolidated.md"
    assert not (vault / "Capture/work/Consolidated.md").exists()       # proposals never write
    assert [c["path"] for c in turn["citations"]] == ["Capture/work/Kickoff.md", open_path]
    assert turn["inputs"]["used"] == ["Capture/work/Kickoff.md", open_path]


def test_edit_mode_changes_notes_in_view_and_drops_the_rest(con, vault, monkeypatch):
    open_path = note(vault, "Notes/work/plan.md", "# Plan\n\nold intro\n")
    linked = note(vault, "Notes/work/todo.md", "- [ ] first\n")
    other = note(vault, "Notes/work/other.md", "untouched\n")
    for path in (linked, other):
        open_source(con, vault, path)
    source = open_source(con, vault, open_path)
    seen = {}
    fake_model(monkeypatch, (
        "Tightened the intro and added a task.\n"
        "FILE: Notes/work/plan.md\n<<<<<<< SEARCH\nold intro\n=======\nnew intro\n>>>>>>> REPLACE\n"
        "FILE: Notes/work/todo.md\n<<<<<<< SEARCH\n=======\n- [ ] second\n>>>>>>> REPLACE\n"
        "FILE: Notes/work/other.md\n<<<<<<< SEARCH\nuntouched\n=======\nx\n>>>>>>> REPLACE\n"), seen)
    turn = copilot.run_turn(con, source["id"], "tighten this and add a task to [[todo]]", [],
                            "quick", edit=True, vault=vault)
    files = {item["path"]: item for item in turn["proposal"]["files"]}
    assert files[open_path]["after"] == "# Plan\n\nnew intro\n"
    assert files[linked]["after"] == "- [ ] first\n\n- [ ] second\n"
    assert turn["proposal"]["dropped"] == [
        {"path": other, "reason": "not part of this request"}]
    assert "=== Notes/work/plan.md · the open note ===\n# Plan" in seen["messages"][0]["content"]
    assert (vault / open_path).read_text() == "# Plan\n\nold intro\n"


def test_ask_mode_keeps_retrieval_and_says_edit_mode_exists(con, vault, monkeypatch):
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    seen = {}
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot.llm, "chat_stream",
                        lambda messages, **k: seen.update(system=messages[0]["content"]) or ("ok", {}))
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    turn = copilot.run_turn(con, source["id"], "what is /usr/bin for?", [], "quick", vault=vault)
    assert copilot.ASK_RIDER in seen["system"] and "SEARCH" not in seen["system"]
    assert turn.get("proposal") is None and turn["skill"] is None


def test_an_unknown_command_is_refused_not_asked(con, vault, monkeypatch):
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    fake_model(monkeypatch, "never", {})
    with pytest.raises(copilot.CopilotError, match="no /quizz skill"):
        copilot.run_turn(con, source["id"], "/quizz me", [], "quick", vault=vault)


def test_a_skill_runs_at_its_own_depth_and_its_prompt_rides_in_history(con, vault, monkeypatch):
    meeting(vault, "Kickoff", "2026-09-01T09:00:00", "ship friday")
    source = open_source(con, vault, meeting(vault, "Review", "2026-09-08T09:00:00", "monday"))
    seen = {}
    fake_model(monkeypatch, "FILE: Capture/work/All.md\n<<<<<<< SEARCH\n=======\nx\n>>>>>>> REPLACE\n", seen)
    turn = copilot.run_turn(con, source["id"], "/consolidate-my-notes", [], "quick", vault=vault)
    assert turn["mode"] == "deep"
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    copilot.run_turn(con, source["id"], "my answer is 4", [
        {"role": "you", "text": "/quiz"}, {"role": "slim", "text": "What is 2+2?"}], "quick", vault=vault)
    assert seen["messages"][1]["content"].startswith("Quiz me on this note. Ask ONE question now")


def test_an_overflowing_prompt_is_never_offered_as_a_proposal(con, vault, monkeypatch):
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    monkeypatch.setattr(copilot.llm, "chat_stream", lambda messages, **k: (
        "FILE: Notes/work/a.md\n<<<<<<< SEARCH\ntext\n=======\nnew\n>>>>>>> REPLACE\n",
        {"ctx_saturated": True}))
    with pytest.raises(copilot.CopilotError, match="too long for one request"):
        copilot.run_turn(con, source["id"], "fix it", [], "quick", edit=True, vault=vault)


def test_section_of_skips_code_fences_and_the_derived_summary():
    text = ("<!-- slim:summary -->\n## Notes\nmodel opinion\n<!-- /slim:summary -->\n"
            "## Notes\n\nsetup:\n```bash\n# install deps\n## Notes\n```\nafter\n## Transcript\nMe: x\n")
    assert copilot.section_of(text, "Notes") == "setup:\n```bash\n# install deps\n## Notes\n```\nafter"


def test_a_partial_link_path_ends_at_a_folder_boundary(con, vault):
    open_path = note(vault, "Notes/work/plan.md", "the plan\n")
    note(vault, "Notes/homework/budget.md", "wrong\n")
    right = note(vault, "Notes/work/sub/budget.md", "right\n")
    for path in ("Notes/homework/budget.md", right):
        open_source(con, vault, path)
    assert copilot.resolve_link(con, vault, "work/sub/budget", open_path) == right
    assert copilot.resolve_link(con, vault, "ork/sub/budget", open_path) is None


def test_a_broken_vault_skill_says_which_file_to_fix(con, vault, monkeypatch):
    (vault / "Skills").mkdir()
    (vault / "Skills/bad.md").write_text("---\ninput: everything\n---\nbody\n")
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    fake_model(monkeypatch, "never", {})
    with pytest.raises(copilot.CopilotError, match=r"/bad cannot run: input must be .*Skills/bad.md"):
        copilot.run_turn(con, source["id"], "/bad", [], "quick", vault=vault)


def test_ask_mode_never_sends_the_selection(con, vault, monkeypatch):
    """A selection left in the note rode along with every Ask question, unseen (review 2026-09-23)."""
    source = open_source(con, vault, note(vault, "Notes/work/a.md", "text\n"))
    seen = {}
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot.llm, "chat_stream",
                        lambda messages, **k: seen.update(user=messages[-1]["content"]) or ("ok", {}))
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    copilot.run_turn(con, source["id"], "what does this mean?", [], "quick",
                     selection="leftover", vault=vault)
    assert seen["user"] == "what does this mean?"
