"""Open-note copilot: one sidebar turn — its window, its trace, and the open note's images. The
loop itself (lookups, edits, sources) is `test_loop.py`'s."""

from pathlib import Path

import pytest

from slim import server, db
from slim.copilot import context, run as edit_run, turn as copilot_turn

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


def lesson(con, vault) -> str:
    write(vault, "Notes/school/lesson.md", "gradient descent")
    return context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)["path"]


@pytest.mark.parametrize("effort", ["auto", "quick", "deep", ""])
def test_an_effort_that_is_not_a_level_is_refused(con, vault, monkeypatch, effort):
    """Pi's four levels, chosen by them. Auto (2026-09-03) spent a whole extra model call deciding
    something they can say in one click; Quick and Deep are read as levels where they arrive (the
    server, a skill's frontmatter), never here. Anything else is refused before any work."""
    seen = scripted(monkeypatch)
    with pytest.raises(copilot_turn.CopilotError, match="effort must be one of off, low, medium, high"):
        copilot_turn.run_turn(con, lesson(con, vault), "q", [], effort, vault=vault)
    assert seen == []


# --- THE WINDOW RULE: the conversation goes whole, its oldest turns first to go ----------------

def test_a_long_answer_reaches_the_model_whole_through_the_server(con, vault, monkeypatch):
    # A 3,123-character answer whose plan began at character 2,424: a cut at 1,000 sent its
    # opening and dropped the plan, so "Ok make the revisions" made a narrow edit.
    answer = ("Here is what the note covers. " * 81)[:2422] + "\n\n## Proposed Structure\n"
    answer += "x" * (3123 - len(answer))
    assert len(answer) == 3123 and answer.index("## Proposed Structure") == 2424
    raw = [{"role": "you", "text": "How could this note be organised?"},
           {"role": "slim", "text": answer}]
    seen = scripted(monkeypatch, ("Done.", []))
    got = copilot_turn.run_turn(con, None, "Ok make the revisions", server.clean_history(raw), "medium",
                                vault=vault)
    assert seen[0]["messages"][2] == {"role": "assistant", "content": answer}
    assert got["history_dropped"] == 0


def test_the_server_keeps_an_answers_transcript_and_drops_a_malformed_one(con, vault, monkeypatch):
    script = [{"role": "user", "content": "q1 as sent"},
              {"role": "assistant", "content": "", "thinking": "look it up",
               "tool_calls": [{"function": {"name": "search", "arguments": {"query": "q"}}}], "extra": 1},
              {"role": "tool", "content": "found"}, {"role": "assistant", "content": "A1"}]
    raw = [{"role": "you", "text": "q1"}, {"role": "slim", "text": "A1", "transcript": script},
           {"role": "you", "text": "q2"}, {"role": "slim", "text": "A2", "transcript": [{"role": "system", "content": "x"}]}]
    history = server.clean_history(raw)
    # An earlier turn's thinking is dropped: the template shows it only after the newest question.
    assert history[1]["transcript"][1] == {"role": "assistant", "content": "",
                                           "tool_calls": [{"function": {"name": "search", "arguments": {"query": "q"}}}]}
    assert "transcript" not in history[3]
    seen = scripted(monkeypatch, ("Done.", []))
    copilot_turn.run_turn(con, None, "q3", history, "medium", vault=vault)
    assert [m["content"] for m in seen[0]["messages"][1:-1]] == ["q1 as sent", "", "found", "A1", "q2", "A2"]


def test_every_turn_is_sent_when_the_conversation_fits(con, vault, monkeypatch):
    history = [{"role": "slim" if i % 2 else "you", "text": f"turn {i}"} for i in range(30)]
    seen = scripted(monkeypatch, ("Done.", []))
    copilot_turn.run_turn(con, None, "Next", server.clean_history(history), "medium", vault=vault)
    assert [m["content"] for m in seen[0]["messages"][1:-1]] == [f"turn {i}" for i in range(30)]


def test_a_conversation_too_long_for_the_window_loses_its_oldest_turns_whole(con, vault, monkeypatch):
    big = copilot_turn.llm.RESIDENT_CTX * copilot_turn.CHARS_PER_TOKEN // 5
    history = [{"role": "slim" if i % 2 else "you", "text": f"{i}" + "w" * big} for i in range(8)]
    seen = scripted(monkeypatch, ("Done.", []))
    got = copilot_turn.run_turn(con, None, "Next", history, "high", vault=vault)
    kept = seen[0]["messages"][1:-1]
    assert 0 < len(kept) < len(history)
    # The newest kept, each whole, in order; the window still holds the reply's and the thinking's room.
    assert [m["content"] for m in kept] == [turn["text"] for turn in history[-len(kept):]]
    assert got["history_dropped"] == len(history) - len(kept)
    chars = sum(len(m["content"]) for m in seen[0]["messages"])
    room = copilot_turn.ANSWER_NUM_PREDICT + edit_run.THINK_BUDGETS["high"] + copilot_turn.TURN_RESERVE
    assert chars / copilot_turn.CHARS_PER_TOKEN + room <= copilot_turn.llm.RESIDENT_CTX


def test_an_old_turns_copy_of_the_open_note_is_named_not_replayed(con, vault, monkeypatch):
    """A chat saved before the store dropped them: the old copy, with its old handles, goes."""
    sent = "=== Notes/a.md · the open note ===\n[b1] # A\n[b2] Old text.\n\n---\n\nshorten it"
    history = [{"role": "you", "text": "(on Notes/a.md) shorten it"},
               {"role": "slim", "text": "Done.", "transcript": [{"role": "user", "content": sent},
                                                                {"role": "assistant", "content": "Done."}]}]
    seen = scripted(monkeypatch, ("Ok.", []))
    copilot_turn.run_turn(con, None, "next", server.clean_history(history), "medium", vault=vault)
    replayed = seen[0]["messages"][1]["content"]
    assert replayed == "[Shown with this question then, not repeated here: Notes/a.md]\n\n---\n\nshorten it"


def test_what_became_of_an_earlier_edit_follows_its_replay(con, vault, monkeypatch):
    script = [{"role": "user", "content": "shorten it"},
              {"role": "assistant", "content": "", "tool_calls": [
                  {"function": {"name": "edit_text", "arguments": {"target": "b2", "old": "x", "new": "y"}}}]},
              {"role": "tool", "content": "ok: changed b2"}]
    said = "[Changed the note: Shorter.; 1 change, 1 undone]\nedited: Notes/a.md (1 change)"
    history = [{"role": "you", "text": "shorten it"}, {"role": "slim", "text": said, "transcript": script}]
    seen = scripted(monkeypatch, ("Ok.", []))
    copilot_turn.run_turn(con, None, "next", server.clean_history(history), "medium", vault=vault)
    messages = seen[0]["messages"]
    assert messages[4] == {"role": "user", "content":
                           "(What became of that edit since: [Changed the note: Shorter.; 1 change, 1 undone])"}


def test_images_count_against_the_window(con, vault, monkeypatch):
    room = copilot_turn.ANSWER_NUM_PREDICT + edit_run.THINK_BUDGET + copilot_turn.TURN_RESERVE
    left = (copilot_turn.llm.RESIDENT_CTX - room - copilot_turn._tokens({"content": copilot_turn.loop_system()})
            - copilot_turn._tokens({"content": "look"}) - 10)
    text = "w" * int((left - 2 * copilot_turn.IMAGE_TOKENS) * copilot_turn.CHARS_PER_TOKEN)
    history = [{"role": "you", "text": "old"}, {"role": "slim", "text": text}]
    scripted(monkeypatch, ("Done.", []), ("Done.", []))
    assert copilot_turn.run_turn(con, None, "look", history, "medium", vault=vault)["history_dropped"] == 0
    got = copilot_turn.run_turn(con, None, "look", history, "medium", vault=vault, images=["A", "B", "C"])
    assert got["history_dropped"] == 2


def test_images_ride_on_user_messages_only(con, vault, monkeypatch):
    seen = scripted(monkeypatch, ("Answer", []))
    copilot_turn.run_turn(
        con, None, "Explain", [{"role": "you", "text": "Earlier", "images": ["old"]},
                               {"role": "slim", "text": "Earlier answer", "images": ["never"]}],
        "medium", vault=vault, images=["current"])
    messages = seen[0]["messages"]
    assert messages[1]["images"] == ["old"]
    assert "images" not in messages[2]
    assert messages[-1]["images"] == ["current"]


def _replying(monkeypatch, content: str, stats: dict):
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", lambda messages, tools=None, **opts: (
        {"content": content, "tool_calls": [], "raw_tool_calls": []}, stats))


def test_thinking_that_eats_the_budget_is_a_clear_refusal(con, vault, monkeypatch):
    _replying(monkeypatch, "", {"thinking_chars": 9000, "thinking_ate_the_budget": True, "output_truncated": True})
    with pytest.raises(copilot_turn.CopilotError, match="thinking"):
        copilot_turn.run_turn(con, None, "Question", [], "high", vault=vault)


def test_truncated_answer_is_reported_not_passed_off_as_complete(con, vault, monkeypatch):
    _replying(monkeypatch, "Half an answer", {"done_reason": "length", "output_truncated": True})
    got = copilot_turn.run_turn(con, None, "Question", [], "medium", vault=vault)
    assert got["verification"]["truncated"] is True


# --- every turn leaves a trace: machine facts only -------------------------------------------

def test_run_turn_writes_a_replayable_trace(con, vault, monkeypatch):
    # The thread JSON keeps only turns that COMPLETED; a trace is the one record of a failed or
    # stopped turn.
    path = lesson(con, vault)
    scripted(monkeypatch, ("An answer.", [], 5))
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append((kind, payload)))
    copilot_turn.run_turn(con, path, "What is this?", [{"role": "you", "text": "hi"}], "medium", vault=vault)
    (kind, payload), = records
    assert kind == "copilot"
    assert payload["source"] == path and payload["open_path"] == path
    assert payload["question"] == "What is this?"
    assert payload["effort"] == "medium" and payload["claimed_edit"] is None
    assert payload["cited"] == [] and payload["looked_up"] == []
    assert payload["history_turns"] == 1
    assert payload["history_dropped"] == 0
    assert payload["timing"]["answer"]["output_tokens"] == 5 and payload["status"] == "answered"
    assert "answer" not in payload and "edit_run" not in payload   # machine facts only; the thread holds the prose


def test_the_trace_counts_the_turns_the_window_dropped(con, vault, monkeypatch):
    path = lesson(con, vault)
    scripted(monkeypatch, ("Answer", []))
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))
    window = "w" * (copilot_turn.llm.RESIDENT_CTX * copilot_turn.CHARS_PER_TOKEN)
    history = [{"role": "you", "text": "a" + window}, {"role": "slim", "text": "b" + window},
               {"role": "you", "text": "short"}, {"role": "slim", "text": "reply"}]
    got = copilot_turn.run_turn(con, path, "What is this?", history, "medium", vault=vault)
    assert got["history_dropped"] == 2
    assert records[0]["history_turns"] == 4 and records[0]["history_dropped"] == 2


def test_run_turn_traces_a_refused_turn(con, vault, monkeypatch):
    path = lesson(con, vault)
    scripted(monkeypatch, ("", []))
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append((kind, payload)))
    with pytest.raises(copilot_turn.CopilotError):
        copilot_turn.run_turn(con, path, "What is this?", [], "medium", vault=vault)
    (kind, payload), = records
    assert kind == "copilot" and payload["status"] == "failed"
    assert "no answer" in payload["error"]


def test_a_turn_that_fails_in_a_callback_is_traced_with_the_error_type(con, vault, monkeypatch):
    path = lesson(con, vault)
    scripted(monkeypatch, ("Answer", []))
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append((kind, payload)))

    def gone(_stage):
        raise BrokenPipeError(32, "Broken pipe")
    with pytest.raises(BrokenPipeError):
        copilot_turn.run_turn(con, path, "What is this?", [], "medium", vault=vault, on_stage=gone)
    (kind, payload), = records
    assert kind == "copilot" and payload["status"] == "failed" and payload["error_type"] == "BrokenPipeError"
    assert "duration_ms" in payload


# --- no profile rides in the prompt -------------------------------------------------------------

def test_the_profile_is_never_injected_even_when_the_file_is_there(con, vault, monkeypatch):
    """⚠ Measured 2026-09-01, three live samples over two wordings of the clause: with
    `Profile/me.md` in context qwen3.6:35b pandered to the profile every time, working a
    career-pivot flourish into an EQUATION explanation. The owner's call, 2026-09-03: it goes.
    Pinning it with `@` is the owner's own act, not injection."""
    (vault / "Profile").mkdir()
    (vault / "Profile/me.md").write_text(
        "---\ntitle: me\n---\n\nI am Alex. Terse answers.\n")
    seen = scripted(monkeypatch, ("q", []))
    copilot_turn.run_turn(con, lesson(con, vault), "q", [], "medium", vault=vault)
    assert seen[0]["messages"][0]["content"] == copilot_turn.loop_system()
    assert "I am Alex" not in str(seen[0]["messages"])


# --- the open note's images ride with every turn --------------------------------------------------

def _png(root: Path, rel: str) -> None:
    from PIL import Image
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color="red").save(str(target))


def test_the_model_sees_the_open_notes_images_and_never_a_siblings(con, vault, monkeypatch):
    """⚠ Until 2026-09-22 the copilot saw only images pasted into the sidebar; 66 of 104 course
    notes embed their equations as screenshots, and it rewrote Ψ(x, y) = θ·f(x, y) in its own
    symbols. What an image says is not searchable, so the open note's ride with every turn, in
    note order; a sibling's never do."""
    write(vault, "Notes/school/lesson.md",
          "Chain rule:\n![[chain.png|300]]\n\nBayes: ![fig](figs/bayes%20rule.png)")
    write(vault, "Notes/school/b.md", "![[other.png]]")
    _png(vault, "Notes/school/Attachements/chain.png")
    _png(vault, "Notes/school/figs/bayes rule.png")
    _png(vault, "Notes/school/other.png")
    context.sync_note(con, vault, "Notes/school/b.md", embed=False)
    path = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)["path"]
    seen = scripted(monkeypatch, ("It multiplies.", []))
    records = []
    monkeypatch.setattr(copilot_turn.trace, "record", lambda kind, payload: records.append(payload))

    copilot_turn.run_turn(con, path, "What does the chain rule say?", [], "medium", vault=vault)

    *_, shown, asked = seen[0]["messages"]
    assert asked["content"].endswith("---\n\nWhat does the chain rule say?") and "images" not in asked
    assert shown["role"] == "user" and len(shown["images"]) == 2
    assert shown["content"].index("`chain.png`") < shown["content"].index("`figs/bayes rule.png`")
    assert "other.png" not in shown["content"]
    assert records[-1]["note_images"] == 2


def test_a_screenshot_heavy_note_sends_only_the_capped_few(con, vault, monkeypatch):
    # ~450 prompt tokens and ~1.3 s of prefill per screenshot (measured 2026-09-22); a lecture
    # note embeds up to 60.
    body = "\n".join(f"Slide {i}\n![[slide-{i:02d}.png]]" for i in range(20))
    write(vault, "Notes/school/lecture.md", body)
    for i in range(20):
        _png(vault, f"Notes/school/Attachements/slide-{i:02d}.png")
    path = context.sync_note(con, vault, "Notes/school/lecture.md", embed=False)["path"]
    seen = scripted(monkeypatch, ("Q1?", []))
    copilot_turn.run_turn(con, path, "Quiz me", [], "medium", vault=vault)
    shown = seen[0]["messages"][-2]
    assert len(shown["images"]) == context.MAX_NOTE_IMAGES
    assert "`slide-00.png`" in shown["content"]


def test_a_note_without_images_adds_no_message(con, vault, monkeypatch):
    write(vault, "Notes/school/plain.md", "just words")
    path = context.sync_note(con, vault, "Notes/school/plain.md", embed=False)["path"]
    seen = scripted(monkeypatch, ("Hm.", []))
    copilot_turn.run_turn(con, path, "Hm?", [], "medium", vault=vault)
    assert [m["role"] for m in seen[0]["messages"]] == ["system", "user"]
