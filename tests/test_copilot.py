"""Open-note copilot: one sidebar turn — the Ask path's messages and window, the sources
call, traces, and the open note's images."""

from pathlib import Path

import pytest

from slim import server, db
from slim.copilot import turn as copilot, context


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


def test_quick_and_deep_use_the_same_resident_with_distinct_thinking(monkeypatch):
    calls = []

    def fake_stream(messages, **kwargs):
        calls.append(kwargs)
        kwargs["on_delta"]("Answer [1]")
        return "Answer [1]", {"model": kwargs["model"], "duration_s": 1}

    monkeypatch.setattr(copilot.llm, "chat_stream", fake_stream)
    evidence = [{"n": 1, "source_id": "s", "path": "Notes/x.md", "title": "x",
                 "heading": None, "text": "Evidence"}]
    quick = copilot.generate_answer("Question", [], evidence, "quick")
    deep = copilot.generate_answer("Question", [], evidence, "deep")
    assert quick["mode"] == "quick"
    assert deep["mode"] == "deep"
    assert calls[0]["model"] == calls[1]["model"] == copilot.llm.MODEL
    assert calls[0]["num_ctx"] == calls[1]["num_ctx"] == copilot.llm.RESIDENT_CTX
    assert calls[0]["think"] is False
    assert calls[1]["think"] is True
    # Thinking tokens bill against num_predict (llm.chat's docstring): deep must budget for
    # reasoning AND the answer, the way summarize's THINKING_PREDICT_MULTIPLIER does.
    assert calls[1]["num_predict"] == copilot.DEEP_NUM_PREDICT
    assert copilot.DEEP_NUM_PREDICT >= 3 * calls[0]["num_predict"]


def test_auto_mode_is_refused(con, vault, monkeypatch):
    """Two depths, chosen by them (2026-09-03). Auto spent a whole extra model call deciding
    something they can say in one click, and resolved to Quick nearly every time — so the cost
    was real and the choice was not. Anything but quick/deep is refused before any work."""
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *a, **k: [])
    with pytest.raises(copilot.CopilotError, match="must be quick or deep"):
        copilot.run_turn(con, source["id"], "q", [], "auto")


def test_sources_come_from_a_bounded_follow_up_on_the_same_prefix(monkeypatch):
    # No markers in the prose and no trailer contract: the pack has no numbers to cite. After
    # the answer, one temperature-0 JSON call on the same messages names the notes used.
    calls = []
    monkeypatch.setattr(copilot.llm, "chat_stream",
                        lambda messages, **kwargs: ("Use arr[0] here; see [2019].", {"duration_s": 1}))

    def fake_json(messages, schema, **kwargs):
        calls.append((messages, schema, kwargs))
        return {"notes": [2]}, {"duration_s": 0.4}

    monkeypatch.setattr(copilot.llm, "chat_json", fake_json)
    evidence = [
        {"n": 1, "source_id": "a", "path": "Notes/a.md", "title": "Alpha", "heading": "One", "text": "E"},
        {"n": 2, "source_id": "a", "path": "Notes/a.md", "title": "Alpha", "heading": "Two", "text": "F"},
        {"n": 3, "source_id": "b", "path": "Notes/b.md", "title": "Beta", "heading": None, "text": "G"},
    ]
    got = copilot.generate_answer("Question", [], evidence, "quick")
    assert got["answer"] == "Use arr[0] here; see [2019]."          # never rewritten
    assert [item["path"] for item in got["citations"]] == ["Notes/b.md"]   # choice 2 = the second NOTE
    (messages, schema, kwargs), = calls
    assert messages[-2] == {"role": "assistant", "content": "Use arr[0] here; see [2019]."}
    assert messages[-1]["content"].endswith("1. Alpha\n2. Beta")
    assert messages[0]["role"] == "system" and "### Alpha § One" in messages[0]["content"]
    assert schema["properties"]["notes"]["items"]["maximum"] == 2
    assert (kwargs["temperature"], kwargs["think"]) == (0.0, False)


def test_a_failed_sources_call_leaves_the_answer_standing(monkeypatch):
    monkeypatch.setattr(copilot.llm, "chat_stream",
                        lambda messages, **kwargs: ("Answer", {"duration_s": 1}))
    monkeypatch.setattr(copilot.llm, "chat_json",
                        lambda *args, **kwargs: (_ for _ in ()).throw(copilot.llm.LLMError("grammar")))
    evidence = [{"n": 1, "source_id": "a", "path": "Notes/a.md", "title": "Alpha", "heading": None, "text": "E"}]
    got = copilot.generate_answer("Question", [], evidence, "quick")
    assert got["answer"] == "Answer" and got["citations"] == []
    # No pack, no call at all.
    calls = []
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: calls.append(1) or ({"notes": []}, {}))
    assert copilot.generate_answer("Question", [], [], "quick")["citations"] == []
    assert calls == []


def test_persona_and_notes_ride_in_the_system_message(monkeypatch):
    sent = {}

    def fake_stream(messages, **kwargs):
        sent["messages"] = messages
        return "Sure.", {"duration_s": 1}

    monkeypatch.setattr(copilot.llm, "chat_stream", fake_stream)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    evidence = [{"n": 1, "source_id": "s", "path": "Notes/school/CS203/Reading Notes.md",
                 "title": "Reading Notes", "heading": "Learning", "text": "Search vs learning."}]
    history = [{"role": "you", "text": "hi"}, {"role": "slim", "text": "hello"}]
    got = copilot.generate_answer("Explain this equation.", history, evidence, "quick")
    system, *rest = sent["messages"]
    assert system["role"] == "system"
    assert system["content"].startswith(copilot.PERSONA)
    assert "About the user" not in system["content"]        # no profile passed, none injected
    assert "### Reading Notes § Learning\nSearch vs learning." in system["content"]
    assert "[1]" not in system["content"]                  # nothing numbered to cite
    assert "Notes/school" not in system["content"]          # titles, never paths
    assert "EVIDENCE" not in system["content"]
    # Their turn is their words alone; the pack no longer rides inside it.
    assert rest[-1] == {"role": "user", "content": "Explain this equation."}
    assert [m["role"] for m in rest] == ["user", "assistant", "user"]
    assert got["answer"] == "Sure." and got["citations"] == []


def _capture(monkeypatch):
    sent = {}

    def fake_stream(messages, **kwargs):
        sent["messages"], sent["kwargs"] = messages, kwargs
        return "Done.", {"duration_s": 1}

    monkeypatch.setattr(copilot.llm, "chat_stream", fake_stream)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    return sent


def test_a_long_answer_reaches_the_model_whole_through_the_server(monkeypatch):
    # A 3,123-character answer whose plan began at character 2,424: a cut at 1,000 sent its
    # opening and dropped the plan, so "Ok make the revisions" made a narrow edit.
    answer = ("Here is what the note covers. " * 81)[:2422] + "\n\n## Proposed Structure\n"
    answer += "x" * (3123 - len(answer))
    assert len(answer) == 3123 and answer.index("## Proposed Structure") == 2424
    raw = [{"role": "you", "text": "How could this note be organised?"},
           {"role": "slim", "text": answer}]
    sent = _capture(monkeypatch)
    got = copilot.generate_answer("Ok make the revisions", server.clean_history(raw), [], "quick")
    assert sent["messages"][2] == {"role": "assistant", "content": answer}
    assert got["history_dropped"] == 0


def test_every_turn_is_sent_when_the_conversation_fits(monkeypatch):
    history = [{"role": "slim" if i % 2 else "you", "text": f"turn {i}"} for i in range(30)]
    sent = _capture(monkeypatch)
    copilot.generate_answer("Next", server.clean_history(history), [], "quick")
    assert [m["content"] for m in sent["messages"][1:-1]] == [f"turn {i}" for i in range(30)]
    assert sent["kwargs"]["num_ctx"] == copilot.llm.RESIDENT_CTX


def test_a_conversation_too_long_for_the_window_loses_its_oldest_turns_whole(monkeypatch):
    big = copilot.llm.RESIDENT_CTX * copilot.CHARS_PER_TOKEN // 5
    history = [{"role": "slim" if i % 2 else "you", "text": f"{i}" + "w" * big} for i in range(8)]
    sent = _capture(monkeypatch)
    got = copilot.generate_answer("Next", history, [], "deep")
    kept = sent["messages"][1:-1]
    assert 0 < len(kept) < len(history)
    # The newest kept, each whole, in order; the window still holds the reply's room.
    assert [m["content"] for m in kept] == [turn["text"] for turn in history[-len(kept):]]
    assert got["history_dropped"] == len(history) - len(kept)
    chars = sum(len(m["content"]) for m in sent["messages"])
    assert chars / copilot.CHARS_PER_TOKEN + copilot.DEEP_NUM_PREDICT <= copilot.llm.RESIDENT_CTX
    assert sent["kwargs"]["num_ctx"] == copilot.llm.RESIDENT_CTX


def test_images_count_against_the_window(monkeypatch):
    room = copilot.llm.RESIDENT_CTX - copilot.QUICK_NUM_PREDICT
    text = "w" * (room - 2 * copilot.IMAGE_TOKENS) * copilot.CHARS_PER_TOKEN
    history = [{"role": "you", "text": "old"}, {"role": "slim", "text": text}]
    sent = _capture(monkeypatch)
    got = copilot.generate_answer("", history, [], "quick", images=["A", "B", "C"])
    assert got["history_dropped"] == 2
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_thinking_that_eats_the_budget_is_a_clear_refusal(monkeypatch):
    monkeypatch.setattr(copilot.llm, "chat_stream", lambda messages, **kwargs: (
        "", {"thinking_chars": 9000, "thinking_ate_the_budget": True, "output_truncated": True}))
    with pytest.raises(copilot.CopilotError, match="thinking"):
        copilot.generate_answer("Question", [], [], "deep")


def test_truncated_answer_is_reported_not_passed_off_as_complete(monkeypatch):
    monkeypatch.setattr(copilot.llm, "chat_stream", lambda messages, **kwargs: (
        "Half an answer", {"done_reason": "length", "output_truncated": True}))
    got = copilot.generate_answer("Question", [], [], "quick")
    assert got["verification"]["truncated"] is True


def test_deep_thinking_heartbeats_the_stage_so_a_stopped_client_is_noticed(monkeypatch):
    # No bytes reach the client while the model reasons, so the plugin's Stop (a destroyed
    # socket) is invisible until content starts. A throttled stage heartbeat is the write
    # that raises BrokenPipe inside the Ollama request and frees _CALL_LOCK.
    def fake_stream(messages, **kwargs):
        for _ in range(3):
            kwargs["on_thinking"](40)
        return "Answer", {"duration_s": 1}

    monkeypatch.setattr(copilot.llm, "chat_stream", fake_stream)
    monkeypatch.setattr(copilot, "HEARTBEAT_S", 0.0)
    stages = []
    copilot.generate_answer("Question", [], [], "deep", on_stage=stages.append)
    assert stages == ["Thinking deeply"] * 4


def answering(monkeypatch, content="Answer", stats=None, seen=None):
    """A model whose gate says no change is wanted, so every turn is an Ask turn; `seen` keeps the
    messages of its first answering call."""
    def chat_tools(messages, tools=None, **opts):
        assert opts.get("fmt") is not None, "only the gate may call with tools here"
        return {"content": '{"change_note": false}', "tool_calls": [], "raw_tool_calls": []}, {}

    def chat_stream(messages, **opts):
        if seen is not None:
            seen.setdefault("messages", list(messages))
        if opts.get("on_delta") and content:
            opts["on_delta"](content)
        return content, dict(stats or {"duration_s": 1})
    monkeypatch.setattr(copilot.llm, "chat_tools", chat_tools)
    monkeypatch.setattr(copilot.llm, "chat_stream", chat_stream)


def test_run_turn_writes_a_replayable_trace(con, vault, monkeypatch):
    # CLAUDE.md: every query writes a replayable trace. The thread JSON keeps only turns
    # that COMPLETED; a trace is the one record of a failed or stopped turn.
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *args, **kwargs: [])
    answering(monkeypatch, stats={"duration_s": 1, "output_tokens": 5})
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": [1]}, {"duration_s": 0.3}))
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append((kind, payload)))
    copilot.run_turn(con, source["id"], "What is this?", [{"role": "you", "text": "hi"}], "quick",
                     vault=vault)
    (kind, payload), = records
    assert kind == "copilot"
    assert payload["source"] == "Notes/school/lesson.md"
    assert payload["question"] == "What is this?"
    assert payload["mode"] == "quick"
    assert payload["evidence"] == ["Notes/school/lesson.md"]
    assert payload["retrieval_levels"] == ["note"]
    assert payload["cited"] == ["Notes/school/lesson.md"]
    assert payload["history_turns"] == 1
    assert payload["history_dropped"] == 0
    assert payload["timing"]["answer"]["output_tokens"] == 5 and payload["status"] == "answered"
    assert "answer" not in payload and "edit_run" not in payload   # machine facts only; the thread holds the prose


def test_the_trace_counts_the_turns_the_window_dropped(con, vault, monkeypatch):
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *args, **kwargs: [])
    answering(monkeypatch)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append(payload))
    window = "w" * (copilot.llm.RESIDENT_CTX * copilot.CHARS_PER_TOKEN)
    history = [{"role": "you", "text": "a" + window}, {"role": "slim", "text": "b" + window},
               {"role": "you", "text": "short"}, {"role": "slim", "text": "reply"}]
    got = copilot.run_turn(con, source["id"], "What is this?", history, "quick", vault=vault)
    assert got["history_dropped"] == 2
    assert records[0]["history_turns"] == 4 and records[0]["history_dropped"] == 2


def test_run_turn_traces_a_refused_turn(con, vault, monkeypatch):
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *args, **kwargs: [])
    answering(monkeypatch, content="")
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append((kind, payload)))
    with pytest.raises(copilot.CopilotError):
        copilot.run_turn(con, source["id"], "What is this?", [], "quick", vault=vault)
    (kind, payload), = records
    assert kind == "copilot" and payload["status"] == "failed"
    assert "no answer" in payload["error"]


# --- M4: any failure leaves a trace record ---------------------------------------------------------

def test_a_turn_that_fails_in_a_callback_is_traced_with_the_error_type(con, vault, monkeypatch):
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *a, **k: [])
    answering(monkeypatch)
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append((kind, payload)))

    def gone(_stage):
        raise BrokenPipeError(32, "Broken pipe")
    with pytest.raises(BrokenPipeError):
        copilot.run_turn(con, source["id"], "What is this?", [], "quick", vault=vault, on_stage=gone)
    (kind, payload), = records
    assert kind == "copilot" and payload["status"] == "failed" and payload["error_type"] == "BrokenPipeError"
    assert "duration_ms" in payload and payload["intent"] is False


# --- nothing rides in the system prompt but the persona and the notes ----------------------

def _answering(monkeypatch, seen: dict):
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *a, **k: [])
    answering(monkeypatch, seen=seen)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))


def test_the_profile_is_never_injected_even_when_the_file_is_there(con, vault, monkeypatch):
    """⚠ Measured 2026-09-01, three live samples over two wordings of the clause: with
    `Profile/me.md` in context qwen3.6:35b pandered to the profile every time, working a
    career-pivot flourish into an EQUATION explanation. The owner's call, 2026-09-03: it goes."""
    (vault / "Profile").mkdir()
    (vault / "Profile/me.md").write_text(
        "---\ntitle: me\n---\n\nI am Alex. Terse answers.\n")
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    seen = {}
    _answering(monkeypatch, seen)
    copilot.run_turn(con, source["id"], "q", [], "quick", vault=vault)
    system = seen["messages"][0]["content"]
    assert system.startswith(copilot.PERSONA)
    assert "I am Alex" not in system
    assert "About the user" not in system


def _png(root: Path, rel: str) -> None:
    from PIL import Image
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color="red").save(str(target))


def test_the_model_sees_the_images_beside_the_passages_it_reads(con, vault, monkeypatch):
    """⚠ Until 2026-09-22 the copilot saw only images pasted into the sidebar; 66 of 104 course
    notes embed their equations as screenshots, and it rewrote Ψ(x, y) = θ·f(x, y) in its own
    symbols. The open note's images ride with its passages; a sibling's never do."""
    write(vault, "Notes/school/lesson.md",
          "Chain rule:\n![[chain.png|300]]\n\nBayes: ![fig](figs/bayes%20rule.png)")
    _png(vault, "Notes/school/Attachements/chain.png")
    _png(vault, "Notes/school/figs/bayes rule.png")
    _png(vault, "Notes/school/other.png")
    source = context.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    sibling = {"n": 9, "source_id": "sib", "path": "Notes/school/b.md", "title": "b",
               "heading": None, "text": "![[other.png]]"}
    real_collect = context.collect_evidence
    monkeypatch.setattr(context, "collect_evidence", lambda *a, **k: (
        lambda got: ([*got[0], sibling], got[1]))(real_collect(*a, **k)))
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *args, **kwargs: [])
    sent = {}
    answering(monkeypatch, seen=sent)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append(payload))

    copilot.run_turn(con, source["id"], "What does the chain rule say?", [], "quick", vault=vault)

    *_, shown, asked = sent["messages"]
    assert asked == {"role": "user", "content": "What does the chain rule say?"}
    assert shown["role"] == "user" and len(shown["images"]) == 2
    assert "`chain.png`" in shown["content"] and "`figs/bayes rule.png`" in shown["content"]
    assert "other.png" not in shown["content"]
    assert records[-1]["note_images"] == 2


def test_a_screenshot_heavy_note_sends_only_the_capped_few(con, vault, monkeypatch):
    # ~450 prompt tokens and ~1.3 s of prefill per screenshot (measured 2026-09-22); a lecture
    # note embeds up to 60.
    body = "\n".join(f"Slide {i}\n![[slide-{i:02d}.png]]" for i in range(20))
    write(vault, "Notes/school/lecture.md", body)
    for i in range(20):
        _png(vault, f"Notes/school/Attachements/slide-{i:02d}.png")
    source = context.sync_note(con, vault, "Notes/school/lecture.md", embed=False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *args, **kwargs: [])
    sent = {}
    answering(monkeypatch, seen=sent)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    copilot.run_turn(con, source["id"], "Quiz me", [], "quick", vault=vault)
    shown = sent["messages"][-2]
    assert len(shown["images"]) == context.MAX_NOTE_IMAGES
    assert "`slide-00.png`" in shown["content"]


def test_a_note_without_images_adds_no_message(con, vault, monkeypatch):
    write(vault, "Notes/school/plain.md", "just words")
    source = context.sync_note(con, vault, "Notes/school/plain.md", embed=False)
    monkeypatch.setattr(context.search_mod, "hybrid", lambda *args, **kwargs: [])
    sent = {}
    answering(monkeypatch, seen=sent)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    copilot.run_turn(con, source["id"], "Hm?", [], "quick", vault=vault)
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_an_image_whose_passage_retrieval_missed_still_rides_after_the_packs(con, vault, monkeypatch):
    """⚠ Measured 2026-09-22 on the real note: its Ψ(x, y) equation exists ONLY as a screenshot,
    so no passage text matches a question about it and the pack skipped that passage. What an
    image says is not searchable; the pack can order the images, never exclude them."""
    write(vault, "Notes/school/ch2.md",
          "## Scoring\n![[psi.png]] for each label\n\n## Naive Bayes\nMLE ![[mle.png]]")
    _png(vault, "Notes/school/psi.png")
    _png(vault, "Notes/school/mle.png")
    source = context.sync_note(con, vault, "Notes/school/ch2.md", embed=False)
    evidence = [{"n": 1, "source_id": source["id"], "path": source["path"], "title": "ch2",
                 "heading": "Naive Bayes", "text": "MLE ![[mle.png]]"}]
    shown, _ = context.open_note_images(vault, source, evidence)
    assert [name for name, _ in shown] == ["mle.png", "psi.png"]
