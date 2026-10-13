"""Open-note copilot: targeted indexing, deterministic scope, and nearby notes."""

from pathlib import Path

import pytest

from slim import chat, copilot, db, threads


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


def test_sync_note_indexes_only_the_requested_file_and_preserves_identity(con, vault, monkeypatch):
    active = write(vault, "Notes/school/cs-201/week-01/lesson.md", "gradient descent")
    other = write(vault, "Notes/school/cs-202/graphs.md", "minimum spanning trees")
    monkeypatch.setattr(copilot.embed_mod, "embed_source", lambda _con, sid: {"embedded": 1})
    first = copilot.sync_note(con, vault, "Notes/school/cs-201/week-01/lesson.md")
    assert first["path"] == "Notes/school/cs-201/week-01/lesson.md"
    assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1
    active.write_text(active.read_text() + "\nNewton updates.\n", encoding="utf-8")
    second = copilot.sync_note(con, vault, "Notes/school/cs-201/week-01/lesson.md")
    assert second["id"] == first["id"]
    assert second["current_hash"] != first["current_hash"]
    assert other.exists()
    assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1


def test_sync_note_refuses_a_path_escape(con, vault):
    with pytest.raises(copilot.CopilotError, match="vault-relative"):
        copilot.sync_note(con, vault, "../outside.md")


def test_scope_levels_come_only_from_the_open_path(monkeypatch):
    monkeypatch.setattr(copilot.config_mod, "registry", lambda: {"school": {"vault_path": "Notes/school"}})
    assert copilot.scope_levels("Notes/school/cs-201/week-01/lesson.md") == [
        ("folder", ("Notes/school/cs-201/week-01",)),
        ("course", ("Notes/school/cs-201", "Capture/school/cs-201")),
        ("subject", ("Notes/school", "Capture/school")),
    ]
    assert copilot.scope_levels("Journal/2026-08-23.md") == [("folder", ("Journal",))]


def test_related_notes_widens_dedupes_and_drops_missing_paths(con, vault, monkeypatch):
    paths = [
        "Notes/school/cs-201/week-01/lesson.md",
        "Notes/school/cs-201/week-01/exercises.md",
        "Notes/school/cs-201/week-02/review.md",
        "Notes/school/cs-202/graphs.md",
    ]
    sources = {}
    for path in paths:
        write(vault, path, f"shared optimization material for {path}")
        sources[path] = copilot.sync_note(con, vault, path, embed=False)
    (vault / paths[1]).unlink()
    monkeypatch.setattr(copilot.config_mod, "registry", lambda: {"school": {"vault_path": "Notes/school"}})

    class Hit:
        def __init__(self, source_id, path, title):
            self.source_id, self.path, self.title = source_id, path, title

    # Ranked vault-wide, as a real hybrid() would: the subject-level hit outranks the
    # course-level one, yet the folder/course/subject ORDER must still decide the output.
    ranked = [Hit(sources[paths[3]]["id"], paths[3], "graphs"),
              Hit(sources[paths[0]]["id"], paths[0], "lesson"),
              Hit(sources[paths[1]]["id"], paths[1], "exercises"),
              Hit(sources[paths[2]]["id"], paths[2], "review")]
    calls = []

    def fake_hybrid(_con, query, *, k, prefixes, **_kw):
        calls.append(query)
        return [hit for hit in ranked
                if any(copilot._matches(hit.path, prefix) for prefix in prefixes)][:k]

    monkeypatch.setattr(copilot.search_mod, "hybrid", fake_hybrid)
    got = copilot.related_notes(con, vault, sources[paths[0]], limit=5)
    assert [(item["path"], item["scope"]) for item in got] == [
        (paths[2], "course"), (paths[3], "subject")]
    # One search over the widest scope, then labelled in code — not one full hybrid()
    # (an embedding call plus an FTS scan of the same 5,000-char query) per level.
    assert len(calls) == 1
    assert len(calls[0]) <= copilot.RELATED_QUERY_CHARS


def test_copilot_thread_round_trip_requires_a_source():
    with pytest.raises(threads.ThreadError, match="source id"):
        threads.save_copilot(
            "chat-1", "Question", [], source_id="", source_path="Note.md",
            reasoning_mode="quick")
    summary = threads.save_copilot(
        "chat-1", "Gradient descent", [{"role": "you", "text": "Explain it"}],
        source_id="source-1", source_path="Notes/school/cs-201/lesson.md",
        reasoning_mode="deep",
    )
    assert summary.source_id == "source-1"
    assert summary.reasoning_mode == "deep"
    loaded = threads.load("chat-1")
    assert loaded["source_path"] == "Notes/school/cs-201/lesson.md"


@pytest.mark.parametrize("mode", ["slow", "auto"])
def test_copilot_thread_rejects_unknown_modes(mode):
    """`auto` is among them since 2026-09-03 — a saved thread may only name a depth they can
    actually pick."""
    with pytest.raises(threads.ThreadError, match="reasoning mode"):
        threads.save_copilot(
            "chat-1", "Question", [], source_id="source-1", source_path="Note.md",
            reasoning_mode=mode)


def test_copilot_threads_are_listed_with_their_source():
    threads.save_copilot(
        "first", "First", [], source_id="source-1", source_path="Notes/one.md",
        reasoning_mode="quick")
    threads.save_copilot(
        "second", "Second", [], source_id="source-2", source_path="Notes/two.md",
        reasoning_mode="deep")
    got = threads.list_copilot_threads()
    assert {item.id for item in got} == {"first", "second"}
    assert all(item.source_id for item in got)


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
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    with pytest.raises(copilot.CopilotError, match="must be quick or deep"):
        copilot.run_turn(con, source["id"], "q", [], "auto")


def test_scope_levels_survive_a_case_only_folder_rename(monkeypatch):
    # APFS is case-insensitive: `notes/School/` IS `Notes/school/`. `_matches` already casefolds;
    # the relative-path step must not raise on the same input.
    monkeypatch.setattr(copilot.config_mod, "registry", lambda: {"school": {"vault_path": "Notes/school"}})
    assert copilot.scope_levels("notes/School/cs-201/week-01/lesson.md") == [
        ("folder", ("notes/School/cs-201/week-01",)),
        ("course", ("Notes/school/cs-201", "Capture/school/cs-201")),
        ("subject", ("Notes/school", "Capture/school")),
    ]


def test_collect_evidence_takes_the_open_note_before_any_sibling(con, monkeypatch):
    # PER_SOURCE_CAP (3) exists so no single note floods a VAULT-WIDE pack. The open note is
    # the subject of every copilot question, so it must never be capped below the pack size.
    class Hit:
        def __init__(self, fragment_id, source_id, path):
            self.fragment_id, self.source_id, self.path = fragment_id, source_id, path
            self.title, self.heading_path, self.text = path, None, f"text {fragment_id}"

    own = [Hit(f"own-{i}", "s1", "Notes/school/a.md") for i in range(12)]
    sibling = [Hit(f"sib-{i}", "s2", "Notes/school/b.md") for i in range(12)]

    def fake_hybrid(_con, _query, *, k, cap=copilot.search_mod.PER_SOURCE_CAP, prefixes):
        per_source, out = {}, []
        for hit in own + sibling:
            if not any(copilot._matches(hit.path, prefix) for prefix in prefixes):
                continue
            per_source[hit.source_id] = per_source.get(hit.source_id, 0) + 1
            if per_source[hit.source_id] <= cap:
                out.append(hit)
        return out[:k]

    monkeypatch.setattr(copilot.search_mod, "hybrid", fake_hybrid)
    monkeypatch.setattr(copilot, "scope_levels", lambda _path: [("folder", ("Notes/school",))])
    evidence, levels = copilot.collect_evidence(
        con, {"id": "s1", "path": "Notes/school/a.md"}, "summarize this note", limit=12)
    assert [item["source_id"] for item in evidence] == ["s1"] * 12
    assert levels == ["note"]


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
    got = copilot.generate_answer("Ok make the revisions", chat.clean_history(raw), [], "quick")
    assert sent["messages"][2] == {"role": "assistant", "content": answer}
    assert got["history_dropped"] == 0


def test_every_turn_is_sent_when_the_conversation_fits(monkeypatch):
    history = [{"role": "slim" if i % 2 else "you", "text": f"turn {i}"} for i in range(30)]
    sent = _capture(monkeypatch)
    copilot.generate_answer("Next", chat.clean_history(history), [], "quick")
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
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
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
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
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
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
    answering(monkeypatch, content="")
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append((kind, payload)))
    with pytest.raises(copilot.CopilotError):
        copilot.run_turn(con, source["id"], "What is this?", [], "quick", vault=vault)
    (kind, payload), = records
    assert kind == "copilot" and payload["status"] == "failed"
    assert "no answer" in payload["error"]


def test_context_refresh_without_nearby_notes_still_embeds(con, vault, monkeypatch):
    # The plugin re-syncs the open note before each question with related=False. The
    # question's vector search then needs the edited fragments EMBEDDED — a fragment
    # without a vector is invisible to retrieval (CLAUDE.md: `embed` is part of the order).
    embedded = []
    monkeypatch.setattr(copilot.embed_mod, "embed_source", lambda _con, sid: embedded.append(sid))
    write(vault, "Notes/school/lesson.md", "gradient descent")
    copilot.context_payload(con, vault, "Notes/school/lesson.md", related=False)
    assert len(embedded) == 1


# --- nothing rides in the system prompt but the persona and the notes ----------------------

def _answering(monkeypatch, seen: dict):
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
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
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
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
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    sibling = {"n": 9, "source_id": "sib", "path": "Notes/school/b.md", "title": "b",
               "heading": None, "text": "![[other.png]]"}
    real_collect = copilot.collect_evidence
    monkeypatch.setattr(copilot, "collect_evidence", lambda *a, **k: (
        lambda got: ([*got[0], sibling], got[1]))(real_collect(*a, **k)))
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
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
    source = copilot.sync_note(con, vault, "Notes/school/lecture.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
    sent = {}
    answering(monkeypatch, seen=sent)
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    copilot.run_turn(con, source["id"], "Quiz me", [], "quick", vault=vault)
    shown = sent["messages"][-2]
    assert len(shown["images"]) == copilot.MAX_NOTE_IMAGES
    assert "`slide-00.png`" in shown["content"]


def test_a_note_without_images_adds_no_message(con, vault, monkeypatch):
    write(vault, "Notes/school/plain.md", "just words")
    source = copilot.sync_note(con, vault, "Notes/school/plain.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
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
    source = copilot.sync_note(con, vault, "Notes/school/ch2.md", embed=False)
    evidence = [{"n": 1, "source_id": source["id"], "path": source["path"], "title": "ch2",
                 "heading": "Naive Bayes", "text": "MLE ![[mle.png]]"}]
    shown, _ = copilot.open_note_images(vault, source, evidence)
    assert [name for name, _ in shown] == ["mle.png", "psi.png"]


# --- one chat that answers or edits (the editing run) -------------------------------------
#
# Ported from the measured prototype's loop tests: each runs a whole turn through run_turn
# against a note sent as the editor's text, with a scripted model.

import json  # noqa: E402

from slim import edits, llm  # noqa: E402

NOTE = "# Title\n\nIntro para.\n\n## A\n\nText a.\n- one\n- two\n\n## B\n\nText b.\n"
NOTE3 = "# Title\n\n## A\n\na\n\n## B\n\nb\n\n## C\n\nc\n"
STEP_TOOLS = ["replace", "edit_text", "insert", "move"]


class Seen(list):
    """The calls a fake model was sent, and apart from them the intent gate's (`gate`)."""
    gate: list


def scripted(monkeypatch, *replies, intent=True):
    """A fake model: each call returns the next (content, [(tool, arguments)][, output tokens])
    and is recorded with what it was sent. The intent gate's structured call says `intent`
    (True, False, a raw reply, or "error" for an Ollama error) and is recorded in `seen.gate`."""
    seen = Seen()
    seen.gate = []

    def chat_tools(messages, tools=None, **opts):
        if opts.get("fmt") is not None:
            seen.gate.append({"messages": [dict(m) for m in messages], "tools": tools, "opts": opts,
                              "held": getattr(llm._HELD, "on", False)})
            if intent == "error":
                raise llm.LLMError("ollama call failed: timed out")
            said = intent if isinstance(intent, str) else json.dumps({"change_note": intent})
            return {"content": said, "tool_calls": [], "raw_tool_calls": []}, {"output_tokens": 5}
        seen.append({"messages": [dict(m) for m in messages],
                     "tools": [t["function"]["name"] for t in tools or []], "opts": opts,
                     "held": getattr(llm._HELD, "on", False), "stream": False})
        content, calls, *tokens = replies[len(seen) - 1]
        return ({"content": content, "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                 "raw_tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]},
                {"duration_s": 1.0, "output_tokens": tokens[0] if tokens else 10})
    def chat_stream(messages, **opts):
        """Today's Ask path: a streamed answer with no tools."""
        seen.append({"messages": [dict(m) for m in messages], "tools": [], "opts": opts,
                     "held": getattr(llm._HELD, "on", False), "stream": True})
        content, _calls, *tokens = replies[len(seen) - 1]
        if opts.get("on_delta") and content:
            opts["on_delta"](content)
        return content, {"duration_s": 1.0, "output_tokens": tokens[0] if tokens else 10}
    monkeypatch.setattr(copilot.llm, "chat_tools", chat_tools)
    monkeypatch.setattr(copilot.llm, "chat_stream", chat_stream)
    return seen


def flaky(monkeypatch, seen, *failing):
    """The scripted model, raising an Ollama error on the given call numbers (1-based)."""
    real, count = copilot.llm.chat_tools, [0]

    def chat_tools(messages, tools=None, **opts):
        if opts.get("fmt") is not None:          # the gate is not counted
            return real(messages, tools, **opts)
        count[0] += 1
        if count[0] in failing:
            raise llm.LLMError("ollama rejected the call: HTTP 500 — XML syntax error on line 10")
        return real(messages, tools, **opts)
    monkeypatch.setattr(copilot.llm, "chat_tools", chat_tools)
    return seen


@pytest.fixture
def turn(con, vault, monkeypatch):
    """run_turn on a note holding `text`, sent as the editor's text and its hash. Returns the
    turn, and every on_edit event as (event, payload)."""
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))

    def go(question, *, text=NOTE, path="Notes/school/n.md", mode="quick", history=(),
           should_stop=lambda: False, selection="", doc=True):
        target = vault / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        source = copilot.sync_note(con, vault, path, embed=False)
        events, deltas = [], []
        sent = {"doc": text, "doc_hash": edits.digest(text)} if doc else {}
        result = copilot.run_turn(con, source["id"], question, list(history), mode, vault=vault,
                                  on_edit=lambda event, payload: events.append((event, payload)),
                                  on_delta=deltas.append, should_stop=should_stop, selection=selection, **sent)
        result["deltas"] = deltas
        result["final"] = text if result["edit_run"] is None else _final(text, result)
        return result, events
    return go


def _final(text, result):
    for change in result["edit_run"]["changes"]:
        text = edits.apply_splices(text, change["splices"])
    return text


def test_a_text_reply_is_an_answer_with_sources_and_nothing_changes(turn, monkeypatch):
    seen = scripted(monkeypatch, ("Lexical ambiguity is…", []), intent=False)
    asked = []
    monkeypatch.setattr(copilot.llm, "chat_json",
                        lambda messages, schema, **k: asked.append(messages) or ({"notes": [1]}, {}))
    got, events = turn("What is lexical ambiguity?")
    assert got["answer"] == "Lexical ambiguity is…" and got["edit_run"] is None and events == []
    assert seen[0]["stream"] and seen[0]["tools"] == [] and len(seen) == 1
    assert [c["path"] for c in got["citations"]] == ["Notes/school/n.md"]
    (sources,) = asked                                     # the Ask path's sources call, on the same prefix
    assert sources[-2] == {"role": "assistant", "content": "Lexical ambiguity is…"}
    assert sources[:-2] == seen[0]["messages"]


def test_an_answer_reaches_the_chat_through_on_delta(con, vault, monkeypatch):
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    scripted(monkeypatch, ("It descends.", []), intent=False)
    deltas, stages = [], []
    copilot.run_turn(con, source["id"], "What is it?", [], "quick", vault=vault,
                     on_delta=deltas.append, on_stage=stages.append)
    assert deltas == ["It descends."] and stages[0] == "Writing"


def test_a_change_turn_carries_the_rider_and_the_note_with_handles(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b3", "markdown": "## A2"})]))
    turn("q", history=[{"role": "you", "text": "hi"}, {"role": "slim", "text": "hello"}])
    system, *rest = seen[0]["messages"]
    assert system["content"] == copilot.PERSONA + "\n\n" + copilot.EDIT_RIDER + "\n\n" + (
        "Notes in view:\n\n=== Notes/school/n.md · the open note ===\nSections:\n"
        "s1  b1–b8  # Title\n  s3  b3–b6  ## A\n  s7  b7–b8  ## B\n\n[b1] # Title\n\n[b2] Intro para.\n\n"
        "[b3] ## A\n\n[b4] Text a.\n[b5] - one\n[b6] - two\n\n[b7] ## B\n\n[b8] Text b.\n")
    assert [m["role"] for m in rest] == ["user", "assistant", "user"] and rest[-1]["content"] == "q"
    opts = seen[0]["opts"]
    assert opts["think"] is False and opts["temperature"] == copilot.STEP_TEMPERATURE == 0.0
    assert opts["num_predict"] == copilot.QUICK_NUM_PREDICT >= copilot.STEP_NUM_PREDICT
    assert copilot.INTENT_GATE is True


def test_the_rider_is_the_measured_one():
    assert "To answer a question, just reply in text and change nothing." in copilot.EDIT_RIDER
    assert "To make one small change, make it right away, pointing at handles." in copilot.EDIT_RIDER
    assert "For a small wording fix, use edit_text instead of retyping the block." in copilot.EDIT_RIDER
    assert "Use the replace, edit_text, insert and move tools" in copilot.EDIT_RIDER
    assert "the note's headings in their final order (each with its # marks)" in copilot.EDIT_RIDER


def test_deep_gives_the_first_edit_call_room_to_think(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b3", "markdown": "## A2"})]))
    turn("q", mode="deep")
    assert seen[0]["opts"]["think"] is True
    assert seen[0]["opts"]["num_predict"] == copilot.QUICK_NUM_PREDICT + copilot.THINK_BUDGET
    assert copilot.THINK_BUDGET == 8192
    assert copilot.QUICK_NUM_PREDICT + copilot.THINK_BUDGET < copilot.RUN_OUTPUT_TOKENS


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
    assert set(change) == {"id", "step", "op", "label", "needs", "splices"}
    assert (change["id"], change["step"], change["op"], change["label"]) == ("c1", 1, "replace", "Replaced b3")
    run = got["edit_run"]
    assert set(run) == {"plan", "steps_done", "changes", "stopped", "outline_met", "fixups",
                        "hash_before", "hash_after", "summary"}
    assert run["steps_done"] == 2 and run["stopped"] is None and run["outline_met"] is True
    assert run["summary"] == "Rename A and move B up." and got["answer"] == run["summary"]
    assert got["final"].index("## B") < got["final"].index("## A renamed")
    assert run["hash_before"] == edits.digest(NOTE) and run["hash_after"] == edits.digest(got["final"])
    assert [c["tools"] for c in seen] == [["plan", *STEP_TOOLS], STEP_TOOLS, STEP_TOOLS]   # no summary call
    assert "Step 2 of 2: Move B above A" in seen[2]["messages"][-1]["content"]
    assert seen[0]["opts"]["think"] is True and seen[1]["opts"]["think"] is False
    assert seen[0]["opts"]["temperature"] == seen[1]["opts"]["temperature"] == copilot.STEP_TEMPERATURE == 0.0
    assert seen[1]["opts"]["num_predict"] == copilot.STEP_NUM_PREDICT == 2048
    assert got["citations"] == []                          # an edit makes no sources call


def test_the_run_holds_the_model_lock_across_its_calls(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Rename A"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    turn("rename A")
    assert [call["held"] for call in seen.gate + seen] == [True, True, True]
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
    monkeypatch.setattr(copilot, "MAX_CALLS", 3)
    many = [("insert", {"after": "b2", "markdown": f"p{i}"}) for i in range(5)]
    scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["a"]})]), ("", many))
    got, events = turn("go")
    assert got["edit_run"]["stopped"] == "call limit" and len(got["edit_run"]["changes"]) == 3
    assert copilot.MAX_CALLS == 3 and events[-1] == ("step", {"index": 1, "state": "partial"})


def test_a_plan_longer_than_the_limit_is_cut_by_code(turn, monkeypatch):
    scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": [f"s{i}" for i in range(12)]})]), *[("", [])] * 8)
    got, _ = turn("go")
    assert len(got["edit_run"]["plan"]["steps"]) == 8 == copilot.MAX_STEPS and got["edit_run"]["steps_done"] == 8


def test_done_during_a_step_finishes_early_after_that_steps_calls(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "Rename A and B.", "steps": ["Rename A", "Rename B", "Check"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"}),
                          ("done", {"summary": "Renamed A; B was fine."})]))
    got, _ = turn("rename")
    run = got["edit_run"]
    assert run["summary"] == "Renamed A; B was fine." and "## A renamed" in got["final"] and len(run["changes"]) == 1
    assert run["steps_done"] == 1 and run["stopped"] is None and len(seen) == 2


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
    assert seen[0]["opts"]["temperature"] == seen[1]["opts"]["temperature"] == 0.0     # the re-ask's too
    assert got["edit_run"] is None and got["answer"] == "A is the first section." and events == []


def test_a_first_call_that_could_not_be_read_is_asked_for_a_plan_once_more(turn, monkeypatch):
    seen = scripted(monkeypatch, ("<tool_call>{broken", []),
                    ("", [("plan", {"intro": "Rename A.", "steps": ["Rename A"]})]),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, _ = turn("rename A")
    assert "refused: your tool call could not be read (a tool call came back as text); write it again" in \
        seen[1]["messages"][-1]["content"]
    assert "## A renamed" in got["final"]


FIXUP = ("Fix-up: make the note's headings match your outline.\n"
         "Change only what is needed to match; don't delete sections to re-add them.\n")


def test_a_fix_up_lists_only_the_headings_missing_or_out_of_order(turn, monkeypatch):
    plan = {"intro": "x", "outline": ["# Title", "## C", "## A", "## B", "## D"], "steps": ["Reorder"]}
    seen = scripted(monkeypatch, ("", [("plan", plan)]),
                    ("", []),                                               # the step does nothing
                    ("", [("move", {"target": "s6", "before": "s2"})]),     # fix-up 1: C first
                    ("", [("insert", {"after": "s4", "markdown": "## D\n\nd."})]))   # fix-up 2: add D
    got, _ = turn("reorder", text=NOTE3)
    assert seen[2]["messages"][-1]["content"].startswith(FIXUP + (
        "Missing:\n1. ## D (your outline puts it after ## B)\n"
        "Out of order:\n1. ## C (your outline puts it after # Title)\n\nSections now:\n"))
    assert FIXUP + "Missing:\n1. ## D (your outline puts it after ## B)\n\nSections now:\n" in \
        seen[3]["messages"][-1]["content"]
    run = got["edit_run"]
    assert seen[2]["tools"] == STEP_TOOLS and run["changes"][0]["step"] == 2
    assert run["outline_met"] is True and run["fixups"] == 2 and run["steps_done"] == 1 and len(seen) == 4


def test_the_outline_is_met_by_its_headings_in_order_with_others_between(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "outline": ["# title", "##   b"], "steps": ["Tidy"]})]),
                    ("", []))
    got, _ = turn("tidy")
    assert got["edit_run"]["outline_met"] is True and got["edit_run"]["fixups"] == 0 and len(seen) == 2


def test_fix_ups_stop_after_two_and_the_miss_is_recorded(turn, monkeypatch):
    plan = {"intro": "x", "outline": ["# Title", "## B", "## A"], "steps": ["Move B"]}
    seen = scripted(monkeypatch, ("", [("plan", plan)]), ("", []), ("", []), ("", []))
    got, _ = turn("go")
    assert got["edit_run"]["fixups"] == 2 == copilot.MAX_FIXUPS and got["edit_run"]["outline_met"] is False
    assert len(seen) == 4


def test_fix_ups_count_against_the_call_limit_not_the_steps(turn, monkeypatch):
    monkeypatch.setattr(copilot, "MAX_CALLS", 2)
    plan = {"intro": "x", "outline": ["# Title", "## B", "## A"], "steps": ["Add a paragraph"]}
    scripted(monkeypatch, ("", [("plan", plan)]),
             ("", [("insert", {"after": "b2", "markdown": "p0"})]),
             ("", [("insert", {"after": "b2", "markdown": "p1"}), ("move", {"target": "s7", "before": "s3"})]))
    got, _ = turn("go")
    assert got["edit_run"]["stopped"] == "call limit" and got["edit_run"]["fixups"] == 1
    assert len(got["edit_run"]["changes"]) == 2


def test_the_outline_is_read_leniently_and_locked_headings_are_left_out(turn, monkeypatch):
    memo = f"# Memo\n\nSummary.\n\n{edits.TRANSCRIPT_MARKER}\nwords words\n"
    for outline in ([" [b1] #   Memo "], ["#Memo", edits.TRANSCRIPT_MARKER]):
        seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "outline": outline, "steps": ["Shorten"]})]),
                        ("", [("replace", {"target": "b2", "markdown": "Short."})]))
        got, _ = turn("shorten", text=memo, path="Capture/m.md")
        assert got["edit_run"]["outline_met"] is True and got["edit_run"]["fixups"] == 0 and len(seen) == 2, outline


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
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append(payload))
    with pytest.raises(llm.LLMError, match="HTTP 500"):
        turn("q")
    assert records[-1]["status"] == "failed"


def test_outline_entries_are_cleaned_of_what_the_model_wrote_around_them():
    assert copilot._outline({"outline": ['"# Softmax Function\\n\\n## Formula\\n\\n## Example"']}) == [
        "# Softmax Function", "## Formula", "## Example"]                          # one quoted string, \n literal
    assert copilot._outline({"outline": ["## Toolkit (New Section)", "## Tools (added)", "## Bag of Words (BoW)"]}) == [
        "## Toolkit", "## Tools", "## Bag of Words (BoW)"]                         # annotations, not words
    assert copilot._outline({"outline": ["# <the note's headings, one per line, in their final order>", "## <…>",
                                         "## Formula"]}) == ["## Formula"]         # the rider's placeholder copied


def test_an_annotation_on_either_side_does_not_make_a_miss():
    assert copilot._misses(["# T", "## Toolkit (New Section)"], ["# T", "## Toolkit"]) == ([], [])
    assert copilot._misses(["# T", "## Toolkit"], ["# T", "## toolkit (added)"]) == ([], [])
    assert copilot._misses(["# T", "## Pipeline (Classical vs. Neural)"], ["# T", "## Pipeline"]) == ([1], [])


def test_a_missing_heading_the_note_has_at_another_level_is_named(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "outline": ["# Title", "# A", "## B"], "steps": ["Promote A"]})]),
                    ("", []), ("", []), ("", []))
    turn("promote A")
    assert "Missing:\n1. # A (your outline puts it after # Title; the note has it as ## A)\n\nSections now:" in \
        seen[2]["messages"][-1]["content"]


def test_the_token_budget_stops_the_run_after_the_reply_that_crosses_it(turn, monkeypatch):
    seen = scripted(monkeypatch,
                    ("", [("plan", {"intro": "x", "steps": ["Rename A", "Move B", "Rename B"]})], 100),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})], 5000),
                    ("", [("move", {"target": "s7", "before": "s3"})], 8000))
    got, _ = turn("go")
    run = got["edit_run"]
    assert run["stopped"] == "token budget" and len(seen) == 3 and run["summary"] == ""
    assert len(run["changes"]) == 2 and got["final"].index("## B") < got["final"].index("## A renamed")
    assert copilot.RUN_OUTPUT_TOKENS == 12000


def test_a_runaway_reply_with_no_calls_stops_the_run_at_once(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("plan", {"intro": "x", "steps": ["Simplify"]})], 300),
                    ("A very long explanation with no tool calls …", [], 12500))
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
    source = copilot.sync_note(con, vault, "Notes/school/h.md", embed=False)
    with pytest.raises(copilot.CopilotError, match="the note changed while sending; try again"):
        copilot.run_turn(con, source["id"], "q", [], "quick", vault=vault, doc=NOTE, doc_hash=edits.digest("other"))


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
    assert "[locked] ## Transcript" in seen[0]["messages"][0]["content"]
    assert "b4 is part of the recording; SLIM never edits it" in seen[2]["messages"][-1]["content"]
    assert got["final"] == memo.replace("one\n", "One line.\n")


def test_an_overflowing_first_call_is_refused_before_anything_lands(turn, monkeypatch):
    def saturated(messages, tools=None, **opts):
        return ({"content": "", "tool_calls": [{"name": "replace", "arguments": {"target": "b3", "markdown": "x"}}],
                 "raw_tool_calls": []}, {"ctx_saturated": True})
    monkeypatch.setattr(copilot.llm, "chat_tools", saturated)
    with pytest.raises(copilot.CopilotError, match="too long for one request"):
        turn("fix it")


def test_an_edit_turn_writes_machine_facts_to_the_trace(turn, monkeypatch):
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append((kind, payload)))
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
    assert facts["stopped"] is None and facts["fixups"] == 0 and facts["outline_met"] is None
    assert [t["output_tokens"] for t in facts["timing"]] == [10, 10, 10, 10]
    assert payload["intent"] is True and payload["timing"]["gate"] == {"output_tokens": 5}
    assert "Text b" not in str(payload) and "A2" not in str(payload)       # facts, never the note's words



# --- the intent gate: one small structured call decides whether this turn may edit -----------

ANSWER = "## Proposed Structure\n\n1. Rename A\n2. Move B up"


def test_the_gate_sees_only_its_own_question_slims_last_answer_and_the_message(turn, monkeypatch):
    seen = scripted(monkeypatch, ("A is the first section.", []), intent=False)
    history = [{"role": "you", "text": "How could this note be organised?"}, {"role": "slim", "text": ANSWER},
               {"role": "you", "text": "a question that failed"}]
    turn("Ok make the revisions", history=history)
    (gate,) = seen.gate
    assert gate["messages"] == [
        {"role": "system", "content": "You decide whether the owner's last message asks to change the note "
                                      "they have open. Reply with JSON only."},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": "Ok make the revisions"}]
    assert gate["tools"] is None and gate["held"] is True              # no note, no evidence, no tools
    assert gate["opts"] == {"think": False, "temperature": 0.0, "num_predict": 20, "fmt": copilot.GATE_SCHEMA}
    seen = scripted(monkeypatch, ("Answer.", []), intent=False)
    turn("What is A?")
    assert seen.gate[0]["messages"][1:] == [{"role": "user", "content": "What is A?"}]   # no history: no answer


def test_no_change_wanted_is_todays_ask_path(turn, monkeypatch):
    seen = scripted(monkeypatch, ("Lexical ambiguity is…", []), intent=False)
    asked = []
    monkeypatch.setattr(copilot.llm, "chat_json",
                        lambda messages, schema, **k: asked.append(messages) or ({"notes": [1]}, {}))
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append(payload))
    got, events = turn("What is lexical ambiguity?")
    (first,) = seen
    system = first["messages"][0]["content"]
    assert first["stream"] and system.startswith(copilot.PERSONA + "\n\n" + copilot.ASK_RIDER)
    assert "[b" not in system and "Sections:" not in system and copilot.EDIT_RIDER not in system
    assert "Intro para." in system                                    # the passages retrieval found
    assert first["opts"]["temperature"] == llm.REPLY_TEMPERATURE and first["opts"]["num_predict"] == \
        copilot.QUICK_NUM_PREDICT
    assert got["deltas"] == ["Lexical ambiguity is…"] and got["edit_run"] is None and events == []
    assert asked and asked[0][:-2] == first["messages"]               # the sources call, on its prefix
    assert [c["path"] for c in got["citations"]] == ["Notes/school/n.md"]
    assert records[-1]["intent"] is False and records[-1]["status"] == "answered"
    assert records[-1]["timing"]["gate"] == {"output_tokens": 5} and "prompt_tokens" in records[-1]["gate"]


def test_a_wanted_change_is_offered_the_tools_at_temperature_zero_and_a_text_reply_is_nudged_once(turn, monkeypatch):
    seen = scripted(monkeypatch, ("I would rename it.", []), ("Still only text.", []))
    got, events = turn("rename A")
    assert [c["tools"] for c in seen] == [["plan", *STEP_TOOLS]] * 2
    assert [c["opts"]["temperature"] for c in seen] == [0.0, 0.0] and "on_delta" not in seen[0]["opts"]
    assert seen[1]["messages"][-1] == {"role": "user", "content": "Use the tools to make the change."}
    assert got["answer"] == "Still only text." and got["edit_run"] is None and events == []
    assert got["deltas"] == ["Still only text."]               # said once, whole
    seen = scripted(monkeypatch, ("I would rename it.", []),
                    ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, _ = turn("rename A")
    assert "## A renamed" in got["final"] and len(seen) == 2


@pytest.mark.parametrize("intent", ["error", "yes, please", '{"change_note": "yes"}'])
def test_a_gate_that_fails_or_cannot_tell_counts_as_a_change(turn, monkeypatch, intent):
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append(payload))
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]), intent=intent)
    got, _ = turn("rename A")
    assert seen[0]["tools"] == ["plan", *STEP_TOOLS] and seen[0]["opts"]["temperature"] == 0.0
    assert "## A renamed" in got["final"] and records[-1]["intent"] is None


def test_an_edit_skill_is_a_wanted_change_without_asking(turn, monkeypatch):
    seen = scripted(monkeypatch, ("", [("replace", {"target": "b5", "markdown": "- **one**"})]))
    turn("/tidy", selection="- one")
    assert seen.gate == [] and seen[0]["tools"] == ["plan", "replace", "edit_text"]
    assert seen[0]["opts"]["temperature"] == 0.0


def test_with_the_gate_off_every_turn_may_edit_at_the_reply_temperature(turn, monkeypatch):
    monkeypatch.setattr(copilot, "INTENT_GATE", False)
    seen = scripted(monkeypatch, ("A is the first section.", []))
    got, _ = turn("what is A?")
    assert seen.gate == [] and len(seen) == 1 and seen[0]["tools"] == ["plan", *STEP_TOOLS]
    assert seen[0]["opts"]["temperature"] == copilot.FIRST_TEMPERATURE and not seen[0]["stream"]
    assert got["answer"] == "A is the first section." and got["deltas"] == ["A is the first section."]


def test_a_turn_stopped_before_the_gate_makes_no_call(turn, monkeypatch):
    seen = scripted(monkeypatch, ("never", []))
    with pytest.raises(copilot.CopilotError, match="stopped"):
        turn("What is A?", should_stop=lambda: True)
    assert seen == [] and seen.gate == []


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
    assert run["summary"] == "I'll tidy this note." and run["fixups"] == 0 and len(seen) == 3
    assert got["final"] == NOTE.replace("## A\n", "## A renamed\n").replace("Text b.", "B.")


@pytest.mark.parametrize("content", ["", "Renamed it.", "Renamed it:\n\n- ## A renamed"])
def test_an_edit_with_short_or_no_prose_stays_a_one_reply_edit(turn, monkeypatch, content):
    seen = scripted(monkeypatch, (content, [("replace", {"target": "b3", "markdown": "## A renamed"})]))
    got, events = turn("rename A")
    assert len(seen) == 1 and got["edit_run"]["plan"]["steps"] == [] and [e for e, _ in events] == [
        "start", "plan", "change"]


def test_list_items_in_any_marker_make_the_steps_and_the_first_sentence_the_intro():
    assert copilot._listed_plan("Plan: do it. Now.\n- a\n* b\n3. c\n4) d") == ("Plan: do it.", ["a", "b", "c", "d"])
    assert copilot._listed_plan("Only one:\n- a") is None and copilot._listed_plan("no list") is None
    assert copilot._listed_plan("\n".join(f"- s{i}" for i in range(12)))[1] == [f"s{i}" for i in range(8)]
