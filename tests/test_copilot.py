"""Open-note copilot: targeted indexing, deterministic scope, and nearby notes."""

from pathlib import Path

import pytest

from slim import copilot, db, threads


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


def test_run_turn_writes_a_replayable_trace(con, vault, monkeypatch):
    # CLAUDE.md: every query writes a replayable trace. The thread JSON keeps only turns
    # that COMPLETED; a trace is the one record of a failed or stopped turn.
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
    monkeypatch.setattr(copilot.llm, "chat_stream", lambda messages, **kwargs: (
        "Answer", {"duration_s": 1, "output_tokens": 5}))
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": [1]}, {"duration_s": 0.3}))
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append((kind, payload)))
    copilot.run_turn(con, source["id"], "What is this?", [{"role": "you", "text": "hi"}], "quick")
    (kind, payload), = records
    assert kind == "copilot"
    assert payload["source"] == "Notes/school/lesson.md"
    assert payload["question"] == "What is this?"
    assert payload["mode"] == "quick"
    assert payload["evidence"] == ["Notes/school/lesson.md"]
    assert payload["retrieval_levels"] == ["note"]
    assert payload["cited"] == ["Notes/school/lesson.md"]
    assert payload["history_turns"] == 1
    assert payload["timing"]["answer"]["output_tokens"] == 5
    assert "answer" not in payload            # machine facts only; the thread holds the prose


def test_run_turn_traces_a_refused_turn(con, vault, monkeypatch):
    write(vault, "Notes/school/lesson.md", "gradient descent")
    source = copilot.sync_note(con, vault, "Notes/school/lesson.md", embed=False)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *args, **kwargs: [])
    monkeypatch.setattr(copilot.llm, "chat_stream", lambda messages, **kwargs: ("", {}))
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append((kind, payload)))
    with pytest.raises(copilot.CopilotError):
        copilot.run_turn(con, source["id"], "What is this?", [], "quick")
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
    def fake_stream(messages, **kwargs):
        seen["system"] = messages[0]["content"]
        return "answer", {}
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot.llm, "chat_stream", fake_stream)
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
    copilot.run_turn(con, source["id"], "q", [], "quick")
    assert seen["system"].startswith(copilot.PERSONA)
    assert "I am Alex" not in seen["system"]
    assert "About the user" not in seen["system"]
