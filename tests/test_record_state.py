"""The recorder's shared state (`record_state`): indexing and the background sweep, cancel,
one writer per note, the bounded job queue and job-local progress."""
import json
import threading
import time

import pytest

from tests.record_fakes import (  # noqa: F401
    NOW, _archive_in_tmp, _draft, _existing, _fakes, _staged)


# --- 2026-08-26: the request indexes ITS note; the vault sweep runs in the background -------

def test_the_request_indexes_only_its_note_and_sweeps_the_vault_afterwards(tmp_path, monkeypatch):
    """Measured 2026-08-26: the full pass inside the request embedded 1,287 fragments for a
    ~50-fragment note — a backlog from 11 other edited notes — and the recording waited for
    it. `ingest_note` + `embed_source` share the sweep's code, so the docstring's old
    objection (a second, less-tested copy) no longer holds."""
    from slim import record_pipeline, record_state
    from slim.index import embed as embed_mod, ingest as ingest_mod
    _fakes(monkeypatch)
    seen = {"note": [], "source": [], "sweep_ingest": 0, "sweep_embed": 0}
    monkeypatch.setattr(ingest_mod, "ingest_note",
                        lambda con, vault, rel, **kw: (seen["note"].append(rel), {"id": "src-1"})[1])
    monkeypatch.setattr(embed_mod, "embed_source",
                        lambda con, sid: (seen["source"].append(sid), {"embedded": 3})[1])
    monkeypatch.setattr(ingest_mod, "ingest", lambda con, vault=None, **k: (
        seen.__setitem__("sweep_ingest", seen["sweep_ingest"] + 1), {"unchanged": 0})[1])
    monkeypatch.setattr(embed_mod, "embed_missing", lambda con, **k: (
        seen.__setitem__("sweep_embed", seen["sweep_embed"] + 1), {"embedded": 0})[1])
    scheduled = []
    monkeypatch.setattr(record_state, "_schedule_sweep", lambda vault: scheduled.append(vault))
    out = record_pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                        declared_type="meeting-note", title="T", when=NOW,
                                        vault=tmp_path)
    assert seen["note"] == [out["note"]]
    assert seen["source"] == ["src-1"]
    assert seen["sweep_ingest"] == 0                     # nothing vault-wide inside the request
    assert scheduled == [tmp_path]
    record_state._sweep_vault(tmp_path)                  # what the scheduled thread runs
    assert (seen["sweep_ingest"], seen["sweep_embed"]) == (1, 1)


def test_a_quiet_sweep_writes_no_trace(tmp_path, monkeypatch):
    """776 `record/sweep` records in a month, all but a few with nothing changed."""
    from slim import record_state, trace
    from slim.index import embed as embed_mod, ingest as ingest_mod
    quiet = {"unchanged": 3, "new": 0, "edited": 0, "renamed": 0, "removed": 0}
    results = iter([(quiet, {"embedded": 0, "already": 3}),
                    ({**quiet, "removed": 1}, {"embedded": 0, "already": 3}),
                    (quiet, {"embedded": 2, "already": 3}),
                    (RuntimeError("disk"), None)])

    def fake_ingest(con, vault=None, **k):
        stats, embedded = next(results)
        if isinstance(stats, Exception):
            raise stats
        fake_ingest.embedded = embedded
        return stats

    monkeypatch.setattr(ingest_mod, "ingest", fake_ingest)
    monkeypatch.setattr(embed_mod, "embed_missing", lambda con, **k: fake_ingest.embedded)
    for _ in range(4):
        record_state._sweep_vault(tmp_path)
    swept = [r for r in trace.last(10) if r.get("stage") == "sweep"]
    assert [(r["status"], r.get("ingest", {}).get("removed"), r.get("embed", {}).get("embedded"))
            for r in swept] == [("ok", 1, 0), ("ok", 0, 2), ("failed", None, None)]


def test_the_sweep_runs_on_a_daemon_thread_and_sweeps_serialize(tmp_path, monkeypatch):
    import threading
    from slim import record_state
    order = []
    def slow_sweep(vault):
        order.append(("start", threading.current_thread().name))
        threading.Event().wait(0.05)
        order.append(("end", threading.current_thread().name))
    monkeypatch.setattr(record_state, "_sweep_vault", lambda vault: record_state._serialized(
        slow_sweep, vault))
    first, second = record_state._schedule_sweep(tmp_path), record_state._schedule_sweep(tmp_path)
    assert first.daemon and second.daemon
    first.join(2); second.join(2)
    assert [kind for kind, _ in order] == ["start", "end", "start", "end"]   # never interleaved


def test_a_broken_single_note_index_never_costs_the_recording(tmp_path, monkeypatch):
    from slim import record_pipeline, record_state
    from slim.index import ingest as ingest_mod
    _fakes(monkeypatch)
    def boom(con, vault, rel):
        raise RuntimeError("index exploded")
    monkeypatch.setattr(ingest_mod, "ingest_note", boom)
    monkeypatch.setattr(record_state, "_schedule_sweep", lambda vault: None)
    out = record_pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                        declared_type="meeting-note", title="T", when=NOW,
                                        vault=tmp_path)
    assert (tmp_path / out["note"]).exists()


# --- 2026-08-26: an accidental Stop must not cost the lecture -----------------------------

def test_cancelling_a_job_stops_the_pass_and_writes_nothing(tmp_path, monkeypatch):
    """Their case: Stop is hit 40 minutes into a lecture by accident. Cancel has to leave the
    recording exactly as it was — ⚠ NOTHING WRITTEN, NOTHING DELETED — because the draft note
    and the staged audio are the only copies of what they said."""
    from slim import record_pipeline, record_state
    from slim.capture import summarize, transcribe
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="forty minutes of lecture", model="fake", audio_seconds=2400.0, wall_seconds=30.0))

    def cancel_midway(*a, **k):
        record_state.request_record_cancel("job-1")
        return summarize.Summary(overview="never used")
    monkeypatch.setattr(summarize, "summarize", cancel_midway)

    audio = _staged(tmp_path)
    draft = _draft(tmp_path)
    record_state._record_progress_start("job-1")
    with pytest.raises(record_state.RecordCancelled):
        record_pipeline.handle_record(audio_rel=audio, draft_rel=draft, declared_type="lecture",
                                      title="T", when=NOW, vault=tmp_path, job_id="job-1")

    assert (tmp_path / audio).is_file(), "the audio is the recording; cancel never deletes it"
    assert (tmp_path / draft).is_file(), "their typed notes survive a cancel"
    assert not list((tmp_path / "Capture").glob("**/*.md")) or \
        all(p.name.endswith(".slim-draft.md") for p in (tmp_path / "Capture").glob("**/*.md"))


def test_cancelling_before_the_expensive_part_is_honoured(tmp_path, monkeypatch):
    """The stages that are not streaming — transcription, the card, the title — are blocking
    calls, so cancel lands at the next boundary. It must at least be CHECKED at each one."""
    from slim import record_pipeline, record_state
    from slim.capture import transcribe
    calls = []
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: (
        calls.append(p), transcribe.Transcript(text="x", model="f", audio_seconds=1.0,
                                               wall_seconds=1.0))[1])
    record_state._record_progress_start("job-2")
    record_state.request_record_cancel("job-2")
    with pytest.raises(record_state.RecordCancelled):
        record_pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                      declared_type="lecture", title="T", when=NOW, vault=tmp_path,
                                      job_id="job-2")
    assert calls == [], "a cancel before transcription does not pay for a transcription"


def test_cancelling_an_unknown_job_says_so(tmp_path):
    from slim import record_state
    assert record_state.request_record_cancel("no-such-job") is False


def test_two_writers_on_one_note_take_turns(tmp_path, monkeypatch):
    """⚠ `ThreadingHTTPServer` runs handlers in parallel and four endpoints write the same note.
    Re-reading before the write closed the long-job case; it does NOT close the case where two
    writers interleave between one another's read and write. One lock per note does."""
    import threading
    from slim import record_review, record_state
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))
    order, errors = [], []

    def notes(text, delay):
        try:
            with record_state._note_lock(tmp_path, note_rel):
                order.append(f"enter {text}")
                time.sleep(delay)
                record_review.handle_record_notes(note=note_rel, notes_md=text, vault=tmp_path)
                order.append(f"leave {text}")
        except Exception as exc:                       # noqa: BLE001 - reported below
            errors.append(exc)

    a = threading.Thread(target=notes, args=("- from A", 0.05))
    b = threading.Thread(target=notes, args=("- from B", 0.0))
    a.start(); time.sleep(0.01); b.start(); a.join(); b.join()

    assert not errors
    assert order == ["enter - from A", "leave - from A", "enter - from B", "leave - from B"]
    assert "- from B" in r.note.read_text(encoding="utf-8"), "the last writer in wins, cleanly"


def test_the_lock_is_per_note_not_global(tmp_path, monkeypatch):
    from slim import record_state
    a = record_state._note_lock(tmp_path, "Capture/a.md")
    b = record_state._note_lock(tmp_path, "Capture/b.md")
    again = record_state._note_lock(tmp_path, "Capture/a.md")
    assert a is again and a is not b


# --- parallel recorder sessions: bounded heavy work and isolated progress -----------------

def test_recording_progress_is_job_local_and_never_exposes_private_reasoning():
    from slim import record_state

    record_state._record_progress_start("meeting-a")
    record_state._record_progress_start("meeting-b")
    record_state._record_progress_update("meeting-a", stage="summarizing", label="Writing A",
                                         kind="thinking", text="private reasoning for A",
                                         metrics={"transcript_words": 41}, )
    record_state._record_progress_update("meeting-b", stage="queued",
                                         label="Queued behind 1 recording",
                                         detail="Waiting for the local recording pipeline", )

    a = record_state.record_progress("meeting-a")
    b = record_state.record_progress("meeting-b")
    assert a["stage"] == "summarizing" and a["metrics"]["transcript_words"] == 41
    assert b["stage"] == "queued" and b["label"] == "Queued behind 1 recording"
    assert "thinking" not in a and "draft" not in a
    assert a["metrics"]["model_activity_events"] == 1
    assert "private reasoning" not in json.dumps(a)

    a["metrics"]["transcript_words"] = 999
    assert record_state.record_progress("meeting-a")["metrics"]["transcript_words"] == 41
    assert record_state.record_progress("meeting-b")["metrics"] == {}


def test_recording_pipeline_is_fifo_and_an_exception_releases_the_next_job():
    from slim import record_state

    pipeline = record_state.RecordingPipelineQueue(limit=3)
    release_a = threading.Event()
    entered_a = threading.Event()
    queued = {"b": threading.Event(), "c": threading.Event()}
    order, positions, errors = [], [], []

    def run(name, *, wait=False, fail=False):
        try:
            with pipeline.slot(
                name,
                on_queued=lambda position: (
                    positions.append((name, position)), queued.get(name, threading.Event()).set()
                ),
            ):
                order.append(name)
                if name == "a":
                    entered_a.set()
                    release_a.wait(2)
                if fail:
                    raise RuntimeError("expected failure")
        except Exception as exc:  # the queue must release even when work fails
            errors.append((name, str(exc)))

    a = threading.Thread(target=run, args=("a",))
    b = threading.Thread(target=run, args=("b",), kwargs={"fail": True})
    c = threading.Thread(target=run, args=("c",))
    a.start(); assert entered_a.wait(1)
    b.start(); assert queued["b"].wait(1)
    c.start(); assert queued["c"].wait(1)
    release_a.set()
    a.join(2); b.join(2); c.join(2)

    assert order == ["a", "b", "c"]
    assert positions == [("b", 1), ("c", 2)]
    assert errors == [("b", "expected failure")]
    assert pipeline.size == 0


def test_recording_pipeline_has_a_hard_admission_bound():
    from slim import record_state

    pipeline = record_state.RecordingPipelineQueue(limit=2)
    release = threading.Event()
    entered = threading.Event()
    queued = threading.Event()

    def hold_first():
        with pipeline.slot("a"):
            entered.set()
            release.wait(2)

    def hold_second():
        with pipeline.slot("b", on_queued=lambda _position: queued.set()):
            pass

    a = threading.Thread(target=hold_first)
    b = threading.Thread(target=hold_second)
    a.start(); assert entered.wait(1)
    b.start(); assert queued.wait(1)
    try:
        with pytest.raises(record_state.RecordQueueFull):
            with pipeline.slot("c"):
                pass
    finally:
        release.set()
        a.join(2); b.join(2)


def test_waiting_recording_job_reports_its_own_queue_position(monkeypatch):
    from slim import record_state

    monkeypatch.setattr(record_state, "_RECORD_PIPELINE", record_state.RecordingPipelineQueue(
        limit=2))
    release = threading.Event()
    entered = {"a": threading.Event(), "b": threading.Event()}

    def run(name):
        with record_state._record_pipeline_slot(name):
            entered[name].set()
            if name == "a":
                release.wait(2)

    record_state._record_progress_start("a")
    record_state._record_progress_start("b")
    a = threading.Thread(target=run, args=("a",))
    b = threading.Thread(target=run, args=("b",))
    a.start(); assert entered["a"].wait(1)
    b.start()
    deadline = time.time() + 1
    while time.time() < deadline and record_state.record_progress("b")["label"] == "Queued":
        time.sleep(0.005)

    queued_state = record_state.record_progress("b")
    assert queued_state["stage"] == "queued"
    assert queued_state["label"] == "Queued behind 1 recording"
    assert entered["b"].is_set() is False

    release.set()
    a.join(2); b.join(2)
    assert entered["b"].is_set(), "the queued job starts when A releases the pipeline"
