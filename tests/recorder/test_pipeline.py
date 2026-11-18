"""One recording through `recorder.pipeline.handle_record`, with the slow parts faked.

Transcription and the dense model have their own suites. What matters here is the PROMISE:
a recording that reaches the server is never lost. A summary or title failure still files the
note; a transcription failure leaves the draft and audio available for retry, because silently
filing an empty transcript is precisely the Notion failure this feature exists to remove.
"""
from pathlib import Path

import pytest

from tests.recorder.fakes import NOW, _archive_in_tmp, _draft, _fakes, _staged  # noqa: F401


# --- POST /api/record -------------------------------------------------------------------


def test_the_http_contract_accepts_exactly_one_notes_source():
    from slim.recorder import pipeline

    legacy = pipeline._record_args({"audio": "a.webm", "notes_md": "- old client",
                                    "type": "lecture", "title": "T"})
    assert legacy["notes_md"] == "- old client" and legacy["draft_rel"] is None

    draft = pipeline._record_args({"audio": "a.webm", "draft": "d.slim-draft.md",
                                   "type": "lecture", "title": "T"})
    assert draft["draft_rel"] == "d.slim-draft.md" and draft["notes_md"] is None

    with pytest.raises(ValueError, match="exactly one"):
        pipeline._record_args({"audio": "a.webm", "draft": "d.slim-draft.md",
                               "notes_md": "two truths", "type": "lecture"})


def test_a_saved_draft_is_the_only_source_of_typed_notes(tmp_path, monkeypatch):
    from slim.recorder import pipeline

    notes = "  # Live notes\n\n- keep trailing spaces  \n\n![[Attachments/Recorder/t/paste.png]]\n"
    draft = _draft(tmp_path, text=notes)
    _fakes(monkeypatch)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), draft_rel=draft, notes_md=None,
                                 declared_type="lecture", title="SVMs", when=NOW,
                                 vault=tmp_path)

    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    start = body.index("## Notes\n\n") + len("## Notes\n\n")
    end = body.index("\n## Transcript", start)
    assert body[start:end] == notes


def test_inline_draft_anchor_is_not_mistaken_for_their_notes(tmp_path, monkeypatch):
    from slim.recorder import pipeline

    notes = "## Follow-up\n\n- exact human note\n"
    draft = _draft(tmp_path, text="````slim-meeting\n\n````\n\n" + notes)
    _fakes(monkeypatch)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), draft_rel=draft,
                                 declared_type="lecture", title="SVMs", when=NOW,
                                 vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert body.count("slim-meeting") == 1
    assert notes.rstrip() in body


def test_a_successful_draft_recording_removes_only_its_staging_files(tmp_path, monkeypatch):
    from slim.recorder import pipeline

    staged = _staged(tmp_path)
    draft = _draft(tmp_path)
    image = tmp_path / "Attachments/Recorder/t/paste.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    _fakes(monkeypatch)

    out = pipeline.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                                 declared_type="lecture", title="SVMs", when=NOW,
                                 vault=tmp_path)

    assert not (tmp_path / staged).exists()
    assert not (tmp_path / draft).exists()
    assert image.read_bytes() == b"png"
    assert (tmp_path / out["note"]).exists()


def test_a_failed_draft_transcript_keeps_both_sources_for_retry(tmp_path, monkeypatch):
    from slim.recorder import pipeline

    staged = _staged(tmp_path)
    draft = _draft(tmp_path)
    _fakes(monkeypatch, transcribe_raises=True)

    with pytest.raises(ValueError, match="transcription failed"):
        pipeline.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)

    assert (tmp_path / staged).exists()
    assert (tmp_path / draft).read_text(encoding="utf-8") == "- a draft note\n"


@pytest.mark.parametrize(
    ("type_tag", "draft_type", "message"),
    [("journal", "lecture", "Journal"), ("lecture", "journal", "Notes/_unfiled")],
)
def test_a_draft_cannot_cross_the_journal_floor(tmp_path, monkeypatch,
                                                type_tag, draft_type, message):
    from slim.recorder import pipeline

    _fakes(monkeypatch)
    with pytest.raises(ValueError, match=message):
        pipeline.handle_record(audio_rel=_staged(tmp_path),
                               draft_rel=_draft(tmp_path, type_tag=draft_type),
                               notes_md=None, declared_type=type_tag, title="T", when=NOW,
                               vault=tmp_path)


def test_a_draft_in_the_old_recordings_folder_is_refused(tmp_path, monkeypatch):
    from slim.recorder import pipeline

    _fakes(monkeypatch)
    staged = _staged(tmp_path)
    new = tmp_path / _draft(tmp_path)
    old = tmp_path / "Capture" / "_unfiled" / new.name
    old.parent.mkdir(parents=True)
    new.rename(old)
    with pytest.raises(ValueError, match="Notes/_unfiled"):
        pipeline.handle_record(audio_rel=staged, draft_rel=old.relative_to(tmp_path).as_posix(),
                               notes_md=None, declared_type="lecture", title="T", when=NOW,
                               vault=tmp_path)


def test_audio_and_draft_must_describe_the_same_recording(tmp_path, monkeypatch):
    from slim.recorder import pipeline

    _fakes(monkeypatch)
    staged = _staged(tmp_path, "audio-id.webm")
    draft = _draft(tmp_path, recording_id="draft-id")
    with pytest.raises(ValueError, match="recording id"):
        pipeline.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)

    assert (tmp_path / staged).exists() and (tmp_path / draft).exists()


def test_a_final_note_collision_keeps_the_draft_and_staged_audio(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.capture import audio

    staged = _staged(tmp_path)
    draft = _draft(tmp_path)
    _fakes(monkeypatch)
    digest = audio.sha256(tmp_path / staged)
    victim = tmp_path / "Notes/school/CS201 ML" / f"2026-08-04--SVMs--{digest[:8]}.md"
    victim.parent.mkdir(parents=True)
    victim.write_text("I was here first", encoding="utf-8")

    with pytest.raises(ValueError, match="overwrite"):
        pipeline.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                               declared_type="lecture", title="SVMs", when=NOW,
                               vault=tmp_path)

    assert victim.read_text(encoding="utf-8") == "I was here first"
    assert (tmp_path / staged).exists() and (tmp_path / draft).exists()


def test_the_note_is_filed_and_the_card_describes_it(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    _fakes(monkeypatch)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="- dual form",
                                 declared_type="lecture", title="SVMs", when=NOW,
                                 vault=tmp_path)
    note = tmp_path / out["note"]
    assert note.exists()
    assert out["card"]["dest_dir"] == "Notes/school/CS201 ML"
    body = note.read_text(encoding="utf-8")
    assert "- dual form" in body and "the kernel trick" in body


def test_the_card_carries_the_folders_and_tag_counts_the_ui_needs(tmp_path, monkeypatch):
    """The card is the whole payload — the plugin must not have to make a second call to
    render the tree or show 'this tag would be new'."""
    from slim.recorder import pipeline
    _fakes(monkeypatch)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="SVMs", when=NOW,
                                 vault=tmp_path)
    assert isinstance(out["card"]["folders"], list)
    assert isinstance(out["card"]["tag_counts"], dict)


def test_a_fresh_card_counts_the_filed_note_in_its_folder_but_not_its_topics(tmp_path, monkeypatch):
    """Folders as the vault is once the note is filed; topics as they were before it, so a
    topic only this recording uses still shows as new."""
    from slim import llm
    from slim.recorder import pipeline
    from slim.capture import summarize, transcribe

    (tmp_path / "Notes" / "school").mkdir(parents=True)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="the kernel trick", model="fake", audio_seconds=10.0, wall_seconds=1.0))
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary(overview="s"))
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: (
        {"dest_dir": "Notes/school", "confident_depth": 1, "type_tag": "lecture", "topics": ["svm"]}, {}))

    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="SVMs", when=NOW, vault=tmp_path)
    assert out["note"].startswith("Notes/school/")
    counts = {f["path"]: f["count"] for f in out["card"]["folders"]}
    assert counts["Notes/school"] == 1
    assert "svm" not in out["card"]["tag_counts"]
    assert "svm" in pipeline._card_lists(tmp_path)["tag_counts"]     # the note is on disk


def test_note_images_resolve_relative_to_the_draft_path(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.vault import note_images

    seen = {}
    monkeypatch.setattr(note_images, "load", lambda text, vault, *, note_rel=None, limit=15: (
        seen.update(note_rel=note_rel), ([], []))[1])
    draft = _draft(tmp_path, text="![[fig-01.png]]\n")
    _fakes(monkeypatch)
    pipeline.handle_record(audio_rel=_staged(tmp_path), draft_rel=draft, notes_md=None,
                           declared_type="lecture", title="SVMs", when=NOW, vault=tmp_path)
    assert seen.get("note_rel") == draft


def test_the_summary_pass_thinks(tmp_path, monkeypatch):
    """The one ratified exception to think=False. If this regresses, invention doubles in the
    text they read and trusts, silently."""
    from slim.recorder import pipeline
    from slim.capture import summarize
    _fakes(monkeypatch)
    seen = {}
    real = summarize.Summary

    def spy(*a, **k):
        seen.update(k)
        return real(overview="s")
    monkeypatch.setattr(summarize, "summarize", spy)
    pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path)
    assert seen.get("think") is True


def test_record_progress_reports_real_pipeline_and_model_activity(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.capture import summarize

    _fakes(monkeypatch)
    events = []

    def streaming_summary(*args, on_activity=None, **kwargs):
        on_activity("thinking", "identify the decision")
        on_activity("draft", '{"overview":"kernel trick"}')
        return summarize.Summary(
            overview="Kernel trick",
            stats={"model": "fake", "duration_s": 2.5,
                   "output_tokens": 20, "decode_tok_s": 10},
        )

    monkeypatch.setattr(summarize, "summarize", streaming_summary)
    pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="SVMs", when=NOW, vault=tmp_path,
                           on_progress=lambda **event: events.append(event), )

    stages = [event["stage"] for event in events]
    assert stages[0:4] == ["preparing", "transcribing", "masking", "transcribed"]
    assert "summarizing" in stages and "filing" in stages and "saving" in stages
    assert any(event.get("kind") == "thinking" and
               event.get("text") == "identify the decision" for event in events)
    assert any(event.get("kind") == "draft" and "kernel trick" in event.get("text", "")
               for event in events)
    summary_done = next(event for event in events if event["stage"] == "summarized")
    assert summary_done["metrics"]["summary_output_tokens"] == 20


def test_a_dead_model_still_files_the_recording(tmp_path, monkeypatch):
    """⚠ THE CORE PROMISE. Summarization failing must cost a summary, never a recording."""
    from slim.recorder import pipeline
    _fakes(monkeypatch, summarize_raises=True)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="- my notes",
                                 declared_type="lecture", title="T", when=NOW,
                                 vault=tmp_path)
    note = tmp_path / out["note"]
    assert note.exists()
    assert "- my notes" in note.read_text(encoding="utf-8")     # their words survived
    assert "the kernel trick" in note.read_text(encoding="utf-8")   # and the transcript


def test_failed_transcription_refuses_and_keeps_the_audio_staged(tmp_path, monkeypatch):
    """⚠ REVERSED 2026-08-23. This used to assert the opposite — that a note was filed with an
    empty transcript so the recording was "recoverable". What that actually produced was a
    SUCCESS card for a recording with no words, no summary, no tags and a timestamp for a name,
    with nothing anywhere saying so. Third time the recorder reported success while losing the
    content it exists to keep.

    A summary or a title DECORATES a note that is already safe. The transcript IS the note."""
    from slim.recorder import pipeline
    _fakes(monkeypatch, transcribe_raises=True)
    staged = _staged(tmp_path)
    with pytest.raises(ValueError, match="transcription failed"):
        pipeline.handle_record(audio_rel=staged, notes_md="- my notes",
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert (tmp_path / staged).exists(), "the audio must stay staged so a retry is possible"
    assert not list(tmp_path.rglob("*.md")), "no half-note may be left behind"


def test_a_staged_file_outside_the_vault_is_refused(tmp_path, monkeypatch):
    """`audio` arrives over HTTP. A traversal would let a caller name any file on the machine
    and have its bytes copied into the vault and transcribed."""
    from slim.recorder import pipeline
    _fakes(monkeypatch)
    with pytest.raises(ValueError):
        pipeline.handle_record(audio_rel="../../etc/passwd", notes_md="",
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)


def test_a_missing_audio_file_fails_before_any_model_call(tmp_path, monkeypatch):
    """A bad path should cost nothing. Without an early check it burns a transcription attempt
    and a dense-model call before dying at write time."""
    from slim.recorder import pipeline
    from slim.capture import transcribe
    called = []
    monkeypatch.setattr(transcribe, "transcribe",
                        lambda *a, **k: called.append(1))
    with pytest.raises(FileNotFoundError):
        pipeline.handle_record(audio_rel="Attachments/_incoming/nope.webm", notes_md="",
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert called == []


def test_a_failed_transcription_does_not_load_the_dense_model(tmp_path, monkeypatch):
    """Classifying an empty string would load ~20 GB and EVICT the resident chat model, so the
    cost of a failed ASR run would land on whatever they did next. It still must not be called —
    the request now fails before it gets there."""
    from slim.recorder import pipeline
    from slim.recorder import suggest
    _fakes(monkeypatch, transcribe_raises=True)
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: pytest.fail("must not be called"))
    with pytest.raises(ValueError):
        pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)


def test_a_vault_reached_through_a_symlink_still_reports_relative_paths(tmp_path, monkeypatch):
    """⚠ Regression, found by a live run and NOT by the tests above: pytest's `tmp_path` is
    already resolved, so a resolved-vs-unresolved path mismatch is invisible here unless a
    symlink is introduced deliberately. On macOS `/var` -> `/private/var` makes that mismatch
    the DEFAULT for any temp-dir vault, and it made apply move the note correctly and then
    return a 400 — the caller sees a failure for work that actually happened."""
    from slim.recorder import pipeline, review
    from slim.capture import audio

    real = tmp_path / "real_vault"
    (real / "Attachments" / "_incoming").mkdir(parents=True)
    (real / "Notes" / "school").mkdir(parents=True)
    link = tmp_path / "linked_vault"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(audio, "AUDIO_ARCHIVE", link / "Attachments" / "Recordings")

    _fakes(monkeypatch, dest="Notes/school")
    (real / "Attachments" / "_incoming" / "s.webm").write_bytes(b"bytes")

    out = pipeline.handle_record(audio_rel="Attachments/_incoming/s.webm", notes_md="",
                                 declared_type="lecture", title="T", when=NOW, vault=link)
    assert not Path(out["note"]).is_absolute()      # relative, not an absolute leak
    assert not Path(out["audio"]).is_absolute()

    moved = review.handle_record_apply(note=out["note"], dest_dir="Notes/school",
                                       type_tag="lecture", topics=["school"], summary_md="s",
                                       vault=link)
    assert not Path(moved["note"]).is_absolute()


def test_the_note_gets_the_whole_summary_not_just_the_overview(tmp_path, monkeypatch):
    """⚠ Found by an end-to-end run, not by review. handle_record wrote only `s.overview`, so
    key points, decisions, action items and open questions were generated and then thrown away.
    It read as the model being lazy when the caller was discarding its work."""
    from slim.recorder import pipeline
    from slim.capture import summarize, transcribe
    from slim.recorder import suggest
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="we discussed the export", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary(
        overview="A short review.",
        key_points=[{"topic": "Refresh cadence",
                     "points": ["Tableau extracts refresh hourly, too slow for the monthly report"]}],
        action_items=[{"task": "Send the tables"}],
        open_questions=["How leadership is classified"]))
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: suggest.Card(
        dest_dir="Notes/work", confident_depth=1, candidates=[], type_tag="meeting-note",
        topics=[]))

    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="meeting-note", title="T", when=NOW,
                                 vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "## Key points" in body
    assert "Tableau extracts refresh hourly" in body
    assert "## Action items" in body
    assert "## Open questions" in body
    assert "## Decisions" not in body        # empty stays omitted — restraint still holds


# --- titles ------------------------------------------------------------------------------

def test_an_untitled_recording_gets_a_real_title_not_a_timestamp(tmp_path, monkeypatch):
    """"Recording 2026-08-06-142530" is a timestamp in a title's slot: it names nothing, and
    the filename embeds it, so the vault fills with rows that all look alike."""
    from slim.recorder import pipeline
    from slim.capture import enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: ({"title": "Kernel methods and the dual form",
                                          "type_tag": "lecture", "topics": [], "project": ""}, {}))
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="", when=NOW, vault=tmp_path)
    assert "Kernel-methods" in out["note"]
    assert "Recording 2026" not in (tmp_path / out["note"]).read_text(encoding="utf-8")


def test_a_title_they_typed_is_never_overridden(tmp_path, monkeypatch):
    """Their words are testimony here exactly as a spoken type is."""
    from slim.recorder import pipeline
    from slim.capture import enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: pytest.fail("must not be called when they gave a title"))
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="My Own Title", when=NOW,
                                 vault=tmp_path)
    assert "My-Own-Title" in out["note"]


def test_a_failed_title_call_still_files_the_recording(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.capture import enrich
    _fakes(monkeypatch)
    def boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(enrich, "enrich", boom)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="", when=NOW, vault=tmp_path)
    assert (tmp_path / out["note"]).exists()


# --- indexing ----------------------------------------------------------------------------

def test_a_filed_recording_is_indexed_so_the_brain_can_reach_it(tmp_path, monkeypatch):
    """Filing makes it findable in OBSIDIAN. It does not reach ask, search, the MCP tools, the
    chat vault lane, or reflect — all of which read SQLite. ⚠ reflect._journals() selects
    type='journal' FROM THE DATABASE, so an unindexed journal is skipped by the nightly
    synthesis silently."""
    from slim.recorder import pipeline, state
    _fakes(monkeypatch)
    called = {}
    monkeypatch.setattr(state, "index_now",
                        lambda v, note: called.update(vault=v, note=note) or {})
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="meeting-note", title="T", when=NOW,
                                 vault=tmp_path)
    assert called.get("vault") == tmp_path
    assert called.get("note") == out["note"]


def test_a_broken_index_never_costs_the_recording(tmp_path, monkeypatch):
    """A note that is filed but not indexed is recoverable with `slim ingest`. One that failed
    to file is not — so indexing must never be able to take the recording down with it."""
    from slim import db
    from slim.recorder import pipeline
    _fakes(monkeypatch)
    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "connect", boom)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="- notes",
                                 declared_type="meeting-note", title="T", when=NOW,
                                 vault=tmp_path)
    assert (tmp_path / out["note"]).exists()
    assert "- notes" in (tmp_path / out["note"]).read_text(encoding="utf-8")


# --- titles (2026-08-23) ---------------------------------------------------------------------
# `cade09a` added "write a real title from the transcript when they typed none" and it was
# UNREACHABLE for two weeks: the plugin sent `Recording <stamp>` for a blank box and the route
# substituted "Recording" too, so every recorded note was named for the clock.

def test_an_untitled_recording_is_named_from_its_transcript(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.capture import enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: ({"title": "Kernel trick and the dual form"}, {}))
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="", when=NOW, vault=tmp_path)
    assert out["title"] == "Kernel trick and the dual form"


def test_a_title_they_typed_is_never_overwritten(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.capture import enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: pytest.fail("the model must not retitle their recording"))
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="SVMs", when=NOW,
                                 vault=tmp_path)
    assert out["title"] == "SVMs"


def test_the_timestamp_name_survives_only_as_a_last_resort(tmp_path, monkeypatch):
    """They typed no title and the titling model was down. The clock is all that is left — but it
    is the THIRD answer, not the first.

    ⚠ A FAILED TRANSCRIPT no longer reaches here at all: it is refused before filing, so the
    only route to a timestamp name is a transcript that worked and a titling call that did not.
    """
    from slim.recorder import pipeline
    from slim.capture import enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model is down")))
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="", when=NOW, vault=tmp_path)
    assert out["title"] == "Recording 2026-08-04 1430"


def test_their_title_is_passed_to_the_card_but_a_generated_one_is_not(tmp_path, monkeypatch):
    """⚠ A model-written title is derived from the same transcript the card already reads, so
    feeding it back adds no evidence and lets one guess reinforce the next."""
    from slim.recorder import pipeline
    from slim.capture import enrich
    from slim.recorder import suggest
    _fakes(monkeypatch)
    seen = []
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: (
        seen.append(k.get("title")),
        suggest.Card(dest_dir="Notes/school", confident_depth=1, candidates=[],
                     type_tag="lecture", topics=[]))[1])
    monkeypatch.setattr(enrich, "enrich", lambda *a, **k: ({"title": "Generated Name"}, {}))

    pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="My Own Title", when=NOW, vault=tmp_path)
    pipeline.handle_record(audio_rel=_staged(tmp_path, "u.webm"), notes_md="",
                           declared_type="lecture", title="", when=NOW, vault=tmp_path)
    assert seen == ["My Own Title", ""]


def test_a_silent_recording_is_refused_the_same_way(tmp_path, monkeypatch):
    """Silence reaches the same worthless note by a different route — a muted mic, the wrong
    input. Nothing is written; ⚠ since 2026-08-26 it is `NoSpeech`, not a failure, because
    saying so LOUDLY was the friction they objected to."""
    from slim.recorder import pipeline, state
    _fakes(monkeypatch, transcript="   ")
    staged = _staged(tmp_path)
    with pytest.raises(state.NoSpeech, match="no words"):
        pipeline.handle_record(audio_rel=staged, notes_md="", declared_type="lecture",
                               title="T", when=NOW, vault=tmp_path)
    assert (tmp_path / staged).exists()


def test_a_failed_SUMMARY_still_files_the_note(tmp_path, monkeypatch):
    """⚠ The line that matters: a summary DECORATES a recording that is already safe and
    findable. Only the transcript is the artifact."""
    from slim.recorder import pipeline
    _fakes(monkeypatch, summarize_raises=True)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="T", when=NOW,
                                 vault=tmp_path)
    assert (tmp_path / out["note"]).exists()
    assert "the kernel trick" in (tmp_path / out["note"]).read_text(encoding="utf-8")


def test_one_recording_asks_for_one_model_at_one_size(tmp_path, monkeypatch):
    """⚠ THE WHOLE PATH AT ONCE, not one caller at a time.

    Each caller's own test asserted its own ctx, and all of them passed
    while the recording path asked Ollama for the SAME dense model at two different sizes.
    Ollama spawns a separate runner per (model, num_ctx), so the second size loads a second
    copy of 19 GB of weights — mid-recording, on the one path where the owner is sitting there
    waiting, and on the machine where a dense pass has already been OOM-killed twice.

    A per-caller assertion cannot see this: every caller is individually correct. Only the
    set of calls one recording makes shows it, which is why this test drives `handle_record`
    and watches the llm boundary rather than any single function.
    """
    from slim import llm
    from slim.recorder import pipeline
    from slim.capture import enrich, transcribe

    # ⚠ ONLY transcription is faked. The real summarize/suggest/enrich must run — faking
    # them is exactly what let this through, since each one is individually correct.
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="the kernel trick", model="fake", audio_seconds=10.0, wall_seconds=1.0))
    monkeypatch.setattr(enrich, "ENABLED", True)   # the autouse fixture disables it; the
                                                   # title call is the third caller under test

    calls = []

    def spy(messages, schema, **kw):
        calls.append((kw.get("model") or llm.MODEL, kw.get("num_ctx")))
        raise llm.LLMError("transport is faked; only the (model, num_ctx) pair is under test")

    monkeypatch.setattr(llm, "chat_json", spy)

    pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                           declared_type="lecture", title="",
                           when=NOW, vault=tmp_path)   # ⚠ no title → enrich runs

    sizes = {}
    for model, num_ctx in calls:
        sizes.setdefault(model, set()).add(num_ctx)
    offenders = {m: sorted(s) for m, s in sizes.items() if len(s) > 1}
    assert not offenders, (
        f"one recording asked for {offenders} — a second (model, num_ctx) pair makes Ollama "
        f"load a second copy of the weights mid-recording. All calls: {calls}")


# --- 2026-08-25: the resident model summarizes recordings; their notes reach it; lectures
# carry no "Open questions" ------------------------------------------------------------

def test_summary_title_and_card_all_use_the_resident_model(tmp_path, monkeypatch):
    """Measured 2026-08-26: of a 305 s wait after Stop, 191 s was the summary, and most of
    that was gemma4:31b LOADING after evicting the resident qwen3.6:35b — then decoding at
    10 tok/s. Their call: no swap. All three calls in the request move together, at the
    resident's one context size, or one recording spawns two runners."""
    from slim import llm
    from slim.recorder import pipeline
    from slim.capture import enrich, summarize, transcribe
    from slim.recorder import suggest
    seen = {}
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="the kernel trick", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: (
        seen.update({"summarize": k}), summarize.Summary(overview="s"))[1])
    monkeypatch.setattr(enrich, "enrich", lambda *a, **k: (
        seen.update({"enrich": k}), ({"title": "Kernel methods", "type_tag": "lecture",
                                      "topics": [], "project": ""}, {}))[1])
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: (
        seen.update({"card": k}), suggest.Card(dest_dir="Notes/school", confident_depth=1,
                                               candidates=[], type_tag="lecture",
                                               topics=[]))[1])
    pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="", when=NOW, vault=tmp_path)
    assert llm.RECORD_MODEL == llm.MODEL                      # the decision, stated once
    assert seen["summarize"]["model"] == llm.RECORD_MODEL
    assert seen["summarize"]["num_ctx"] == llm.RESIDENT_CTX
    assert seen["enrich"]["model"] == llm.RECORD_MODEL
    assert seen["enrich"]["num_ctx"] == llm.RESIDENT_CTX
    assert seen["card"]["model"] == llm.RECORD_MODEL


def test_their_notes_reach_the_summarizer(tmp_path, monkeypatch):
    """They tell it what they noticed and how they name things. The transcript stays the source."""
    from slim.recorder import pipeline
    from slim.capture import summarize
    seen = {}
    _fakes(monkeypatch)
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: (
        seen.update(k), summarize.Summary(overview="s"))[1])
    pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="- dual form\n- RBF case",
                           declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert seen["notes"] == "- dual form\n- RBF case"


def test_a_lecture_carries_no_open_questions_section(tmp_path, monkeypatch):
    """The summarizer reads the transcript, so for a lecture `open_questions` can only ever
    hold the LECTURER's questions — rhetorical ones they pose and answers, examples on a slide.
    Rendered under "Open questions" they read as the owner's confusions (2026-08-26, NLP lecture).
    A meeting's unresolved question is real information and stays."""
    from slim.recorder import pipeline
    from slim.capture import summarize, transcribe
    from slim.recorder import suggest
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="does a three year old know english", model="fake", audio_seconds=9.0,
        wall_seconds=1.0))
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary(
        overview="A lecture.",
        key_points=[{"topic": "Planned languages",
                     "points": ["Non-natural languages are planned"]}],
        open_questions=["Does a three-year-old know English?"]))
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: suggest.Card(
        dest_dir="Notes/school", confident_depth=1, candidates=[], type_tag="lecture",
        topics=[]))
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="T", when=NOW,
                                 vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "## Key points" in body
    assert "## Open questions" not in body
    assert "three-year-old" not in body


# --- 2026-08-26: silence is an OUTCOME, not an error -------------------------------------

def test_silence_is_reported_as_silence_not_as_a_failure(tmp_path, monkeypatch):
    """Their call, and they are right: "there were no words" is not an error. It was reaching them as
    a red screen asking whether to delete their files — a decision they should never have to make
    over a recording that simply had nothing in it. The note is still not written (an empty
    note is worthless), but the caller has to be able to tell silence from a real failure."""
    from slim.recorder import pipeline, state
    from slim.capture import transcribe
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="   ", model="fake", audio_seconds=30.0, wall_seconds=1.0))
    with pytest.raises(state.NoSpeech) as caught:
        pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert "no words" in str(caught.value).casefold()


def test_a_real_transcription_failure_is_still_an_error(tmp_path, monkeypatch):
    """Silence being quiet must not make a broken ASR quiet too — that one they do need to see."""
    from slim.recorder import pipeline
    from slim.capture import transcribe

    def boom(*a, **k):
        raise RuntimeError("ASR exploded")
    monkeypatch.setattr(transcribe, "transcribe", boom)
    with pytest.raises(ValueError, match="transcription failed"):
        pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                               declared_type="lecture", title="T", when=NOW, vault=tmp_path)


def test_a_leftover_live_session_never_blocks_the_next_recording(monkeypatch):
    """⚠ `live_start` REFUSED when any other session existed, and only ever cleaned up the one
    with the same id. A session left behind by a crash, a killed server or a recording that
    never called stop therefore turned off live captions for every recording afterwards — with
    one line of grey status text as the only sign. A recorder has exactly one live session; a
    new one replaces whatever is there."""
    from slim.capture import transcribe

    class Fake:
        def __init__(self, *a, **k):
            self.closed = False

        def close(self):
            self.closed = True
            return ""
    monkeypatch.setattr(transcribe, "LiveTranscriber", Fake)
    transcribe._LIVE_SESSIONS.clear()

    transcribe.live_start("stale-one")
    stale = transcribe._LIVE_SESSIONS["stale-one"]
    transcribe.live_start("the-new-recording")          # must not raise

    assert stale.closed, "the abandoned session is closed, not left running"
    assert list(transcribe._LIVE_SESSIONS) == ["the-new-recording"]
    transcribe._LIVE_SESSIONS.clear()


def test_an_evening_recording_is_dated_today_not_tomorrow():
    """⚠ A CALENDAR DAY ONLY EXISTS IN A TIMEZONE.

    `handle_record` builds `when` as `datetime.now(timezone.utc)`, and both the `date:` field
    and the FILENAME used to call `strftime` on it directly. East of UTC that is harmless; at
    UTC-4 every recording after 20:00 local is stamped with tomorrow.

    Found 2026-08-27 in real notes: three CS203 lectures recorded on the evening of the 26th
    (20:38, 20:58, 21:17 local) all carried `date: 2026-08-27` — the same date as the lecture
    recorded at 07:00 the next morning. Because the date is the filename prefix, the evening
    lectures also SORTED ahead of the next morning's, which is how they noticed.

    ⚠ THE CLAIM IS ABOUT A MACHINE AT UTC-4, and `local_date` reads the MACHINE's zone, not the
    instant's — so the zone is pinned here. Without the pin this test passes only east of the
    dateline that suits it and fails on a UTC CI runner, which is exactly what happened.
    """
    import os, time
    from datetime import datetime, timedelta, timezone
    from slim.vault import notefile

    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        eastern = timezone(timedelta(hours=-4))
        evening = datetime(2026, 8, 26, 21, 17, tzinfo=eastern)     # 2026-08-27T01:17Z

        assert evening.astimezone(timezone.utc).strftime("%Y-%m-%d") == "2026-08-27"  # the old bug
        assert notefile.local_date(evening) == "2026-08-26"        # their day, and their sort order

        # A UTC-aware instant for the same moment must resolve to the same LOCAL day.
        assert notefile.local_date(evening.astimezone(timezone.utc)) == \
            evening.astimezone().strftime("%Y-%m-%d")
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


# --- PII: masked before any model reads a recording ---------------------------------------

SSN_TEXT = "my ssn is 219-09-9999"
SSN_TOKENS = [("my", 0.0, 0.3), (" ssn", 0.3, 0.6), (" is", 0.6, 0.8), (" 219-09-9999", 1.0, 2.5)]


def test_models_never_see_the_raw_value(tmp_path, monkeypatch):
    from slim.capture import enrich, summarize
    from slim.recorder import pipeline, suggest
    from slim.vault.frontmatter import parse_frontmatter
    _fakes(monkeypatch, transcript=SSN_TEXT, tokens=SSN_TOKENS)
    seen = []
    monkeypatch.setattr(summarize, "summarize", lambda text, *a, **k: (
        seen.append(text), summarize.Summary(overview="s"))[1])
    monkeypatch.setattr(enrich, "enrich", lambda text, *a, **k: (
        seen.append(text), ({"title": "Named"}, {}))[1])
    monkeypatch.setattr(suggest, "build_card", lambda text, *a, **k: (
        seen.append(text), suggest.Card(dest_dir="Notes/_unfiled", confident_depth=1,
                                        candidates=[], type_tag="meeting", topics=[]))[1])
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="meeting", title="", when=NOW, vault=tmp_path)
    assert len(seen) == 3 and not any("219-09-9999" in t for t in seen)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "219-09-9999" not in body and "my ssn is [SSN]" in body
    fm, _ = parse_frontmatter(body)
    assert fm["pii_masked"] == ["ssn:1"] and fm["pii_by"] == "pii-1"
    assert "pii_partial" not in fm
    assert out["pii"] == {"masked": {"ssn": 1}, "partial": False}


def test_a_clean_recording_says_it_was_checked(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter
    _fakes(monkeypatch)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="lecture", title="SVMs", when=NOW, vault=tmp_path)
    fm, _ = parse_frontmatter((tmp_path / out["note"]).read_text(encoding="utf-8"))
    assert fm["pii_masked"] == [] and fm["pii_by"] == "pii-1"
    assert out["pii"] == {"masked": {}, "partial": False}


def test_password_failure_still_files_partial(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter

    def down(window):
        raise RuntimeError("model is down")
    _fakes(monkeypatch, transcript="the door code is 4471 okay", ask_model=down)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="meeting", title="T", when=NOW, vault=tmp_path)
    fm, _ = parse_frontmatter((tmp_path / out["note"]).read_text(encoding="utf-8"))
    assert fm["pii_partial"] == "true"
    assert out["pii"]["partial"] is True


def test_secret_spans_silenced_before_archive(tmp_path, monkeypatch):
    from slim.capture import audio
    from slim.recorder import pipeline
    _fakes(monkeypatch, transcript=SSN_TEXT, tokens=SSN_TOKENS)
    calls = []

    def fake_silence(path, spans, pad=0.15):
        calls.append((path, spans, path.is_file()))
    monkeypatch.setattr(audio, "silence", fake_silence)
    staged = _staged(tmp_path)
    pipeline.handle_record(audio_rel=staged, notes_md="", declared_type="meeting",
                           title="T", when=NOW, vault=tmp_path)
    assert calls == [(tmp_path / staged, [(1.0, 2.5)], True)], "the staged copy, before archiving"


def test_cancel_before_save_leaves_audio_untouched(tmp_path, monkeypatch):
    from slim.capture import audio, summarize
    from slim.recorder import pipeline, state
    _fakes(monkeypatch, transcript=SSN_TEXT, tokens=SSN_TOKENS)

    def cancel_during(*a, **k):
        state.request_cancel("job-pii")
        return summarize.Summary(overview="s")
    monkeypatch.setattr(summarize, "summarize", cancel_during)
    monkeypatch.setattr(audio, "silence", lambda *a, **k: pytest.fail("silenced after a cancel"))
    staged = _staged(tmp_path)
    state.progress_start("job-pii")
    with pytest.raises(state.RecordCancelled):
        pipeline.handle_record(audio_rel=staged, notes_md="", declared_type="meeting",
                               title="T", when=NOW, vault=tmp_path, job_id="job-pii")
    assert (tmp_path / staged).read_bytes() == b"fake-audio-bytes"


def test_a_failed_silence_still_files_partial(tmp_path, monkeypatch):
    import json
    import subprocess

    from slim import trace
    from slim.capture import audio
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter
    _fakes(monkeypatch, transcript=SSN_TEXT, tokens=SSN_TOKENS)

    def boom(path, spans, pad=0.15):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])
    monkeypatch.setattr(audio, "silence", boom)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="meeting", title="T", when=NOW, vault=tmp_path)
    fm, _ = parse_frontmatter((tmp_path / out["note"]).read_text(encoding="utf-8"))
    assert fm["pii_partial"] == "true" and out["pii"]["partial"] is True
    logged = "".join(p.read_text() for p in trace.TRACE_DIR.rglob("*") if p.is_file())
    assert "silence_failed" in logged and "219-09-9999" not in logged


def test_a_secret_with_no_token_times_is_partial(tmp_path, monkeypatch):
    """A cache from before token times existed cannot place the secret in the audio."""
    from slim.recorder import pipeline
    _fakes(monkeypatch, transcript=SSN_TEXT)            # no tokens
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="meeting", title="T", when=NOW, vault=tmp_path)
    assert out["pii"]["partial"] is True


def test_notes_section_never_masked(tmp_path, monkeypatch):
    from slim.recorder import pipeline
    _fakes(monkeypatch, transcript=SSN_TEXT, tokens=SSN_TOKENS)
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="their ssn 219-09-9999",
                                 declared_type="meeting", title="T", when=NOW, vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "## Notes\n\ntheir ssn 219-09-9999" in body


def test_appended_segment_is_masked(tmp_path, monkeypatch):
    from slim.capture import pii, transcribe
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter
    from tests.recorder.fakes import _existing
    r = _existing(tmp_path, monkeypatch)
    before = r.note.read_text(encoding="utf-8")
    old = before[before.index("verbatim words"):].split("\n", 1)[0]
    monkeypatch.setattr(pii, "ask_model", lambda window: None)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="write to jane.doe@example.com today", model="fake", audio_seconds=9.0,
        wall_seconds=1.0))
    pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[_staged(tmp_path, "a-002.webm")], vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    assert "jane.doe@example.com" not in body and "write to [EMAIL] today" in body
    assert old == "verbatim words" and "verbatim words\n" in body
    fm, _ = parse_frontmatter(body)
    assert fm["pii_masked"] == ["email:1"]


def test_appended_counts_add_to_the_notes_own(tmp_path, monkeypatch):
    from slim.capture import pii, transcribe
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter, set_frontmatter
    from tests.recorder.fakes import _existing
    r = _existing(tmp_path, monkeypatch)
    r.note.write_text(set_frontmatter(r.note.read_text(encoding="utf-8"), {
        "pii_masked": ["email:2", "ssn:1"], "pii_by": "pii-1"}), encoding="utf-8")
    monkeypatch.setattr(pii, "ask_model", lambda window: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="write to jane.doe@example.com, the door code is 4471", model="fake",
        audio_seconds=9.0, wall_seconds=1.0))
    pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[_staged(tmp_path, "a-002.webm")], vault=tmp_path)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm["pii_masked"] == ["email:3", "ssn:1"]
    assert fm["pii_by"] == "pii-1" and fm["pii_partial"] == "true"


def test_appended_audio_is_silenced_before_it_is_archived(tmp_path, monkeypatch):
    from slim.capture import audio, pii, transcribe
    from slim.recorder import pipeline
    from tests.recorder.fakes import _existing
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(pii, "ask_model", lambda window: None)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text=SSN_TEXT, model="fake", audio_seconds=9.0, wall_seconds=1.0, tokens=SSN_TOKENS))
    calls = []
    monkeypatch.setattr(audio, "silence",
                        lambda path, spans, pad=0.15: calls.append((path, spans, path.is_file())))
    staged = _staged(tmp_path, "a-002.webm")
    pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[staged], vault=tmp_path)
    assert calls == [(tmp_path / staged, [(1.0, 2.5)], True)]


def test_paused_preview_masks_without_model(tmp_path, monkeypatch):
    import json

    from slim.capture import pii
    from slim.recorder import pipeline
    monkeypatch.setattr(pii, "ask_model", lambda window: pytest.fail("the preview is instant"))
    staged = _staged(tmp_path, "r-001.webm")
    (tmp_path / staged).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": "my ssn is 219-09-9999 and the password is hunter2"}),
        encoding="utf-8")
    out = pipeline.handle_record_transcript(audio_rel=[staged], vault=tmp_path)
    assert "219-09-9999" not in out["text"] and "my ssn is [SSN]" in out["text"]


def test_masking_off_files_verbatim_and_claims_nothing(tmp_path, monkeypatch):
    """`pii: {mask: []}` turns masking off: a note never checked must not say it was, or a
    later `slim mask` would skip it."""
    from slim import config
    from slim.capture import audio, pii
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter
    _fakes(monkeypatch, transcript=SSN_TEXT, tokens=SSN_TOKENS)
    monkeypatch.setattr(config, "PII_MASK", ())
    monkeypatch.setattr(pii, "ask_model", lambda window: pytest.fail("masking is off"))
    monkeypatch.setattr(audio, "silence", lambda *a, **k: pytest.fail("masking is off"))
    stages = []
    out = pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                 declared_type="meeting", title="T", when=NOW, vault=tmp_path,
                                 on_progress=lambda stage, **k: stages.append(stage))
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "my ssn is 219-09-9999" in body
    fm, _ = parse_frontmatter(body)
    assert not [k for k in fm if k.startswith("pii_")]
    assert out["pii"] == {"masked": {}, "partial": False}
    assert "masking" not in stages


def test_masking_off_leaves_an_append_and_the_preview_verbatim(tmp_path, monkeypatch):
    import json

    from slim import config
    from slim.capture import audio, pii, transcribe
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter, set_frontmatter
    from tests.recorder.fakes import _existing
    monkeypatch.setattr(config, "PII_MASK", ())
    monkeypatch.setattr(pii, "ask_model", lambda window: pytest.fail("masking is off"))
    monkeypatch.setattr(audio, "silence", lambda *a, **k: pytest.fail("masking is off"))
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text=SSN_TEXT, model="fake", audio_seconds=9.0, wall_seconds=1.0, tokens=SSN_TOKENS))
    pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[_staged(tmp_path, "a-002.webm")], vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(body)
    assert "my ssn is 219-09-9999" in body and not [k for k in fm if k.startswith("pii_")]

    staged = _staged(tmp_path, "r-001.webm")
    (tmp_path / staged).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": SSN_TEXT}), encoding="utf-8")
    out = pipeline.handle_record_transcript(audio_rel=[staged], vault=tmp_path)
    assert out["text"] == SSN_TEXT


def test_an_append_onto_a_never_masked_note_never_claims_the_old_words(tmp_path, monkeypatch):
    """No `pii_by` before means the older transcript was never checked: `slim mask` must still
    pick the note up, so the append adds counts and partial, never `pii_by`."""
    from slim.capture import pii, transcribe
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter
    from tests.recorder.fakes import _existing
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(pii, "ask_model", lambda window: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="write to jane.doe@example.com, the door code is 4471", model="fake",
        audio_seconds=9.0, wall_seconds=1.0))
    pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[_staged(tmp_path, "a-002.webm")], vault=tmp_path)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm["pii_masked"] == ["email:1"] and fm["pii_partial"] == "true"
    assert "pii_by" not in fm


def test_an_append_onto_a_masked_note_keeps_it_current_and_partial(tmp_path, monkeypatch):
    from slim.capture import pii, transcribe
    from slim.recorder import pipeline
    from slim.vault.frontmatter import parse_frontmatter, set_frontmatter
    from tests.recorder.fakes import _existing
    r = _existing(tmp_path, monkeypatch)
    r.note.write_text(set_frontmatter(r.note.read_text(encoding="utf-8"), {
        "pii_masked": ["ssn:1"], "pii_by": "pii-0", "pii_partial": "true"}), encoding="utf-8")
    monkeypatch.setattr(pii, "ask_model", lambda window: None)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="nothing to hide here", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[_staged(tmp_path, "a-002.webm")], vault=tmp_path)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm["pii_masked"] == ["ssn:1"] and fm["pii_by"] == pii.VERSION
    assert fm["pii_partial"] == "true", "the note's own partial survives a clean append"
