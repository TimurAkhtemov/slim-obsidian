"""A recording in segments: the resume, the append to a filed note, the paused transcript,
the per-segment cache, and the label a one-sided segment carries beside a labelled one."""
import json
from pathlib import Path

import pytest

from tests.record_fakes import (  # noqa: F401
    NOW, _archive_in_tmp, _draft, _existing, _fakes, _segments_tx, _staged)


def test_a_resumed_recording_is_one_note_from_several_segments(tmp_path, monkeypatch):
    """A second MediaRecorder session writes its own container header, so appending to one
    .webm risks a file nothing can play. Segments sidestep that: several audio files, one
    recording id, one note, transcripts joined in order."""
    from slim import record_pipeline
    from slim.capture import transcribe
    seen = []

    def fake_tx(p, model=None):
        seen.append(Path(p).name)
        return transcribe.Transcript(text=f"part {len(seen)}", model="fake",
                                     audio_seconds=60.0, wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
    _fakes(monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)

    out = record_pipeline.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                                   _staged(tmp_path, "r-002.webm")],
                                        notes_md="", declared_type="lecture", title="T", when=NOW,
                                        vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert seen == ["r-001.webm", "r-002.webm"], "in the order they recorded them"
    assert "part 1\n\npart 2" in body, "one transcript, no seam marker"


def test_every_segment_is_archived_and_named_in_the_note(tmp_path, monkeypatch):
    from slim import record_pipeline
    from slim.vault.frontmatter import parse_frontmatter
    _fakes(monkeypatch)
    out = record_pipeline.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                                   _staged(tmp_path, "r-002.webm")],
                                        notes_md="", declared_type="lecture", title="T", when=NOW,
                                        vault=tmp_path)
    fm, _ = parse_frontmatter((tmp_path / out["note"]).read_text(encoding="utf-8"))
    assert isinstance(fm.get("audio"), list) and len(fm["audio"]) == 2
    assert len(fm.get("audio_sha256")) == 2
    assert not (tmp_path / "Attachments/_incoming/r-001.webm").exists(), "staging is cleared"


def test_one_segment_still_writes_a_plain_scalar(tmp_path, monkeypatch):
    """300 notes already carry `audio:` as a scalar. A single-segment recording keeps writing
    one, so nothing downstream has to learn a new shape for the common case."""
    from slim import record_pipeline
    from slim.vault.frontmatter import parse_frontmatter
    _fakes(monkeypatch)
    out = record_pipeline.handle_record(audio_rel=_staged(tmp_path), notes_md="",
                                        declared_type="lecture", title="T", when=NOW,
                                        vault=tmp_path)
    fm, _ = parse_frontmatter((tmp_path / out["note"]).read_text(encoding="utf-8"))
    assert isinstance(fm.get("audio"), str)


def test_a_transcribed_segment_is_not_transcribed_twice(tmp_path, monkeypatch):
    """⚠ THE REASON CANCEL IS AFFORDABLE. Without a cache, cancelling after a 40-minute
    transcription makes them pay for that transcription again on resume — which is most of the
    wait they cancelled to avoid."""
    from slim import record_pipeline, record_state
    from slim.capture import transcribe
    calls = []

    def fake_tx(p, model=None):
        calls.append(Path(p).name)
        return transcribe.Transcript(text="the first forty minutes", model="fake",
                                     audio_seconds=2400.0, wall_seconds=90.0)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
    from slim.capture import summarize as summarize_mod
    monkeypatch.setattr(summarize_mod, "summarize",
                        lambda *a, **k: (_ for _ in ()).throw(record_state.RecordCancelled()))
    first = _staged(tmp_path, "r-001.webm")
    record_state._record_progress_start("job-3")
    with pytest.raises(record_state.RecordCancelled):
        record_pipeline.handle_record(audio_rel=first, notes_md="", declared_type="lecture",
                                      title="T", when=NOW, vault=tmp_path, job_id="job-3")
    assert calls == ["r-001.webm"]

    _fakes(monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
    out = record_pipeline.handle_record(audio_rel=[first, _staged(tmp_path, "r-002.webm")],
                                        notes_md="", declared_type="lecture", title="T", when=NOW,
                                        vault=tmp_path)
    assert calls == ["r-001.webm", "r-002.webm"], "the cancelled segment is reused, not redone"
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "the first forty minutes" in body
    assert not list((tmp_path / "Attachments/_incoming").glob("*.json")), "caches are cleaned up"


def test_the_wire_takes_one_path_or_several():
    """An older plugin sends a string. Both shapes reach the same code."""
    from slim import record_pipeline
    one = record_pipeline._record_args({"audio": "a.webm", "notes_md": "", "type": "lecture"})
    many = record_pipeline._record_args(
        {"audio": ["a.webm", "b.webm"], "notes_md": "", "type": "lecture"})
    assert one["audio_rel"] == "a.webm"
    assert many["audio_rel"] == ["a.webm", "b.webm"]
    with pytest.raises(ValueError):
        record_pipeline._record_args({"audio": [], "notes_md": "", "type": "lecture"})


def test_a_segment_still_belongs_to_its_draft(tmp_path, monkeypatch):
    """⚠ SHIPPED BROKEN, 2026-08-26, and it killed the first real recording after the change.

    `_draft_source` proves the audio and the draft are the same recording by comparing the
    audio filename's stem with the draft's id. Segments made the stem `<id>-001`, which never
    equals `<id>`, so EVERY recording died with "audio and draft recording ids do not match"
    — and Try again could not help, because nothing about it was transient.

    The real lesson is about the test suite, not the parse: the multi-segment tests all passed
    `notes_md`, and the draft tests all used one segment, so the ONE combination the plugin
    actually sends was the one combination nothing covered. Two guards were widened for
    segments; this was the third and it was found by them, not by us."""
    from slim import record_pipeline
    _fakes(monkeypatch)
    rid = "20260826T150802-782569ba"
    draft = _draft(tmp_path, recording_id=rid)
    out = record_pipeline.handle_record(
        audio_rel=[_staged(tmp_path, f"{rid}-001.webm"), _staged(tmp_path, f"{rid}-002.webm")],
        draft_rel=draft, declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert out["note"]


def test_a_mismatched_recording_id_is_still_refused(tmp_path, monkeypatch):
    """The guard exists to stop one recording's audio being filed with another's notes. It has
    to keep saying no — widening it for segments must not turn it off."""
    from slim import record_pipeline
    _fakes(monkeypatch)
    draft = _draft(tmp_path, recording_id="20260826T150802-782569ba")
    with pytest.raises(ValueError, match="do not match"):
        record_pipeline.handle_record(
            audio_rel=_staged(tmp_path, "20260826T999999-cafebabe-001.webm"), draft_rel=draft,
            declared_type="lecture", title="T", when=NOW, vault=tmp_path)


def test_the_transcript_so_far_can_be_read_back_without_transcribing(tmp_path, monkeypatch):
    """The paused screen has to show what was captured, or "1 segment saved" is a claim they have
    no way to check — and after a cancel that claim is the whole point. Reads the per-segment
    caches ONLY: asking for a transcript must never start a transcription."""
    from slim import record_pipeline
    from slim.capture import transcribe
    calls = []
    monkeypatch.setattr(transcribe, "transcribe", lambda *a, **k: calls.append(1))
    first, second = _staged(tmp_path, "r-001.webm"), _staged(tmp_path, "r-002.webm")
    (tmp_path / first).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": "the first part", "words": 3}), encoding="utf-8")

    out = record_pipeline.handle_record_transcript(audio_rel=[first, second], vault=tmp_path)

    assert out["text"] == "the first part"
    assert out["pending"] == 1, "the segment with no cache is reported, not invented"
    assert calls == [], "reading never transcribes"


def test_resuming_a_filed_note_appends_audio_and_words(tmp_path, monkeypatch):
    """Their ask: resume should be available when the review card is waiting, and on any note
    that is already finished. The note exists, its audio is archived and staging is clear, so
    this is an APPEND — not a second recording, and not a rewrite."""
    from slim import record_pipeline
    from slim.capture import transcribe
    from slim.vault.frontmatter import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="and ten minutes more", model="fake", audio_seconds=600.0, wall_seconds=5.0))

    out = record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                               audio_rel=[_staged(tmp_path, "more-001.webm")],
                                               vault=tmp_path)

    body = r.note.read_text(encoding="utf-8")
    assert "verbatim words" in body and "and ten minutes more" in body
    fm, _ = parse_frontmatter(body)
    assert isinstance(fm.get("audio"), list) and len(fm["audio"]) == 2
    assert out["words"] == 4
    assert not (tmp_path / "Attachments/_incoming/more-001.webm").exists()


def test_an_appended_segment_leaves_the_summary_alone(tmp_path, monkeypatch):
    """Their call: the summary keeps describing what it described until they presse Retry summary.
    Nothing that costs a minute of model time runs unasked."""
    from slim import record_pipeline
    from slim.capture import transcribe
    r = _existing(tmp_path, monkeypatch)     # written with summary_md="s"
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="more words", model="fake", audio_seconds=60.0, wall_seconds=1.0))
    record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                         audio_rel=[_staged(tmp_path, "more-001.webm")],
                                         vault=tmp_path)
    assert "\ns\n" in r.note.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_appending_silence_is_refused_and_keeps_the_audio(tmp_path, monkeypatch):
    """Same rule as the recording path: a transcript with no words is refused, not filed."""
    from slim import record_pipeline, record_state
    from slim.capture import transcribe
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="   ", model="fake", audio_seconds=60.0, wall_seconds=1.0))
    staged = _staged(tmp_path, "more-001.webm")
    with pytest.raises(record_state.NoSpeech, match="no words"):
        record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                             audio_rel=[staged], vault=tmp_path)
    assert (tmp_path / staged).is_file(), "the audio stays staged for a retry"
    assert "and ten minutes" not in r.note.read_text(encoding="utf-8")


def test_appending_silence_is_the_same_quiet_outcome(tmp_path, monkeypatch):
    from slim import record_pipeline, record_state
    from slim.capture import transcribe
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    staged = _staged(tmp_path, "more-001.webm")
    with pytest.raises(record_state.NoSpeech):
        record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                             audio_rel=[staged], vault=tmp_path)
    assert (tmp_path / staged).is_file()


def test_an_append_does_not_clobber_notes_saved_while_it_ran(tmp_path, monkeypatch):
    from slim import record_pipeline, record_review
    from slim.capture import transcribe
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))

    def slow_tx(p, model=None):
        record_review.handle_record_notes(note=note_rel, notes_md="- typed while it ran",
                                          vault=tmp_path)
        return transcribe.Transcript(text="more words", model="fake", audio_seconds=9.0,
                                     wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", slow_tx)

    record_pipeline.handle_record_append(note=note_rel,
                                         audio_rel=[_staged(tmp_path, "more-001.webm")],
                                         vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    assert "- typed while it ran" in body
    assert "more words" in body


def test_cancelling_a_one_segment_resume_actually_cancels(tmp_path, monkeypatch):
    """⚠ Cancel was checked only at the top of the per-segment loop, and a resume normally has
    exactly ONE segment — so the check ran once, before a transcription that can take forty
    minutes, and never again. They were told "nothing is written or deleted" and the note grew."""
    from slim import record_pipeline, record_state
    from slim.capture import transcribe

    def cancel_during(p, model=None):
        record_state.request_record_cancel("job-append")
        return transcribe.Transcript(text="more words", model="fake", audio_seconds=9.0,
                                     wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", cancel_during)
    r = _existing(tmp_path, monkeypatch)
    before = r.note.read_text(encoding="utf-8")
    staged = _staged(tmp_path, "more-001.webm")
    record_state._record_progress_start("job-append")

    with pytest.raises(record_state.RecordCancelled):
        record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                             audio_rel=[staged], job_id="job-append",
                                             vault=tmp_path)

    assert r.note.read_text(encoding="utf-8") == before, "nothing written, as promised"
    assert (tmp_path / staged).is_file(), "nothing deleted, as promised"


def test_append_never_deletes_the_audio_it_just_recorded(tmp_path, monkeypatch):
    """⚠ The cleanup unlinked every input path. Point `audio` at a file that is ALREADY its
    content-addressed archive and `archive_audio` returns that same path — so the note ends up
    naming a recording the same request deleted."""
    from slim import record_pipeline
    from slim.capture import transcribe
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="more words", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    archived = tmp_path / "Attachments" / "Recordings" / "keep.webm"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(b"the only copy")

    with pytest.raises(ValueError, match="staged"):
        record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                             audio_rel=[str(archived.relative_to(tmp_path))],
                                             vault=tmp_path)
    assert archived.is_file(), "an archived recording is never an input to append"


def test_every_segment_must_belong_to_the_same_recording(tmp_path, monkeypatch):
    """Only the FIRST segment was checked against the draft id, so two different recordings
    could be merged into one note and both staged files deleted."""
    from slim import record_pipeline
    _fakes(monkeypatch)
    draft = _draft(tmp_path, recording_id="alpha")
    with pytest.raises(ValueError, match="do not match"):
        record_pipeline.handle_record(audio_rel=[_staged(tmp_path, "alpha-001.webm"),
                                                 _staged(tmp_path, "bravo-001.webm")],
                                      draft_rel=draft, declared_type="lecture", title="T", when=NOW,
                                      vault=tmp_path)


def test_a_transcript_cache_is_bound_to_the_audio_it_transcribed(tmp_path, monkeypatch):
    """The cache was trusted by PATH. Different bytes at the same staged name returned the old
    words with no transcription at all."""
    from slim import record_pipeline
    from slim.capture import audio, transcribe
    calls = []

    def tx(p, model=None):
        calls.append(1)
        return transcribe.Transcript(text=f"take {len(calls)}", model="fake",
                                     audio_seconds=1.0, wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", tx)
    monkeypatch.setattr(audio, "normalize_container", lambda p: None)
    staged = tmp_path / "Attachments" / "_incoming" / "same-001.webm"
    staged.parent.mkdir(parents=True, exist_ok=True)

    staged.write_bytes(b"first audio")
    first, cached = record_pipeline._transcribe_segment(staged, transcribe)
    assert first["text"] == "take 1" and not cached
    again, cached = record_pipeline._transcribe_segment(staged, transcribe)
    assert again["text"] == "take 1" and cached, "same bytes, same words, no second pass"

    staged.write_bytes(b"DIFFERENT audio")
    third, cached = record_pipeline._transcribe_segment(staged, transcribe)
    assert third["text"] == "take 2" and not cached, "different bytes are a different recording"


def test_speaker_provenance_survives_the_segment_cache(tmp_path, monkeypatch):
    """The cache hands back the words without transcribing. It must hand back how their
    speakers were decided too, or a Try again files a labelled transcript that says nothing
    about where its labels came from."""
    from slim import record_pipeline
    from slim.capture import audio, transcribe

    _fakes(monkeypatch)
    monkeypatch.setattr(audio, "normalize_container", lambda p: p)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="**Me:** so\n\n**Them:** yes", model="fake", audio_seconds=10.0, wall_seconds=1.0,
        speakers_by="channels"))
    staged = tmp_path / "Attachments/_incoming/t-001.webm"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"audio")

    first, cached = record_pipeline._transcribe_segment(staged, transcribe)
    assert (first["speakers_by"], cached) == ("channels", False)
    again, cached = record_pipeline._transcribe_segment(staged, transcribe)
    assert (again["speakers_by"], cached) == ("channels", True)


def test_the_one_speaker_survives_the_segment_cache(tmp_path, monkeypatch):
    from slim import record_pipeline
    from slim.capture import audio, transcribe

    _fakes(monkeypatch)
    monkeypatch.setattr(audio, "normalize_container", lambda p: p)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="and one more thing", model="fake", audio_seconds=10.0, wall_seconds=1.0,
        solo_speaker="Me"))
    staged = tmp_path / "Attachments/_incoming/t-001.webm"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"audio")

    record_pipeline._transcribe_segment(staged, transcribe)
    again, cached = record_pipeline._transcribe_segment(staged, transcribe)
    assert (again["solo_speaker"], cached) == ("Me", True)


def test_a_one_speaker_segment_in_a_labelled_recording_carries_its_label(tmp_path, monkeypatch):
    """Unlabelled, their closing monologue reads as the last speaker still talking."""
    from slim import record_pipeline
    _fakes(monkeypatch)
    _segments_tx(monkeypatch, ("**Me:** so\n\n**Them:** yes", "channels", ""),
                 ("and one more thing", "", "Me"))

    out = record_pipeline.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                                   _staged(tmp_path, "r-002.webm")],
                                        notes_md="", declared_type="meeting-note", title="T",
                                        when=NOW, vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "**Them:** yes\n\n**Me:** and one more thing" in body


def test_the_labelled_segment_can_come_second(tmp_path, monkeypatch):
    from slim import record_pipeline
    _fakes(monkeypatch)
    _segments_tx(monkeypatch, ("they opened", "", "Them"),
                 ("**Me:** so\n\n**Them:** yes", "channels", ""))

    out = record_pipeline.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                                   _staged(tmp_path, "r-002.webm")],
                                        notes_md="", declared_type="meeting-note", title="T",
                                        when=NOW, vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "**Them:** they opened\n\n**Me:** so" in body


def test_a_lecture_in_two_segments_stays_unlabelled(tmp_path, monkeypatch):
    """Every segment one-sided: nothing to tell apart, so no labels at all."""
    from slim import record_pipeline
    _fakes(monkeypatch)
    _segments_tx(monkeypatch, ("part one", "", "Them"), ("part two", "", "Them"))

    out = record_pipeline.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                                   _staged(tmp_path, "r-002.webm")],
                                        notes_md="", declared_type="lecture", title="T", when=NOW,
                                        vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "part one\n\npart two" in body and "**Them:**" not in body


def test_resuming_a_labelled_note_labels_a_one_speaker_segment(tmp_path, monkeypatch):
    from slim import record, record_pipeline
    r = _existing(tmp_path, monkeypatch)
    r.note.write_text(record.append_audio_frontmatter(
        r.note.read_text(encoding="utf-8"), rels=[], digests=[], seconds=0.0,
        speakers_by="channels"), encoding="utf-8")
    _segments_tx(monkeypatch, ("and one more thing", "", "Me"))

    record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                         audio_rel=[_staged(tmp_path, "more-001.webm")],
                                         vault=tmp_path)
    assert "**Me:** and one more thing" in r.note.read_text(encoding="utf-8")


def test_resuming_an_unlabelled_note_adds_no_label(tmp_path, monkeypatch):
    from slim import record_pipeline
    r = _existing(tmp_path, monkeypatch)
    _segments_tx(monkeypatch, ("and ten minutes more", "", "Them"))

    record_pipeline.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                         audio_rel=[_staged(tmp_path, "more-001.webm")],
                                         vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    assert "and ten minutes more" in body and "**Them:**" not in body


def test_the_paused_transcript_labels_segments_the_same_way(tmp_path):
    from slim import record_pipeline
    first, second = _staged(tmp_path, "r-001.webm"), _staged(tmp_path, "r-002.webm")
    (tmp_path / first).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": "**Me:** so\n\n**Them:** yes", "speakers_by": "channels"}),
        encoding="utf-8")
    (tmp_path / second).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": "and more", "solo_speaker": "Me"}), encoding="utf-8")

    out = record_pipeline.handle_record_transcript(audio_rel=[first, second], vault=tmp_path)
    assert out["text"] == "**Me:** so\n\n**Them:** yes\n\n**Me:** and more"
