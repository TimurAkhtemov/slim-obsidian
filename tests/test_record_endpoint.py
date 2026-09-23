"""The recorder endpoints, with the slow parts faked.

Transcription and the dense model have their own suites. What matters here is the PROMISE:
a recording that reaches the server is never lost. A summary or title failure still files the
note; a transcription failure leaves the draft and audio available for retry, because silently
filing an empty transcript is precisely the Notion failure this feature exists to remove.
"""
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

NOW = datetime(2026, 8, 4, 14, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _archive_in_tmp(tmp_path, monkeypatch):
    from slim import inbox
    monkeypatch.setattr(inbox, "AUDIO_ARCHIVE", tmp_path / "Attachments" / "Recordings")


def _staged(tmp_path: Path, name: str = "t.webm") -> str:
    p = tmp_path / "Attachments" / "_incoming" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-audio-bytes")
    return str(p.relative_to(tmp_path))


def _draft(tmp_path: Path, *, recording_id: str = "t", type_tag: str = "lecture",
           text: str = "- a draft note\n") -> str:
    parent = tmp_path / ("Journal" if type_tag == "journal" else "Capture/_unfiled")
    p = parent / f"2026-08-23--recording--{recording_id}.slim-draft.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p.relative_to(tmp_path))


def _fakes(monkeypatch, *, transcript="the kernel trick", summary="Derived the kernel trick.",
           dest="Notes/school/CS201 ML",
           transcribe_raises=False, summarize_raises=False):
    from slim import suggest, summarize, transcribe

    def fake_tx(p, model=None):
        if transcribe_raises:
            raise RuntimeError("ASR exploded")
        return transcribe.Transcript(text=transcript, model="fake",
                                     audio_seconds=10.0, wall_seconds=1.0)

    def fake_sum(*a, **k):
        if summarize_raises:
            raise RuntimeError("model is down")
        return summarize.Summary(overview=summary)

    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
    monkeypatch.setattr(summarize, "summarize", fake_sum)
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: suggest.Card(
        dest_dir=dest, confident_depth=2, candidates=["CS201 ML", "CS202 RL"],
        type_tag=k.get("declared_type") or "lecture", topics=["school"]))


# --- POST /api/record -------------------------------------------------------------------


def test_the_http_contract_accepts_exactly_one_notes_source():
    from slim import chat

    legacy = chat._record_args({"audio": "a.webm", "notes_md": "- old client",
                                "type": "lecture", "title": "T"})
    assert legacy["notes_md"] == "- old client" and legacy["draft_rel"] is None

    draft = chat._record_args({"audio": "a.webm", "draft": "d.slim-draft.md",
                               "type": "lecture", "title": "T"})
    assert draft["draft_rel"] == "d.slim-draft.md" and draft["notes_md"] is None

    with pytest.raises(ValueError, match="exactly one"):
        chat._record_args({"audio": "a.webm", "draft": "d.slim-draft.md",
                           "notes_md": "two truths", "type": "lecture"})


def test_a_saved_draft_is_the_only_source_of_typed_notes(tmp_path, monkeypatch):
    from slim import chat

    notes = "  # Live notes\n\n- keep trailing spaces  \n\n![[Attachments/Recorder/t/paste.png]]\n"
    draft = _draft(tmp_path, text=notes)
    _fakes(monkeypatch)
    out = chat.handle_record(audio_rel=_staged(tmp_path), draft_rel=draft,
                             notes_md=None, declared_type="lecture", title="SVMs",
                             when=NOW, vault=tmp_path)

    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    start = body.index("## Notes\n\n") + len("## Notes\n\n")
    end = body.index("\n## Transcript", start)
    assert body[start:end] == notes


def test_inline_draft_anchor_is_not_mistaken_for_their_notes(tmp_path, monkeypatch):
    from slim import chat

    notes = "## Follow-up\n\n- exact human note\n"
    draft = _draft(tmp_path, text="````slim-meeting\n\n````\n\n" + notes)
    _fakes(monkeypatch)
    out = chat.handle_record(audio_rel=_staged(tmp_path), draft_rel=draft,
                             declared_type="lecture", title="SVMs",
                             when=NOW, vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert body.count("slim-meeting") == 1
    assert notes.rstrip() in body


def test_a_successful_draft_recording_removes_only_its_staging_files(tmp_path, monkeypatch):
    from slim import chat

    staged = _staged(tmp_path)
    draft = _draft(tmp_path)
    image = tmp_path / "Attachments/Recorder/t/paste.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    _fakes(monkeypatch)

    out = chat.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                             declared_type="lecture", title="SVMs", when=NOW, vault=tmp_path)

    assert not (tmp_path / staged).exists()
    assert not (tmp_path / draft).exists()
    assert image.read_bytes() == b"png"
    assert (tmp_path / out["note"]).exists()


def test_a_failed_draft_transcript_keeps_both_sources_for_retry(tmp_path, monkeypatch):
    from slim import chat

    staged = _staged(tmp_path)
    draft = _draft(tmp_path)
    _fakes(monkeypatch, transcribe_raises=True)

    with pytest.raises(ValueError, match="transcription failed"):
        chat.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                           declared_type="lecture", title="T", when=NOW, vault=tmp_path)

    assert (tmp_path / staged).exists()
    assert (tmp_path / draft).read_text(encoding="utf-8") == "- a draft note\n"


@pytest.mark.parametrize(
    ("type_tag", "draft_type", "message"),
    [("journal", "lecture", "Journal"), ("lecture", "journal", "Capture/_unfiled")],
)
def test_a_draft_cannot_cross_the_journal_floor(tmp_path, monkeypatch,
                                                type_tag, draft_type, message):
    from slim import chat

    _fakes(monkeypatch)
    with pytest.raises(ValueError, match=message):
        chat.handle_record(audio_rel=_staged(tmp_path),
                           draft_rel=_draft(tmp_path, type_tag=draft_type), notes_md=None,
                           declared_type=type_tag, title="T", when=NOW, vault=tmp_path)


def test_audio_and_draft_must_describe_the_same_recording(tmp_path, monkeypatch):
    from slim import chat

    _fakes(monkeypatch)
    staged = _staged(tmp_path, "audio-id.webm")
    draft = _draft(tmp_path, recording_id="draft-id")
    with pytest.raises(ValueError, match="recording id"):
        chat.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                           declared_type="lecture", title="T", when=NOW, vault=tmp_path)

    assert (tmp_path / staged).exists() and (tmp_path / draft).exists()


def test_a_final_note_collision_keeps_the_draft_and_staged_audio(tmp_path, monkeypatch):
    from slim import chat, inbox

    staged = _staged(tmp_path)
    draft = _draft(tmp_path)
    _fakes(monkeypatch)
    digest = inbox._sha256(tmp_path / staged)
    victim = tmp_path / "Notes/school/CS201 ML" / f"2026-08-04--SVMs--{digest[:8]}.md"
    victim.parent.mkdir(parents=True)
    victim.write_text("I was here first", encoding="utf-8")

    with pytest.raises(ValueError, match="overwrite"):
        chat.handle_record(audio_rel=staged, draft_rel=draft, notes_md=None,
                           declared_type="lecture", title="SVMs", when=NOW, vault=tmp_path)

    assert victim.read_text(encoding="utf-8") == "I was here first"
    assert (tmp_path / staged).exists() and (tmp_path / draft).exists()

def test_the_note_is_filed_and_the_card_describes_it(tmp_path, monkeypatch):
    from slim import chat
    _fakes(monkeypatch)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="- dual form",
                             declared_type="lecture", title="SVMs", when=NOW, vault=tmp_path)
    note = tmp_path / out["note"]
    assert note.exists()
    assert out["card"]["dest_dir"] == "Notes/school/CS201 ML"
    body = note.read_text(encoding="utf-8")
    assert "- dual form" in body and "the kernel trick" in body


def test_the_card_carries_the_folders_and_tag_counts_the_ui_needs(tmp_path, monkeypatch):
    """The card is the whole payload — the plugin must not have to make a second call to
    render the tree or show 'this tag would be new'."""
    from slim import chat
    _fakes(monkeypatch)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                            title="SVMs", when=NOW, vault=tmp_path)
    assert isinstance(out["card"]["folders"], list)
    assert isinstance(out["card"]["tag_counts"], dict)


def test_the_summary_pass_thinks(tmp_path, monkeypatch):
    """The one ratified exception to think=False. If this regresses, invention doubles in the
    text they read and trusts, silently."""
    from slim import chat, summarize
    _fakes(monkeypatch)
    seen = {}
    real = summarize.Summary

    def spy(*a, **k):
        seen.update(k)
        return real(overview="s")
    monkeypatch.setattr(summarize, "summarize", spy)
    chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                       title="T", when=NOW, vault=tmp_path)
    assert seen.get("think") is True


def test_record_progress_reports_real_pipeline_and_model_activity(tmp_path, monkeypatch):
    from slim import chat, summarize

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
    chat.handle_record(
        audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture", title="SVMs",
        when=NOW, vault=tmp_path, on_progress=lambda **event: events.append(event),
    )

    stages = [event["stage"] for event in events]
    assert stages[0:3] == ["preparing", "transcribing", "transcribed"]
    assert "summarizing" in stages and "filing" in stages and "saving" in stages
    assert any(event.get("kind") == "thinking" and
               event.get("text") == "identify the decision" for event in events)
    assert any(event.get("kind") == "draft" and "kernel trick" in event.get("text", "")
               for event in events)
    summary_done = next(event for event in events if event["stage"] == "summarized")
    assert summary_done["metrics"]["summary_output_tokens"] == 20


def test_a_dead_model_still_files_the_recording(tmp_path, monkeypatch):
    """⚠ THE CORE PROMISE. Summarization failing must cost a summary, never a recording."""
    from slim import chat
    _fakes(monkeypatch, summarize_raises=True)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="- my notes",
                             declared_type="lecture", title="T", when=NOW, vault=tmp_path)
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
    from slim import chat
    _fakes(monkeypatch, transcribe_raises=True)
    staged = _staged(tmp_path)
    with pytest.raises(ValueError, match="transcription failed"):
        chat.handle_record(audio_rel=staged, notes_md="- my notes",
                           declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert (tmp_path / staged).exists(), "the audio must stay staged so a retry is possible"
    assert not list(tmp_path.rglob("*.md")), "no half-note may be left behind"


def test_a_staged_file_outside_the_vault_is_refused(tmp_path, monkeypatch):
    """`audio` arrives over HTTP. A traversal would let a caller name any file on the machine
    and have its bytes copied into the vault and transcribed."""
    from slim import chat
    _fakes(monkeypatch)
    with pytest.raises(ValueError):
        chat.handle_record(audio_rel="../../etc/passwd", notes_md="", declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path)


# --- POST /api/record/apply -------------------------------------------------------------

def _existing(tmp_path, monkeypatch, dest="Capture/_unfiled"):
    from slim import record
    src = tmp_path / "Attachments" / "_incoming" / "a.webm"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"audio")
    return record.write_recording(
        audio_src=src, title="T", when=NOW, type_tag="lecture", transcript="verbatim words",
        notes_md="", summary_md="s", topics=["school"],
        dest_dir=dest, filed_by_slim=True, vault=tmp_path)


def test_an_edit_moves_the_note_and_marks_the_fields_as_theirs(tmp_path, monkeypatch):
    """⚠ Writing the new value is not enough. A model pass re-judges what it INFERRED but never
    overrides a declared value, so a card edit that only writes the value gets reverted by a
    later pass — they correct the same note twice and concludes correcting does not work."""
    from slim import chat
    from slim.chunk import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    out = chat.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                   dest_dir="Notes/school/CS201 ML", type_tag="lecture",
                                   topics=["school", "svm"], summary_md="s",
                                   vault=tmp_path)
    moved = tmp_path / out["note"]
    assert moved.exists() and not r.note.exists()
    fm, _ = parse_frontmatter(moved.read_text(encoding="utf-8"))
    assert "filed_by" not in fm                 # absence means THEY declared it
    assert fm.get("subject_by") == "declared"
    assert fm.get("review_status") == "complete"


def test_approval_without_edits_completes_review_and_owns_the_filing(tmp_path, monkeypatch):
    """Approving the card as it stands is agreeing with where SLIM put it. `filed_by: slim`
    left behind made "SLIM guessed" and "they agreed" the same note. Nothing else changes:
    no move, no rewrite of the body."""
    from slim import chat
    from slim.chunk import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    before = r.note.read_text(encoding="utf-8")
    before_fm, _ = parse_frontmatter(before)
    assert before_fm.get("filed_by") == "slim"

    out = chat.handle_record_review(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)

    after = (tmp_path / out["note"]).read_text(encoding="utf-8")
    after_fm, _ = parse_frontmatter(after)
    assert (tmp_path / out["note"]) == r.note
    assert after.replace("review_status: complete", "review_status: pending") == \
        before.replace("filed_by: slim\n", "")
    expected = {k: v for k, v in before_fm.items() if k != "filed_by"}
    assert after_fm == {**expected, "review_status": "complete"}


def test_the_transcript_is_untouched_by_an_edit(tmp_path, monkeypatch):
    """The card edits interpretations. The raw record is never rewritten — no exceptions."""
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    chat.handle_record_apply(note=str(r.note.relative_to(tmp_path)), dest_dir="Capture/_unfiled",
                             type_tag="idea", topics=["x"], summary_md="s",
                             vault=tmp_path)
    assert "verbatim words" in r.note.read_text(encoding="utf-8")


def test_review_edits_persist_both_summary_and_notes(tmp_path, monkeypatch):
    from slim import chat, record
    r = _existing(tmp_path, monkeypatch)
    chat.handle_record_apply(
        note=str(r.note.relative_to(tmp_path)), dest_dir="Capture/_unfiled",
        type_tag="lecture", topics=["school"],
        summary_md="Edited AI summary.", notes_md="- note added during review",
        vault=tmp_path)
    text = r.note.read_text(encoding="utf-8")
    assert "Edited AI summary." in text
    assert "- note added during review" in text
    block = record.split_meeting_block(text)
    assert block is not None
    assert block.body.rstrip().endswith("verbatim words")


def test_applying_never_silently_overwrites_another_note(tmp_path, monkeypatch):
    """Both movers grew this guard on 2026-08-04 after two notes could resolve to one
    destination and one was lost. A third mover must not reintroduce it."""
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    victim = tmp_path / "Notes/school/CS201 ML" / r.note.name
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("I was here first", encoding="utf-8")
    with pytest.raises(ValueError):
        chat.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                 dest_dir="Notes/school/CS201 ML", type_tag="lecture",
                                 topics=[], summary_md="s", vault=tmp_path)
    assert victim.read_text(encoding="utf-8") == "I was here first"


def test_a_missing_audio_file_fails_before_any_model_call(tmp_path, monkeypatch):
    """A bad path should cost nothing. Without an early check it burns a transcription attempt
    and a dense-model call before dying at write time."""
    from slim import chat, transcribe
    called = []
    monkeypatch.setattr(transcribe, "transcribe",
                        lambda *a, **k: called.append(1))
    with pytest.raises(FileNotFoundError):
        chat.handle_record(audio_rel="Attachments/_incoming/nope.webm", notes_md="",
                           declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert called == []


def test_a_failed_transcription_does_not_load_the_dense_model(tmp_path, monkeypatch):
    """Classifying an empty string would load ~20 GB and EVICT the resident chat model, so the
    cost of a failed ASR run would land on whatever they did next. It still must not be called —
    the request now fails before it gets there."""
    from slim import chat, suggest
    _fakes(monkeypatch, transcribe_raises=True)
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: pytest.fail("must not be called"))
    with pytest.raises(ValueError):
        chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path)


def test_a_vault_reached_through_a_symlink_still_reports_relative_paths(tmp_path, monkeypatch):
    """⚠ Regression, found by a live run and NOT by the tests above: pytest's `tmp_path` is
    already resolved, so a resolved-vs-unresolved path mismatch is invisible here unless a
    symlink is introduced deliberately. On macOS `/var` -> `/private/var` makes that mismatch
    the DEFAULT for any temp-dir vault, and it made apply move the note correctly and then
    return a 400 — the caller sees a failure for work that actually happened."""
    from slim import chat, inbox

    real = tmp_path / "real_vault"
    (real / "Attachments" / "_incoming").mkdir(parents=True)
    (real / "Notes" / "school").mkdir(parents=True)
    link = tmp_path / "linked_vault"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(inbox, "AUDIO_ARCHIVE", link / "Attachments" / "Recordings")

    _fakes(monkeypatch, dest="Notes/school")
    (real / "Attachments" / "_incoming" / "s.webm").write_bytes(b"bytes")

    out = chat.handle_record(audio_rel="Attachments/_incoming/s.webm", notes_md="",
                             declared_type="lecture", title="T", when=NOW, vault=link)
    assert not Path(out["note"]).is_absolute()      # relative, not an absolute leak
    assert not Path(out["audio"]).is_absolute()

    moved = chat.handle_record_apply(note=out["note"], dest_dir="Notes/school",
                                     type_tag="lecture", topics=["school"],
                                     summary_md="s", vault=link)
    assert not Path(moved["note"]).is_absolute()


def test_the_note_gets_the_whole_summary_not_just_the_overview(tmp_path, monkeypatch):
    """⚠ Found by an end-to-end run, not by review. handle_record wrote only `s.overview`, so
    key points, decisions, action items and open questions were generated and then thrown away.
    It read as the model being lazy when the caller was discarding its work."""
    from slim import chat, suggest, summarize, transcribe
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="we discussed the export", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary(
        overview="A short review.",
        key_points=[{"topic": "Refresh cadence",
                     "points": ["Tableau extracts refresh hourly, too slow for the monthly report"]}],
        action_items=[{"task": "Send the tables"}],
        open_questions=["How leadership is classified"]))
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: suggest.Card(
        dest_dir="Capture/work", confident_depth=1, candidates=[], type_tag="meeting-note",
        topics=[]))

    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="meeting-note",
                             title="T", when=NOW, vault=tmp_path)
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
    from slim import chat, enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: ({"title": "Kernel methods and the dual form",
                                          "type_tag": "lecture", "topics": [], "project": ""}, {}))
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="", when=NOW, vault=tmp_path)
    assert "Kernel-methods" in out["note"]
    assert "Recording 2026" not in (tmp_path / out["note"]).read_text(encoding="utf-8")


def test_a_title_they_typed_is_never_overridden(tmp_path, monkeypatch):
    """Their words are testimony here exactly as a spoken type is."""
    from slim import chat, enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: pytest.fail("must not be called when they gave a title"))
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="My Own Title", when=NOW, vault=tmp_path)
    assert "My-Own-Title" in out["note"]


def test_a_failed_title_call_still_files_the_recording(tmp_path, monkeypatch):
    from slim import chat, enrich
    _fakes(monkeypatch)
    def boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(enrich, "enrich", boom)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="", when=NOW, vault=tmp_path)
    assert (tmp_path / out["note"]).exists()


def test_editing_the_title_renames_the_file_and_keeps_date_and_digest(tmp_path, monkeypatch):
    """The filename is what they read in the file tree and the quick switcher. A note retitled
    only in frontmatter is still wrongly named where it counts."""
    from slim import chat
    from slim.chunk import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    date, _slug, digest = r.note.stem.split("--")
    out = chat.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                   dest_dir="Capture/_unfiled", type_tag="lecture",
                                   topics=[], summary_md="s",
                                   title="Contoso Screen", vault=tmp_path)
    new = tmp_path / out["note"]
    assert new.exists() and not r.note.exists()
    assert new.stem.startswith(date) and new.stem.endswith(digest)   # both preserved
    assert "Contoso-Screen" in new.name
    fm, _ = parse_frontmatter(new.read_text(encoding="utf-8"))
    assert fm.get("title") == "Contoso Screen"


# --- indexing ----------------------------------------------------------------------------

def test_a_filed_recording_is_indexed_so_the_brain_can_reach_it(tmp_path, monkeypatch):
    """Filing makes it findable in OBSIDIAN. It does not reach ask, search, the MCP tools, the
    chat vault lane, or reflect — all of which read SQLite. ⚠ reflect._journals() selects
    type='journal' FROM THE DATABASE, so an unindexed journal is skipped by the nightly
    synthesis silently."""
    from slim import chat
    _fakes(monkeypatch)
    called = {}
    monkeypatch.setattr(chat, "_index_now", lambda v, note: called.update(vault=v, note=note) or {})
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="meeting-note",
                             title="T", when=NOW, vault=tmp_path)
    assert called.get("vault") == tmp_path
    assert called.get("note") == out["note"]


def test_the_card_reindexes_after_it_commits(tmp_path, monkeypatch):
    """The card changes both content and path, so the index must be refreshed after it, not
    only after the initial filing."""
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(chat, "_index_now",
                        lambda v, note, **kw: called.append((v, note, kw.get("moved_from"))) or {})
    original = str(r.note.relative_to(tmp_path))
    moved = chat.handle_record_apply(note=original, dest_dir="Capture/work",
                                     type_tag="lecture", topics=[], summary_md="s",
                                     vault=tmp_path)
    # The MOVED path, and where it came from: the index keeps the note's id through the move.
    assert called == [(tmp_path, moved["note"], original)]


def test_filing_keeps_the_source_id_a_copilot_chat_was_opened_on(tmp_path, monkeypatch):
    """The copilot can be opened on the draft before the card is applied, and its threads are
    keyed by source id. Apply changes content AND path in one ingest — the one sequence the
    hash rename rule cannot follow — so without a hint the note is re-minted and every chat
    on it is orphaned (2026-09-22)."""
    from slim import chat, db, embed, ingest
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(embed, "embed_source", lambda con, sid: 0)
    monkeypatch.setattr(chat, "_schedule_sweep", lambda vault: None)
    original = str(r.note.relative_to(tmp_path))
    con = db.connect()
    draft_id = ingest.ingest_note(con, tmp_path, original)["id"]
    con.close()
    out = chat.handle_record_apply(note=original, dest_dir="Notes/school", type_tag="lecture",
                                   topics=["school"], summary_md="new summary", vault=tmp_path)
    con = db.connect()
    rows = con.execute("SELECT id, path FROM sources WHERE deleted=0").fetchall()
    con.close()
    assert [(row["id"], row["path"]) for row in rows] == [(draft_id, out["note"])]


def test_a_broken_index_never_costs_the_recording(tmp_path, monkeypatch):
    """A note that is filed but not indexed is recoverable with `slim ingest`. One that failed
    to file is not — so indexing must never be able to take the recording down with it."""
    from slim import chat, db
    _fakes(monkeypatch)
    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(db, "connect", boom)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="- notes",
                             declared_type="meeting-note", title="T", when=NOW, vault=tmp_path)
    assert (tmp_path / out["note"]).exists()
    assert "- notes" in (tmp_path / out["note"]).read_text(encoding="utf-8")


# --- titles (2026-08-23) ---------------------------------------------------------------------
# `cade09a` added "write a real title from the transcript when they typed none" and it was
# UNREACHABLE for two weeks: the plugin sent `Recording <stamp>` for a blank box and the route
# substituted "Recording" too, so every recorded note was named for the clock.

def test_an_untitled_recording_is_named_from_its_transcript(tmp_path, monkeypatch):
    from slim import chat, enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: ({"title": "Kernel trick and the dual form"}, {}))
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="", when=NOW, vault=tmp_path)
    assert out["title"] == "Kernel trick and the dual form"


def test_a_title_they_typed_is_never_overwritten(tmp_path, monkeypatch):
    from slim import chat, enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: pytest.fail("the model must not retitle their recording"))
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="SVMs", when=NOW, vault=tmp_path)
    assert out["title"] == "SVMs"


def test_the_timestamp_name_survives_only_as_a_last_resort(tmp_path, monkeypatch):
    """They typed no title and the titling model was down. The clock is all that is left — but it
    is the THIRD answer, not the first.

    ⚠ A FAILED TRANSCRIPT no longer reaches here at all: it is refused before filing, so the
    only route to a timestamp name is a transcript that worked and a titling call that did not.
    """
    from slim import chat, enrich
    _fakes(monkeypatch)
    monkeypatch.setattr(enrich, "enrich",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model is down")))
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="", when=NOW, vault=tmp_path)
    assert out["title"] == "Recording 2026-08-04 1430"


def test_their_title_is_passed_to_the_card_but_a_generated_one_is_not(tmp_path, monkeypatch):
    """⚠ A model-written title is derived from the same transcript the card already reads, so
    feeding it back adds no evidence and lets one guess reinforce the next."""
    from slim import chat, enrich, suggest
    _fakes(monkeypatch)
    seen = []
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: (
        seen.append(k.get("title")),
        suggest.Card(dest_dir="Notes/school", confident_depth=1, candidates=[],
                     type_tag="lecture", topics=[]))[1])
    monkeypatch.setattr(enrich, "enrich", lambda *a, **k: ({"title": "Generated Name"}, {}))

    chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                       title="My Own Title", when=NOW, vault=tmp_path)
    chat.handle_record(audio_rel=_staged(tmp_path, "u.webm"), notes_md="",
                       declared_type="lecture", title="", when=NOW, vault=tmp_path)
    assert seen == ["My Own Title", ""]


def test_a_silent_recording_is_refused_the_same_way(tmp_path, monkeypatch):
    """Silence reaches the same worthless note by a different route — a muted mic, the wrong
    input. Nothing is written; ⚠ since 2026-08-26 it is `NoSpeech`, not a failure, because
    saying so LOUDLY was the friction they objected to."""
    from slim import chat
    _fakes(monkeypatch, transcript="   ")
    staged = _staged(tmp_path)
    with pytest.raises(chat.NoSpeech, match="no words"):
        chat.handle_record(audio_rel=staged, notes_md="", declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path)
    assert (tmp_path / staged).exists()


def test_a_failed_SUMMARY_still_files_the_note(tmp_path, monkeypatch):
    """⚠ The line that matters: a summary DECORATES a recording that is already safe and
    findable. Only the transcript is the artifact."""
    from slim import chat
    _fakes(monkeypatch, summarize_raises=True)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="T", when=NOW, vault=tmp_path)
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
    from slim import chat, enrich, llm, transcribe

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

    chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                       title="", when=NOW, vault=tmp_path)   # ⚠ no title → enrich runs

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
    from slim import chat, enrich, llm, suggest, summarize, transcribe
    seen = {}
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="the kernel trick", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: (
        seen.update({"summarize": k}), summarize.Summary(overview="s"))[1])
    monkeypatch.setattr(enrich, "enrich", lambda *a, **k: (
        seen.update({"enrich": k}), ({"title": "Kernel methods", "type_tag": "lecture",
                                      "topics": [], "project": ""}, {}))[1])
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: (
        seen.update({"card": k}), suggest.Card(dest_dir="Capture/school", confident_depth=1,
                                               candidates=[], type_tag="lecture",
                                               topics=[]))[1])
    chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                       title="", when=NOW, vault=tmp_path)
    assert llm.RECORD_MODEL == llm.MODEL                      # the decision, stated once
    assert seen["summarize"]["model"] == llm.RECORD_MODEL
    assert seen["summarize"]["num_ctx"] == llm.RESIDENT_CTX
    assert seen["enrich"]["model"] == llm.RECORD_MODEL
    assert seen["enrich"]["num_ctx"] == llm.RESIDENT_CTX
    assert seen["card"]["model"] == llm.RECORD_MODEL


def test_their_notes_reach_the_summarizer(tmp_path, monkeypatch):
    """They tell it what they noticed and how they name things. The transcript stays the source."""
    from slim import chat, summarize
    seen = {}
    _fakes(monkeypatch)
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: (
        seen.update(k), summarize.Summary(overview="s"))[1])
    chat.handle_record(audio_rel=_staged(tmp_path), notes_md="- dual form\n- RBF case",
                       declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert seen["notes"] == "- dual form\n- RBF case"


def test_a_lecture_carries_no_open_questions_section(tmp_path, monkeypatch):
    """The summarizer reads the transcript, so for a lecture `open_questions` can only ever
    hold the LECTURER's questions — rhetorical ones they pose and answers, examples on a slide.
    Rendered under "Open questions" they read as the owner's confusions (2026-08-26, NLP lecture).
    A meeting's unresolved question is real information and stays."""
    from slim import chat, suggest, summarize, transcribe
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="does a three year old know english", model="fake", audio_seconds=9.0,
        wall_seconds=1.0))
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary(
        overview="A lecture.",
        key_points=[{"topic": "Planned languages",
                     "points": ["Non-natural languages are planned"]}],
        open_questions=["Does a three-year-old know English?"]))
    monkeypatch.setattr(suggest, "build_card", lambda *a, **k: suggest.Card(
        dest_dir="Capture/school", confident_depth=1, candidates=[], type_tag="lecture",
        topics=[]))
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="T", when=NOW, vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "## Key points" in body
    assert "## Open questions" not in body
    assert "three-year-old" not in body


# --- 2026-08-26: the request indexes ITS note; the vault sweep runs in the background -------

def test_the_request_indexes_only_its_note_and_sweeps_the_vault_afterwards(tmp_path, monkeypatch):
    """Measured 2026-08-26: the full pass inside the request embedded 1,287 fragments for a
    ~50-fragment note — a backlog from 11 other edited notes — and the recording waited for
    it. `ingest_note` + `embed_source` share the sweep's code, so the docstring's old
    objection (a second, less-tested copy) no longer holds."""
    from slim import chat, embed as embed_mod, ingest as ingest_mod
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
    monkeypatch.setattr(chat, "_schedule_sweep", lambda vault: scheduled.append(vault))
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="meeting-note",
                             title="T", when=NOW, vault=tmp_path)
    assert seen["note"] == [out["note"]]
    assert seen["source"] == ["src-1"]
    assert seen["sweep_ingest"] == 0                     # nothing vault-wide inside the request
    assert scheduled == [tmp_path]
    chat._sweep_vault(tmp_path)                          # what the scheduled thread runs
    assert (seen["sweep_ingest"], seen["sweep_embed"]) == (1, 1)


def test_the_sweep_runs_on_a_daemon_thread_and_sweeps_serialize(tmp_path, monkeypatch):
    import threading
    from slim import chat
    order = []
    def slow_sweep(vault):
        order.append(("start", threading.current_thread().name))
        threading.Event().wait(0.05)
        order.append(("end", threading.current_thread().name))
    monkeypatch.setattr(chat, "_sweep_vault", lambda vault: chat._serialized(slow_sweep, vault))
    first, second = chat._schedule_sweep(tmp_path), chat._schedule_sweep(tmp_path)
    assert first.daemon and second.daemon
    first.join(2); second.join(2)
    assert [kind for kind, _ in order] == ["start", "end", "start", "end"]   # never interleaved


def test_a_broken_single_note_index_never_costs_the_recording(tmp_path, monkeypatch):
    from slim import chat, ingest as ingest_mod
    _fakes(monkeypatch)
    def boom(con, vault, rel):
        raise RuntimeError("index exploded")
    monkeypatch.setattr(ingest_mod, "ingest_note", boom)
    monkeypatch.setattr(chat, "_schedule_sweep", lambda vault: None)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="meeting-note",
                             title="T", when=NOW, vault=tmp_path)
    assert (tmp_path / out["note"]).exists()


# --- 2026-08-26: their notes save on their own, and a summary can be retried ----------------

def test_saving_notes_touches_nothing_but_the_notes(tmp_path, monkeypatch):
    """The finished block now edits their notes in place, so it needs a save path that is NOT
    `apply`: apply rewrites frontmatter, drops `filed_by` and can MOVE the file. Typing a note
    is not a filing decision, and an autosave that files is an autosave that surprises."""
    from slim import chat
    from slim.chunk import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    before = r.note.read_text(encoding="utf-8")

    out = chat.handle_record_notes(note=str(r.note.relative_to(tmp_path)),
                                   notes_md="- typed after the fact", vault=tmp_path)

    after = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert (tmp_path / out["note"]) == r.note, "the note does not move"
    assert "- typed after the fact" in after
    assert "verbatim words" in after                     # the transcript is untouched
    fm_before, _ = parse_frontmatter(before)
    fm_after, _ = parse_frontmatter(after)
    assert fm_after == fm_before, "saving notes is not a filing decision"


def test_retrying_the_summary_rewrites_only_the_summary(tmp_path, monkeypatch):
    """Notion's Retry summary, and their reason for wanting it: iterate on the summarizer
    without re-recording a lecture."""
    from slim import chat, summarize
    r = _existing(tmp_path, monkeypatch)
    seen = {}

    def fake_sum(transcript, **kw):
        seen.update(transcript=transcript, **kw)
        return summarize.Summary(overview="A better summary.",
                                 key_points=[{"topic": "Kernels", "points": ["The dual form."]}])
    monkeypatch.setattr(summarize, "summarize", fake_sum)

    out = chat.handle_record_summary(note=str(r.note.relative_to(tmp_path)),
                                     instructions="organise by concept", vault=tmp_path)

    after = r.note.read_text(encoding="utf-8")
    assert "A better summary." in after and "### Kernels" in after
    assert "verbatim words" in after                     # the raw record survives a retry
    assert seen["transcript"].strip() == "verbatim words"
    assert seen["instructions"] == "organise by concept"
    assert seen["think"] is True                         # the recorder's ratified setting
    assert out["summary"].startswith("A better summary.")


def test_a_failed_retry_keeps_the_summary_it_has(tmp_path, monkeypatch):
    """⚠ THE THIRD TIME THIS RULE HAS HAD TO BE WRITTEN. A recording that reports success while
    losing what it exists to keep is the recorder's worst failure, and a retry that blanks a
    good summary because the model died is the same shape. Refuse; do not write."""
    from slim import chat, summarize
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(summarize, "summarize",
                        lambda *a, **k: summarize.Summary(valid=False, error="ctx_saturated"))
    with pytest.raises(ValueError, match="ctx_saturated"):
        chat.handle_record_summary(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)
    assert "\ns\n" in r.note.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_a_retry_on_a_lecture_still_drops_open_questions(tmp_path, monkeypatch):
    """The lecture rule lives in render(), and the retry path has to read `type` from the note
    to reach it — a rhetorical question the lecturer asked and answered reads, under an 'Open
    questions' heading, as one of THEIR confusions."""
    from slim import chat, summarize
    r = _existing(tmp_path, monkeypatch)          # written with type_tag="lecture"
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary(
        overview="o", key_points=[{"topic": "T", "points": ["p"]}],
        open_questions=["Does a dictionary know English?"]))
    chat.handle_record_summary(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)
    assert "## Open questions" not in r.note.read_text(encoding="utf-8")


def test_correcting_the_type_to_a_lecture_drops_its_open_questions(tmp_path, monkeypatch):
    """MEASURED IN THE LIVE PLUGIN, 2026-08-26. The summary is rendered against the type the
    CARD guessed; the type they settle on is written afterwards. Guess `meeting-note`, correct
    it to `lecture`, and the note ends up typed lecture while carrying a section a lecture is
    not supposed to have — the lecturer's rhetorical questions, reading as their own."""
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    summary = ("An overview.\n\n## Key points\n\n### Topic\n- a point\n\n"
               "## Open questions\n- What is understanding?\n")

    chat.handle_record_apply(note=str(r.note.relative_to(tmp_path)), dest_dir="Capture/school",
                             type_tag="lecture", topics=[], summary_md=summary,
                             vault=tmp_path)

    after = (tmp_path / "Capture/school" / r.note.name).read_text(encoding="utf-8")
    assert "## Open questions" not in after
    assert "- a point" in after                      # only the one section goes


def test_a_meeting_keeps_the_open_questions_it_was_given(tmp_path, monkeypatch):
    """A meeting's unresolved question is real information. The drop is scoped to the types
    where the field cannot mean what its heading says."""
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    chat.handle_record_apply(note=str(r.note.relative_to(tmp_path)), dest_dir="Capture/work",
                             type_tag="meeting-note", topics=[],
                             summary_md="o\n\n## Open questions\n- who owns it?",
                             vault=tmp_path)
    after = (tmp_path / "Capture/work" / r.note.name).read_text(encoding="utf-8")
    assert "## Open questions" in after and "who owns it?" in after


# --- 2026-08-26: an accidental Stop must not cost the lecture -----------------------------

def test_cancelling_a_job_stops_the_pass_and_writes_nothing(tmp_path, monkeypatch):
    """Their case: Stop is hit 40 minutes into a lecture by accident. Cancel has to leave the
    recording exactly as it was — ⚠ NOTHING WRITTEN, NOTHING DELETED — because the draft note
    and the staged audio are the only copies of what they said."""
    from slim import chat, summarize, transcribe
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="forty minutes of lecture", model="fake", audio_seconds=2400.0, wall_seconds=30.0))

    def cancel_midway(*a, **k):
        chat.request_record_cancel("job-1")
        return summarize.Summary(overview="never used")
    monkeypatch.setattr(summarize, "summarize", cancel_midway)

    audio = _staged(tmp_path)
    draft = _draft(tmp_path)
    chat._record_progress_start("job-1")
    with pytest.raises(chat.RecordCancelled):
        chat.handle_record(audio_rel=audio, draft_rel=draft, declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path, job_id="job-1")

    assert (tmp_path / audio).is_file(), "the audio is the recording; cancel never deletes it"
    assert (tmp_path / draft).is_file(), "their typed notes survive a cancel"
    assert not list((tmp_path / "Capture").glob("**/*.md")) or \
        all(p.name.endswith(".slim-draft.md") for p in (tmp_path / "Capture").glob("**/*.md"))


def test_cancelling_before_the_expensive_part_is_honoured(tmp_path, monkeypatch):
    """The stages that are not streaming — transcription, the card, the title — are blocking
    calls, so cancel lands at the next boundary. It must at least be CHECKED at each one."""
    from slim import chat, transcribe
    calls = []
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: (
        calls.append(p), transcribe.Transcript(text="x", model="f", audio_seconds=1.0,
                                               wall_seconds=1.0))[1])
    chat._record_progress_start("job-2")
    chat.request_record_cancel("job-2")
    with pytest.raises(chat.RecordCancelled):
        chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path, job_id="job-2")
    assert calls == [], "a cancel before transcription does not pay for a transcription"


def test_cancelling_an_unknown_job_says_so(tmp_path):
    from slim import chat
    assert chat.request_record_cancel("no-such-job") is False


def test_a_resumed_recording_is_one_note_from_several_segments(tmp_path, monkeypatch):
    """A second MediaRecorder session writes its own container header, so appending to one
    .webm risks a file nothing can play. Segments sidestep that: several audio files, one
    recording id, one note, transcripts joined in order."""
    from slim import chat, transcribe
    seen = []

    def fake_tx(p, model=None):
        seen.append(Path(p).name)
        return transcribe.Transcript(text=f"part {len(seen)}", model="fake",
                                     audio_seconds=60.0, wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
    _fakes(monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)

    out = chat.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                        _staged(tmp_path, "r-002.webm")],
                             notes_md="", declared_type="lecture", title="T", when=NOW,
                             vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert seen == ["r-001.webm", "r-002.webm"], "in the order they recorded them"
    assert "part 1\n\npart 2" in body, "one transcript, no seam marker"


def test_every_segment_is_archived_and_named_in_the_note(tmp_path, monkeypatch):
    from slim import chat
    from slim.chunk import parse_frontmatter
    _fakes(monkeypatch)
    out = chat.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
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
    from slim import chat
    from slim.chunk import parse_frontmatter
    _fakes(monkeypatch)
    out = chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                             title="T", when=NOW, vault=tmp_path)
    fm, _ = parse_frontmatter((tmp_path / out["note"]).read_text(encoding="utf-8"))
    assert isinstance(fm.get("audio"), str)


def test_a_transcribed_segment_is_not_transcribed_twice(tmp_path, monkeypatch):
    """⚠ THE REASON CANCEL IS AFFORDABLE. Without a cache, cancelling after a 40-minute
    transcription makes them pay for that transcription again on resume — which is most of the
    wait they cancelled to avoid."""
    from slim import chat, transcribe
    calls = []

    def fake_tx(p, model=None):
        calls.append(Path(p).name)
        return transcribe.Transcript(text="the first forty minutes", model="fake",
                                     audio_seconds=2400.0, wall_seconds=90.0)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
    from slim import summarize as summarize_mod
    monkeypatch.setattr(summarize_mod, "summarize",
                        lambda *a, **k: (_ for _ in ()).throw(chat.RecordCancelled()))
    first = _staged(tmp_path, "r-001.webm")
    chat._record_progress_start("job-3")
    with pytest.raises(chat.RecordCancelled):
        chat.handle_record(audio_rel=first, notes_md="", declared_type="lecture", title="T",
                           when=NOW, vault=tmp_path, job_id="job-3")
    assert calls == ["r-001.webm"]

    _fakes(monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
    out = chat.handle_record(audio_rel=[first, _staged(tmp_path, "r-002.webm")],
                             notes_md="", declared_type="lecture", title="T", when=NOW,
                             vault=tmp_path)
    assert calls == ["r-001.webm", "r-002.webm"], "the cancelled segment is reused, not redone"
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "the first forty minutes" in body
    assert not list((tmp_path / "Attachments/_incoming").glob("*.json")), "caches are cleaned up"


def test_the_wire_takes_one_path_or_several():
    """An older plugin sends a string. Both shapes reach the same code."""
    from slim import chat
    one = chat._record_args({"audio": "a.webm", "notes_md": "", "type": "lecture"})
    many = chat._record_args({"audio": ["a.webm", "b.webm"], "notes_md": "", "type": "lecture"})
    assert one["audio_rel"] == "a.webm"
    assert many["audio_rel"] == ["a.webm", "b.webm"]
    with pytest.raises(ValueError):
        chat._record_args({"audio": [], "notes_md": "", "type": "lecture"})


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
    from slim import chat
    _fakes(monkeypatch)
    rid = "20260826T150802-782569ba"
    draft = _draft(tmp_path, recording_id=rid)
    out = chat.handle_record(
        audio_rel=[_staged(tmp_path, f"{rid}-001.webm"), _staged(tmp_path, f"{rid}-002.webm")],
        draft_rel=draft, declared_type="lecture", title="T", when=NOW, vault=tmp_path)
    assert out["note"]


def test_a_mismatched_recording_id_is_still_refused(tmp_path, monkeypatch):
    """The guard exists to stop one recording's audio being filed with another's notes. It has
    to keep saying no — widening it for segments must not turn it off."""
    from slim import chat
    _fakes(monkeypatch)
    draft = _draft(tmp_path, recording_id="20260826T150802-782569ba")
    with pytest.raises(ValueError, match="do not match"):
        chat.handle_record(audio_rel=_staged(tmp_path, "20260826T999999-cafebabe-001.webm"),
                           draft_rel=draft, declared_type="lecture", title="T", when=NOW,
                           vault=tmp_path)


def test_the_transcript_so_far_can_be_read_back_without_transcribing(tmp_path, monkeypatch):
    """The paused screen has to show what was captured, or "1 segment saved" is a claim they have
    no way to check — and after a cancel that claim is the whole point. Reads the per-segment
    caches ONLY: asking for a transcript must never start a transcription."""
    from slim import chat, transcribe
    calls = []
    monkeypatch.setattr(transcribe, "transcribe", lambda *a, **k: calls.append(1))
    first, second = _staged(tmp_path, "r-001.webm"), _staged(tmp_path, "r-002.webm")
    (tmp_path / first).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": "the first part", "words": 3}), encoding="utf-8")

    out = chat.handle_record_transcript(audio_rel=[first, second], vault=tmp_path)

    assert out["text"] == "the first part"
    assert out["pending"] == 1, "the segment with no cache is reported, not invented"
    assert calls == [], "reading never transcribes"


def test_resuming_a_filed_note_appends_audio_and_words(tmp_path, monkeypatch):
    """Their ask: resume should be available when the review card is waiting, and on any note
    that is already finished. The note exists, its audio is archived and staging is clear, so
    this is an APPEND — not a second recording, and not a rewrite."""
    from slim import chat, transcribe
    from slim.chunk import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="and ten minutes more", model="fake", audio_seconds=600.0, wall_seconds=5.0))

    out = chat.handle_record_append(note=str(r.note.relative_to(tmp_path)),
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
    from slim import chat, transcribe
    r = _existing(tmp_path, monkeypatch)     # written with summary_md="s"
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="more words", model="fake", audio_seconds=60.0, wall_seconds=1.0))
    chat.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                              audio_rel=[_staged(tmp_path, "more-001.webm")], vault=tmp_path)
    assert "\ns\n" in r.note.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_appending_silence_is_refused_and_keeps_the_audio(tmp_path, monkeypatch):
    """Same rule as the recording path: a transcript with no words is refused, not filed."""
    from slim import chat, transcribe
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="   ", model="fake", audio_seconds=60.0, wall_seconds=1.0))
    staged = _staged(tmp_path, "more-001.webm")
    with pytest.raises(chat.NoSpeech, match="no words"):
        chat.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[staged], vault=tmp_path)
    assert (tmp_path / staged).is_file(), "the audio stays staged for a retry"
    assert "and ten minutes" not in r.note.read_text(encoding="utf-8")


# --- 2026-08-26: silence is an OUTCOME, not an error -------------------------------------

def test_silence_is_reported_as_silence_not_as_a_failure(tmp_path, monkeypatch):
    """Their call, and they are right: "there were no words" is not an error. It was reaching them as
    a red screen asking whether to delete their files — a decision they should never have to make
    over a recording that simply had nothing in it. The note is still not written (an empty
    note is worthless), but the caller has to be able to tell silence from a real failure."""
    from slim import chat, transcribe
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="   ", model="fake", audio_seconds=30.0, wall_seconds=1.0))
    with pytest.raises(chat.NoSpeech) as caught:
        chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path)
    assert "no words" in str(caught.value).casefold()


def test_a_real_transcription_failure_is_still_an_error(tmp_path, monkeypatch):
    """Silence being quiet must not make a broken ASR quiet too — that one they do need to see."""
    from slim import chat, transcribe

    def boom(*a, **k):
        raise RuntimeError("ASR exploded")
    monkeypatch.setattr(transcribe, "transcribe", boom)
    with pytest.raises(ValueError, match="transcription failed"):
        chat.handle_record(audio_rel=_staged(tmp_path), notes_md="", declared_type="lecture",
                           title="T", when=NOW, vault=tmp_path)


def test_appending_silence_is_the_same_quiet_outcome(tmp_path, monkeypatch):
    from slim import chat, transcribe
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    staged = _staged(tmp_path, "more-001.webm")
    with pytest.raises(chat.NoSpeech):
        chat.handle_record_append(note=str(r.note.relative_to(tmp_path)), audio_rel=[staged],
                                  vault=tmp_path)
    assert (tmp_path / staged).is_file()


# --- 2026-08-26, found in review: long jobs must not write what they read minutes ago -----

def test_a_retry_does_not_clobber_notes_saved_while_it_ran(tmp_path, monkeypatch):
    """⚠ The notes editor autosaves on blur, and a retry takes 60-300s. Reading the note at
    entry and writing that same text back at exit reverts anything they typed in between —
    silently, and only sometimes, which is the worst shape a bug can have."""
    from slim import chat, summarize
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))

    def slow_summary(*a, **k):
        # what the blur-autosave does while the model is thinking
        chat.handle_record_notes(note=note_rel, notes_md="- typed while it ran", vault=tmp_path)
        return summarize.Summary(overview="A new summary.")
    monkeypatch.setattr(summarize, "summarize", slow_summary)

    chat.handle_record_summary(note=note_rel, vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    assert "- typed while it ran" in body, "their words outrank a stale read"
    assert "A new summary." in body


def test_an_append_does_not_clobber_notes_saved_while_it_ran(tmp_path, monkeypatch):
    from slim import chat, transcribe
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))

    def slow_tx(p, model=None):
        chat.handle_record_notes(note=note_rel, notes_md="- typed while it ran", vault=tmp_path)
        return transcribe.Transcript(text="more words", model="fake", audio_seconds=9.0,
                                     wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", slow_tx)

    chat.handle_record_append(note=note_rel, audio_rel=[_staged(tmp_path, "more-001.webm")],
                              vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    assert "- typed while it ran" in body
    assert "more words" in body


def test_a_retry_never_resurrects_a_note_that_moved(tmp_path, monkeypatch):
    """`write_text` CREATES. If `apply` moved the note while the retry was thinking, writing
    the old path would put a second, stale copy of the recording back into the vault — and
    index it."""
    from slim import chat, summarize
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))

    def moves_it(*a, **k):
        chat.handle_record_apply(note=note_rel, dest_dir="Capture/work", type_tag="meeting-note",
                                 topics=[], summary_md="s", vault=tmp_path)
        return summarize.Summary(overview="A new summary.")
    monkeypatch.setattr(summarize, "summarize", moves_it)

    with pytest.raises(FileNotFoundError):
        chat.handle_record_summary(note=note_rel, vault=tmp_path)
    assert not r.note.exists(), "the old path stays gone"


def test_cancelling_a_one_segment_resume_actually_cancels(tmp_path, monkeypatch):
    """⚠ Cancel was checked only at the top of the per-segment loop, and a resume normally has
    exactly ONE segment — so the check ran once, before a transcription that can take forty
    minutes, and never again. They were told "nothing is written or deleted" and the note grew."""
    from slim import chat, transcribe

    def cancel_during(p, model=None):
        chat.request_record_cancel("job-append")
        return transcribe.Transcript(text="more words", model="fake", audio_seconds=9.0,
                                     wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", cancel_during)
    r = _existing(tmp_path, monkeypatch)
    before = r.note.read_text(encoding="utf-8")
    staged = _staged(tmp_path, "more-001.webm")
    chat._record_progress_start("job-append")

    with pytest.raises(chat.RecordCancelled):
        chat.handle_record_append(note=str(r.note.relative_to(tmp_path)), audio_rel=[staged],
                                  job_id="job-append", vault=tmp_path)

    assert r.note.read_text(encoding="utf-8") == before, "nothing written, as promised"
    assert (tmp_path / staged).is_file(), "nothing deleted, as promised"


def test_an_autosave_does_not_sweep_the_vault(tmp_path, monkeypatch):
    """⚠ `_index_now`'s tail schedules a full-vault ingest + embed. On a note that is seconds
    old that is right — the sweep retires the old path's row. On every blur of the notes field
    it is a dozen full-vault hash-and-embed passes for a dozen click-aways, each competing with
    the resident model for the box. Index the note; leave the vault alone."""
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    sweeps = []
    monkeypatch.setattr(chat, "_schedule_sweep", lambda v: sweeps.append(v))

    chat.handle_record_notes(note=str(r.note.relative_to(tmp_path)), notes_md="- typed",
                             vault=tmp_path)
    assert sweeps == [], "typing a note is not a reason to re-embed the vault"


# --- 2026-08-26, second review: the destructive set ---------------------------------------

def test_an_empty_but_valid_summary_never_deletes_the_one_that_is_there(tmp_path, monkeypatch):
    """⚠ `valid` means SCHEMA-valid. `Summary()` passes and renders to "", which
    `replace_review_sections` writes as "no summary" — deleting the fence and its contents.
    "A failed retry keeps the summary it has" has to cover the model returning nothing."""
    from slim import chat, summarize
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary())
    with pytest.raises(ValueError, match="empty"):
        chat.handle_record_summary(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)
    assert "\ns\n" in r.note.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_append_never_deletes_the_audio_it_just_recorded(tmp_path, monkeypatch):
    """⚠ The cleanup unlinked every input path. Point `audio` at a file that is ALREADY its
    content-addressed archive and `archive_audio` returns that same path — so the note ends up
    naming a recording the same request deleted."""
    from slim import chat, transcribe
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="more words", model="fake", audio_seconds=9.0, wall_seconds=1.0))
    archived = tmp_path / "Attachments" / "Recordings" / "keep.webm"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_bytes(b"the only copy")

    with pytest.raises(ValueError, match="staged"):
        chat.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                                  audio_rel=[str(archived.relative_to(tmp_path))],
                                  vault=tmp_path)
    assert archived.is_file(), "an archived recording is never an input to append"


def test_every_segment_must_belong_to_the_same_recording(tmp_path, monkeypatch):
    """Only the FIRST segment was checked against the draft id, so two different recordings
    could be merged into one note and both staged files deleted."""
    from slim import chat
    _fakes(monkeypatch)
    draft = _draft(tmp_path, recording_id="alpha")
    with pytest.raises(ValueError, match="do not match"):
        chat.handle_record(audio_rel=[_staged(tmp_path, "alpha-001.webm"),
                                      _staged(tmp_path, "bravo-001.webm")],
                           draft_rel=draft, declared_type="lecture", title="T", when=NOW,
                           vault=tmp_path)


def test_a_transcript_cache_is_bound_to_the_audio_it_transcribed(tmp_path, monkeypatch):
    """The cache was trusted by PATH. Different bytes at the same staged name returned the old
    words with no transcription at all."""
    from slim import chat, transcribe
    calls = []

    def tx(p, model=None):
        calls.append(1)
        return transcribe.Transcript(text=f"take {len(calls)}", model="fake",
                                     audio_seconds=1.0, wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", tx)
    monkeypatch.setattr(transcribe, "normalize_container", lambda p: None)
    staged = tmp_path / "Attachments" / "_incoming" / "same-001.webm"
    staged.parent.mkdir(parents=True, exist_ok=True)

    staged.write_bytes(b"first audio")
    first, cached = chat._transcribe_segment(staged, transcribe)
    assert first["text"] == "take 1" and not cached
    again, cached = chat._transcribe_segment(staged, transcribe)
    assert again["text"] == "take 1" and cached, "same bytes, same words, no second pass"

    staged.write_bytes(b"DIFFERENT audio")
    third, cached = chat._transcribe_segment(staged, transcribe)
    assert third["text"] == "take 2" and not cached, "different bytes are a different recording"


def test_speaker_provenance_survives_the_segment_cache(tmp_path, monkeypatch):
    """The cache hands back the words without transcribing. It must hand back how their
    speakers were decided too, or a Try again files a labelled transcript that says nothing
    about where its labels came from."""
    from slim import chat, transcribe

    _fakes(monkeypatch)
    monkeypatch.setattr(transcribe, "normalize_container", lambda p: p)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="**Me:** so\n\n**Them:** yes", model="fake", audio_seconds=10.0, wall_seconds=1.0,
        speakers_by="channels"))
    staged = tmp_path / "Attachments/_incoming/t-001.webm"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"audio")

    first, cached = chat._transcribe_segment(staged, transcribe)
    assert (first["speakers_by"], cached) == ("channels", False)
    again, cached = chat._transcribe_segment(staged, transcribe)
    assert (again["speakers_by"], cached) == ("channels", True)


def test_the_one_speaker_survives_the_segment_cache(tmp_path, monkeypatch):
    from slim import chat, transcribe

    _fakes(monkeypatch)
    monkeypatch.setattr(transcribe, "normalize_container", lambda p: p)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="and one more thing", model="fake", audio_seconds=10.0, wall_seconds=1.0,
        solo_speaker="Me"))
    staged = tmp_path / "Attachments/_incoming/t-001.webm"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"audio")

    chat._transcribe_segment(staged, transcribe)
    again, cached = chat._transcribe_segment(staged, transcribe)
    assert (again["solo_speaker"], cached) == ("Me", True)


def _segments_tx(monkeypatch, *results):
    """One Transcript per segment, in order: (text, speakers_by, solo_speaker)."""
    from slim import transcribe
    queue = list(results)

    def fake_tx(p, model=None):
        text, by, solo = queue.pop(0)
        return transcribe.Transcript(text=text, model="fake", audio_seconds=60.0,
                                     wall_seconds=1.0, speakers_by=by, solo_speaker=solo)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)


def test_a_one_speaker_segment_in_a_labelled_recording_carries_its_label(tmp_path, monkeypatch):
    """Unlabelled, their closing monologue reads as the last speaker still talking."""
    from slim import chat
    _fakes(monkeypatch)
    _segments_tx(monkeypatch, ("**Me:** so\n\n**Them:** yes", "channels", ""),
                 ("and one more thing", "", "Me"))

    out = chat.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                        _staged(tmp_path, "r-002.webm")],
                             notes_md="", declared_type="meeting-note", title="T", when=NOW,
                             vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "**Them:** yes\n\n**Me:** and one more thing" in body


def test_the_labelled_segment_can_come_second(tmp_path, monkeypatch):
    from slim import chat
    _fakes(monkeypatch)
    _segments_tx(monkeypatch, ("they opened", "", "Them"),
                 ("**Me:** so\n\n**Them:** yes", "channels", ""))

    out = chat.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                        _staged(tmp_path, "r-002.webm")],
                             notes_md="", declared_type="meeting-note", title="T", when=NOW,
                             vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "**Them:** they opened\n\n**Me:** so" in body


def test_a_lecture_in_two_segments_stays_unlabelled(tmp_path, monkeypatch):
    """Every segment one-sided: nothing to tell apart, so no labels at all."""
    from slim import chat
    _fakes(monkeypatch)
    _segments_tx(monkeypatch, ("part one", "", "Them"), ("part two", "", "Them"))

    out = chat.handle_record(audio_rel=[_staged(tmp_path, "r-001.webm"),
                                        _staged(tmp_path, "r-002.webm")],
                             notes_md="", declared_type="lecture", title="T", when=NOW,
                             vault=tmp_path)
    body = (tmp_path / out["note"]).read_text(encoding="utf-8")
    assert "part one\n\npart two" in body and "**Them:**" not in body


def test_resuming_a_labelled_note_labels_a_one_speaker_segment(tmp_path, monkeypatch):
    from slim import chat, record
    r = _existing(tmp_path, monkeypatch)
    r.note.write_text(record.append_audio_frontmatter(
        r.note.read_text(encoding="utf-8"), rels=[], digests=[], seconds=0.0,
        speakers_by="channels"), encoding="utf-8")
    _segments_tx(monkeypatch, ("and one more thing", "", "Me"))

    chat.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                              audio_rel=[_staged(tmp_path, "more-001.webm")], vault=tmp_path)
    assert "**Me:** and one more thing" in r.note.read_text(encoding="utf-8")


def test_resuming_an_unlabelled_note_adds_no_label(tmp_path, monkeypatch):
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    _segments_tx(monkeypatch, ("and ten minutes more", "", "Them"))

    chat.handle_record_append(note=str(r.note.relative_to(tmp_path)),
                              audio_rel=[_staged(tmp_path, "more-001.webm")], vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    assert "and ten minutes more" in body and "**Them:**" not in body


def test_the_paused_transcript_labels_segments_the_same_way(tmp_path):
    from slim import chat
    first, second = _staged(tmp_path, "r-001.webm"), _staged(tmp_path, "r-002.webm")
    (tmp_path / first).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": "**Me:** so\n\n**Them:** yes", "speakers_by": "channels"}),
        encoding="utf-8")
    (tmp_path / second).with_suffix(".webm.transcript.json").write_text(
        json.dumps({"text": "and more", "solo_speaker": "Me"}), encoding="utf-8")

    out = chat.handle_record_transcript(audio_rel=[first, second], vault=tmp_path)
    assert out["text"] == "**Me:** so\n\n**Them:** yes\n\n**Me:** and more"


def test_two_writers_on_one_note_take_turns(tmp_path, monkeypatch):
    """⚠ `ThreadingHTTPServer` runs handlers in parallel and four endpoints write the same note.
    Re-reading before the write closed the long-job case; it does NOT close the case where two
    writers interleave between one another's read and write. One lock per note does."""
    import threading
    from slim import chat
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))
    order, errors = [], []

    def notes(text, delay):
        try:
            with chat._note_lock(tmp_path, note_rel):
                order.append(f"enter {text}")
                time.sleep(delay)
                chat.handle_record_notes(note=note_rel, notes_md=text, vault=tmp_path)
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
    from slim import chat
    a = chat._note_lock(tmp_path, "Capture/a.md")
    b = chat._note_lock(tmp_path, "Capture/b.md")
    again = chat._note_lock(tmp_path, "Capture/a.md")
    assert a is again and a is not b


def test_a_leftover_live_session_never_blocks_the_next_recording(monkeypatch):
    """⚠ `live_start` REFUSED when any other session existed, and only ever cleaned up the one
    with the same id. A session left behind by a crash, a killed server or a recording that
    never called stop therefore turned off live captions for every recording afterwards — with
    one line of grey status text as the only sign. A recorder has exactly one live session; a
    new one replaces whatever is there."""
    from slim import transcribe

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


# --- parallel recorder sessions: bounded heavy work and isolated progress -----------------

def test_recording_progress_is_job_local_and_never_exposes_private_reasoning():
    from slim import chat

    chat._record_progress_start("meeting-a")
    chat._record_progress_start("meeting-b")
    chat._record_progress_update(
        "meeting-a", stage="summarizing", label="Writing A", kind="thinking",
        text="private reasoning for A", metrics={"transcript_words": 41},
    )
    chat._record_progress_update(
        "meeting-b", stage="queued", label="Queued behind 1 recording",
        detail="Waiting for the local recording pipeline",
    )

    a = chat.record_progress("meeting-a")
    b = chat.record_progress("meeting-b")
    assert a["stage"] == "summarizing" and a["metrics"]["transcript_words"] == 41
    assert b["stage"] == "queued" and b["label"] == "Queued behind 1 recording"
    assert "thinking" not in a and "draft" not in a
    assert a["metrics"]["model_activity_events"] == 1
    assert "private reasoning" not in json.dumps(a)

    a["metrics"]["transcript_words"] = 999
    assert chat.record_progress("meeting-a")["metrics"]["transcript_words"] == 41
    assert chat.record_progress("meeting-b")["metrics"] == {}


def test_recording_pipeline_is_fifo_and_an_exception_releases_the_next_job():
    from slim import chat

    pipeline = chat.RecordingPipelineQueue(limit=3)
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
    from slim import chat

    pipeline = chat.RecordingPipelineQueue(limit=2)
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
        with pytest.raises(chat.RecordQueueFull):
            with pipeline.slot("c"):
                pass
    finally:
        release.set()
        a.join(2); b.join(2)


def test_waiting_recording_job_reports_its_own_queue_position(monkeypatch):
    from slim import chat

    monkeypatch.setattr(chat, "_RECORD_PIPELINE", chat.RecordingPipelineQueue(limit=2))
    release = threading.Event()
    entered = {"a": threading.Event(), "b": threading.Event()}

    def run(name):
        with chat._record_pipeline_slot(name):
            entered[name].set()
            if name == "a":
                release.wait(2)

    chat._record_progress_start("a")
    chat._record_progress_start("b")
    a = threading.Thread(target=run, args=("a",))
    b = threading.Thread(target=run, args=("b",))
    a.start(); assert entered["a"].wait(1)
    b.start()
    deadline = time.time() + 1
    while time.time() < deadline and chat.record_progress("b")["label"] == "Queued":
        time.sleep(0.005)

    queued_state = chat.record_progress("b")
    assert queued_state["stage"] == "queued"
    assert queued_state["label"] == "Queued behind 1 recording"
    assert entered["b"].is_set() is False

    release.set()
    a.join(2); b.join(2)
    assert entered["b"].is_set(), "the queued job starts when A releases the pipeline"


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
    from slim import inbox

    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        eastern = timezone(timedelta(hours=-4))
        evening = datetime(2026, 8, 26, 21, 17, tzinfo=eastern)     # 2026-08-27T01:17Z

        assert evening.astimezone(timezone.utc).strftime("%Y-%m-%d") == "2026-08-27"  # the old bug
        assert inbox.local_date(evening) == "2026-08-26"        # their day, and their sort order

        # A UTC-aware instant for the same moment must resolve to the same LOCAL day.
        assert inbox.local_date(evening.astimezone(timezone.utc)) == \
            evening.astimezone().strftime("%Y-%m-%d")
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()
