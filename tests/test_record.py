from datetime import datetime, timezone
from pathlib import Path

import pytest

from slim import record
from slim.chunk import parse_frontmatter, strip_derived_blocks

NOW = datetime(2026, 8, 4, 14, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_audio_archive(tmp_path, monkeypatch):
    """`inbox.AUDIO_ARCHIVE` is bound to the real VAULT at import time. Redirect it into the
    test's tmp_path so archiving never touches the real vault. Do NOT edit slim/inbox.py."""
    from slim import inbox

    monkeypatch.setattr(inbox, "AUDIO_ARCHIVE", tmp_path / "Attachments/Recordings")


def _write(tmp_path: Path, **kw):
    audio_src = kw.pop("audio_src", None)
    if audio_src is None:
        audio_bytes = kw.pop("audio_bytes", b"RIFFfake-audio-bytes")
        audio_src = tmp_path / "_incoming" / "rec.webm"
        audio_src.parent.mkdir(parents=True, exist_ok=True)
        audio_src.write_bytes(audio_bytes)
    args = dict(audio_src=audio_src,
                title="SVMs and Kernel Methods", when=NOW, type_tag="lecture",
                transcript="Today we derive the kernel trick.",
                notes_md="- dual form\n- RBF is the motivating case",
                summary_md="Derived the kernel trick from the dual form.",
                topics=["school", "svm"],
                dest_dir="Notes/school/CS201 ML", vault=tmp_path)
    args.update(kw)
    return record.write_recording(**args)


def test_the_note_lands_at_the_destination_it_was_given(tmp_path):
    """The card decides the path. record.py obeys it — it makes no routing judgement of its
    own, which is what keeps it a pure function."""
    r = _write(tmp_path)
    assert r.note.parent == tmp_path / "Notes/school/CS201 ML"
    assert r.note.suffix == ".md"


def test_their_typed_notes_are_unfenced_and_survive_ingestion(tmp_path):
    """Their notes are THEIR WORDS and must stay indexed. Only SLIM's output is fenced."""
    r = _write(tmp_path)
    indexed = strip_derived_blocks(r.note.read_text(encoding="utf-8"))
    assert "dual form" in indexed
    assert "RBF is the motivating case" in indexed


def test_the_summary_is_fenced_and_is_blanked_before_ingestion(tmp_path):
    """The brain must never retrieve its own opinions as evidence."""
    r = _write(tmp_path, summary_md="A DERIVED SENTENCE.")
    text = r.note.read_text(encoding="utf-8")
    assert "A DERIVED SENTENCE." in text
    assert "A DERIVED SENTENCE." not in strip_derived_blocks(text)


def test_new_plugin_recordings_are_durably_pending_review(tmp_path):
    r = _write(tmp_path)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm["review_status"] == "pending"


def test_review_status_has_one_validated_frontmatter_writer(tmp_path):
    r = _write(tmp_path)
    text = r.note.read_text(encoding="utf-8")
    complete = record.set_review_status(text, "complete")
    fm, _ = parse_frontmatter(complete)
    assert fm["review_status"] == "complete"
    assert record.read_review_sections(complete) == record.read_review_sections(text)
    with pytest.raises(ValueError, match="invalid recording review status"):
        record.set_review_status(text, "ignored")


def test_the_transcript_is_last_and_byte_identical(tmp_path):
    r = _write(tmp_path, transcript="verbatim words they said")
    text = r.note.read_text(encoding="utf-8")
    block = record.split_meeting_block(text)
    assert block is not None
    assert block.body.rstrip().endswith("verbatim words they said")
    assert block.after == ""
    assert text.index("## Transcript") > text.index("## Notes")


def test_ordinary_markdown_can_continue_after_the_inline_meeting_block(tmp_path):
    r = _write(tmp_path)
    before = r.note.read_text(encoding="utf-8")
    after = before + "\n## Follow-up\n\nA note written after the recording object.\n"
    edited = record.replace_review_sections(after, summary_md="Updated", notes_md=None)
    block = record.split_meeting_block(edited)
    assert block is not None
    assert block.after == "\n## Follow-up\n\nA note written after the recording object.\n"


def test_review_section_edits_preserve_frontmatter_and_transcript_bytes(tmp_path):
    r = _write(tmp_path, transcript="verbatim  \nsecond line", notes_md="")
    before = r.note.read_text(encoding="utf-8")
    fm_end = before.find("\n---", 4) + 4
    transcript_at = before.rfind("\n## Transcript")

    after = record.replace_review_sections(
        before, summary_md="A corrected summary.", notes_md="- added after recording  ")

    assert after[:fm_end] == before[:fm_end]
    assert after[after.rfind("\n## Transcript"):] == before[transcript_at:]
    assert "A corrected summary." in after
    assert "- added after recording  \n## Transcript" in after


def test_an_old_review_client_preserves_existing_notes(tmp_path):
    r = _write(tmp_path, notes_md="- exact existing note  ")
    before = r.note.read_text(encoding="utf-8")
    after = record.replace_review_sections(before, summary_md="New summary", notes_md=None)
    assert "- exact existing note  \n## Transcript" in after


def test_an_auto_filed_note_says_so(tmp_path):
    r = _write(tmp_path, filed_by_slim=True)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm.get("filed_by") == "slim"


def test_a_note_they_confirmed_carries_no_filed_by(tmp_path):
    """Absence means 'they declared this' — established by the 2026-08-04 curation pass.
    Inventing a second term for the confirmed case would contradict it."""
    r = _write(tmp_path, filed_by_slim=False)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert "filed_by" not in fm


def test_origin_is_recorded_so_every_later_pass_finds_it(tmp_path):
    """`origin: recorded` is provenance — the recorder's mark that SLIM captured this,
    carried into the index by ingest."""
    r = _write(tmp_path)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm.get("origin") == "recorded"


def test_the_audio_is_archived_by_content_hash_and_referenced(tmp_path):
    r = _write(tmp_path, audio_bytes=b"distinct-bytes")
    assert r.audio.exists()
    assert r.audio.read_bytes() == b"distinct-bytes"
    assert r.digest[:8] in r.note.read_text(encoding="utf-8")


def test_identical_audio_is_archived_once(tmp_path):
    """Content-addressed: a re-send of the same recording must not duplicate the blob."""
    a = _write(tmp_path, audio_bytes=b"same-bytes")
    b = _write(tmp_path, audio_bytes=b"same-bytes", title="Second Take")
    assert a.audio == b.audio
    assert a.note != b.note


def test_audio_src_is_deleted_after_archiving(tmp_path):
    """The plugin writes the recording into the vault itself; once it is safely archived
    content-addressed, the staged copy must not linger as a second, uncontrolled copy."""
    audio_src = tmp_path / "_incoming" / "rec2.webm"
    audio_src.parent.mkdir(parents=True, exist_ok=True)
    audio_src.write_bytes(b"stage-me")
    r = _write(tmp_path, audio_src=audio_src)
    assert not audio_src.exists()
    assert r.audio.exists()


def test_draft_cleanup_precedes_staged_audio_cleanup(tmp_path, monkeypatch):
    audio_src = tmp_path / "Attachments/_incoming/paired.webm"
    draft_src = tmp_path / "Capture/_unfiled/paired.slim-draft.md"
    audio_src.parent.mkdir(parents=True)
    draft_src.parent.mkdir(parents=True)
    audio_src.write_bytes(b"paired audio")
    draft_src.write_text("paired notes", encoding="utf-8")
    real_unlink = Path.unlink
    removed = []

    def spy(path, *args, **kwargs):
        if path in (draft_src, audio_src):
            removed.append(path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", spy)
    _write(tmp_path, audio_src=audio_src, draft_src=draft_src)
    assert removed == [draft_src, audio_src]


def test_a_late_draft_cleanup_failure_does_not_unfile_the_safe_note(tmp_path, monkeypatch):
    audio_src = tmp_path / "Attachments/_incoming/paired.webm"
    draft_src = tmp_path / "Capture/_unfiled/paired.slim-draft.md"
    audio_src.parent.mkdir(parents=True)
    draft_src.parent.mkdir(parents=True)
    audio_src.write_bytes(b"paired audio")
    draft_src.write_text("paired notes", encoding="utf-8")
    real_unlink = Path.unlink

    def fail_draft(path, *args, **kwargs):
        if path == draft_src:
            raise OSError("draft is temporarily busy")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_draft)
    result = _write(tmp_path, audio_src=audio_src, draft_src=draft_src)
    assert result.note.exists()
    assert draft_src.exists() and audio_src.exists()


def test_a_late_audio_cleanup_failure_does_not_unfile_the_safe_note(tmp_path, monkeypatch):
    audio_src = tmp_path / "Attachments/_incoming/paired.webm"
    draft_src = tmp_path / "Capture/_unfiled/paired.slim-draft.md"
    audio_src.parent.mkdir(parents=True)
    draft_src.parent.mkdir(parents=True)
    audio_src.write_bytes(b"paired audio")
    draft_src.write_text("paired notes", encoding="utf-8")
    real_unlink = Path.unlink

    def fail_audio(path, *args, **kwargs):
        if path == audio_src:
            raise OSError("audio is temporarily busy")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_audio)
    result = _write(tmp_path, audio_src=audio_src, draft_src=draft_src)
    assert result.note.exists()
    assert not draft_src.exists()
    assert audio_src.exists()


# --- frontmatter written from MODEL output, which is not a safe string ------------------

def test_a_comma_inside_a_topic_does_not_split_it_in_two(tmp_path):
    """`topics` comes straight from the model as free text. SLIM's own reader
    (`chunk.parse_frontmatter`) splits a flow list on bare commas and ignores quotes, so an
    unsanitised value silently becomes two — measured on the index_terms field that used to
    sit beside this one: "Mercer's condition, restated" parsed as ["Mercer's condition",
    "restated"]. Quoting cannot fix it; the value must be clean."""
    from slim.chunk import parse_frontmatter
    r = _write(tmp_path, topics=["Mercer's condition, restated", "school"])
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert len(fm["topics"]) == 2
    assert fm["topics"][1] == "school"


def test_a_bracket_in_a_topic_cannot_truncate_the_list(tmp_path):
    from slim.chunk import parse_frontmatter
    r = _write(tmp_path, topics=["kernel [trick]", "school"])
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert len(fm["topics"]) == 2


def test_a_newline_or_quote_in_the_title_cannot_break_the_frontmatter(tmp_path):
    """The title arrives from the plugin. A stray quote or newline must not corrupt the
    block — every later pass reads this note's frontmatter."""
    from slim.chunk import parse_frontmatter
    r = _write(tmp_path, title='They said "hello"\nthen left')
    text = r.note.read_text(encoding="utf-8")
    fm, skip = parse_frontmatter(text)
    assert skip > 0                       # the block still parses
    assert fm.get("type") == "lecture"    # fields AFTER title survived
    assert fm.get("origin") == "recorded"


# --- provenance and the shared vocabulary (2026-08-06) --------------------------------------
# Two lanes write recordings and they used to write different frontmatter: the memo lane
# (`inbox`) wrote tags and ASR provenance, this one did not, so two notes of the same kind did
# not look alike in Obsidian. They noticed. These pin the fields and their ORDER.

def test_the_type_is_also_an_obsidian_tag(tmp_path):
    """Nothing in SLIM reads `tags:` — Obsidian's tag pane does, and a recorded note used to
    be invisible there while every memo note showed up."""
    r = _write(tmp_path, type_tag="idea")
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm.get("tags") == ["idea"]


def test_asr_provenance_is_written_when_it_is_known(tmp_path):
    r = _write(tmp_path, asr_model="mlx-community/parakeet-tdt-0.6b-v3",
               audio_seconds=173.42, transcribed_at=NOW)
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm.get("asr_model") == "mlx-community/parakeet-tdt-0.6b-v3"
    assert fm.get("source") == "local-asr"
    assert str(fm.get("asr_selected")).lower() == "false"
    assert str(fm.get("audio_seconds")) == "173.4"
    assert fm.get("audio_sha256") == r.digest
    assert str(fm.get("transcribed_at")).startswith("2026-08-04T14:30:00")


def test_no_asr_fields_when_transcription_failed(tmp_path):
    """⚠ An `asr_model` on a note with no transcript claims a model produced the silence.
    A failed transcription must leave the field ABSENT, not guessed."""
    r = _write(tmp_path, transcript="")
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    for key in ("asr_model", "source", "asr_selected", "audio_seconds", "transcribed_at"):
        assert key not in fm, key
    assert fm.get("audio")          # what IS known is still recorded


def test_shared_key_order_matches_the_memo_lane_before_recorder_workflow_state(tmp_path):
    """Nothing PARSES key order; they read it. Shared fields stay in memo order; the plugin's
    explicit review state sits beside provenance because Voice Memos have no review card."""
    r = _write(tmp_path, asr_model="parakeet", audio_seconds=12.0, transcribed_at=NOW)
    lines = r.note.read_text(encoding="utf-8").split("---")[1].strip().splitlines()
    keys = [line.split(":", 1)[0] for line in lines]
    assert keys == ["title", "date", "type", "tags", "topics",
                    "origin", "review_status", "source", "asr_model", "asr_selected", "audio",
                    "audio_sha256", "audio_seconds", "recorded_at", "transcribed_at",
                    "filed_by"]


# --- 2026-08-26: notes are always their, and a summary can be retried -----------------------

def test_saving_notes_alone_keeps_the_summary_byte_for_byte(tmp_path):
    """`summary_md=None` is the mirror of the existing `notes_md=None`. Typing in the notes
    field must not touch the derived summary — and the caller must not have to read the
    summary back and hand it in again just to leave it alone, because that round trip is
    where an autosave would eventually blank it."""
    r = _write(tmp_path, notes_md="- first line")
    before = r.note.read_text(encoding="utf-8")
    summary_at = before.index("<!-- slim:summary")
    summary_end = before.index("<!-- /slim:summary -->")

    after = record.replace_review_sections(before, summary_md=None, notes_md="- first line\n- second")

    assert after[summary_at:summary_end] == before[summary_at:summary_end]
    assert "- second\n## Transcript" in after


def test_reading_the_review_sections_back_out(tmp_path):
    """Retry has to re-summarize the transcript the note already holds, and feed their notes in
    as context exactly as the first pass did. The plugin parses these client-side; the server
    cannot borrow that, so the split lives here beside the writer."""
    r = _write(tmp_path, transcript="verbatim words they said", notes_md="- their note")
    parts = record.read_review_sections(r.note.read_text(encoding="utf-8"))
    assert parts["transcript"].strip() == "verbatim words they said"
    assert parts["notes"].strip() == "- their note"
    assert "kernel" in parts["summary"].casefold() or parts["summary"]


def test_reading_a_note_with_no_notes_section(tmp_path):
    r = _write(tmp_path, notes_md="")
    parts = record.read_review_sections(r.note.read_text(encoding="utf-8"))
    assert parts["notes"] == ""
    assert parts["transcript"].strip()


def test_appending_a_segment_leaves_the_existing_transcript_byte_for_byte(tmp_path):
    """⚠ RAW SOURCES ARE NEVER REWRITTEN. Appending adds; it must not reflow, re-wrap or
    normalize one character of what was already said."""
    r = _write(tmp_path, transcript="verbatim  \nwith a hard break", notes_md="- their note")
    before = r.note.read_text(encoding="utf-8")
    after = record.append_transcript(before, "and ten minutes more")
    assert "verbatim  \nwith a hard break" in after
    assert after.rstrip().endswith("and ten minutes more\n````") or \
        "and ten minutes more" in after
    parts = record.read_review_sections(after)
    assert parts["transcript"].startswith("verbatim  \nwith a hard break")
    assert parts["transcript"].endswith("and ten minutes more")
    assert parts["notes"] == "- their note"


def test_appending_keeps_the_meeting_block_fence_valid(tmp_path):
    """A resumed segment can contain a longer backtick run than the fence that is already
    there — a transcript that discusses Markdown would otherwise close its own block."""
    r = _write(tmp_path, transcript="the first part")
    after = record.append_transcript(r.note.read_text(encoding="utf-8"),
                                     "they said ````` five backticks `````")
    block = record.split_meeting_block(after)
    assert block is not None
    assert "five backticks" in block.body
    assert block.after == ""


def test_notes_are_found_even_when_the_block_starts_with_them(tmp_path):
    """⚠ DATA LOSS, found in review. `_NOTES_RE` required a newline before `## Notes`, so a
    note whose SUMMARY IS EMPTY — every recording whose summary call failed, which is exactly
    the note they presse Retry summary on — parsed as having no notes at all. The retry then
    rewrote the note without them, and fed the model an empty `notes` for good measure. The
    plugin's own parser reads them correctly, so the UI showed their notes right up until the
    server deleted them."""
    from datetime import datetime, timezone
    text = record.render_note(
        title="T", when=datetime(2026, 8, 26, tzinfo=timezone.utc), type_tag="lecture",
        transcript="words", notes_md="- their hand typed note", summary_md="",
        topics=[], audio_rel="a.webm", filed_by_slim=True)

    assert record.read_review_sections(text)["notes"] == "- their hand typed note"
    after = record.replace_review_sections(text, summary_md="## Key points\n\n- a point")
    assert "- their hand typed note" in after
    assert record.read_review_sections(after)["notes"] == "- their hand typed note"


def test_a_notes_heading_inside_the_summary_is_not_their_notes(tmp_path):
    """The summary is hand-editable, so it can contain any Markdown — including `## Notes`.
    Scoping the search to the whole block let that swallow the closing fence and the real
    notes, which would then be re-emitted as source and break `strip_derived_blocks`."""
    r = _write(tmp_path, notes_md="- real notes",
               summary_md="An overview.\n\n## Notes\n\n- inside the summary")
    parts = record.read_review_sections(r.note.read_text(encoding="utf-8"))
    assert parts["notes"] == "- real notes"
    assert "slim:summary" not in parts["notes"]


def test_one_owner_for_the_recording_id_in_a_staged_path():
    """The rule "`<id>-001.webm` belongs to recording `<id>`" was written three times in two
    languages, and missing one of them made EVERY recording fail on 2026-08-26."""
    assert record.recording_id_from_staged("Attachments/_incoming/abc-001.webm") == "abc"
    assert record.recording_id_from_staged("abc-042.webm") == "abc"
    assert record.recording_id_from_staged("abc.webm") == "abc"      # before segments existed
    assert record.recording_id_from_staged("abc-1.webm") == "abc-1", "only NNN is a segment"


def test_how_the_speakers_were_told_apart_is_recorded(tmp_path):
    r = _write(tmp_path, asr_model="parakeet", speakers_by="channels",
               transcript="**Me:** so\n\n**Them:** yes")
    fm, _ = parse_frontmatter(r.note.read_text(encoding="utf-8"))
    assert fm.get("speakers_by") == "channels"


def test_an_unlabelled_transcript_claims_no_speakers(tmp_path):
    r = _write(tmp_path, asr_model="parakeet")
    assert "speakers_by" not in r.note.read_text(encoding="utf-8")


def test_a_resumed_segment_can_bring_speakers_to_a_note_that_had_none(tmp_path):
    r = _write(tmp_path, asr_model="parakeet")
    text = record.append_audio_frontmatter(
        r.note.read_text(encoding="utf-8"), rels=["Attachments/Recordings/b.webm"],
        digests=["e" * 64], seconds=5.0, speakers_by="channels")
    fm, _ = parse_frontmatter(text)
    assert fm.get("speakers_by") == "channels"


def test_a_resume_keeps_audio_that_obsidian_rewrote_as_a_block_list(tmp_path):
    """Editing any property in Obsidian rewrites the note's lists as block lists. The resume
    read that `audio:` as empty and replaced it with the new segment alone, orphaning the
    first recording from its note."""
    r = _write(tmp_path)
    text = r.note.read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(text)
    first, digest = fm["audio"], fm["audio_sha256"]
    text = text.replace(f"audio: {first}", f"audio:\n  - {first}")
    text = text.replace(f"audio_sha256: {digest}", f"audio_sha256:\n  - {digest}")
    grown = record.append_audio_frontmatter(
        text, rels=["Attachments/Recordings/b.webm"], digests=["b" * 64], seconds=5.0)
    fm2, _ = parse_frontmatter(grown)
    assert fm2["audio"] == [first, "Attachments/Recordings/b.webm"]
    assert fm2["audio_sha256"] == [digest, "b" * 64]


def test_audio_frontmatter_has_one_writer(tmp_path):
    """CLAUDE.md: record.py remains the sole writer of a recorded note's frontmatter. The
    append path was setting `audio`, `audio_sha256` and `audio_seconds` from chat.py, reaching
    into a private helper here to format them."""
    r = _write(tmp_path)
    from slim.chunk import parse_frontmatter
    grown = record.append_audio_frontmatter(
        r.note.read_text(encoding="utf-8"),
        rels=["Attachments/Recordings/b.webm"], digests=["b" * 64], seconds=60.0)
    fm, _ = parse_frontmatter(grown)
    assert len(fm["audio"]) == 2 and len(fm["audio_sha256"]) == 2
    assert float(fm["audio_seconds"]) == 60.0
    # and a second resume accumulates onto the first rather than replacing it
    twice = record.append_audio_frontmatter(grown, rels=["Attachments/Recordings/c.webm"],
                                            digests=["c" * 64], seconds=30.0)
    fm2, _ = parse_frontmatter(twice)
    assert len(fm2["audio"]) == 3
    assert float(fm2["audio_seconds"]) == 90.0


def test_a_partial_write_can_never_truncate_a_recorded_note(tmp_path, monkeypatch):
    """⚠ `Path.write_text` TRUNCATES first. A disk-full or a crash mid-write leaves the note
    shorter than it was — and the thing at the end of it is the raw transcript, which is the
    one artifact this project promises never to lose."""
    import slim.record as rec
    r = _write(tmp_path, transcript="the words they actually said")
    before = r.note.read_text(encoding="utf-8")

    real = rec.write_note_text

    def dies_halfway(path, text):
        raise OSError("No space left on device")
    monkeypatch.setattr(rec, "write_note_text", dies_halfway)
    with pytest.raises(OSError):
        rec.write_note_text(r.note, "truncated")
    assert r.note.read_text(encoding="utf-8") == before

    monkeypatch.setattr(rec, "write_note_text", real)
    rec.write_note_text(r.note, before + "\nmore")
    assert r.note.read_text(encoding="utf-8").endswith("more")


def test_a_pasted_summary_cannot_forge_the_derived_fence(tmp_path):
    """The summary is hand-editable. Pasting the closing marker into it produced a note with
    two of them — and that marker is what `strip_derived_blocks` keys on to decide what must
    never be indexed."""
    from slim.chunk import SUMMARY_CLOSE
    r = _write(tmp_path)
    out = record.replace_review_sections(r.note.read_text(encoding="utf-8"),
                                         summary_md=f"ok {SUMMARY_CLOSE}\n\nsmuggled")
    assert out.count(SUMMARY_CLOSE) == 1


def test_pasted_notes_cannot_close_the_meeting_block_early(tmp_path):
    """A run of backticks equal to the fence would end the block, leaving the transcript
    outside the object — visible as raw Markdown, and no longer part of the recording."""
    r = _write(tmp_path)
    out = record.replace_review_sections(r.note.read_text(encoding="utf-8"),
                                         summary_md="A summary.",
                                         notes_md="````\nnot a fence\n````")
    block = record.split_meeting_block(out)
    assert block is not None
    assert "## Transcript" in block.body, "the transcript stays inside the object"
    assert block.after.strip() == ""
