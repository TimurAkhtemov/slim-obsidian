"""The card's edits to a filed recording (`record_review`): apply, approval, the notes
autosave and the summary retry, none of which may lose what the note already holds."""
import pytest

from tests.record_fakes import _archive_in_tmp, _existing  # noqa: F401


# --- POST /api/record/apply -------------------------------------------------------------

def test_an_edit_moves_the_note_and_marks_the_fields_as_theirs(tmp_path, monkeypatch):
    """⚠ Writing the new value is not enough. A model pass re-judges what it INFERRED but never
    overrides a declared value, so a card edit that only writes the value gets reverted by a
    later pass — they correct the same note twice and concludes correcting does not work."""
    from slim.recorder import review
    from slim.vault.frontmatter import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    out = review.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
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
    from slim.recorder import review
    from slim.vault.frontmatter import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    before = r.note.read_text(encoding="utf-8")
    before_fm, _ = parse_frontmatter(before)
    assert before_fm.get("filed_by") == "slim"

    out = review.handle_record_review(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)

    after = (tmp_path / out["note"]).read_text(encoding="utf-8")
    after_fm, _ = parse_frontmatter(after)
    assert (tmp_path / out["note"]) == r.note
    assert after.replace("review_status: complete", "review_status: pending") == \
        before.replace("filed_by: slim\n", "")
    expected = {k: v for k, v in before_fm.items() if k != "filed_by"}
    assert after_fm == {**expected, "review_status": "complete"}


def test_the_transcript_is_untouched_by_an_edit(tmp_path, monkeypatch):
    """The card edits interpretations. The raw record is never rewritten — no exceptions."""
    from slim.recorder import review
    r = _existing(tmp_path, monkeypatch)
    review.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                      dest_dir="Capture/_unfiled", type_tag="idea", topics=["x"],
                                      summary_md="s", vault=tmp_path)
    assert "verbatim words" in r.note.read_text(encoding="utf-8")


def test_review_edits_persist_both_summary_and_notes(tmp_path, monkeypatch):
    from slim.recorder import note as record, review
    r = _existing(tmp_path, monkeypatch)
    review.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                      dest_dir="Capture/_unfiled", type_tag="lecture",
                                      topics=["school"], summary_md="Edited AI summary.",
                                      notes_md="- note added during review", vault=tmp_path)
    text = r.note.read_text(encoding="utf-8")
    assert "Edited AI summary." in text
    assert "- note added during review" in text
    block = record.split_meeting_block(text)
    assert block is not None
    assert block.body.rstrip().endswith("verbatim words")


def test_applying_never_silently_overwrites_another_note(tmp_path, monkeypatch):
    """Both movers grew this guard on 2026-08-04 after two notes could resolve to one
    destination and one was lost. A third mover must not reintroduce it."""
    from slim.recorder import review
    r = _existing(tmp_path, monkeypatch)
    victim = tmp_path / "Notes/school/CS201 ML" / r.note.name
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_text("I was here first", encoding="utf-8")
    with pytest.raises(ValueError):
        review.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                          dest_dir="Notes/school/CS201 ML", type_tag="lecture",
                                          topics=[], summary_md="s", vault=tmp_path)
    assert victim.read_text(encoding="utf-8") == "I was here first"


def test_editing_the_title_renames_the_file_and_keeps_date_and_digest(tmp_path, monkeypatch):
    """The filename is what they read in the file tree and the quick switcher. A note retitled
    only in frontmatter is still wrongly named where it counts."""
    from slim.recorder import review
    from slim.vault.frontmatter import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    date, _slug, digest = r.note.stem.split("--")
    out = review.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                            dest_dir="Capture/_unfiled", type_tag="lecture",
                                            topics=[], summary_md="s", title="Contoso Screen",
                                            vault=tmp_path)
    new = tmp_path / out["note"]
    assert new.exists() and not r.note.exists()
    assert new.stem.startswith(date) and new.stem.endswith(digest)   # both preserved
    assert "Contoso-Screen" in new.name
    fm, _ = parse_frontmatter(new.read_text(encoding="utf-8"))
    assert fm.get("title") == "Contoso Screen"


def test_the_card_reindexes_after_it_commits(tmp_path, monkeypatch):
    """The card changes both content and path, so the index must be refreshed after it, not
    only after the initial filing."""
    from slim.recorder import review, state
    r = _existing(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(state, "_index_now",
                        lambda v, note, **kw: called.append((v, note, kw.get("moved_from"))) or {})
    original = str(r.note.relative_to(tmp_path))
    moved = review.handle_record_apply(note=original, dest_dir="Capture/work",
                                              type_tag="lecture", topics=[], summary_md="s",
                                              vault=tmp_path)
    # The MOVED path, and where it came from: the index keeps the note's id through the move.
    assert called == [(tmp_path, moved["note"], original)]


def test_filing_keeps_the_source_id_a_copilot_chat_was_opened_on(tmp_path, monkeypatch):
    """The copilot can be opened on the draft before the card is applied, and its threads are
    keyed by source id. Apply changes content AND path in one ingest — the one sequence the
    hash rename rule cannot follow — so without a hint the note is re-minted and every chat
    on it is orphaned (2026-09-22)."""
    from slim import db
    from slim.recorder import review, state
    from slim.index import embed, ingest
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(embed, "embed_source", lambda con, sid: 0)
    monkeypatch.setattr(state, "_schedule_sweep", lambda vault: None)
    original = str(r.note.relative_to(tmp_path))
    con = db.connect()
    draft_id = ingest.ingest_note(con, tmp_path, original)["id"]
    con.close()
    out = review.handle_record_apply(note=original, dest_dir="Notes/school",
                                            type_tag="lecture", topics=["school"],
                                            summary_md="new summary", vault=tmp_path)
    con = db.connect()
    rows = con.execute("SELECT id, path FROM sources WHERE deleted=0").fetchall()
    con.close()
    assert [(row["id"], row["path"]) for row in rows] == [(draft_id, out["note"])]


# --- 2026-08-26: their notes save on their own, and a summary can be retried ----------------

def test_saving_notes_touches_nothing_but_the_notes(tmp_path, monkeypatch):
    """The finished block now edits their notes in place, so it needs a save path that is NOT
    `apply`: apply rewrites frontmatter, drops `filed_by` and can MOVE the file. Typing a note
    is not a filing decision, and an autosave that files is an autosave that surprises."""
    from slim.recorder import review
    from slim.vault.frontmatter import parse_frontmatter
    r = _existing(tmp_path, monkeypatch)
    before = r.note.read_text(encoding="utf-8")

    out = review.handle_record_notes(note=str(r.note.relative_to(tmp_path)),
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
    from slim.recorder import review
    from slim.capture import summarize
    r = _existing(tmp_path, monkeypatch)
    seen = {}

    def fake_sum(transcript, **kw):
        seen.update(transcript=transcript, **kw)
        return summarize.Summary(overview="A better summary.",
                                 key_points=[{"topic": "Kernels", "points": ["The dual form."]}])
    monkeypatch.setattr(summarize, "summarize", fake_sum)

    out = review.handle_record_summary(note=str(r.note.relative_to(tmp_path)),
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
    from slim.recorder import review
    from slim.capture import summarize
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(summarize, "summarize",
                        lambda *a, **k: summarize.Summary(valid=False, error="ctx_saturated"))
    with pytest.raises(ValueError, match="ctx_saturated"):
        review.handle_record_summary(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)
    assert "\ns\n" in r.note.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_a_retry_on_a_lecture_still_drops_open_questions(tmp_path, monkeypatch):
    """The lecture rule lives in render(), and the retry path has to read `type` from the note
    to reach it — a rhetorical question the lecturer asked and answered reads, under an 'Open
    questions' heading, as one of THEIR confusions."""
    from slim.recorder import review
    from slim.capture import summarize
    r = _existing(tmp_path, monkeypatch)          # written with type_tag="lecture"
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary(
        overview="o", key_points=[{"topic": "T", "points": ["p"]}],
        open_questions=["Does a dictionary know English?"]))
    review.handle_record_summary(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)
    assert "## Open questions" not in r.note.read_text(encoding="utf-8")


def test_correcting_the_type_to_a_lecture_drops_its_open_questions(tmp_path, monkeypatch):
    """MEASURED IN THE LIVE PLUGIN, 2026-08-26. The summary is rendered against the type the
    CARD guessed; the type they settle on is written afterwards. Guess `meeting-note`, correct
    it to `lecture`, and the note ends up typed lecture while carrying a section a lecture is
    not supposed to have — the lecturer's rhetorical questions, reading as their own."""
    from slim.recorder import review
    r = _existing(tmp_path, monkeypatch)
    summary = ("An overview.\n\n## Key points\n\n### Topic\n- a point\n\n"
               "## Open questions\n- What is understanding?\n")

    review.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                      dest_dir="Capture/school", type_tag="lecture", topics=[],
                                      summary_md=summary, vault=tmp_path)

    after = (tmp_path / "Capture/school" / r.note.name).read_text(encoding="utf-8")
    assert "## Open questions" not in after
    assert "- a point" in after                      # only the one section goes


def test_a_meeting_keeps_the_open_questions_it_was_given(tmp_path, monkeypatch):
    """A meeting's unresolved question is real information. The drop is scoped to the types
    where the field cannot mean what its heading says."""
    from slim.recorder import review
    r = _existing(tmp_path, monkeypatch)
    review.handle_record_apply(note=str(r.note.relative_to(tmp_path)),
                                      dest_dir="Capture/work", type_tag="meeting-note", topics=[],
                                      summary_md="o\n\n## Open questions\n- who owns it?",
                                      vault=tmp_path)
    after = (tmp_path / "Capture/work" / r.note.name).read_text(encoding="utf-8")
    assert "## Open questions" in after and "who owns it?" in after


# --- 2026-08-26, found in review: long jobs must not write what they read minutes ago -----

def test_a_retry_does_not_clobber_notes_saved_while_it_ran(tmp_path, monkeypatch):
    """⚠ The notes editor autosaves on blur, and a retry takes 60-300s. Reading the note at
    entry and writing that same text back at exit reverts anything they typed in between —
    silently, and only sometimes, which is the worst shape a bug can have."""
    from slim.recorder import review
    from slim.capture import summarize
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))

    def slow_summary(*a, **k):
        # what the blur-autosave does while the model is thinking
        review.handle_record_notes(note=note_rel, notes_md="- typed while it ran",
                                          vault=tmp_path)
        return summarize.Summary(overview="A new summary.")
    monkeypatch.setattr(summarize, "summarize", slow_summary)

    review.handle_record_summary(note=note_rel, vault=tmp_path)
    body = r.note.read_text(encoding="utf-8")
    assert "- typed while it ran" in body, "their words outrank a stale read"
    assert "A new summary." in body


def test_a_retry_never_resurrects_a_note_that_moved(tmp_path, monkeypatch):
    """`write_text` CREATES. If `apply` moved the note while the retry was thinking, writing
    the old path would put a second, stale copy of the recording back into the vault — and
    index it."""
    from slim.recorder import review
    from slim.capture import summarize
    r = _existing(tmp_path, monkeypatch)
    note_rel = str(r.note.relative_to(tmp_path))

    def moves_it(*a, **k):
        review.handle_record_apply(note=note_rel, dest_dir="Capture/work",
                                          type_tag="meeting-note", topics=[], summary_md="s",
                                          vault=tmp_path)
        return summarize.Summary(overview="A new summary.")
    monkeypatch.setattr(summarize, "summarize", moves_it)

    with pytest.raises(FileNotFoundError):
        review.handle_record_summary(note=note_rel, vault=tmp_path)
    assert not r.note.exists(), "the old path stays gone"


def test_an_autosave_does_not_sweep_the_vault(tmp_path, monkeypatch):
    """⚠ `_index_now`'s tail schedules a full-vault ingest + embed. On a note that is seconds
    old that is right — the sweep retires the old path's row. On every blur of the notes field
    it is a dozen full-vault hash-and-embed passes for a dozen click-aways, each competing with
    the resident model for the box. Index the note; leave the vault alone."""
    from slim.recorder import review, state
    r = _existing(tmp_path, monkeypatch)
    sweeps = []
    monkeypatch.setattr(state, "_schedule_sweep", lambda v: sweeps.append(v))

    review.handle_record_notes(note=str(r.note.relative_to(tmp_path)), notes_md="- typed",
                                      vault=tmp_path)
    assert sweeps == [], "typing a note is not a reason to re-embed the vault"


# --- 2026-08-26, second review: the destructive set ---------------------------------------

def test_an_empty_but_valid_summary_never_deletes_the_one_that_is_there(tmp_path, monkeypatch):
    """⚠ `valid` means SCHEMA-valid. `Summary()` passes and renders to "", which
    `replace_review_sections` writes as "no summary" — deleting the fence and its contents.
    "A failed retry keeps the summary it has" has to cover the model returning nothing."""
    from slim.recorder import review
    from slim.capture import summarize
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary())
    with pytest.raises(ValueError, match="empty"):
        review.handle_record_summary(note=str(r.note.relative_to(tmp_path)), vault=tmp_path)
    assert "\ns\n" in r.note.read_text(encoding="utf-8").replace("\r\n", "\n")
