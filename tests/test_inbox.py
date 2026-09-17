"""Tests for the §16 audio pipeline.

The rules under test are the ones that protect the archive. This pipeline will one day be the
only thing between a recording and the vault — the Notion pull learned what that costs.

ASR itself is stubbed: these pin the PIPELINE's contract, not parakeet's accuracy (that is
M8's bake-off, and it needs 3.6 hours of audio).
"""
import json
from pathlib import Path

import pytest

from slim import inbox as inbox_mod
from slim import transcribe as tx


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A throwaway vault + brain dir. Never touches the real ones."""
    v = tmp_path / "vault"
    data = tmp_path / "data"
    (v / "Inbox" / "Journal").mkdir(parents=True)

    monkeypatch.setattr(inbox_mod, "VAULT", v)
    monkeypatch.setattr(inbox_mod, "INBOX", {"journal": v / "Inbox" / "Journal"})
    monkeypatch.setattr(inbox_mod, "AUDIO_ARCHIVE", v / "Attachments" / "Recordings")
    monkeypatch.setattr(inbox_mod, "TMP_AUDIO_DIR", data / "audio" / "tmp")
    monkeypatch.setattr(inbox_mod, "MANIFEST", data / "audio" / "manifest.json")

    # Stub the ASR and the ffmpeg hop: this is a pipeline test.
    monkeypatch.setattr(inbox_mod.tx, "to_wav16k", lambda src, dst: src)
    monkeypatch.setattr(inbox_mod.tx, "transcribe", lambda p, model=None: tx.Transcript(
        text="So we agreed Priya would send the roster by Friday.",
        model=model or tx.MODEL, audio_seconds=600.0, wall_seconds=10.0))
    monkeypatch.setattr(inbox_mod, "recorded_at",
                        lambda p: __import__("datetime").datetime(2026, 7, 14, 9, 30).astimezone())
    return v


def drop(vault: Path, name: str, content: bytes = b"fake audio") -> Path:
    """Drop one recording into the memo lane's ONE inbox. There is no second kind."""
    p = inbox_mod.INBOX["journal"] / name
    p.write_bytes(content)
    return p


def test_a_memo_lands_in_the_floor_with_its_transcript(vault):
    """The phone lane writes journals and nothing else (2026-09-03). Its transcript has no
    spoken type, so nothing routes it out and it keeps the floor."""
    drop(vault, "morning thought.m4a")
    (r,) = inbox_mod.process()
    assert r.status == "written"
    assert r.note.parent == vault / "Journal"

    text = r.note.read_text()
    assert "type: journal" in text
    assert "## Transcript" in text          # the marker `record.py` and `chunk.py` split on
    assert "Priya would send the roster" in text


def test_no_summary_is_ever_written(vault):
    """A summary is a DERIVED artifact; ingesting it would feed the brain its own opinions as
    evidence. And M4 selected no summarization model. The pipeline writes the transcript only."""
    drop(vault, "standup.m4a")
    (r,) = inbox_mod.process()
    body = r.note.read_text().lower()
    for derived in ("## summary", "## action items", "## decisions", "## key points"):
        assert derived not in body


def test_rerunning_is_idempotent_by_content(vault):
    drop(vault, "sync.m4a", b"identical bytes")
    (first,) = inbox_mod.process()
    assert first.status == "written"

    # Same audio arrives again under a different name — it is the SAME recording.
    drop(vault, "sync copy.m4a", b"identical bytes")
    (second,) = inbox_mod.process()
    assert second.status == "skipped"
    assert second.note == first.note


def test_never_overwrites_an_existing_note(vault):
    """The Notion pull could destroy a transcript by overwriting in place. Not here."""
    drop(vault, "sync.m4a", b"aaa")
    (first,) = inbox_mod.process()
    original = first.note.read_text()

    # Forge the collision: different audio, same computed note path.
    first.note.write_text(original + "\nhuman edit that must survive\n")
    drop(vault, "sync.m4a", b"bbb")
    import hashlib
    digest = hashlib.sha256(b"bbb").hexdigest()
    monkeyed = inbox_mod._note_path(inbox_mod.recorded_at(Path(".")), "sync", digest)
    monkeyed.parent.mkdir(parents=True, exist_ok=True)
    monkeyed.write_text("a note that already exists")

    (r,) = inbox_mod.process()
    assert r.status == "refused"
    assert monkeyed.read_text() == "a note that already exists"
    assert "human edit that must survive" in first.note.read_text()


def test_raw_audio_is_preserved_in_the_synced_vault(vault):
    a = drop(vault, "sync.m4a", b"precious recording")
    inbox_mod.process()

    assert not a.exists(), "processed audio should leave the active Inbox"
    archived = list(inbox_mod.AUDIO_ARCHIVE.glob("*.m4a"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"precious recording"   # byte-identical, never rewritten
    assert archived[0].is_relative_to(vault)


def test_corrupt_existing_archive_never_deletes_the_source(vault):
    a = drop(vault, "sync.m4a", b"precious recording")
    digest = inbox_mod._sha256(a)
    archived = inbox_mod.AUDIO_ARCHIVE / f"{digest}.m4a"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        inbox_mod.process()

    assert a.read_bytes() == b"precious recording"
    assert not list(vault.rglob("*.md"))


def test_dry_run_writes_nothing(vault):
    drop(vault, "sync.m4a")
    (r,) = inbox_mod.process(dry_run=True)
    assert r.status == "written" and not r.note.exists()
    assert not inbox_mod.MANIFEST.exists()


def test_the_processed_audio_always_leaves_the_inbox(vault):
    """`--keep-audio` is gone (2026-09-03): the source is archived in the vault before the
    Inbox copy is removed, so a second copy in staging was a flag with nothing to protect."""
    a = drop(vault, "sync.m4a")
    inbox_mod.process()
    assert not a.exists()


def test_frontmatter_records_that_the_asr_was_never_selected(vault):
    """Deterministic truth in metadata, not inference. M8 selected NOBODY; a future
    re-transcription must be able to find every note produced by an unvalidated model."""
    drop(vault, "sync.m4a")
    (r,) = inbox_mod.process()
    text = r.note.read_text()
    assert "asr_selected: false" in text
    assert f"asr_model: {tx.MODEL}" in text
    assert "date: 2026-07-14" in text          # feeds ingest's authored_at
    assert "audio_sha256:" in text


def test_journal_note_gains_spoken_tags_from_the_opening(vault, monkeypatch):
    """P2: a journal memo's spoken opening ("this is a journal") becomes a `tags:` line,
    parsed deterministically in code. The transcript body is never modified."""
    monkeypatch.setattr(inbox_mod.tx, "transcribe", lambda p, model=None: tx.Transcript(
        text="This is a journal. Today I felt good about the SLIM project.",
        model=tx.MODEL, audio_seconds=30.0, wall_seconds=2.0))
    drop(vault, "morning.m4a")
    (r,) = inbox_mod.process()
    text = r.note.read_text()
    assert "tags: [journal]" in text
    assert "This is a journal. Today I felt good about the SLIM project." in text


def test_an_untyped_memo_carries_no_tags_line(vault):
    """One spoken vocabulary (2026-09-03): with nothing said and nothing inferred there is no
    type to write, so the note says so by omission rather than by inventing a triage tag."""
    drop(vault, "unlabeled.m4a")            # fixture transcript has no type word
    (r,) = inbox_mod.process()
    text = r.note.read_text()
    assert "tags:" not in text
    assert "tagged_by: none" in text


def test_manifest_links_note_to_archived_audio(vault):
    drop(vault, "sync.m4a")
    (r,) = inbox_mod.process()
    manifest = json.loads(inbox_mod.MANIFEST.read_text())
    (entry,) = manifest.values()
    assert entry["note"] == str(r.note)
    assert Path(entry["audio"]).exists()


def test_swept_hash_suffix_is_stripped_from_the_note_title(vault):
    """The voice-memo sweep names its inbox copies `<stem>--<hash8>`; the note title must
    not repeat that suffix (the note path re-appends the digest, and a vault note named
    `…--f70f13c6--f70f13c6` is forever — caught on the first live e2e run)."""
    drop(vault, "morning-thoughts--a1b2c3d4.m4a")
    (r,) = inbox_mod.process()
    assert 'title: "morning-thoughts"' in r.note.read_text()
    assert "a1b2c3d4--" not in r.note.name, \
        "the sweep suffix must not survive into the note filename ahead of the digest"
