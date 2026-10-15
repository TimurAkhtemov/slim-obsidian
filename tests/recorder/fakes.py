"""Fakes shared by the recorder suites, with the slow parts faked: no model is ever called.

`_archive_in_tmp` is autouse, so a suite that imports it by name gets it.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

NOW = datetime(2026, 8, 4, 14, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _archive_in_tmp(tmp_path, monkeypatch):
    from slim.capture import audio
    monkeypatch.setattr(audio, "AUDIO_ARCHIVE", tmp_path / "Attachments" / "Recordings")


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
    from slim.capture import summarize, transcribe
    from slim.recorder import suggest

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


def _existing(tmp_path, monkeypatch, dest="Capture/_unfiled"):
    from slim.recorder import note
    src = tmp_path / "Attachments" / "_incoming" / "a.webm"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"audio")
    return note.write_recording(
        audio_src=src, title="T", when=NOW, type_tag="lecture", transcript="verbatim words",
        notes_md="", summary_md="s", topics=["school"],
        dest_dir=dest, filed_by_slim=True, vault=tmp_path)


def _segments_tx(monkeypatch, *results):
    """One Transcript per segment, in order: (text, speakers_by, solo_speaker)."""
    from slim.capture import transcribe
    queue = list(results)

    def fake_tx(p, model=None):
        text, by, solo = queue.pop(0)
        return transcribe.Transcript(text=text, model="fake", audio_seconds=60.0,
                                     wall_seconds=1.0, speakers_by=by, solo_speaker=solo)
    monkeypatch.setattr(transcribe, "transcribe", fake_tx)
