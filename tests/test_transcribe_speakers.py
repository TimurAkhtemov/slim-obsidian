"""`transcribe()` used to keep `.text` and throw the token times away. They are what makes
"who said this" answerable."""

from types import SimpleNamespace

import pytest

from slim import speakers, transcribe


class _FakeASR:
    def transcribe(self, path, chunk_duration, overlap_duration):
        tokens = [SimpleNamespace(text=" so", start=0.0, end=0.5),
                  SimpleNamespace(text=" yes", start=2.0, end=2.5)]
        return SimpleNamespace(text=" so yes", tokens=tokens)


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch):
    monkeypatch.setattr(transcribe, "ensure_ffmpeg_on_path", lambda: None)
    monkeypatch.setattr(transcribe, "audio_seconds", lambda p: 3.0)
    monkeypatch.setattr(transcribe, "_asr_model", lambda name, lane: _FakeASR())


def test_a_two_sided_recording_comes_back_labelled(tmp_path, monkeypatch):
    seen = {}

    def fake_label(path, tokens):
        seen["tokens"] = [t.text for t in tokens]
        return "**Me:** so\n\n**Them:** yes", ""

    monkeypatch.setattr(speakers, "label", fake_label)
    t = transcribe.transcribe(tmp_path / "a.webm")
    assert t.text == "**Me:** so\n\n**Them:** yes"
    assert t.speakers_by == "channels"
    assert seen["tokens"] == [" so", " yes"]


def test_nothing_to_tell_apart_leaves_the_text_exactly_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr(speakers, "label", lambda path, tokens: (None, ""))
    t = transcribe.transcribe(tmp_path / "a.webm")
    assert t.text == "so yes"
    assert t.speakers_by == "" and t.solo_speaker == ""


def test_one_side_speaking_stays_unlabelled_but_says_who(tmp_path, monkeypatch):
    monkeypatch.setattr(speakers, "label", lambda path, tokens: (None, "Them"))
    t = transcribe.transcribe(tmp_path / "a.webm")
    assert (t.text, t.speakers_by, t.solo_speaker) == ("so yes", "", "Them")


def test_a_labelling_failure_never_costs_the_transcript(tmp_path, monkeypatch):
    """The transcript IS the note. Labels decorate it."""
    from slim import trace

    traced = []
    monkeypatch.setattr(trace, "record", lambda kind, facts: traced.append((kind, facts)))

    def boom(path, tokens):
        raise RuntimeError("ffmpeg could not decode a.webm")

    monkeypatch.setattr(speakers, "label", boom)
    t = transcribe.transcribe(tmp_path / "a.webm")
    assert t.text == "so yes" and t.speakers_by == ""
    assert traced == [("record", {"stage": "speakers", "status": "failed",
                                  "error": "ffmpeg could not decode a.webm"})]
