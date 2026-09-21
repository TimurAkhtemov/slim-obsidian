"""Who spoke is decided by which channel was making sound — a measurement, not a model."""

import shutil
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from slim import speakers

QUIET, LOUD = -90.0, -20.0

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def tok(text, start, end):
    return SimpleNamespace(text=text, start=start, end=end)


def _wav(path, left_spans, right_spans, seconds=6, channels=2):
    """A 16 kHz WAV: a 440 Hz tone on the left inside left_spans, 880 Hz on the right."""
    sr = 16000
    t = np.arange(seconds * sr) / sr
    sides = []
    for freq, spans in ((440, left_spans), (880, right_spans)):
        gate = np.zeros_like(t)
        for a, b in spans:
            gate[int(a * sr):int(b * sr)] = 1.0
        sides.append(0.3 * np.sin(2 * np.pi * freq * t) * gate)
    data = np.stack(sides[:channels], axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767).astype("<i2").tobytes())
    return path


def levels(seconds, mic=(), system=()):
    """(2, frames) dBFS: QUIET everywhere, LOUD inside each (start, end) span."""
    frames = int(seconds / speakers.FRAME_SECONDS)
    out = np.full((2, frames), QUIET)
    for row, spans in ((0, mic), (1, system)):
        for a, b in spans:
            out[row, int(a / speakers.FRAME_SECONDS):int(b / speakers.FRAME_SECONDS)] = LOUD
    return out


def test_a_token_is_labelled_by_the_channel_that_was_making_sound():
    lv = levels(6, mic=[(0, 2)], system=[(3, 5)])
    tokens = [tok(" hello", 0.5, 0.9), tok(" there", 3.5, 3.9)]
    assert speakers.label_tokens(tokens, lv) == [speakers.ME, speakers.THEM]


def test_system_audio_wins_when_both_channels_are_loud():
    """On speakers the far end leaks into the microphone — but only while the far end is
    talking, so those moments are already Them."""
    lv = levels(4, mic=[(0, 4)], system=[(1, 3)])
    assert speakers.label_tokens([tok(" bleed", 1.5, 1.9)], lv) == [speakers.THEM]


def test_a_token_in_silence_inherits_the_speaker_before_it():
    lv = levels(6, system=[(0, 2)])
    tokens = [tok(" so", 0.5, 0.9), tok(" anyway", 4.0, 4.4)]
    assert speakers.label_tokens(tokens, lv) == [speakers.THEM, speakers.THEM]


def test_a_leading_token_in_silence_takes_the_first_decided_speaker():
    lv = levels(6, mic=[(2, 4)])
    tokens = [tok(" um", 0.2, 0.4), tok(" right", 2.5, 2.9)]
    assert speakers.label_tokens(tokens, lv) == [speakers.ME, speakers.ME]


def test_the_tail_of_their_speech_is_still_theirs():
    """Room reverb outlasts the digital signal: just after system audio stops, the microphone
    is still loud with THEIR voice. The hangover keeps that tail from becoming Me."""
    lv = levels(4, mic=[(0, 2.2)], system=[(0, 2.0)])
    assert speakers.label_tokens([tok(" end", 2.02, 2.18)], lv) == [speakers.THEM]


def test_a_token_past_the_end_of_the_audio_does_not_crash():
    lv = levels(1, mic=[(0, 1)])
    assert speakers.label_tokens([tok(" late", 5.0, 5.2)], lv) == [speakers.ME]


def test_no_frames_means_no_labels():
    assert speakers.label_tokens([tok(" x", 0, 1)], np.zeros((2, 0))) == [""]


def test_consecutive_tokens_from_one_speaker_are_one_turn():
    tokens = [tok(" so", 0, .3), tok(" what", .3, .6), tok(" yes", 2, 2.5), tok(" indeed", 2.5, 3)]
    labels = [speakers.ME, speakers.ME, speakers.THEM, speakers.THEM]
    assert speakers.turns(tokens, labels) == [(speakers.ME, "so what"), (speakers.THEM, "yes indeed")]


def test_a_flicker_shorter_than_a_turn_joins_the_turn_before_it():
    tokens = [tok(" we", 0, .5), tok(" ship", .5, 1.0), tok(" uh", 1.0, 1.1),
              tok(" on", 1.1, 1.6), tok(" friday", 1.6, 2.2)]
    labels = [speakers.THEM, speakers.THEM, speakers.ME, speakers.THEM, speakers.THEM]
    assert speakers.turns(tokens, labels) == [(speakers.THEM, "we ship uh on friday")]


def test_a_flicker_at_the_very_start_joins_the_turn_after_it():
    tokens = [tok(" uh", 0, .1), tok(" hello", .1, .8), tok(" everyone", .8, 1.5)]
    labels = [speakers.ME, speakers.THEM, speakers.THEM]
    assert speakers.turns(tokens, labels) == [(speakers.THEM, "uh hello everyone")]


def test_turns_render_as_labelled_paragraphs():
    text = speakers.render([(speakers.ME, "so what"), (speakers.THEM, "yes indeed")])
    assert text == "**Me:** so what\n\n**Them:** yes indeed"


def test_a_word_split_across_pieces_is_not_labelled_in_the_middle():
    """Parakeet's tokens are sentencepiece PIECES, and a piece's start time can lead its own
    audio by up to ~100-250 ms. At a Me -> Them boundary the first piece of the new speaker's
    word can land while the system channel is still silent — decide per word, not per piece,
    or the word splits between two speakers."""
    lv = levels(11, mic=[(0, 9.0)], system=[(9.30, 11)])
    tokens = [tok(" Okay", 0.5, 0.9),
              tok(" Ne", 9.20, 9.36), tok("vert", 9.36, 9.60),
              tok("hel", 9.60, 9.76), tok("ess", 9.76, 9.92)]
    labels = speakers.label_tokens(tokens, lv)
    assert labels == [speakers.ME] + [speakers.THEM] * 4
    assert speakers.turns(tokens, labels) == [
        (speakers.ME, "Okay"), (speakers.THEM, "Nevertheless"),
    ]


def test_punctuation_never_opens_a_turn():
    """A piece that does not start with a space — here trailing punctuation — joins the word
    before it rather than being decided on its own."""
    lv = levels(2, mic=[(0, 0.5)], system=[(0.48, 2)])
    tokens = [tok(" Hello", 0.0, 0.48), tok(".", 0.48, 0.52), tok(" world", 0.6, 1.0)]
    labels = speakers.label_tokens(tokens, lv)
    for _, text in speakers.turns(tokens, labels):
        assert not text.startswith(".")


def test_a_leading_piece_with_no_space_still_opens_a_word():
    lv = levels(1, mic=[(0, 1)])
    assert speakers.label_tokens([tok("Hi", 0.0, 0.5)], lv) == [speakers.ME]


@needs_ffmpeg
def test_levels_come_back_per_channel(tmp_path):
    lv = speakers.channel_levels(_wav(tmp_path / "a.wav", [(0, 2)], [(3, 5)]))
    assert lv.shape == (2, 300)
    frame = lambda s: int(s / speakers.FRAME_SECONDS)  # noqa: E731
    assert lv[0, frame(1)] > -20 and lv[1, frame(1)] < -80
    assert lv[1, frame(4)] > -20 and lv[0, frame(4)] < -80


@needs_ffmpeg
def test_a_two_sided_recording_is_labelled(tmp_path):
    path = _wav(tmp_path / "a.wav", [(0, 2)], [(3, 5)])
    tokens = [tok(" so", 0.4, 0.9), tok(" what", 0.9, 1.5), tok(" yes", 3.4, 4.0)]
    assert speakers.label(path, tokens) == "**Me:** so what\n\n**Them:** yes"


@needs_ffmpeg
def test_one_speaker_is_not_labelled_at_all(tmp_path):
    """A lecture is all Them. A label on every paragraph of it is noise."""
    path = _wav(tmp_path / "a.wav", [], [(0, 5)])
    assert speakers.label(path, [tok(" today", 0.5, 1.0), tok(" kernels", 3.0, 3.6)]) is None


@needs_ffmpeg
def test_a_mono_file_is_never_analysed(tmp_path, monkeypatch):
    path = _wav(tmp_path / "a.wav", [(0, 2)], [], channels=1)
    monkeypatch.setattr(speakers, "channel_levels",
                        lambda p: pytest.fail("a mono file has no sides to compare"))
    assert speakers.channel_count(path) == 1
    assert speakers.label(path, [tok(" memo", 0.5, 1.0)]) is None


def test_no_tokens_means_nothing_to_label(tmp_path):
    assert speakers.label(tmp_path / "missing.webm", []) is None
