"""The recorder's live transcript is provisional, local, and independently disposable."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from slim import transcribe


@pytest.fixture(autouse=True)
def _clear_live_sessions():
    transcribe._LIVE_SESSIONS.clear()
    yield
    for session in list(transcribe._LIVE_SESSIONS.values()):
        session.close()
    transcribe._LIVE_SESSIONS.clear()


def test_live_session_streams_pcm_and_closes(monkeypatch):
    events = []

    class FakeLive:
        def __init__(self, model=None):
            events.append(("start", model))

        def add_pcm16(self, pcm):
            events.append(("chunk", pcm))
            return "words so far"

        def close(self):
            events.append(("stop",))
            return "final provisional words"

    monkeypatch.setattr(transcribe, "LiveTranscriber", FakeLive)
    transcribe.live_start("recording-1")
    assert transcribe.live_add("recording-1", b"\x00\x00") == "words so far"
    assert transcribe.live_stop("recording-1") == "final provisional words"
    assert events == [("start", None), ("chunk", b"\x00\x00"), ("stop",)]


def test_a_new_live_session_evicts_the_previous_one(monkeypatch):
    """The MLX constraint is unchanged: exactly one live stream exists at a time.

    ⚠ WHAT CHANGED (2026-08-26) IS WHO WINS. This used to REFUSE a second session, and it
    cleaned up only a session with the same id — so one left behind by a crash, a killed server
    or a recording that never called stop disabled live captions for every recording after it,
    with a line of grey status text as the only sign. There is one recorder, so a new session
    now evicts the old one first. The invariant still holds: the old stream is closed BEFORE
    the new one is constructed, so two never coexist.
    """
    closed, made = [], []

    class FakeLive:
        def __init__(self, model=None):
            made.append(len(transcribe._LIVE_SESSIONS))

        def close(self):
            closed.append(True)
            return ""

    monkeypatch.setattr(transcribe, "LiveTranscriber", FakeLive)
    transcribe._LIVE_SESSIONS.clear()
    transcribe.live_start("first")
    transcribe.live_start("second")

    assert closed == [True], "the abandoned stream is closed, not left mutating attention"
    assert made == [0, 0], "each stream is built only once the registry is empty"
    assert list(transcribe._LIVE_SESSIONS) == ["second"]
    transcribe._LIVE_SESSIONS.clear()


def test_real_live_wrapper_keeps_mlx_stream_on_one_worker_thread(monkeypatch):
    thread_ids = []

    class FakeStream:
        result = SimpleNamespace(text="words so far")

        def __enter__(self):
            thread_ids.append(("enter", threading.get_ident()))

        def add_audio(self, _samples):
            thread_ids.append(("add", threading.get_ident()))

        def __exit__(self, *_args):
            thread_ids.append(("exit", threading.get_ident()))

    fake_stream = FakeStream()
    monkeypatch.setattr(transcribe, "ensure_ffmpeg_on_path", lambda: None)
    fake_mx = SimpleNamespace(array=lambda value: value)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setattr(
        transcribe,
        "_asr_model",
        lambda _name, _lane="batch": SimpleNamespace(transcribe_stream=lambda: fake_stream),
    )

    live = transcribe.LiveTranscriber()
    assert live.add_pcm16(b"\x00\x00") == "words so far"
    assert live.close() == "words so far"
    assert {thread_id for _, thread_id in thread_ids} == {thread_ids[0][1]}
    assert thread_ids[0][1] != threading.get_ident()


def test_final_parakeet_pass_does_not_wait_for_the_live_stream(monkeypatch):
    """⚠ THIS TEST WAS INVERTED ON 2026-08-27, AND THE INVERSION IS THE FEATURE.

    It used to assert `was_blocked` — one cached MLX model behind one lock, with the live
    stream owning it for the WHOLE capture. The consequence was the opposite of what parallel
    meetings are for: meeting A's final transcription waited for meeting B to STOP RECORDING,
    so A's summary could not arrive while B ran.

    Each lane now has its own instance (`lane` is a cache key) and its own lock. Measured
    2026-08-27: a second instance costs 1.21 GB, and both lanes driven from two threads on one
    GPU returned text byte-identical to their solo baselines, with no error.
    """
    batch_called = threading.Event()
    lanes: list[str] = []

    class FakeStream:
        result = SimpleNamespace(text="live words")

        def __enter__(self):
            return self

        def add_audio(self, _samples):
            pass

        def __exit__(self, *_args):
            return None

    class FakeModel:
        def transcribe_stream(self):
            return FakeStream()

        def transcribe(self, *_args, **_kwargs):
            batch_called.set()
            return SimpleNamespace(text="final words")

    monkeypatch.setattr(transcribe, "ensure_ffmpeg_on_path", lambda: None)
    monkeypatch.setattr(transcribe, "audio_seconds", lambda _path: 1.0)
    fake_mx = SimpleNamespace(array=lambda value: value)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    def fake_asr(_name, lane="batch"):
        lanes.append(lane)
        return FakeModel()

    monkeypatch.setattr(transcribe, "_asr_model", fake_asr)

    live = transcribe.LiveTranscriber()          # a capture is running and owns the live lane
    result = []
    batch = threading.Thread(target=lambda: result.append(transcribe.transcribe(Path("a.webm"))))
    batch.start()
    ran_while_live_was_open = batch_called.wait(5.0)
    live.close()
    batch.join(5)

    assert ran_while_live_was_open, "batch ASR must NOT queue behind a live Parakeet stream"
    assert result and result[0].text == "final words"
    assert sorted(lanes) == ["batch", "live"], "each lane must ask for its OWN instance"
    assert transcribe._ASR_LIVE_LOCK is not transcribe._ASR_BATCH_LOCK
    assert result[0].text == "final words"


def test_neither_lane_can_be_asked_for_without_naming_itself():
    """`lane` is REQUIRED, and that is a memory guard, not a style rule.

    `lru_cache` keys on the call as written, so a defaulted `lane` would make `_asr_model(n)`
    and `_asr_model(n, "batch")` two separate entries — a third 1.21 GB Parakeet instance that
    neither `_ASR_LIVE_LOCK` nor `_ASR_BATCH_LOCK` guards, i.e. a mutable MLX model shared with
    no lock at all.

    The wiring itself (transcribe -> "batch", the live worker -> "live", two distinct locks) is
    covered by `test_final_parakeet_pass_does_not_wait_for_the_live_stream`, which fails if
    either lane is put back behind one instance or one lock.
    """
    import inspect

    lane = inspect.signature(transcribe._asr_model.__wrapped__).parameters["lane"]
    assert lane.default is inspect.Parameter.empty, "an omitted lane is a third unguarded model"
    assert transcribe._ASR_LIVE_LOCK is not transcribe._ASR_BATCH_LOCK
