"""The ASR model must be usable from a thread other than the one that loaded it.

`parakeet_mlx.from_pretrained` returns weights that are LAZY — `mx.load` plus an `astype`,
never evaluated — and MLX streams are thread-local, so every pending load is bound to the
thread that called `_asr_model`. `ThreadingHTTPServer` hands each request its own thread and
`_asr_model` is an `lru_cache`, so the instance one request creates is reused by the next.
A first run on speech evaluates every weight in the creating thread and hides this; a first
run on SILENCE never emits a token, the prediction-network weights stay lazy, and the next
request dies with `There is no Stream(cpu, N) in current thread` (2026-08-27, live: two silent
test recordings, then a real one that failed three times, and live captions went with it).

Real model, real threads — a fake cannot carry a stream.
"""

import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from slim import transcribe

parakeet_mlx = pytest.importorskip("parakeet_mlx")

_CACHED_MODEL = Path.home() / ".cache/huggingface/hub" / ("models--" + transcribe.MODEL.replace("/", "--"))

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("say") is None or not _CACHED_MODEL.exists(),
    reason="needs ffmpeg, macOS `say`, and the Parakeet weights already in the HF cache",
)


def _spoken_clip(tmp_path: Path) -> Path:
    aiff = tmp_path / "clip.aiff"
    subprocess.run(["say", "-v", "Albert", "-o", str(aiff),
                    "The transform maps a signal from the time domain to the frequency domain."],
                   check=True)
    wav = tmp_path / "clip.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                    "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


def _in_thread(fn):
    out = {}

    def target():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - the assertion needs the exception
            out["error"] = exc
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def test_a_model_loaded_on_one_thread_transcribes_on_another(tmp_path):
    clip = _spoken_clip(tmp_path)
    transcribe._asr_model.cache_clear()
    try:
        _in_thread(lambda: transcribe._asr_model(transcribe.MODEL, "batch"))   # request 1 loads
        text = _in_thread(lambda: transcribe.transcribe(clip).text)           # request 2 uses
    finally:
        transcribe._asr_model.cache_clear()
    assert "domain" in text.lower(), text
