"""Local speech-to-text: Parakeet via MLX, two instances (live captions, batch).

The model is an unselected default — see experiment M8-transcription in the vault's slim-docs. Called by
`chat.py` for the recorder and by `inbox` for the memo lane. Never writes to the vault.
"""
from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Where ffmpeg installs itself when PATH cannot be trusted. Homebrew on Apple silicon
# first, Intel second; anything else has to be on PATH.
FF_PREFIXES = ("/opt/homebrew/bin", "/usr/local/bin")


@lru_cache(maxsize=None)
def ffbin(name: str) -> str:
    """Absolute path to an ffmpeg-family binary.

    ⚠ **PATH IS NOT THE SAME FOR EVERY LANE, and that is what broke this.** Obsidian is
    launched from the Dock, so the `slim chat` it spawns inherits launchd's minimal PATH, with
    no `/opt/homebrew/bin` on it: `ffprobe` was not found and a real recording died at the
    first line of `transcribe()` with `[Errno 2]` (2026-08-06). From a terminal the same code
    works. Behavior differs by WHO STARTED THE PROCESS, and nothing in the code says so.
    Resolving the absolute path here fixes every lane at once.
    """
    found = shutil.which(name)
    if found:
        return found
    for prefix in FF_PREFIXES:
        candidate = Path(prefix) / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"{name} not found on PATH or in {', '.join(FF_PREFIXES)}. Install it with "
        f"`brew install ffmpeg`. (If it IS installed, the process that started this server "
        f"has a minimal PATH — that is what this lookup exists to survive.)")


def ensure_ffmpeg_on_path() -> None:
    """Put ffmpeg's directory on PATH, for code that does its OWN lookup.

    ⚠ Resolving our own calls is not enough: `parakeet_mlx.audio.load_audio` runs
    `shutil.which("ffmpeg")` itself, so the ENVIRONMENT has to be right, not just our argv
    (2026-08-22).
    """
    directory = str(Path(ffbin("ffmpeg")).parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if directory not in parts:
        os.environ["PATH"] = os.pathsep.join([directory, *parts])


# UNSELECTED default — see the module docstring. The only lane that varies is live vs batch.
MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# parakeet's `chunk_duration` defaults to None, which means "the whole file as ONE tensor".
# Attention is quadratic in sequence length, so a 36-minute meeting asks Metal for 45.9 GB
# and dies. Whisper windows internally at 30 s and so never shows the problem; parakeet has
# to be TOLD. This is not a tuning knob — it is the difference between working and crashing.
CHUNK_SECONDS = 120.0
OVERLAP_SECONDS = 15.0  # parakeet-mlx's own default

AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".mp4", ".mov",
                  ".caf", ".flac", ".opus", ".webm", ".ogg"}


@dataclass
class Transcript:
    text: str
    model: str
    audio_seconds: float
    wall_seconds: float
    # How the speakers in `text` were told apart ("channels"), or "" when it carries no labels.
    speakers_by: str = ""

    @property
    def rtf(self) -> float:
        return self.audio_seconds / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def words(self) -> int:
        return len(self.text.split())


def audio_seconds(path: Path) -> float:
    """Duration in seconds.

    ⚠ A container written incrementally by a recorder has NO duration in its header —
    `MediaRecorder` streams webm and never finalizes it, so ffprobe answers `N/A`. That was
    EVERY recording the plugin made, and `float("N/A")` raised from the first line of
    `transcribe()`. `normalize_container` fixes the cause; this fallback exists because a crash
    here kills a whole pass over a file that is fine.
    """
    out = subprocess.run(
        [ffbin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        # ffprobe's own last line ("Invalid data found when processing input"), not the
        # CalledProcessError repr — that is the full argv plus the absolute vault path, and the
        # plugin's error screen prints whatever arrives here verbatim (measured 2026-08-23).
        lines = [l for l in out.stderr.strip().splitlines() if l.strip()]
        reason = lines[-1] if lines else f"exit status {out.returncode}"
        prefix = f"{path}: "
        if reason.startswith(prefix):
            reason = reason[len(prefix):]
        raise ValueError(f"ffprobe could not read {path.name}: {reason}")
    raw = out.stdout.strip()
    try:
        return float(raw)
    except ValueError:
        pass

    decoded = subprocess.run([ffbin("ffmpeg"), "-i", str(path), "-f", "null", "-"],
                             capture_output=True, text=True)
    marks = re.findall(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)", decoded.stderr)
    if not marks:
        raise ValueError(f"could not determine duration of {path.name}")
    h, m, s = marks[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def normalize_container(path: Path) -> Path:
    """Rewrite a stream-written container so it carries a duration, in place. Returns `path`.

    A REMUX, not a re-encode: `-c copy` rewrites the container while ffmpeg, having read the
    whole stream, now knows its length. Measured on a 37-minute meeting — **0.12 s, lossless,
    same size**. Worth doing even though `audio_seconds` no longer crashes without it: Obsidian's
    own player cannot seek a file whose length it does not know.
    """
    if not path.exists():
        return path
    probe = subprocess.run(
        [ffbin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    if probe.stdout.strip() not in ("", "N/A"):
        return path                                   # already seekable; nothing to do

    fixed = path.with_name(f".{path.name}.remux{path.suffix}")
    try:
        subprocess.run([ffbin("ffmpeg"), "-y", "-loglevel", "error", "-i", str(path),
                        "-c", "copy", str(fixed)], check=True)
        fixed.replace(path)
    except Exception:                                  # noqa: BLE001 - the original still works
        fixed.unlink(missing_ok=True)
    return path


def to_wav16k(src: Path, dst: Path) -> Path:
    """Any recording -> 16 kHz mono WAV. Voice Memos hands us m4a; a call recorder may hand
    us stereo at 48 kHz. Every ASR wants 16 kHz mono, so normalize once, here, rather than
    letting each runtime resample differently."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffbin("ffmpeg"), "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )
    return dst


# ONE INSTANCE PER LANE, and the lane is part of the cache key. A parakeet model object is
# mutable, so sharing one across live captions and final transcription needs a lock — and a
# live stream owns the model for the WHOLE capture, so that lock made one meeting's
# transcription wait for another to stop recording.
#
# Measured 2026-08-27: a second instance costs 1.21 GB (bf16), two sit at 2.41 GB active and
# peak 4.22 GB through a 271 s batch run, and driven from two threads on one GPU both lanes
# returned text byte-identical to their solo baselines.
@lru_cache(maxsize=4)
def _asr_model(name: str, lane: str):
    """One local ASR model instance per lane. `lane` is a cache key, not an MLX argument.

    ⚠ REQUIRED, NOT DEFAULTED. `lru_cache` keys on the call as written, so `_asr_model(n)`
    and `_asr_model(n, "batch")` would be two entries — a third 1.21 GB instance that
    neither lane lock guards, shared mutable state with no lock at all.

    ⚠ EVALUATED HERE, ON THE LOADING THREAD. `from_pretrained` returns LAZY weights (an
    `mx.load` plus an `astype`, never run), and MLX streams are thread-local, so every pending
    load is bound to the thread that called this. The cache hands the instance to other
    threads — every HTTP request is one, and each live capture gets its own worker. A first
    run on speech evaluated everything and hid it; a first run on silence emitted no token,
    the prediction-network weights stayed lazy, and the next request from another thread
    died with "There is no Stream(cpu, N) in current thread" (2026-08-27, three times on one
    recording, live captions with it). Evaluated arrays cross threads freely."""
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    model = from_pretrained(name)
    mx.eval(model.parameters())
    return model


# One lock per lane, guarding that lane's own instance — never each other's. The live lock
# spans a whole capture and blocks only a second live capture, which the plugin's capture lease
# already prevents. The batch lock keeps queued recordings off one object at once.
_ASR_LIVE_LOCK = threading.Lock()
_ASR_BATCH_LOCK = threading.Lock()


class LiveTranscriber:
    """Incremental, local-only PCM transcription for the recorder's live caption surface.

    MLX streams are thread-affine, and ``ThreadingHTTPServer`` handles ``start`` and later
    ``chunk`` requests on different threads. A dedicated worker owns the stream for its whole
    lifetime; request threads only exchange PCM bytes and text with it.
    """

    def __init__(self, model: str | None = None):
        ensure_ffmpeg_on_path()
        self.model_name = model or MODEL
        self._commands: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._final_text = ""
        self.closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="slim-live-transcriber",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def _run(self) -> None:
        with _ASR_LIVE_LOCK:
            try:
                import mlx.core as mx

                stream = _asr_model(self.model_name, "live").transcribe_stream()
                stream.__enter__()
            except BaseException as exc:  # noqa: BLE001 - propagate model startup to the caller
                self._startup_error = exc
                self._ready.set()
                return
            self._ready.set()

            while True:
                action, payload, response = self._commands.get()
                try:
                    if action == "add":
                        import numpy as np

                        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                        stream.add_audio(mx.array(samples))
                        response.put((True, stream.result.text.strip()))
                    else:
                        self._final_text = stream.result.text.strip()
                        stream.__exit__(None, None, None)
                        response.put((True, self._final_text))
                        return
                except BaseException as exc:  # noqa: BLE001 - return worker failures to HTTP
                    response.put((False, exc))
                    if action == "close":
                        return

    def _request(self, action: str, payload: bytes = b"") -> str:
        response: queue.Queue = queue.Queue(maxsize=1)
        self._commands.put((action, payload, response))
        ok, value = response.get()
        if not ok:
            raise value
        return value

    def add_pcm16(self, pcm: bytes) -> str:
        if self.closed:
            raise RuntimeError("live transcript is already closed")
        if not pcm or len(pcm) % 2:
            raise ValueError("live transcript chunks must contain 16-bit PCM samples")
        return self._request("add", pcm)

    def close(self) -> str:
        if self.closed:
            return self._final_text
        self.closed = True
        self._final_text = self._request("close")
        self._thread.join()
        return self._final_text


_LIVE_SESSIONS: dict[str, LiveTranscriber] = {}
_LIVE_LOCK = threading.RLock()


def live_start(session_id: str, model: str | None = None) -> None:
    """Begin live captions for one recording, replacing whatever came before.

    ⚠ A NEW SESSION EVICTS THE OLD. Refusing while another session existed meant one left
    behind by a crash disabled live captions for every recording afterwards, with a line of
    grey status text as the only sign. There is one recorder, so there is one live session.
    """
    with _LIVE_LOCK:
        for stale_id in list(_LIVE_SESSIONS):
            stale = _LIVE_SESSIONS.pop(stale_id)
            try:
                stale.close()
            except Exception:                    # noqa: BLE001 - a dead worker must not block
                pass
        _LIVE_SESSIONS[session_id] = LiveTranscriber(model)


def live_add(session_id: str, pcm: bytes) -> str:
    with _LIVE_LOCK:
        session = _LIVE_SESSIONS.get(session_id)
        if session is None:
            raise KeyError("no such live transcript")
        return session.add_pcm16(pcm)


def live_stop(session_id: str) -> str:
    with _LIVE_LOCK:
        session = _LIVE_SESSIONS.pop(session_id, None)
        return session.close() if session is not None else ""


def transcribe(path: Path, model: str | None = None) -> Transcript:
    """Transcribe an audio file. Long-form safe: chunked, so an hour-long meeting works."""
    ensure_ffmpeg_on_path()          # parakeet_mlx shells out to ffmpeg by bare name
    model = model or MODEL

    secs = audio_seconds(path)
    t0 = time.perf_counter()
    with _ASR_BATCH_LOCK:
        asr = _asr_model(model, "batch")
        result = asr.transcribe(
            str(path),
            chunk_duration=CHUNK_SECONDS,
            overlap_duration=OVERLAP_SECONDS,
        )
    text, speakers_by = result.text.strip(), ""
    # Outside the ASR lock: this is ffmpeg and numpy, and must not hold up live captions.
    from . import speakers, trace
    try:
        labelled = speakers.label(path, result.tokens)
    except Exception as exc:                       # noqa: BLE001 - labels decorate, never gate
        trace.record("record", {"stage": "speakers", "status": "failed", "error": str(exc)})
        labelled = None
    if labelled:
        text, speakers_by = labelled, speakers.PROVENANCE
    wall = time.perf_counter() - t0

    return Transcript(text=text, model=model, audio_seconds=secs, wall_seconds=wall,
                      speakers_by=speakers_by)
