"""Audio files: the ffmpeg binaries, a recording's container and duration, its hash and archive.

Shared by both lanes and by `speakers` and `transcribe`; no speech recognition happens here.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from ..config import VAULT

AUDIO_ARCHIVE = VAULT / "Attachments" / "Recordings"

AUDIO_SUFFIXES = {".m4a", ".mp3", ".wav", ".aac", ".mp4", ".mov",
                  ".caf", ".flac", ".opus", ".webm", ".ogg"}


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def archive_audio(source: Path, digest: str) -> Path:
    """Copy raw source bytes into the synced vault before the Inbox copy may be removed.

    The temporary file and hash verification make a partial copy distinguishable from a
    preserved source. An existing content-addressed path is trusted only after re-hashing it.
    """
    archived = AUDIO_ARCHIVE / f"{digest}{source.suffix.lower()}"
    archived.parent.mkdir(parents=True, exist_ok=True)
    if archived.exists():
        if sha256(archived) != digest:
            raise RuntimeError(f"audio archive hash mismatch: {archived}")
        return archived

    partial = archived.with_name(f".{archived.name}.part")
    try:
        with source.open("rb") as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256(partial) != digest:
            raise RuntimeError(f"audio archive copy failed verification: {source}")
        partial.replace(archived)
    finally:
        partial.unlink(missing_ok=True)
    return archived
