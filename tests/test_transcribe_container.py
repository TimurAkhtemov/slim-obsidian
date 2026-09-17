"""The duration bug that cost a real 37-minute meeting its transcript (2026-08-05).

`MediaRecorder` streams its container and never finalizes it, so the webm the Obsidian plugin
writes carries NO duration. `ffprobe` answers "N/A", `float("N/A")` raised on the first line of
`transcribe()`, and the recording produced no transcript, no summary and no tags — while the
audio sat on disk, complete and fine. The same missing duration also left Obsidian's own audio
player unable to seek the file.

These tests synthesize the exact condition: piping ffmpeg's output through stdout produces a
non-seekable container without a duration header, the same shape a browser recorder writes.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from slim import transcribe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")


def _probe(path: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    return out.stdout.strip()


def _streamed_webm(path: Path, seconds: int = 3) -> Path:
    """A container written the way a recorder writes one: piped, so never finalized."""
    with path.open("wb") as fh:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"sine=frequency=440:duration={seconds}", "-f", "webm", "pipe:1"],
                       stdout=fh, check=True)
    return path


def _seekable_webm(path: Path, seconds: int = 3) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"sine=frequency=440:duration={seconds}", str(path)], check=True)
    return path


def test_the_fixture_really_reproduces_the_bug(tmp_path):
    """If this ever stops being N/A the other tests below prove nothing."""
    assert _probe(_streamed_webm(tmp_path / "s.webm")) == "N/A"


def test_audio_seconds_survives_a_container_with_no_duration(tmp_path):
    """The crash was here: float('N/A'). It killed the whole pass over a perfectly good file."""
    got = transcribe.audio_seconds(_streamed_webm(tmp_path / "s.webm", seconds=3))
    assert 2.5 < got < 3.6


def test_a_corrupt_file_fails_with_ffprobe_s_reason_not_a_subprocess_repr(tmp_path):
    """Measured 2026-08-23: the plugin's error screen printed a CalledProcessError repr — the
    full ffprobe argv and the absolute vault path — where ffprobe's own one line would do."""
    bad = tmp_path / "2026-08-23_11-42-00.webm"
    bad.write_bytes(b"\x00\x01not a container" * 64)
    with pytest.raises(ValueError) as e:
        transcribe.audio_seconds(bad)
    msg = str(e.value)
    assert "2026-08-23_11-42-00.webm" in msg
    assert "Invalid data found" in msg                 # ffprobe's reason survives
    assert "returned non-zero" not in msg and "ffprobe'," not in msg
    assert str(tmp_path) not in msg                    # the name, not the whole path


def test_normalize_container_makes_the_file_seekable(tmp_path):
    """Fixes the cause rather than working around it — and fixes it for Obsidian's audio
    player too, not just for ASR."""
    p = _streamed_webm(tmp_path / "s.webm", seconds=3)
    assert _probe(p) == "N/A"
    transcribe.normalize_container(p)
    assert 2.5 < float(_probe(p)) < 3.6


def test_normalize_container_is_lossless(tmp_path):
    """A REMUX, not a re-encode. Measured on the real meeting: 0.12 s versus 15 s and a
    generation loss for transcoding to AAC."""
    p = _streamed_webm(tmp_path / "s.webm", seconds=3)
    before = p.stat().st_size
    transcribe.normalize_container(p)
    assert p.stat().st_size == pytest.approx(before, rel=0.1)


def test_a_seekable_file_is_left_completely_alone(tmp_path):
    """Voice Memos m4a and every already-good file must not be rewritten on every ingest."""
    p = _seekable_webm(tmp_path / "ok.webm", seconds=3)
    before = p.read_bytes()
    transcribe.normalize_container(p)
    assert p.read_bytes() == before


def test_a_missing_file_is_not_an_error(tmp_path):
    """Called on the ingest path, where the caller already reports a missing file properly."""
    transcribe.normalize_container(tmp_path / "gone.webm")


# --- binary resolution (2026-08-22) ---------------------------------------------------------
# A real recording died at `[Errno 2] No such file or directory: 'ffprobe'` because Obsidian is
# launched from the Dock and the `slim chat` it spawns inherits launchd's minimal PATH. The same
# code run from a terminal works. These do NOT need ffmpeg installed, so they run everywhere.

@pytest.fixture(autouse=True)
def _clear_ffbin_cache():
    transcribe.ffbin.cache_clear()
    yield
    transcribe.ffbin.cache_clear()


def test_ffbin_prefers_what_is_on_path(monkeypatch):
    monkeypatch.setattr(transcribe.shutil, "which", lambda n: f"/somewhere/{n}")
    assert transcribe.ffbin("ffprobe") == "/somewhere/ffprobe"


def test_ffbin_finds_homebrew_when_path_is_the_dock_minimum(tmp_path, monkeypatch):
    """⚠ THE ACTUAL FAILURE: PATH is `/usr/bin:/bin:/usr/sbin:/sbin`, homebrew is not on it,
    and the binary is sitting in /opt/homebrew/bin the whole time."""
    brew = tmp_path / "homebrew"
    brew.mkdir()
    (brew / "ffprobe").write_text("#!/bin/sh\n")
    monkeypatch.setattr(transcribe.shutil, "which", lambda n: None)
    monkeypatch.setattr(transcribe, "FF_PREFIXES", (str(brew),))
    assert transcribe.ffbin("ffprobe") == str(brew / "ffprobe")


def test_ffbin_says_what_to_do_when_it_is_genuinely_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe.shutil, "which", lambda n: None)
    monkeypatch.setattr(transcribe, "FF_PREFIXES", (str(tmp_path / "nope"),))
    with pytest.raises(FileNotFoundError) as e:
        transcribe.ffbin("ffmpeg")
    assert "brew install ffmpeg" in str(e.value)


def test_ffmpeg_is_put_on_path_for_dependencies_that_look_it_up_themselves(tmp_path, monkeypatch):
    """⚠ `parakeet_mlx` calls `shutil.which("ffmpeg")` itself. Resolving only OUR argv left a
    real lecture with an empty transcript, filed as a success (2026-08-22)."""
    brew = tmp_path / "brew"
    brew.mkdir()
    (brew / "ffmpeg").write_text("#!/bin/sh\n")
    monkeypatch.setattr(transcribe.shutil, "which", lambda n: None)
    monkeypatch.setattr(transcribe, "FF_PREFIXES", (str(brew),))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    transcribe.ensure_ffmpeg_on_path()
    assert transcribe.os.environ["PATH"].split(":")[0] == str(brew)
