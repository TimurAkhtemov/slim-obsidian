"""Who was speaking when — for a recording whose two sides were captured apart.

The recorder plugin writes the microphone to the LEFT channel and the machine's own audio to
the RIGHT (`obsidian/main.js: buildStream`). So "me or them" is a fact about the capture: look
at which channel was making sound while a word was spoken. No model, nothing to hallucinate.

The system channel decides, because it is a digital copy: when nobody on the call is talking
it is silent, with no room in it. The microphone is a room — and on speakers it also hears
THEM. That leak exists only while they are talking, which is exactly when the system channel
already says Them.

⚠ THE SEAM IS `label_tokens`. It is the only function that decides a speaker. Telling remote
speakers apart (or people sharing one microphone) is a voice-clustering model that replaces
it; `turns` and `render` do not change, and neither does anything downstream.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ME, THEM = "Me", "Them"
PROVENANCE = "channels"            # the note's `speakers_by`

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.02

# ⚠ STARTING VALUES, to be replaced by measurement on real recordings (headphones AND
# speakers) — see the plan's acceptance task. Speech sits around -30..-15 dBFS; a call app's
# comfort noise around -65.
ACTIVE_DBFS = -50.0                # a channel "is making sound" at or above this
HANGOVER_SECONDS = 0.2             # system audio still counts this long after it stops
MIN_TURN_SECONDS = 0.4             # a shorter run is a flicker, not a turn


def _hangover(active: np.ndarray) -> np.ndarray:
    out = active.copy()
    lit = np.flatnonzero(active)
    for k in range(1, int(round(HANGOVER_SECONDS / FRAME_SECONDS)) + 1):
        ahead = lit + k
        out[ahead[ahead < len(out)]] = True
    return out


def label_tokens(tokens, levels: np.ndarray) -> list[str]:
    """One label per token. `levels` is (2, frames) dBFS: row 0 microphone, row 1 system."""
    frames = levels.shape[1]
    if not frames:
        return [""] * len(tokens)
    mic = levels[0] >= ACTIVE_DBFS
    system = _hangover(levels[1] >= ACTIVE_DBFS)
    labels, last = [], ""
    for t in tokens:
        a = min(int(t.start / FRAME_SECONDS), frames - 1)
        b = min(max(a + 1, int(np.ceil(t.end / FRAME_SECONDS))), frames)
        if system[a:b].mean() >= 0.5:
            last = THEM
        elif mic[a:b].any():
            last = ME
        labels.append(last)               # silence on both: whoever was speaking still is
    first = next((label for label in labels if label), "")
    return [label or first for label in labels]


def turns(tokens, labels: list[str]) -> list[tuple[str, str]]:
    """Runs of one speaker, with flickers folded into the turn they interrupt."""
    runs: list[list] = []
    for t, label in zip(tokens, labels):
        if runs and runs[-1][0] == label:
            runs[-1][1].append(t)
        else:
            runs.append([label, [t]])

    def flicker(run) -> bool:
        return run[1][-1].end - run[1][0].start < MIN_TURN_SECONDS

    if len(runs) > 1 and flicker(runs[0]):
        runs[1][1] = runs[0][1] + runs[1][1]
        runs.pop(0)
    folded: list[list] = []
    for run in runs:
        if folded and (flicker(run) or folded[-1][0] == run[0]):
            folded[-1][1].extend(run[1])
        else:
            folded.append([run[0], list(run[1])])
    return [(label, "".join(t.text for t in toks).strip()) for label, toks in folded]


def render(turns: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"**{label}:** {text}" for label, text in turns)


def channel_count(path: Path) -> int:
    from .transcribe import ffbin

    out = subprocess.run([ffbin("ffprobe"), "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=channels", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def channel_levels(path: Path) -> np.ndarray:
    """(2, frames) dBFS, one value per 20 ms: row 0 microphone (left), row 1 system (right).

    Streamed in ten-second blocks: an hour of 16 kHz stereo is 230 MB as samples and 2 MB as
    levels, and this runs beside a 25 GB model.
    """
    from .transcribe import ffbin

    frame = int(SAMPLE_RATE * FRAME_SECONDS)
    frame_bytes = frame * 2 * 2                       # two channels of s16le
    proc = subprocess.Popen([ffbin("ffmpeg"), "-nostdin", "-loglevel", "error", "-i", str(path),
                             "-f", "s16le", "-ac", "2", "-ar", str(SAMPLE_RATE), "-"],
                            stdout=subprocess.PIPE)
    rows = []
    while block := proc.stdout.read(frame_bytes * 500):
        block = block[:len(block) - len(block) % frame_bytes]
        if not block:
            continue
        pcm = np.frombuffer(block, dtype="<i2").reshape(-1, frame, 2).astype(np.float32) / 32768
        rows.append(np.sqrt((pcm ** 2).mean(axis=1)))
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg could not decode {path.name}")
    rms = np.concatenate(rows) if rows else np.zeros((0, 2), dtype=np.float32)
    return 20 * np.log10(np.maximum(rms, 1e-10)).T


def label(path: Path, tokens) -> str | None:
    """The transcript with speakers, or None when there are not two sides to tell apart:
    a mono file, or a recording where only one of them ever spoke."""
    tokens = list(tokens)
    if not tokens or channel_count(path) != 2:
        return None
    got = turns(tokens, label_tokens(tokens, channel_levels(path)))
    if len({who for who, _ in got}) < 2:
        return None
    return render(got)
