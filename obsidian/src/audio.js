// The recording's audio: channels, 16 kHz PCM for live captions, and digital silence.
"use strict";

const MIC_CHANNEL = 0;      // left  — the server labels it Me
const SYSTEM_CHANNEL = 1;   // right — the server labels it Them

// One caption request per this much audio: 2.0 s of 16 kHz PCM16. The server re-encodes ~20 s
// of context on EVERY request whatever the chunk holds (parakeet-mlx's streaming design, ~170 ms
// per call on this machine), so the number of requests IS the cost — and a request that carries
// 85 ms of audio costs the same as one that carries two seconds.
const LIVE_FLUSH_BYTES = 64000;

function pcm16k(input, inputRate) {
  const source = input || new Float32Array();
  const ratio = Math.max(1, Number(inputRate || 16000) / 16000);
  const length = Math.floor(source.length / ratio);
  const out = Buffer.alloc(length * 2);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(source.length, Math.floor((i + 1) * ratio));
    let value = 0;
    for (let j = start; j < end; j++) value += source[j];
    value = Math.max(-1, Math.min(1, value / Math.max(1, end - start)));
    out.writeInt16LE(value < 0 ? value * 32768 : value * 32767, i * 2);
  }
  return out;
}

/* Digital silence — exact zeros — on the channels the file gets, said WHILE it can be fixed.
 *
 * ⚠ THIS WAS FOUND AT THE END, NOT DURING (2026-09-24): 157 s of zeros on both channels, and
 * the refusal of the empty transcript was the first anyone knew of it. The same refusal on
 * 09-04, 09-09 and 09-17. For "I record and forget", a meeting you think is being recorded is
 * the worst failure there is.
 *
 * ONE level, and only when BOTH channels are zeros: then nothing at all is being recorded. The
 * mic ALONE is zeros whenever echo cancellation does its job — 09-28, a lecture played on the
 * Mac: mic -91 dB for 20+ min, system -23.5, and the recording was fine. A line for that fired
 * on every lecture, so it was cut (2026-09-29). System silence alone is nobody else talking.
 *
 * -80 dBFS PEAK per second. Above what a dead input emits through Chromium's processing (one
 * 16-bit step is -90 dBFS; AEC3's comfort-noise floor is -96), well below a real room after
 * noise suppression, and a channel needs to cross it only ONCE a second to count. 5 s at the
 * start of each segment, when a broken capture is still cheap to fix; 20 s once anything has
 * been heard, because a pause is not zeros but a device lost mid-meeting is the same loss. */
const SILENT_PEAK = 10 ** (-80 / 20);
const SILENT_START_S = 5;
const SILENT_LATER_S = 20;

function captureSilence(watch, peaks, seconds = 1) {
  const prev = watch || { heard: false, silent: 0, warn: false };
  const sound = peaks.mic >= SILENT_PEAK || peaks.system >= SILENT_PEAK;
  const silent = sound ? 0 : prev.silent + seconds;
  const limit = prev.heard ? SILENT_LATER_S : SILENT_START_S;
  return { heard: prev.heard || sound, silent, warn: silent >= limit };
}

module.exports = { MIC_CHANNEL, SYSTEM_CHANNEL, LIVE_FLUSH_BYTES, pcm16k, SILENT_PEAK, SILENT_START_S,
        SILENT_LATER_S, captureSilence };
