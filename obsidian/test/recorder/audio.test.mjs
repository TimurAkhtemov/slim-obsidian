// The recording's audio: live-caption PCM and the digital-silence alarm.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { SPEECH, ZERO } from "../recorder-harness.mjs";

const require = createRequire(import.meta.url);
const { captureSilence, pcm16k } = require("../../src/recorder/audio.js");

test("live captions downsample browser audio to local 16 kHz PCM", () => {
  const input = new Float32Array(48000).fill(0.5);
  const pcm = pcm16k(input, 48000);
  assert.equal(pcm.length, 32000);
  assert.ok(Math.abs(pcm.readInt16LE(0) - 16384) < 2);
});

function feed(peaks, watch = null) {
  for (const p of peaks) watch = captureSilence(watch, p);
  return watch;
}

// --- the digital-silence warning -------------------------------------------------------------
// 2026-09-24: 157 s of zeros on both channels, and the empty-transcript refusal at the end was
// the first anyone knew. A meeting you think is being recorded is the worst failure there is.

test("five seconds of zeros on BOTH channels says nothing is being recorded; four do not", () => {
  assert.equal(feed(Array(4).fill(ZERO)).warn, false);
  assert.equal(feed(Array(5).fill(ZERO)).warn, true);
});

test("a silent mic beside live system audio is never a warning: echo cancellation at work", () => {
  // 09-28: a lecture played on the Mac, mic zeros for 20+ min, system fine. It still worked.
  assert.equal(feed(Array(60).fill({ mic: 0, system: SPEECH })).warn, false);
});

test("system silence alone is never a warning: nobody else is talking", () => {
  assert.equal(feed(Array(60).fill({ mic: SPEECH, system: 0 })).warn, false);
});

test("a real noise floor is never silence, even with nobody speaking", () => {
  // -70 dBFS: a quiet room after noise suppression; the machine playing nothing at all.
  const floor = 10 ** (-70 / 20);
  assert.equal(feed(Array(60).fill({ mic: floor, system: 0 })).warn, false);
});

test("the alarm clears the moment either side is heard", () => {
  const warned = feed(Array(6).fill(ZERO));
  assert.equal(warned.warn, true);
  assert.equal(captureSilence(warned, { mic: SPEECH, system: 0 }).warn, false);
  assert.equal(captureSilence(warned, { mic: 0, system: SPEECH }).warn, false);
});

test("once anything was heard, 20 s of zeros before the alarm, not 5", () => {
  // A pause is not zeros — but a device lost mid-meeting is the same loss.
  const heard = feed([{ mic: SPEECH, system: 0 }]);
  assert.equal(feed(Array(19).fill(ZERO), heard).warn, false);
  assert.equal(feed(Array(20).fill(ZERO), heard).warn, true);
});
