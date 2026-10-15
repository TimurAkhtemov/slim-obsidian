// Live captions while recording: flushing, restarts, coalescing, carrying across segments.
import assert from "node:assert/strict";
import test from "node:test";
import { view } from "../recorder-harness.mjs";

test("resuming carries the earlier transcript and the elapsed time forward", () => {
  // Seen live: resume showed an empty transcript pane and a timer back at 00:00:00, so the
  // first forty minutes looked lost even though the server had them cached.
  const v = view();
  v.liveTranscript = "the first part";
  v.elapsed = 41;
  v.carryTranscriptAndClock();
  assert.equal(v.liveCarry, "the first part");
  assert.equal(v.elapsedCarry, 41);

  v.liveTranscript = "the second part";
  assert.equal(v.shownTranscript(), "the first part\n\nthe second part");
  v.elapsed = v.elapsedCarry + 5;
  assert.equal(v.elapsed, 46, "the clock counts the whole recording, not this segment");
});

test("the live transcript keeps flushing after a pause and resume", async () => {
  const v = view();
  v.recordingId = "rec-live";
  const posts = [];
  v.plugin.postLiveTranscript = async (body) => {
    posts.push(body.action);
    return { transcript: `words ${posts.filter((a) => a === "chunk").length}` };
  };
  let onaudio = null;
  const node = () => ({ connect() {}, disconnect() {} });
  v.audioCtx = {
    sampleRate: 16000,
    destination: node(),
    createMediaStreamSource: () => node(),
    createGain: () => ({ ...node(), gain: { value: 1 } }),
    createScriptProcessor: () => {
      const p = node();
      Object.defineProperty(p, "onaudioprocess", {
        set(fn) { onaudio = fn; }, get() { return onaudio; },
      });
      return p;
    },
  };
  v.startLiveTranscript({});
  await v.liveTranscriptQueue;

  const frame = { inputBuffer: { getChannelData: () => new Float32Array(4096).fill(0.2) } };
  const speak = async () => {
    for (let i = 0; i < 10; i += 1) onaudio(frame);
    await v.liveTranscriptQueue;
  };

  await speak();
  const before = posts.filter((a) => a === "chunk").length;
  assert.ok(before > 0, "audio reaches the transcriber before the pause");

  v.recorder = { state: "recording", pause() { this.state = "paused"; },
                 resume() { this.state = "recording"; } };
  v.togglePause();                                  // Pause
  await speak();
  assert.equal(posts.filter((a) => a === "chunk").length, before, "paused sends nothing");

  v.togglePause();                                  // Resume
  await speak();
  assert.ok(posts.filter((a) => a === "chunk").length > before,
            "⚠ after resume the live transcript must start growing again");
});

test("a lost live session is restarted instead of switching captions off", async () => {
  // The chunk handler treated EVERY error as fatal and set liveTranscriptActive = false, so one
  // failed POST — a restarted server, an evicted session — killed captions for the rest of the
  // recording, with grey status text as the only sign.
  const v = view();
  v.recordingId = "rec-1";
  let failed = false;
  const actions = [];
  v.plugin.postLiveTranscript = async (body) => {
    actions.push(body.action);
    if (body.action === "chunk" && !failed) {
      failed = true;
      throw new Error("/api/record/live returned 400: no such live transcript");
    }
    return { transcript: "recovered words" };
  };
  v.liveTranscriptActive = true;
  v.livePcm = [Buffer.alloc(64000)];
  v.livePcmBytes = 64000;

  await v.flushLiveTranscript();
  await v.liveTranscriptQueue;

  assert.equal(v.liveTranscriptActive, true, "captions survive one lost session");
  assert.ok(actions.includes("start"), `it restarts the session: ${actions.join(",")}`);
});

test("the caption pane says which of the three states it is in", () => {
  const v = view();
  v.recorder = { state: "paused" };
  v.liveTranscriptActive = true;
  assert.match(v.liveStatusText(), /paused/i);

  v.recorder = { state: "recording" };
  v.liveTranscriptStatus = "Live transcript · draft";
  assert.match(v.liveStatusText(), /draft/i);

  v.liveTranscriptActive = false;
  assert.match(v.liveStatusText(), /off|unavailable|stopped/i);
});

test("live captions coalesce instead of queueing behind a slow round trip", async () => {
  // ⚠ Captions flushed every 2s onto an UNBOUNDED promise chain. One round trip slower than
  // 2s and every later flush waits behind it, so the lag compounds for the whole recording —
  // "it doesn't keep up", exactly as they described.
  const v = view();
  v.liveTranscriptActive = true;
  v.recordingId = "r";
  let inFlight = 0;
  let maxInFlight = 0;
  const sizes = [];
  let release;
  const gate = new Promise((r) => { release = r; });
  v.plugin.postLiveTranscript = async (body) => {
    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    sizes.push(Buffer.from(body.pcm16, "base64").length);
    await gate;
    inFlight -= 1;
    return { transcript: "words" };
  };

  v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
  v.flushLiveTranscript();
  for (let i = 0; i < 4; i += 1) {              // four more flushes while the first is stuck
    v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
    v.flushLiveTranscript();
  }
  release();
  await new Promise((r) => setTimeout(r, 0));
  await v.liveTranscriptQueue;

  assert.equal(maxInFlight, 1, "never more than one request in flight");
  assert.ok(sizes.length <= 2, `the backlog coalesced into ${sizes.length} sends`);
});

test("a decoder that revises its own prefix never duplicates the caption", () => {
  // ⚠ THE BUG THE OWNER REPORTED, 2026-08-27: captions "repeat excessively and spam words".
  //
  // The rule used to be "growth replaces, anything else appends", written against a
  // rolling-window ASR. Parakeet does not roll — it REWRITES what it already emitted as more
  // audio arrives. Measured over 99 s of speech: 19 of 49 chunks revised their prefix and 0
  // shrank, so the append branch fired constantly and appended the WHOLE cumulative string,
  // compounding each time. The pane showed 37,785 chars of a 1,477-char transcript (25.6x).
  //
  // The stream is cumulative — a server contract, not an assumption — so: replace, always.
  const v = view();
  v.applyLiveTranscript("The live caption lane feeds six.");
  v.applyLiveTranscript("The live caption lane feeds sixteen kilohertz audio into a stream.");
  // A real revision: "sixteen kilohertz" is rewritten as "16 kHz". This is the case that used
  // to append everything again.
  v.applyLiveTranscript("The live caption lane feeds 16 kHz audio into a streaming parakeet.");
  assert.equal(v.liveTranscript,
               "The live caption lane feeds 16 kHz audio into a streaming parakeet.");
  assert.equal(v.shownTranscript(), v.liveTranscript, "nothing is carried mid-segment");

  // Punctuation-only revision of the tail — the most common shape by far.
  v.applyLiveTranscript("The live caption lane feeds 16 kHz audio into a streaming parakeet model.");
  assert.ok(!v.liveTranscript.includes("parakeet. The"), "no duplicated prefix");
});

test("a live session that restarts keeps the words already on screen", () => {
  // The append branch was really protecting THIS case: a restarted session decodes from zero,
  // so a bare replace would erase the first half of a lecture. It belongs here, not in the
  // per-chunk rule — carry, then let the new session grow from empty.
  const v = view();
  v.applyLiveTranscript("everything they said before the restart");
  v.carryTranscript();
  assert.equal(v.liveTranscript, "");
  assert.equal(v.shownTranscript(), "everything they said before the restart");

  v.applyLiveTranscript("what the new session hears");
  assert.equal(v.shownTranscript(),
               "everything they said before the restart\n\nwhat the new session hears");
});

test("carrying mid-segment does not touch the clock", () => {
  // A restart happens INSIDE a segment, with the clock still running; only a segment boundary
  // may bank the elapsed time. One helper doing both would silently reset their timer.
  const v = view();
  v.elapsed = 42;
  v.elapsedCarry = 0;
  v.applyLiveTranscript("words");
  v.carryTranscript();
  assert.equal(v.elapsedCarry, 0, "a restart must not bank the clock");
  v.applyLiveTranscript("more words");
  v.carryTranscriptAndClock();
  assert.equal(v.elapsedCarry, 42, "a segment boundary still banks it");
});

// --- 2026-08-28: three things they noticed in a week of real use ---------------------------------

test("a caption round trip that returns with less than 2 s buffered waits for the tap", async () => {
  // ⚠ THE COIL WHINE. The `finally` re-flushed WHATEVER had arrived during the request, so after
  // the first 2 s chunk the loop ran at one request per round trip: 2.7 requests/s measured on a
  // real recording, each re-encoding 20 s of context — GPU 60 % busy on silence, pulsing at
  // 3 Hz. Per call cost is flat (~170 ms), so the count of calls is the whole cost. At the tap's
  // own 2 s floor: 0.5/s and 9 %.
  const v = view();
  v.liveTranscriptActive = true;
  v.recordingId = "r";
  const sizes = [];
  v.plugin.postLiveTranscript = async (body) => {
    sizes.push(Buffer.from(body.pcm16, "base64").length);
    // One tap callback lands mid-flight. Once: with the old `finally` this fake would feed
    // the loop forever, which is exactly what the real tap did.
    if (sizes.length === 1) { v.livePcm.push(Buffer.alloc(2730)); v.livePcmBytes += 2730; }
    return { transcript: "" };
  };

  v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
  await v.flushLiveTranscript();
  await v.liveTranscriptQueue;

  assert.deepEqual(sizes, [64000], "85 ms that arrived mid-flight waits for the 2 s floor");
  assert.equal(v.livePcmBytes, 2730, "and stays buffered for the tap to send");
});

test("a backlog that reached the floor while a request was out still goes at once", async () => {
  // The coalescing that fixed "captions fall behind" (2026-08-26) is kept: a slow round trip
  // sends fewer, larger chunks — it must not wait for yet another 2 s on top of the lag.
  const v = view();
  v.liveTranscriptActive = true;
  v.recordingId = "r";
  const sizes = [];
  v.plugin.postLiveTranscript = async (body) => {
    sizes.push(Buffer.from(body.pcm16, "base64").length);
    if (sizes.length === 1) { v.livePcm.push(Buffer.alloc(70000)); v.livePcmBytes += 70000; }
    return { transcript: "" };
  };

  v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
  await v.flushLiveTranscript();
  await v.liveTranscriptQueue;

  assert.deepEqual(sizes, [64000, 70000]);
});
