// Capture: system audio with no device picker, the stereo split, the silence watch, the clock.
import assert from "node:assert/strict";
import test from "node:test";
import { fakeContentEl, findClass, installed, notices } from "../fakes.mjs";
import { FakeStream, SPEECH, ZERO, view, wiring } from "../recorder-harness.mjs";

const okDisplay = async () => new FakeStream(["video", "audio"]);

// --- system audio ---------------------------------------------------------------------------

test("system audio is captured with no device selection at all", async () => {
  const v = view({ display: okDisplay });
  const stream = await v.openSystemAudio();
  assert.equal(stream.getAudioTracks().length, 1);
});

test("the video track is stopped — we only ever wanted audio", async () => {
  // ⚠ `video` is a REQUIRED field satisfied with Obsidian's own frame. Leaving the track
  // running would keep a capture alive for a feature that never wanted one.
  const v = view({ display: okDisplay });
  const stream = await v.openSystemAudio();
  assert.equal(stream.getVideoTracks()[0].stopped, true);
  assert.equal(stream.getAudioTracks()[0].stopped, false);
});

test("the handler is removed from the shared session afterwards", async () => {
  const v = view({ display: okDisplay });
  await v.openSystemAudio();
  assert.equal(installed.handler, null);
});

test("the handler is removed even when the request throws", async () => {
  // ⚠ THE ONE THAT MATTERS: `session.defaultSession` is shared with the whole app. A handler
  // left installed silently hijacks any other getDisplayMedia call in Obsidian.
  const v = view({ display: async () => { throw new Error("denied"); } });
  await assert.rejects(() => v.openSystemAudio());
  assert.equal(installed.handler, null);
});

test("a stream with no audio track is an error, not a silent success", async () => {
  const v = view({ display: async () => new FakeStream(["video"]) });
  await assert.rejects(() => v.openSystemAudio(), /no audio track/);
});

test("the handler asks for loopback audio and the requesting frame as video", async () => {
  let got = null;
  const v = view({
    display: async () => {
      installed.handler({ frame: "FRAME", audioRequested: true }, (streams) => { got = streams; });
      return new FakeStream(["video", "audio"]);
    },
  });
  await v.openSystemAudio();
  assert.deepEqual(got, { video: "FRAME", audio: "loopback" });
});

// --- the mix ---------------------------------------------------------------------------------

test("'Record my voice' off means the microphone is never opened", async () => {
  let asked = false;
  const v = view({
    recordVoice: false,
    display: okDisplay,
    user: async () => { asked = true; return new FakeStream(["audio"]); },
  });
  await v.buildStream();
  assert.equal(asked, false);
});

test("losing system audio still records the voice, and SAYS SO", async () => {
  // Degrading silently is how a two-hour loss goes unnoticed.
  const v = view({ display: async () => { throw new Error("denied"); } });
  const stream = await v.buildStream();
  assert.ok(stream);
  assert.equal(notices.length, 1);
  assert.match(notices[0], /could not capture system audio/i);
});

test("no voice and no system audio refuses to record rather than capturing silence", async () => {
  const v = view({
    recordVoice: false,
    display: async () => { throw new Error("denied"); },
  });
  await assert.rejects(() => v.buildStream(), /no audio source/);
});

test("the two sides are kept APART: microphone left, system audio right", async () => {
  // Blended, nobody can ever say who spoke. Apart, "me or them" is a fact of the capture.
  const mic = new FakeStream(["audio"]);
  const sys = new FakeStream(["video", "audio"]);
  const v = view({ user: async () => mic, display: async () => sys });
  await v.buildStream();
  assert.equal(wiring.inputs, 2);
  assert.deepEqual(wiring.routed, [[mic, 0], [sys, 1]]);
  assert.ok(wiring.out, "the merger feeds the recorded stream");
});

test("system audio stays on the RIGHT even when the microphone is off", async () => {
  // The server reads the right channel as Them. A lone source must not slide to the left.
  const sys = new FakeStream(["video", "audio"]);
  const v = view({ recordVoice: false, display: async () => sys });
  await v.buildStream();
  assert.deepEqual(wiring.routed, [[sys, 1]]);
});

function silenceCtx(levels) {
  const made = { connected: [], disconnected: [] };
  const node = (name) => ({
    name,
    connect(target, output = 0) { made.connected.push([name, target.name, output]); },
    disconnect() { made.disconnected.push(name); },
  });
  let analysers = 0;
  const ctx = {
    createMediaStreamSource: () => node("source"),
    createChannelSplitter: () => node("split"),
    createAnalyser: () => {
      const side = analysers++ === 0 ? "mic" : "system";
      return Object.assign(node(side), {
        fftSize: 2048,
        getFloatTimeDomainData(data) { data.fill(0); data[0] = levels[side]; },
      });
    },
    close() {},
  };
  return { ctx, made };
}

test("the watch reads the recorded stream: LEFT is the mic, RIGHT is the machine", () => {
  const v = view();
  const { ctx, made } = silenceCtx({ ...ZERO });
  v.audioCtx = ctx;
  v.setupSilenceWatch(new FakeStream(["audio"]));
  assert.deepEqual(made.connected,
    [["source", "split", 0], ["split", "mic", 0], ["split", "system", 1]]);
});

test("'Record my voice' off means the left side is empty on purpose: nothing to watch", () => {
  const v = view({ recordVoice: false });
  const { ctx, made } = silenceCtx({ ...ZERO });
  v.audioCtx = ctx;
  v.setupSilenceWatch(new FakeStream(["audio"]));
  v.checkSilence();
  assert.equal(made.connected.length, 0);
  assert.equal(v.silenceWatch, null);
});

test("dead capture shows the alarm inside the recording, and the recording continues", () => {
  const v = view();
  const levels = { ...ZERO };
  v.audioCtx = silenceCtx(levels).ctx;
  v.recorder = { state: "recording" };
  v.state = "recording";
  v.contentEl = fakeContentEl();
  v.setupSilenceWatch(new FakeStream(["audio"]));
  for (let i = 0; i < 5; i++) v.checkSilence();
  assert.ok(findClass(v.contentEl, "slim-silence-warning"));
  assert.equal(v.state, "recording");
  levels.system = SPEECH;
  v.checkSilence();
  assert.equal(findClass(v.contentEl, "slim-silence-warning"), null);
});

test("two recordings keep their own watch", () => {
  const a = view();
  const b = view();
  a.audioCtx = silenceCtx({ ...ZERO }).ctx;
  b.audioCtx = silenceCtx({ mic: SPEECH, system: SPEECH }).ctx;
  a.setupSilenceWatch(new FakeStream(["audio"]));
  b.setupSilenceWatch(new FakeStream(["audio"]));
  for (let i = 0; i < 5; i++) { a.checkSilence(); b.checkSilence(); }
  assert.equal(a.silenceWatch.warn, true);
  assert.equal(b.silenceWatch.warn, false);
});

test("stopping disconnects the silence watch", () => {
  const v = view();
  const { ctx, made } = silenceCtx({ ...ZERO });
  v.audioCtx = ctx;
  v.plugin.postLiveTranscript = async () => ({});
  v.setupSilenceWatch(new FakeStream(["audio"]));
  v.teardownStream();
  assert.deepEqual(made.disconnected.sort(), ["mic", "source", "split", "system"]);
  assert.equal(v.silenceTap, null);
  v.checkSilence();   // a late timer tick after teardown is harmless
});

test("the clock reads the wall, not the count of timer ticks", () => {
  // Seen live 2026-10-01: a 51-minute meeting with the display asleep read about 30 minutes.
  // Chromium throttles timers in an occluded window; the audio kept recording, the ticks did not.
  const realNow = Date.now;
  let now = 1_000_000;
  Date.now = () => now;
  try {
    const v = view();
    v.recorder = { state: "recording", pause() { this.state = "paused"; },
                   resume() { this.state = "recording"; } };
    v.clock = { base: 41, since: now };            // a resumed segment, 41 s carried
    now += 10 * 60_000;                            // ten minutes and a single late tick
    v.tick();
    assert.equal(v.elapsed, 41 + 600);

    v.togglePause();
    now += 5 * 60_000;                             // paused time is not recording time
    v.togglePause();
    now += 30_500;
    v.teardownStream();                            // stop reads the wall, not the last tick
    assert.equal(v.elapsed, 41 + 600 + 30);
    v.carryTranscriptAndClock();
    assert.equal(v.elapsedCarry, 671);
  } finally {
    Date.now = realNow;
  }
});
