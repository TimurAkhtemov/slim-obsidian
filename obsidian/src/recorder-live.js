// RecorderView: live captions while recording.
"use strict";

const { LIVE_FLUSH_BYTES, pcm16k } = require("./audio");

module.exports = {
  /* One recording's captions and clock, across every segment of it. */
  shownTranscript() {
    return [this.liveCarry, this.liveTranscript].filter(Boolean).join("\n\n");
  },

  /* Fold what is on screen into the carry. Used by a segment boundary AND by a mid-segment
   * session restart — the restart must NOT touch the clock, which is still running. */
  carryTranscript() {
    this.liveCarry = [this.liveCarry, this.liveTranscript].filter(Boolean).join("\n\n");
    this.liveTranscript = "";
  },

  carryTranscriptAndClock() {
    this.carryTranscript();
    this.elapsedCarry = this.elapsed || 0;
  },

  /* What the caption pane says. Three states, and they must be distinguishable: PAUSED (they did
   * that and nothing is wrong), running, and off. "Live transcript · draft" sitting frozen
   * because captions had quietly died is what sent them looking for a bug in the wrong place. */
  liveStatusText() {
    if (!this.liveTranscriptActive) return "Live transcript off — recording continues";
    if (this.recorder && this.recorder.state === "paused") return "Paused — not listening";
    return this.liveTranscriptStatus || "Live transcript · draft";
  },

  /* Fold one caption result into what is on screen.
   *
   * ⚠ THE STREAM IS CUMULATIVE, AND IT REVISES ITS OWN PREFIX. That second half is what this
   * used to get wrong. The rule was "growth replaces, anything else appends", written to guard
   * against a rolling-window ASR — but parakeet does not roll, it REWRITES: as more audio
   * arrives, `"...produces stable."` becomes `"...produces stable text we can compare..."` and
   * `"sixteen kilohertz"` becomes `"16 kHz"`. Every rewrite fails `startsWith`, so the append
   * branch appended the WHOLE cumulative text to what was already there, and compounded on the
   * next chunk. Measured 2026-08-27 over 99 s of speech: 19 of 49 chunks revised their prefix,
   * and the pane showed 37,785 chars of a 1,477-char transcript — 25.6x, which is what the owner
   * saw as captions spamming repeated words.
   *
   * So: REPLACE, always. Server contract, not an assumption — `/api/record/live` returns
   * `stream.result.text`, the full text for that session, and 0 of those 49 chunks shrank.
   * A session that RESTARTS does start empty, which is the case the old append branch was
   * really protecting; that is now handled where it belongs, by folding the words on screen
   * into `liveCarry` before asking for a new session, so nothing vanishes.
   */
  applyLiveTranscript(text) {
    const next = String(text || "").trim();
    if (!next) return;
    this.liveTranscript = next;
  },

  updateLiveTranscriptUI() {
    if (!this.contentEl) return;
    const status = this.contentEl.querySelector(".slim-live-transcript-status");
    if (status) status.setText(this.liveStatusText());
    const text = this.contentEl.querySelector(".slim-live-transcript-text");
    if (text) {
      text.setText(this.shownTranscript() || "Listening for speech…");
      if (typeof text.scrollTo === "function") text.scrollTo({ top: text.scrollHeight });
      else text.scrollTop = text.scrollHeight;
    }
  },

  startLiveTranscript(mixed) {
    this.liveTranscript = "";   // the carry holds the earlier segments; see shownTranscript()
    this.liveTranscriptStatus = "Starting local captions…";
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveTranscriptActive = true;
    this.liveTranscriptPaused = false;
    this.updateLiveTranscriptUI();
    this.liveTranscriptQueue = this.plugin.postLiveTranscript({
      action: "start", session_id: this.recordingId,
    }).then(() => {
      this.liveTranscriptStatus = "Live transcript · draft";
      this.updateLiveTranscriptUI();
    }).catch((e) => {
      console.error("[slim] live transcript start", e);
      this.liveTranscriptActive = false;
      this.liveTranscriptStatus = "Live transcript unavailable — recording continues";
      this.updateLiveTranscriptUI();
    });
    try {
      if (!this.audioCtx || typeof this.audioCtx.createScriptProcessor !== "function") return;
      const source = this.audioCtx.createMediaStreamSource(mixed);
      const processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
      const silent = this.audioCtx.createGain();
      silent.gain.value = 0;
      processor.onaudioprocess = (event) => {
        if (!this.liveTranscriptActive || this.liveTranscriptPaused) return;
        const channel = event.inputBuffer.getChannelData(0);
        const pcm = pcm16k(channel, this.audioCtx && this.audioCtx.sampleRate);
        if (!pcm.length) return;
        this.livePcm.push(pcm);
        this.livePcmBytes += pcm.length;
        if (this.livePcmBytes >= LIVE_FLUSH_BYTES) this.flushLiveTranscript();
      };
      source.connect(processor);
      processor.connect(silent);
      silent.connect(this.audioCtx.destination);
      this.liveProcessor = { source, processor, silent };
    } catch (e) {
      console.error("[slim] live transcript tap", e);
    }
  },

  /* One caption round trip at a time, and the audio waits in the buffer rather than in a
   * queue of requests.
   *
   * ⚠ THIS IS WHY CAPTIONS FELL BEHIND AND NEVER CAUGHT UP. A flush fired every 2 s of audio
   * and chained onto an unbounded promise queue; one round trip slower than 2 s and every
   * later flush waited behind it, so the lag compounded for the length of the recording.
   * Coalescing bounds the lag to a single round trip: a slow machine sends fewer, larger
   * chunks and the captions jump forward instead of drifting away.
   */
  flushLiveTranscript() {
    if (!this.liveTranscriptActive || !this.livePcmBytes) return this.liveTranscriptQueue;
    if (this.liveInFlight) return this.liveTranscriptQueue;   // audio keeps accumulating
    const pcm = Buffer.concat(this.livePcm, this.livePcmBytes);
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveInFlight = true;
    this.liveTranscriptQueue = this.liveTranscriptQueue.then(async () => {
      if (!this.liveTranscriptActive) return;
      const result = await this.plugin.postLiveTranscript({
        action: "chunk", session_id: this.recordingId, pcm16: pcm.toString("base64"),
      });
      if (result && typeof result.transcript === "string") {
        this.applyLiveTranscript(result.transcript);
        this.liveTranscriptStatus = "Live transcript · draft";
        this.updateLiveTranscriptUI();
      }
    }).catch(async (e) => {
      // ⚠ NOT EVERY FAILURE IS FATAL. A lost session — a restarted server, an eviction, a
      // recording that never closed its predecessor — used to switch captions off for the rest
      // of the meeting behind one line of grey text. Ask for a new session and carry on; the
      // words already on screen are kept, and the authoritative transcript never depended on
      // this surface anyway.
      console.error("[slim] live transcript chunk", e);
      if (!this.liveTranscriptActive || this.liveRestarting) return;
      this.liveRestarting = true;
      // ⚠ CARRY BEFORE RESTARTING. A new session decodes from zero, and `applyLiveTranscript`
      // replaces — so without this the pane would drop everything said before the restart,
      // which is the "lost the first half of a lecture" failure, not a new one.
      this.carryTranscript();
      try {
        await this.plugin.postLiveTranscript({ action: "start", session_id: this.recordingId });
        this.liveTranscriptStatus = "Live transcript · draft";
      } catch (restartError) {
        console.error("[slim] live transcript restart", restartError);
        this.liveTranscriptActive = false;
      } finally {
        this.liveRestarting = false;
        this.updateLiveTranscriptUI();
      }
    });
    // ⚠ ALWAYS cleared, on every path. Left set by a failure, captions stop for the rest of
    // the recording and the pane simply freezes — no error, no status change.
    this.liveTranscriptQueue = this.liveTranscriptQueue.finally(() => {
      this.liveInFlight = false;
      // A backlog that reached the floor while the request was out goes now — that is the
      // coalescing, and it bounds the lag to one round trip. Anything smaller waits for the
      // tap, which flushes at the same floor.
      //
      // ⚠ THIS LINE WAS THE COIL WHINE (2026-08-28). It used to send WHATEVER had arrived, so
      // after the first 2 s chunk the loop ran at one request per round trip — 2.7 requests/s
      // measured on a real recording, each carrying ~340 ms of audio and each re-encoding 20 s
      // of context: GPU 60 % busy on silence, pulsing at 3 Hz. Faster machine, more requests.
      // At the floor: 0.5 requests/s and 9 %.
      if (this.liveTranscriptActive && this.livePcmBytes >= LIVE_FLUSH_BYTES) {
        this.flushLiveTranscript();
      }
    });
    return this.liveTranscriptQueue;
  },

  async stopLiveTranscript() {
    if (!this.liveTranscriptActive) return;
    await this.flushLiveTranscript();
    this.liveTranscriptActive = false;
    try {
      const result = await this.plugin.postLiveTranscript({
        action: "stop", session_id: this.recordingId,
      });
      if (result && result.transcript) this.liveTranscript = result.transcript;
    } catch (e) {
      console.error("[slim] live transcript stop", e);
    }
  },
};
