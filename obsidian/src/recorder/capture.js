// RecorderView: microphone and system audio, the clock, the meter, and audio on disk.
"use strict";

const { Notice } = require("obsidian");
const nodePath = require("path");
const { MIC_CHANNEL, SYSTEM_CHANNEL, captureSilence } = require("./audio");
const { clockSeconds, hhmmss } = require("./clock");
const { STAGING_DIR, recordingId } = require("./meeting-note");

module.exports = {
  // ---- recording -----------------------------------------------------------------------

  /* ⚠ The two inputs want OPPOSITE treatment, and getting this backwards is worse than
   * leaving it off.
   *
   * The microphone is in a room: chair movement, keyboard, fans. Chromium ships real DSP for
   * exactly that, and unasked-for it is not reliably on. So the mic gets noise suppression,
   * echo cancellation and auto gain.
   *
   * The loopback input is NOT a room — it is a clean digital copy of what the machine is
   * playing. Noise suppression there chews on the far end's voices, and echo cancellation
   * would try to remove the speaker audio, which is the entire thing we are recording. So it
   * is captured raw.
   */
  async openMic() {
    return navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  },

  /* What the machine is PLAYING, straight from macOS. No virtual device, no routing.
   *
   * Electron >= 39 defaults to Apple's Core Audio Tap API for desktop audio capture on
   * macOS >= 14.2, so `audio: "loopback"` returns a track labelled "System audio" carrying the
   * real mix. Measured 2026-08-22 against a YouTube lecture: peak 0.0 dBFS, RMS -18.7 dBFS —
   * within half a decibel of the same lecture captured through BlackHole.
   *
   * ⚠ `video` is REQUIRED and must be a WebFrameMain or a DesktopCapturerSource. We hand back
   * the requesting frame — Obsidian's own window — and stop the track immediately. That is not
   * a trick to save code: asking desktopCapturer for a SCREEN drags in screen-recording
   * permission for a feature that only ever wanted audio.
   *
   * ⚠ The handler lives on the SHARED default session, so it is installed for exactly one
   * request and always removed, including when the request throws. Leaving it installed would
   * silently hijack any other getDisplayMedia call in Obsidian.
   */
  async openSystemAudio() {
    const { session } = require("@electron/remote");
    session.defaultSession.setDisplayMediaRequestHandler((request, callback) => {
      callback({ video: request.frame, audio: "loopback" });
    });
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      stream.getVideoTracks().forEach((track) => track.stop());
      if (!stream.getAudioTracks().length) throw new Error("no audio track was returned");
      return stream;
    } finally {
      session.defaultSession.setDisplayMediaRequestHandler(null);
    }
  },

  /* Both sides of the conversation in ONE file, and kept APART inside it.
   *
   * A microphone records the room; on headphones that is only them, and the transcript ends up
   * a monologue with the other half missing. So the machine's own output is captured too —
   * directly, via `openSystemAudio`.
   *
   * ⚠ THIS USED TO COST THEM A DEVICE SETUP, and that was the whole complaint (2026-08-22):
   * `getUserMedia` sees only INPUT devices, so reaching system output meant installing
   * BlackHole to impersonate a microphone and building a Multi-Output Device — a NEW one per
   * pair of headphones — to fork the sound so they could still hear it. Notion is one click.
   * Nothing about that was necessary: the Electron Obsidian ships could already ask macOS.
   * The lesson is the pattern, not the fix — reach for what the HOST can do before reaching
   * for the universal workaround.
   *
   * The two are opened separately and joined here with Web Audio — one to each channel —
   * rather than depending on a pre-built Aggregate Device having the right channel layout.
   * It degrades honestly: system audio can fail and you still get the mic, with a Notice
   * saying so.
   *
   * ⚠ LEFT IS THE MICROPHONE, RIGHT IS THE MACHINE — `slim/capture/speakers.py` reads them
   * that way. They used to be summed, and once summed nobody can say who spoke. Apart, "me or
   * them" is a fact of the capture, decided on the server with no model. A missing source leaves its
   * channel silent; it never moves to the other side. The caption tap and the level meter
   * each ask for ONE channel, so Web Audio downmixes for them — leave them alone. The downmix
   * is half of each side, so the level bar reads ~6 dB lower than before for the same voice.
   */
  async buildStream() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.audioCtx = new Ctx();
    const dest = this.audioCtx.createMediaStreamDestination();
    const sides = this.audioCtx.createChannelMerger(2);
    sides.connect(dest);
    this.streams = [];

    let sources = 0;

    if (this.recordVoice) {
      const mic = await this.openMic();
      this.streams.push(mic);
      this.audioCtx.createMediaStreamSource(mic).connect(sides, 0, MIC_CHANNEL);
      sources++;
    }

    try {
      const sys = await this.openSystemAudio();
      this.streams.push(sys);
      this.audioCtx.createMediaStreamSource(sys).connect(sides, 0, SYSTEM_CHANNEL);
      sources++;
    } catch (e) {
      // Losing the far side is bad; losing the recording is worse. Carry on with the mic and
      // SAY SO — ⚠ degrading silently is how a two-hour loss goes unnoticed. There is
      // deliberately no fall back to device pickers: that apparatus is what this removed.
      console.error("[slim] system audio", e);
      new Notice(this.recordVoice
        ? "SLIM: could not capture system audio — recording your voice only."
        : "SLIM: could not capture system audio, and 'Record my voice' is off.");
    }

    if (!sources) throw new Error("no audio source available — nothing would be recorded");
    return dest.stream;
  },

  /* One recording, several audio files. A second MediaRecorder session writes its own
   * container header, so appending a resumed take to the first file risks a recording nothing
   * can play; a numbered segment beside it cannot. The server joins their transcripts. */
  nextSegmentPath() {
    const n = String((this.segments || []).length + 1).padStart(3, "0");
    return `${STAGING_DIR}/${this.recordingId}-${n}.webm`;
  },

  async start() {
    // Idempotent for the holder; a setup draft whose lease a Resume took over is refused here.
    if (this.manager) this.manager.claimCapture(this);
    if (!this.recordingId) this.recordingId = recordingId();
    // ⚠ A FILED NOTE IS ITS OWN DRAFT. `ensureDraftLocation` creates one whenever `draftRel`
    // is empty, and it is empty for a filed note — so a resume used to mint an orphan in
    // Capture/_unfiled, switch the editor to it, and lose everything typed while resuming.
    if (!this.appendTo) await this.ensureDraftLocation();
    const mixed = await this.buildStream();
    if (!this.segments) this.segments = [];
    this.audioRel = this.nextSegmentPath();
    this.segments.push(this.audioRel);
    try {
      let mime = "audio/webm;codecs=opus";
      if (typeof MediaRecorder.isTypeSupported === "function" &&
          !MediaRecorder.isTypeSupported(mime)) {
        mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      }
      this.recorder = mime ? new MediaRecorder(mixed, { mimeType: mime })
                           : new MediaRecorder(mixed);
      this.chunks = [];
      this.pending = [];
      this.writeQueue = Promise.resolve();
      this.warnedWrite = false;
      await this.ensureStagingDirs();
      this.openDiskFile();
      this.recorder.ondataavailable = (e) => { if (e.data && e.data.size) this.onChunk(e.data); };
      // A timeslice means the buffer is flushed periodically rather than held whole until stop —
      // and every flushed chunk is appended to the file on disk immediately. A crash mid-meeting
      // then costs the last five seconds, not the meeting.
      this.recorder.start(5000);
    } catch (e) {
      try { await this.removeUntouchedDraft(); }
      catch (cleanupError) { console.error("[slim] draft cleanup", cleanupError); }
      this.teardownStream();
      throw e;
    }

    this.setupMeter(mixed);
    this.setupSilenceWatch(mixed);
    this.elapsed = this.elapsedCarry || 0;
    this.clock = { base: this.elapsed, since: Date.now() };
    this.timer = window.setInterval(() => this.tick(), 1000);
    if (typeof this.plugin.registerInterval === "function") this.plugin.registerInterval(this.timer);

    this.state = "recording";
    this.render();
    this.startLiveTranscript(mixed);
  },

  tick() {
    if (!this.recorder || this.recorder.state !== "recording") return;
    this.elapsed = clockSeconds(this.clock, Date.now());
    const el = this.contentEl && this.contentEl.querySelector(".slim-timer");
    if (el) el.setText(hhmmss(this.elapsed));
    this.checkSilence();
  },

  setupMeter(mixed) {
    try {
      const src = this.audioCtx.createMediaStreamSource(mixed);
      const analyser = this.audioCtx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        if (!this.audioCtx || this.state !== "recording") return;
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i] - 128));
        const bar = this.contentEl.querySelector(".slim-level-fill");
        if (bar) bar.style.width = `${Math.min(100, (peak / 128) * 160)}%`;
        window.requestAnimationFrame(tick);
      };
      window.requestAnimationFrame(tick);
    } catch (e) {
      console.error("[slim] meter", e);   // a missing level bar must never stop a recording
    }
  },

  /* Both channels of the stream MediaRecorder is writing — what the file gets, not what the
   * devices claim. Read once a second by the clock's timer, not per frame: rAF stops in a hidden
   * window, and a meeting is exactly when Obsidian is behind the call. The largest analyser
   * window (32768 samples, ~0.7 s) sees most of each second. 'Record my voice' off leaves the
   * left side empty on purpose, and system silence alone is never a warning — so there is
   * nothing to watch. */
  setupSilenceWatch(mixed) {
    this.silenceWatch = null;
    if (!this.recordVoice) return;
    try {
      const source = this.audioCtx.createMediaStreamSource(mixed);
      const split = this.audioCtx.createChannelSplitter(2);
      const mic = this.audioCtx.createAnalyser();
      const system = this.audioCtx.createAnalyser();
      mic.fftSize = 32768;
      system.fftSize = 32768;
      source.connect(split);
      split.connect(mic, MIC_CHANNEL);
      split.connect(system, SYSTEM_CHANNEL);
      this.silenceTap = { source, split, mic, system };
      this.silenceSamples = new Float32Array(mic.fftSize);
    } catch (e) {
      console.error("[slim] silence watch", e);   // a missing warning must never stop a recording
    }
  },

  checkSilence() {
    if (!this.silenceTap) return;
    const peak = (analyser) => {
      const data = this.silenceSamples;
      analyser.getFloatTimeDomainData(data);
      let max = 0;
      for (let i = 0; i < data.length; i++) max = Math.max(max, Math.abs(data[i]));
      return max;
    };
    const was = this.silenceWatch ? this.silenceWatch.warn : false;
    this.silenceWatch = captureSilence(this.silenceWatch, {
      mic: peak(this.silenceTap.mic), system: peak(this.silenceTap.system),
    });
    if (this.silenceWatch.warn !== was) this.render();
  },

  togglePause() {
    if (!this.recorder) return;
    if (this.recorder.state === "recording") {
      this.recorder.pause();
      this.liveTranscriptPaused = true;
      if (this.clock) {
        this.clock = { base: clockSeconds(this.clock, Date.now()), since: null };
        this.elapsed = this.clock.base;
      }
    } else if (this.recorder.state === "paused") {
      this.recorder.resume();
      this.liveTranscriptPaused = false;
      if (this.clock) this.clock.since = Date.now();
    }
    this.render();
  },

  teardownStream() {
    if (this.timer) { window.clearInterval(this.timer); this.timer = null; }
    // The last tick can be a throttled minute old; stop the clock at the wall's reading.
    if (this.clock) { this.elapsed = clockSeconds(this.clock, Date.now()); this.clock = null; }
    // EVERY input, not just the mic — a live loopback stream keeps the device busy and the
    // recording indicator on long after the pane is gone.
    this.streams.forEach((s) => s.getTracks().forEach((t) => t.stop()));
    this.streams = [];
    if (this.liveTranscriptActive) {
      this.liveTranscriptActive = false;
      this.plugin.postLiveTranscript({ action: "stop", session_id: this.recordingId })
        .catch((e) => console.debug("[slim] live transcript cleanup", e));
    }
    if (this.liveProcessor) {
      for (const node of Object.values(this.liveProcessor)) {
        try { if (node) node.disconnect(); } catch (e) { /* already disconnected */ }
      }
      this.liveProcessor = null;
    }
    if (this.silenceTap) {
      for (const node of Object.values(this.silenceTap)) {
        try { node.disconnect(); } catch (e) { /* already disconnected */ }
      }
      this.silenceTap = null;
      this.silenceSamples = null;
    }
    if (this.audioCtx) { try { this.audioCtx.close(); } catch (e) { /* already closed */ } }
    this.audioCtx = null;
  },

  async ensureStagingDirs() {
    await this.ensureVaultDirs(STAGING_DIR);
  },

  /* Open the file the chunks will be appended to, and prove it is writable BEFORE recording —
   * discovering a permissions problem five seconds into a meeting is discovering it too late.
   *
   * Node's `fs`, not the Vault API, because Obsidian offers only create and overwrite: there is
   * no appendBinary. `isDesktopOnly: true` in the manifest is precisely the licence for this.
   * The path is still a vault-relative path resolved against the vault's own base, and the
   * containment check keeps it that way.
   */
  openDiskFile() {
    try {
      const fs = require("fs");
      const adapter = this.app.vault.adapter;
      if (typeof adapter.getBasePath !== "function") throw new Error("no base path");
      const base = nodePath.resolve(adapter.getBasePath());
      const abs = nodePath.resolve(base, this.audioRel);
      if (abs !== base && !abs.startsWith(base + nodePath.sep)) {
        throw new Error(`${this.audioRel} resolves outside the vault`);
      }
      fs.mkdirSync(nodePath.dirname(abs), { recursive: true });
      fs.writeFileSync(abs, Buffer.alloc(0));   // create/truncate, so appends start from zero
      this.fs = fs;
      this.audioAbs = abs;
      this.diskWrite = true;
    } catch (e) {
      // ⚠ SAY SO. Falling back silently is exactly how a two-hour loss goes unnoticed: the
      // recording still works, so nothing looks wrong until the day something crashes.
      console.error("[slim] disk flush unavailable", e);
      this.diskWrite = false;
      new Notice(
        "SLIM: cannot write audio to disk while recording — holding it in memory until you " +
        "press Stop. A crash or a quit before then loses the whole recording. " +
        `(${(e && e.message) || e})`, 20000);
    }
  },

  onChunk(blob) {
    if (!this.diskWrite) { this.chunks.push(blob); return; }
    // Serialized: `Blob.arrayBuffer()` is async and the webm's clusters must land in the order
    // MediaRecorder produced them, or the container is garbage.
    // The `.catch` is not decoration: a rejected queue swallows every LATER chunk silently,
    // because `.then` on a rejected promise never runs.
    this.writeQueue = this.writeQueue
      .then(() => this.appendChunk(blob))
      .catch((e) => { console.error("[slim] write queue", e); });
  },

  async appendChunk(blob) {
    try {
      this.pending.push(Buffer.from(await blob.arrayBuffer()));
    } catch (e) {
      console.error("[slim] chunk read", e);
      return;
    }
    this.flushPending();
  },

  /* Append everything the file is still owed. `pending` normally holds exactly one buffer and
   * empties immediately; it only grows when a write fails, and the next chunk then retries the
   * lot in order. That keeps a transient failure (a full disk, an iCloud stall) from tearing a
   * hole in the middle of the container. */
  flushPending() {
    if (!this.pending.length) return;
    try {
      this.fs.appendFileSync(this.audioAbs, this.pending.length === 1
        ? this.pending[0] : Buffer.concat(this.pending));
      this.pending = [];
    } catch (e) {
      console.error("[slim] append", e);
      if (!this.warnedWrite) {
        this.warnedWrite = true;
        new Notice("SLIM: a chunk of audio did not reach disk. Still recording, and the next " +
                   "chunk retries it — but check free space.", 15000);
      }
    }
  },

  async writeAudio(blob) {
    // The in-memory fallback, used only when `openDiskFile` failed. Written by Obsidian itself;
    // the audio never travels over HTTP (the server caps bodies at 1 MB and a meeting is tens
    // of megabytes). From the moment this returns, the recording exists on disk.
    await this.ensureStagingDirs();
    await this.app.vault.createBinary(this.audioRel, await blob.arrayBuffer());
    return this.audioRel;
  },
};
