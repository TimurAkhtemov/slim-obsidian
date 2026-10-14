// RecorderView: the meeting block's screens, one per state.
"use strict";

const { Notice } = require("obsidian");
const { hhmmss } = require("./clock");
const { STAGING_DIR, TYPES } = require("./meeting-note");

module.exports = {
  renderSavedLocation(root) {
    if (!this.filedNotePath) return;
    const row = root.createDiv({ cls: "slim-saved-location" });
    const copy = row.createDiv({ cls: "slim-saved-location-copy" });
    copy.createSpan({ cls: "slim-saved-location-label", text: "Saved in the vault" });
    copy.createSpan({ cls: "slim-saved-location-path", text: this.filedNotePath });
    const reveal = row.createEl("button", { text: "Reveal" });
    reveal.addEventListener("click", () => {
      if (typeof this.plugin.revealMeetingPath !== "function") return;
      this.plugin.revealMeetingPath(this.filedNotePath)
        .catch((e) => new Notice(`SLIM: could not reveal the note — ${e.message || e}`));
    });
  },

  /* Retry the summary against the transcript the note already holds — Notion's button, and
   * the reason they asked for it: iterate on the summarizer without re-recording a lecture.
   * The instructions field is the SLIM-safe half of Notion's: nothing tells the model what
   * KIND of recording this is, they do, in their own words, for this one retry. */
  renderRetryRow(root) {
    const row = root.createDiv({ cls: "slim-retry" });
    const field = row.createDiv({ cls: "slim-field slim-retry-field" });
    field.createEl("label", { text: "Instructions (optional)" });
    const input = field.createEl("input", { type: "text", cls: "slim-retry-instructions" });
    input.value = this.retryInstructions || "";
    input.setAttr("placeholder", "organise by concept, keep the formulas…");
    input.addEventListener("input", (e) => { this.retryInstructions = e.target.value; });

    const button = row.createEl("button", { text: "Retry summary" });
    button.disabled = !!this.retrying;
    button.addEventListener("click", () => this.retrySummary());
    // Always present, even empty: the progress ticks update it in place rather than
    // re-rendering the card, so it has to be there before the first tick arrives.
    row.createSpan({ cls: "slim-retry-status", text: this.retryStatus || "" });
    return row;
  },

  renderPaused(root) {
    const n = (this.segments || []).length;
    this.renderObjectHeader(root, "SLIM meeting note", this.title.trim() || "Recording paused",
                            "Paused");
    // Its own padded body: the header pads itself, so content hung directly off the root sat
    // flush against the left edge.
    const body = root.createDiv({ cls: "slim-object-body slim-paused" });
    body.createEl("p", { cls: "slim-hint",
      text: `${n} segment${n === 1 ? "" : "s"} saved, and your notes with them. Carry on ` +
            "recording into this same note, or finish it now." });

    // "1 segment saved" is a claim they cannot check. Show the words.
    const pane = body.createDiv({ cls: "slim-live-transcript" });
    const head = pane.createDiv({ cls: "slim-live-transcript-head" });
    head.createSpan({ cls: "slim-object-section-label", text: "Transcript so far" });
    const status = head.createSpan({ cls: "slim-live-transcript-status", text: "Reading…" });
    const text = pane.createEl("p", { cls: "slim-live-transcript-text",
                                      text: this.pausedTranscript || "Reading the transcript…" });
    if (this.pausedTranscript == null && typeof this.plugin.recordTranscript === "function") {
      this.plugin.recordTranscript({ audio: this.segments || [] }).then((out) => {
        this.pausedTranscript = out.text || "";
        status.setText(out.pending
          ? `${out.words} words · ${out.pending} segment${out.pending === 1 ? "" : "s"} not transcribed yet`
          : `${out.words} words`);
        text.setText(this.pausedTranscript || "Nothing was transcribed before you cancelled.");
      }).catch((e) => {
        console.error("[slim] paused transcript", e);
        status.setText("could not read it");
        text.setText("The audio is safe; the transcript is not available yet.");
      });
    } else {
      status.setText("");
      text.setText(this.pausedTranscript || "Nothing was transcribed before you cancelled.");
    }

    const row = body.createDiv({ cls: "slim-row" });
    const resume = row.createEl("button", { cls: "mod-cta", text: "Resume recording" });
    resume.addEventListener("click", () => this.resumeRecording());
    // Always offered: a cancel that left them with no way forward would be a worse trap than
    // the wait they cancelled.
    const finish = row.createEl("button", { text: "Finish" });
    finish.addEventListener("click", async () => {
      this.freezeEditor(true);
      this.state = "processing";
      this.processingProgress = {
        stage: "queued", label: "Starting the local pipeline",
        detail: "Your recording and notes are already safe", steps: [], metrics: {},
      };
      this.render();
      try {
        await this.submitRecording(this.segments);
      } catch (e) {
        console.error("[slim] finish", e);
        this.errorText = String((e && e.message) || e);
        this.state = "error";
        this.freezeEditor(false);
      }
      this.render();
    });
  },

  renderObjectHeader(root, eyebrow, title, status) {
    const head = root.createDiv({ cls: "slim-object-head" });
    const copy = head.createDiv({ cls: "slim-object-head-copy" });
    copy.createDiv({ cls: "slim-object-eyebrow", text: eyebrow });
    copy.createEl("h3", { text: title });
    const badge = head.createDiv({ cls: "slim-object-status" });
    badge.createSpan({ cls: "slim-object-status-dot" });
    badge.createSpan({ text: status });
    return head;
  },

  renderProperties(root) {
    const c = this.result && this.result.card;
    if (!c) return;
    const props = root.createDiv({ cls: "slim-object-properties" });
    const property = (label, value) => {
      const p = props.createDiv({ cls: "slim-object-property" });
      p.createSpan({ cls: "slim-object-property-label", text: label });
      p.createSpan({ cls: "slim-object-property-value", text: value || "—" });
    };
    property("Location", c.dest_dir);
    property("Type", TYPES.find(([value]) => value === c.type_tag)?.[1] || c.type_tag);
    property("Topics", (c.topics || []).join(", "));
  },

  renderIdle(root) {
    this.renderObjectHeader(root, "SLIM meeting note", "Set up this recording", "Saved locally");
    const setup = root.createDiv({ cls: "slim-object-setup" });

    const t = setup.createDiv({ cls: "slim-field" });
    t.createEl("label", { text: "Title" });
    const titleInput = t.createEl("input", { type: "text", placeholder: "What is this?" });
    titleInput.value = this.title;
    titleInput.addEventListener("input", (e) => { this.title = e.target.value; });

    const ty = setup.createDiv({ cls: "slim-field" });
    ty.createEl("label", { text: "Type" });
    const sel = ty.createEl("select");
    TYPES.forEach(([value, label]) => {
      const o = sel.createEl("option", { text: label });
      o.value = value;
      if (value === this.type) o.selected = true;
    });
    sel.addEventListener("change", (e) => {
      this.type = e.target.value;
      this.plugin.settings.lastType = this.type;
      this.plugin.saveSettings();
    });
    ty.createEl("p", { cls: "slim-hint",
      text: "Chosen before you speak, so nothing has to guess it later." });

    const v = setup.createDiv({ cls: "slim-field slim-toggle" });
    const box = v.createEl("input", { type: "checkbox" });
    box.checked = this.recordVoice;
    v.createEl("label", { text: "Record my voice" });
    box.addEventListener("change", (e) => {
      this.recordVoice = e.target.checked;
      this.plugin.settings.recordVoice = this.recordVoice;
      this.plugin.saveSettings();
    });
    v.createEl("p", { cls: "slim-hint",
      text: "What the computer is playing is always captured — no device setup, whatever your "
          + "output is set to. Turn this off for a lecture you are only watching." });

    const btn = setup.createEl("button", { cls: "mod-cta slim-big", text: "Start recording" });
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try { await this.start(); }
      catch (e) {
        console.error("[slim] start", e);
        new Notice(`SLIM: could not start recording — ${e.message || e}`);
        btn.disabled = false;
      }
    });
  },

  renderRecording(root) {
    const paused = this.recorder && this.recorder.state === "paused";
    this.renderObjectHeader(root, "SLIM meeting note", this.title.trim() || "Untitled recording",
                            paused ? "Paused" : "Recording");
    const console = root.createDiv({ cls: "slim-recording-console" });
    const toolbar = console.createDiv({ cls: "slim-recording-toolbar" });
    const head = toolbar.createDiv({ cls: "slim-rec-head" });
    head.createSpan({ cls: `slim-dot ${paused ? "is-paused" : ""}` });
    head.createSpan({ cls: "slim-timer", text: hhmmss(this.elapsed) });
    head.createSpan({ cls: "slim-state", text: paused ? "Paused" : "Live" });
    const row = toolbar.createDiv({ cls: "slim-row slim-recording-actions" });
    const p = row.createEl("button", { text: paused ? "Resume" : "Pause" });
    p.addEventListener("click", () => this.togglePause());
    const s = row.createEl("button", { cls: "mod-cta", text: "Stop" });
    s.addEventListener("click", async () => {
      s.disabled = true; p.disabled = true;
      await this.stop();
    });
    if (this.silenceWatch && this.silenceWatch.warn) {
      console.createDiv({
        cls: "slim-silence-warning", attr: { role: "alert" },
        text: "Nothing is being recorded — the microphone and system audio are both silent. " +
              "Check the input device (System Settings › Sound) and Obsidian's microphone " +
              "permission (Privacy & Security › Microphone). Recording continues.",
      });
    }
    const level = console.createDiv({ cls: "slim-level" });
    level.createDiv({ cls: "slim-level-fill" });
    const transcript = console.createDiv({ cls: "slim-live-transcript" });
    const transcriptHead = transcript.createDiv({ cls: "slim-live-transcript-head" });
    transcriptHead.createSpan({ cls: "slim-object-section-label", text: "Live transcript" });
    transcriptHead.createSpan({ cls: "slim-live-transcript-status",
      text: this.liveStatusText() });
    transcript.createEl("p", { cls: "slim-live-transcript-text",
      text: this.shownTranscript() || "Listening for speech…" });
    const bridge = console.createDiv({ cls: "slim-editor-bridge" });
    bridge.createDiv({ cls: "slim-object-section-label", text: "Your notes" });
    bridge.createEl("p", {
      text: "Keep writing in the native Markdown editor below. Every keystroke is already saved to this recording.",
    });
  },

  renderProcessing(root) {
    this.renderObjectHeader(root, "SLIM meeting note", this.title.trim() || "Untitled recording",
                            "Processing locally");
    const progress = this.processingProgress || {};
    const w = root.createDiv({ cls: "slim-processing" });
    const current = w.createDiv({ cls: "slim-processing-current" });
    current.createDiv({ cls: "slim-spinner" });
    const copy = current.createDiv();
    copy.createEl("h3", { text: progress.label || "Working on your note…" });
    copy.createEl("p", { cls: "slim-hint",
      text: progress.detail || "Your recording and notes are safe while SLIM finishes locally." });
    // Stop was hit by accident, 40 minutes into a lecture. Cancelling parks the session with
    // every segment and their notes intact instead of making them wait out the whole pipeline.
    const cancel = current.createEl("button", { cls: "slim-cancel", text: "Cancel" });
    cancel.addEventListener("click", () => this.cancelProcessing());

    const metrics = progress.metrics || {};
    const facts = [];
    if (metrics.transcript_words != null) facts.push(`${metrics.transcript_words} words`);
    if (metrics.transcription_seconds != null) facts.push(`transcribed in ${Number(metrics.transcription_seconds).toFixed(1)}s`);
    if (metrics.summary_output_tokens != null) facts.push(`${metrics.summary_output_tokens} summary tokens`);
    if (metrics.summary_decode_tok_s != null) facts.push(`${metrics.summary_decode_tok_s} tok/s`);

    const activity = w.createEl("details", { cls: "slim-thinking" });
    const activitySummary = activity.createEl("summary");
    activitySummary.createSpan({ cls: "slim-thinking-spark" });
    activitySummary.createSpan({ text: metrics.model_activity_events ? "Thinking locally" : "What SLIM is doing" });
    activitySummary.createSpan({ cls: "slim-thinking-stage", text: progress.label || "Starting" });
    activity.createEl("p", { cls: "slim-hint",
      text: metrics.model_activity_events
        ? "The local model is working. Its private reasoning is not shown."
        : "Stage and performance updates appear here as the local pipeline runs." });
    if (facts.length) activity.createDiv({ cls: "slim-thinking-meta", text: facts.join(" · ") });
  },

  renderError(root) {
    this.renderObjectHeader(root, "SLIM meeting note",
                            this.title.trim() || this.recordingId || "Untitled recording",
                            "Processing stopped · sources preserved");
    root.createEl("h3", { text: "That did not work" });
    root.createEl("p", { text: this.errorText || "Unknown error." });
    root.createEl("p", { cls: "slim-hint",
      text: "The audio is at " + ((this.segments || []).join(", ") || this.audioRel || STAGING_DIR) +
            (this.draftRel ? " and your notes are at " + this.draftRel : "") +
            ". Nothing is lost. " +
            "If the server is the problem, check the plugin's repo and uv settings — it is " +
            "started from here, on port " + this.plugin.settings.port + "." });
    const row = root.createDiv({ cls: "slim-row slim-error-actions" });
    const again = row.createEl("button", { cls: "mod-cta", text: "Try again" });
    again.addEventListener("click", async () => {
      // ⚠ EVERY segment, not the one that happens to be current. Posting `this.audioRel` would
      // re-file the last segment of a resumed recording and silently drop the earlier ones,
      // under a screen that says nothing is lost.
      const all = (this.segments || []).length ? this.segments
                : (this.audioRel ? [this.audioRel] : []);
      if (!all.length) return;
      try {
        await this.saveDraft();
        this.freezeEditor(true);
        this.state = "processing";
        this.processingProgress = {
          stage: "queued", label: "Retrying the local pipeline",
          detail: "Using the same saved audio and notes", steps: [], metrics: {},
        };
        this.render();
        await this.submitRecording(all);
      } catch (e) {
        console.error("[slim] retry", e);
        this.errorText = String((e && e.message) || e);
        this.state = "error";
        this.freezeEditor(false);
      }
      this.render();
    });
    const discard = row.createEl("button", { cls: "mod-warning", text: "Delete & start over" });
    discard.addEventListener("click", async () => {
      again.disabled = true; discard.disabled = true;
      try { await this.discardSession(true); }
      catch (e) {
        console.error("[slim] discard", e);
        new Notice(`SLIM: could not discard the recording — ${e.message || e}`);
        again.disabled = false; discard.disabled = false;
      }
    });
    const close = row.createEl("button", { text: "Keep files & close" });
    close.addEventListener("click", () => this.closeSession());
  },

  renderFinished(root) {
    const title = (this.result && this.result.title) || this.cardTitle || "Meeting note";
    this.renderObjectHeader(root, "SLIM meeting note", title, "Finished");
    this.renderSavedLocation(root);
    this.renderProperties(root);
    this.renderSource(root, false);
    this.renderRetryRow(root);
    const row = root.createDiv({ cls: "slim-row slim-finished-actions" });
    const more = row.createEl("button", { text: "Resume recording" });
    more.addEventListener("click", () => this.startAppendRecording());
    if (this.result && this.result.card) {
      const edit = row.createEl("button", { text: "Edit review details" });
      edit.addEventListener("click", async () => {
        try {
          await this.saveDraft();
          await this.loadNoteParts();
          this.result.summary = this.noteParts.summary;
          this.result.notes_md = this.noteParts.notes;
          this.summaryEditing = false;
          this.state = "card";
          this.render();
        } catch (e) {
          console.error("[slim] reopen review", e);
          new Notice(`SLIM: could not reopen the review — ${e.message || e}`);
        }
      });
    }
    root.createEl("p", { cls: "slim-native-editor-hint",
      text: "This block is part of the note. Keep writing normally before or after it." });
  },
};
