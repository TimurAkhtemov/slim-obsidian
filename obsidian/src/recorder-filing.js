// RecorderView: stop, file, append, apply the card, retry, cancel, resume.
"use strict";

const { MarkdownView, Notice } = require("obsidian");
const nodePath = require("path");
const { recordingId } = require("./meeting-note");

module.exports = {
  async stop() {
    if (!this.recorder) return;
    const blob = await new Promise((resolve) => {
      this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: "audio/webm" }));
      this.recorder.stop();
    });
    await this.stopLiveTranscript();
    this.teardownStream();
    if (this.manager) this.manager.releaseCapture(this);
    // This segment is over. Fold its captions and its seconds into the recording's running
    // totals so a resume continues the same transcript and the same clock.
    this.carryTranscriptAndClock();

    try {
      // The last chunk arrives just before `onstop`, so the queue is drained after it, not
      // before — otherwise the final five seconds are posted without having been written.
      let rel;
      if (this.diskWrite) {
        await this.writeQueue;
        this.flushPending();
        rel = this.audioRel;
        if (this.pending.length) {
          new Notice("SLIM: the tail of this recording could not be written — the file is " +
                     "playable but may be missing its last seconds.", 20000);
        }
      } else {
        rel = await this.writeAudio(blob);
      }

      // Persist both sources before changing the UI or posting either path. In particular, the
      // in-memory fallback has no audio file until `writeAudio` returns, so a draft-save error
      // must happen after that commit point rather than discard the only copy of the recording.
      await this.saveDraft();
      this.freezeEditor(true);
      this.state = "processing";
      this.processingProgress = {
        stage: "queued", label: "Starting the local pipeline",
        detail: "Your recording and notes are already safe", steps: [], metrics: {},
      };
      this.render();
      await this.submitRecording(this.segments.length ? this.segments : [rel]);
    } catch (e) {
      console.error("[slim] stop", e);
      this.errorText = String((e && e.message) || e);
      this.state = "error";
      this.freezeEditor(false);
    }
    this.render();
  },

  /* Post one staged recording and move to its card.
   *
   * Separated from stop() so RETRY runs exactly the same request. ⚠ The server now REFUSES a
   * recording it could not transcribe rather than filing an empty note and returning 200
   * (2026-08-23), which makes a retry meaningful: the audio is still in `_incoming`, and the
   * usual causes — Ollama not up, the model still loading — are fixed in the time it takes to
   * read the error. Before that change there was nothing to retry, because the server had
   * already reported success.
   */
  async submitRecording(rel) {
    const rels = Array.isArray(rel) ? rel : [rel];
    // Resuming a note that already exists is an APPEND, not a second recording: its audio is
    // archived, its staging cleared, and a new note would split one lecture across two files.
    if (this.appendTo) return this.submitAppend(rels);
    const body = { audio: rels.length > 1 ? rels : rels[0],
                   type: this.type, title: this.title.trim() };
    if (this.draftRel) {
      await this.saveDraft();
      body.draft = this.draftRel;
    } else {
      body.notes_md = "";
    }
    // ⚠ ONE JOB PER ATTEMPT. Reusing the recording id meant the progress poll could find the
    // PREVIOUS attempt's record — finished, and marked cancelled — and stop watching at once,
    // so a perfectly healthy run displayed "Cancelled" from beginning to end.
    const base = this.recordingId ||
      nodePath.posix.basename(rels[0], nodePath.posix.extname(rels[0]));
    this.attempt = (this.attempt || 0) + 1;
    const jobId = `${base}-${this.attempt}`;
    this.jobId = jobId;
    if (typeof this.plugin.watchRecording === "function") {
      body.job_id = jobId;
      this.stopProgressWatch = this.plugin.watchRecording(jobId, (progress) => {
        this.processingProgress = progress;
        if (this.state === "processing") this.render();
      });
    }
    try {
      this.result = await this.plugin.postRecording(body);
    } catch (e) {
      // ⚠ NEITHER OF THESE IS AN ERROR SCREEN. A cancel is their instruction; a silent take is
      // an outcome, not a failure. Both leave every file where it was, so both park in
      // `paused` — the ordinary way forward — instead of asking them to choose between
      // deleting and keeping their own audio over a recording that simply had nothing in it.
      if (e && (e.cancelled || e.silent)) {
        if (e.silent) new Notice(`SLIM: ${e.message}`);
        this.freezeEditor(false);
        this.state = "paused";
        this.pausedTranscript = null;      // re-read; a resume adds words
        this.errorText = "";
        this.processingProgress = null;
        this.render();
        return;
      }
      throw e;
    } finally {
      if (this.stopProgressWatch) this.stopProgressWatch();
      this.stopProgressWatch = null;
    }
    this.cardTitle = this.result.title || "";
    // ⚠ INDEX BEFORE OPENING. `openFile` fires `file-open`, and the handler adopts any recorded
    // note it cannot find in the index as a NEW session — a second card on the same block,
    // approved separately, and a "review still pending" notice on close for the one they never
    // saw (2026-09-05).
    const fromDraft = Boolean(this.draftRel);
    this.draftRel = "";
    this.filedNotePath = this.result.note;
    if (this.manager) this.manager.indexSession(this);
    if (fromDraft) await this.loadEditorFile(this.filedNotePath);
    await this.loadNoteParts();
    this.result.notes_md = this.noteParts.notes;
    this.freezeEditor(false);
    this.activeTab = "summary";
    this.state = "card";
  },

  /* Record more into a note that is already filed — from the review card, or from any
   * finished meeting note. The summary is deliberately NOT re-run (their call): it keeps
   * describing what it described until they presse Retry summary. */
  async startAppendRecording() {
    const back = this.state;
    this.appendTo = this.filedNotePath;
    // ⚠ COME BACK WHERE THEY WERE. Resuming from the review card used to file the note on Stop,
    // skipping the approval step they were standing in.
    this.resumeReturnState = this.state === "card" ? "card" : "finished";
    this.recordingId = recordingId();
    this.segments = [];
    // The pane shows the RECORDING's transcript, not this segment's: resuming a filed note
    // with an empty pane made the words already in the note look gone.
    this.liveCarry = (this.noteParts && this.noteParts.transcript) || "";
    this.elapsedCarry = 0;
    this.state = "recording";
    this.render();
    try {
      if (this.manager) this.manager.claimCapture(this);
      await this.start();
    } catch (e) {
      // ⚠ Permission denied or no device leaves a recording UI with no recorder behind it, and
      // an `appendTo` pointing at a note nothing is going to append to. Put them back.
      console.error("[slim] resume recording", e);
      new Notice(`SLIM: could not start recording — ${e.message || e}`);
      this.appendTo = "";
      this.resumeReturnState = "";
      this.segments = [];
      this.liveCarry = "";
      this.state = back;
      if (this.manager) this.manager.releaseCapture(this);
      this.render();
    }
  },

  async submitAppend(rels) {
    const note = this.appendTo;
    this.attempt = (this.attempt || 0) + 1;
    const jobId = `${this.recordingId}-append-${this.attempt}`;
    this.jobId = jobId;
    let stop = null;
    if (typeof this.plugin.watchRecording === "function") {
      stop = this.plugin.watchRecording(jobId, (progress) => {
        this.processingProgress = progress;
        if (this.state === "processing") this.render();
      });
    }
    try {
      await this.plugin.appendRecording({ note, audio: rels, job_id: jobId });
    } catch (e) {
      if (e && (e.cancelled || e.silent)) {
        // A silent resumed segment adds nothing and costs nothing: say so, and put them back
        // where they were rather than on a failure screen. ⚠ Its staged file has no words in it,
        // so it is dropped here — left behind it would sit in _incoming forever, unassociated
        // with any note, and the next Resume resets `segments` out from under it.
        if (e.silent) {
          new Notice(`SLIM: ${e.message}`);
          for (const seg of this.segments || []) {
            await this.removeStagedSegment(seg).catch(() => {});
          }
          this.segments = [];
          this.appendTo = "";
        }
        this.freezeEditor(false);
        this.state = e.silent ? (this.resumeReturnState || "finished") : "paused";
        this.pausedTranscript = null;
        this.errorText = "";
        this.render();
        return;
      }
      throw e;
    } finally {
      if (stop) stop();
    }
    // ⚠ The POST is the commit point: the words are in the note. Everything below is opening
    // it again, and a failure there is a display problem, not a reason to re-run an append
    // that already happened.
    this.appendTo = "";
    this.segments = [];
    this.filedNotePath = note;
    try {
      await this.loadEditorFile(note);
      await this.loadNoteParts();
    } catch (e) {
      console.error("[slim] reopen after append", e);
      new Notice(`SLIM: the recording was added to ${note}, but the note could not be opened — `
                 + `${e.message || e}`);
    }
    if (this.result) this.result.notes_md = this.noteParts.notes;
    this.freezeEditor(false);
    this.activeTab = "transcript";
    this.state = this.resumeReturnState || "finished";
    this.resumeReturnState = "";
    this.render();
  },

  async applyCard() {
    // Retitling renames the file, so the server's response carries the new path — keep it, or
    // "Open note" and any later apply would point at a file that no longer exists.
    const c = this.result.card;
    this.filingFrom = this.filedNotePath;
    let moved;
    try {
      moved = await this.plugin.applyCard({
        note: this.filedNotePath,
        dest_dir: c.dest_dir,
        type_tag: c.type_tag,
        topics: c.topics || [],
        summary_md: this.result.summary || "",
        notes_md: this.result.notes_md == null ? this.noteParts.notes : this.result.notes_md,
        title: this.cardTitle || "",
      });
    } finally {
      this.filingFrom = "";
    }
    if (moved && moved.note) {
      this.filedNotePath = moved.note;
      this.result.note = moved.note;
      if (this.manager) this.manager.indexSession(this);
      // A move is a RENAME, and Obsidian renames in place: the leaf keeps its file object and
      // the object's path changes, so the note is already on screen. Reopening it reloaded an
      // editor for nothing — and it was the one step approve-with-edits took that
      // approve-without did not, when "Cannot set properties of null" surfaced after changing
      // the suggested location (2026-08-28).
      const view = this.editorLeaf && this.editorLeaf.view;
      const shown = view instanceof MarkdownView && view.file && view.file.path === moved.note;
      if (!shown) {
        // The server has filed the note by now; that is the fact. A pane that cannot be
        // refreshed is a console matter — reported as a failed filing, it re-enabled Approve,
        // left the session at `card`, and the next leaf change announced a pending review
        // for a note that was done.
        try { await this.loadEditorFile(this.filedNotePath); }
        catch (e) { console.error("[slim] refresh after filing", e); }
      }
    }
    this.result.title = this.cardTitle || this.result.title;
  },

  /* Their notes, written back on their own. NOT `applyCard`: apply rewrites frontmatter, drops
   * `filed_by` and can MOVE the note. Typing is not a filing decision. */
  async saveNotes(baseline) {
    const value = (this.result && this.result.notes_md) || "";
    if (!this.filedNotePath || value === baseline.text) return;   // every blur is not a write
    // Closing the note blurs the field AND unloads the block; one edit is still one write.
    if (value === baseline.saving) return;
    baseline.saving = value;
    try {
      await this.plugin.saveNotes({ note: this.filedNotePath, notes_md: value });
      baseline.text = value;
      this.noteParts.notes = value;
    } catch (e) {
      console.error("[slim] save notes", e);
      new Notice(`SLIM: could not save your notes — ${e.message || e}`);
    } finally {
      if (baseline.saving === value) baseline.saving = null;
    }
  },

  /* Blur never fires when the field is torn down with the caret still in it — the note
   * closed, the block re-rendered, Obsidian quit. Those paths call this instead. */
  flushNotes() {
    return this.notesBaseline ? this.saveNotes(this.notesBaseline) : Promise.resolve();
  },

  updateRetryStatusUI() {
    if (!this.contentEl || typeof this.contentEl.querySelector !== "function") return;
    const el = this.contentEl.querySelector(".slim-retry-status");
    if (el && typeof el.setText === "function") el.setText(this.retryStatus || "");
  },

  async retrySummary() {
    if (this.retrying || !this.filedNotePath) return;
    this.retrying = true;
    this.retryStartedAt = Date.now();
    this.retryStatus = "Re-reading the transcript… 0s";
    this.render();
    const jobId = `retry-${this.recordingId || "note"}-${this.retryCount = (this.retryCount || 0) + 1}`;
    let stop = null;
    if (typeof this.plugin.watchRecording === "function") {
      stop = this.plugin.watchRecording(jobId, (progress) => {
        // The LABEL, not the detail: while the model streams, `detail` is the fixed string
        // "Live local-model activity", which says nothing. The seconds are what make a
        // minute-long wait read as progress rather than a stall.
        const elapsed = Math.round((Date.now() - this.retryStartedAt) / 1000);
        const label = progress.label || "Rewriting the summary";
        this.retryStatus = `${label}… ${elapsed}s`;
        // ⚠ IN PLACE, NEVER render(). This fires every 600 ms while the state is card or
        // finished — the states whose DOM holds the live notes editor, the summary editor and
        // the instructions field. A full rebuild every tick tore them out from under the caret
        // a few hundred times per retry.
        this.updateRetryStatusUI();
      });
    }
    try {
      const target = this.filedNotePath;
      const out = await this.plugin.retrySummary({
        note: target,
        instructions: (this.retryInstructions || "").trim(),
        job_id: jobId,
      });
      // ⚠ They may have clicked into another note during the minute this took. The note on disk
      // is correct either way; what must not happen is A's summary being painted into B's view.
      if (this.filedNotePath !== target) return;
      this.result.summary = out.summary;
      this.noteParts.summary = out.summary;
      this.activeTab = "summary";
    } catch (e) {
      // ⚠ The old summary stays. A retry that blanks a good summary because the model died is
      // the recorder reporting a loss as progress, which is its worst failure mode.
      console.error("[slim] retry summary", e);
      new Notice(`SLIM: the summary was not rewritten — ${e.message || e}`);
    } finally {
      if (stop) stop();
      this.retrying = false;
      this.retryStatus = "";
      this.render();
    }
  },

  async cancelProcessing() {
    const jobId = this.jobId || this.recordingId;
    if (!jobId || typeof this.plugin.cancelRecording !== "function") return;
    try {
      const out = await this.plugin.cancelRecording({ job_id: jobId });
      // ⚠ The click can beat the POST that registers the job, and the server then has nothing
      // to flag. Announcing success there is a lie that ends with a filed note.
      if (out && out.cancelling === false) {
        new Notice("SLIM: could not stop it — the job had not started yet. Try Cancel again in " +
                   "a moment.");
        return;
      }
      // Cancelling is cooperative: the summary stream stops mid-generation, but transcription
      // and the filing card are blocking calls that finish their current step first.
      new Notice("SLIM: stopping after the current step. Nothing is written or deleted.");
    } catch (e) {
      console.error("[slim] cancel", e);
      new Notice(`SLIM: could not cancel — ${e.message || e}`);
    }
  },

  async resumeRecording() {
    try {
      if (this.manager) this.manager.claimCapture(this);
      this.state = "recording";
      this.render();
      await this.start();
    } catch (e) {
      if (this.manager) this.manager.releaseCapture(this);
      console.error("[slim] resume", e);
      this.errorText = String((e && e.message) || e);
      this.state = "error";
      this.render();
    }
  },
};
