// RecorderView: Delete, from any state, behind a second click.
"use strict";

const { Notice } = require("obsidian");
const { ARCHIVE_DIR, RECORDER_ATTACHMENTS } = require("./meeting-note");

module.exports = {
  /* Delete this recording outright, from any state: they asked for it on the object itself,
   * and the alternative was waiting for it to file and then deleting by hand in the file
   * explorer (2026-09-30). Everything goes to Obsidian's trash, so it stays recoverable.
   * A filed note takes its archived audio and pasted images with it, and the index is told. */
  async deleteRecording() {
    if (this.filing || this.retrying) {
      new Notice("SLIM: wait for the current step to finish, then delete.");
      return;
    }
    if (this.state === "processing") return this.deleteAfterProcessing();
    this.deleting = true;
    try {
      await this.abandonCapture();
      if (this.appendTo) return await this.discardTake();
      const note = this.filedNotePath;
      const filed = note ? await this.filedSources(note) : [];
      await this.trashStagedSources();
      for (const path of filed) await this.trashManagedPath(path);
      if (note) {
        await this.trashManagedPath(note);
        if (typeof this.plugin.forgetRecording === "function") {
          // The note is gone either way; a stale index entry is the next sweep's to clear.
          await this.plugin.forgetRecording({ note })
            .catch((e) => console.error("[slim] forget", e));
        }
      }
      this.resetSession();
      if (this.manager) this.manager.removeSession(this);
      new Notice("SLIM: deleted the recording.");
    } finally {
      this.deleting = false;
    }
  },

  /* Deletion is cooperative like Cancel: the job stops after its current step, or files if it
   * was nearly done — and whichever way it ends, `submitRecording` deletes what it left. */
  async deleteAfterProcessing() {
    this.deleteWhenDone = true;
    this.render();
    new Notice("SLIM: deleting after the current step.");
    const jobId = this.jobId || this.recordingId;
    if (!jobId || typeof this.plugin.cancelRecording !== "function") return;
    await this.plugin.cancelRecording({ job_id: jobId })
      .catch((e) => console.error("[slim] cancel for delete", e));
  },

  async deleteWhenStopped() {
    this.deleteWhenDone = false;
    this.freezeEditor(false);
    this.state = this.filedNotePath ? "card" : "paused";
    this.processingProgress = null;
    await this.deleteRecording();
  },

  /* The archived audio and pasted images a filed note names. Read before the note goes, and
   * only from the two folders the recorder writes. Its audio comes from the note's own text:
   * a note filed a moment ago may not be in Obsidian's metadata cache yet. The archive is
   * content-addressed, so a file another note also names stays: deleting this note must not
   * take that one's source. */
  async filedSources(note) {
    const file = this.app.vault.getAbstractFileByPath(note);
    if (!file) return [];
    const text = await this.app.vault.read(file);
    const head = text.startsWith("---\n") ? text.slice(0, text.indexOf("\n---", 3) + 1) : "";
    const archived = new RegExp(`${ARCHIVE_DIR}/[^\\s"',\\]]+`, "g");
    const elsewhere = new Set();
    if (typeof this.app.vault.getMarkdownFiles === "function" && this.app.metadataCache) {
      for (const other of this.app.vault.getMarkdownFiles()) {
        if (other.path === note) continue;
        const cache = this.app.metadataCache.getFileCache(other);
        const audio = cache && cache.frontmatter && cache.frontmatter.audio;
        (Array.isArray(audio) ? audio : [audio]).forEach((p) => elsewhere.add(String(p || "")));
      }
    }
    const paths = [...head.matchAll(archived)].map((m) => m[0]).filter((p) => !elsewhere.has(p));
    const pasted = new RegExp(`${RECORDER_ATTACHMENTS}/([A-Za-z0-9_-]+)/`, "g");
    for (const m of text.matchAll(pasted)) paths.push(`${RECORDER_ATTACHMENTS}/${m[1]}`);
    return [...new Set(paths)];
  },

  /* Stop capturing without filing anything: the recorder's own stop, minus the submit. */
  async abandonCapture() {
    if (!this.recorder || this.recorder.state === "inactive") return;
    await new Promise((resolve) => {
      this.recorder.onstop = resolve;
      this.recorder.stop();
    });
    this.recorder = null;
    // The last chunk lands just before `onstop`; let it reach disk before its file is trashed.
    if (this.writeQueue) await this.writeQueue.catch(() => {});
    this.teardownStream();
    if (this.manager) this.manager.releaseCapture(this);
  },

  /* A resumed segment deleted before it was added: the note keeps what it had. */
  async discardTake() {
    for (const seg of this.segments || []) await this.removeStagedSegment(seg).catch(() => {});
    this.segments = [];
    this.appendTo = "";
    this.liveTranscript = "";
    this.liveCarry = "";
    this.freezeEditor(false);
    this.state = this.resumeReturnState || "finished";
    this.resumeReturnState = "";
    this.render();
    new Notice("SLIM: discarded that take. The note is as it was.");
  },

  /* One Delete for every state. Armed, not confirmed in a modal: the first click turns it into
   * "Click again to delete" for four seconds, so a stray click costs nothing and a meant one
   * costs one more. The arming lives on the session so a re-render mid-count keeps it. */
  renderDeleteButton(row) {
    const label = this.appendTo ? "Discard take" : "Delete";
    const armed = Date.now() < (this.deleteArmedUntil || 0);
    const button = row.createEl("button", {
      cls: "mod-warning slim-delete",
      text: this.deleteWhenDone ? "Deleting…" : armed ? "Click again to delete" : label,
    });
    button.disabled = Boolean(this.deleteWhenDone);
    button.addEventListener("click", async () => {
      if (Date.now() >= (this.deleteArmedUntil || 0)) {
        this.deleteArmedUntil = Date.now() + 4000;
        button.setText("Click again to delete");
        window.setTimeout(() => {
          if (Date.now() >= (this.deleteArmedUntil || 0)) button.setText(label);
        }, 4000);
        return;
      }
      this.deleteArmedUntil = 0;
      button.disabled = true;
      try {
        await this.deleteRecording();
      } catch (e) {
        console.error("[slim] delete", e);
        new Notice(`SLIM: could not delete the recording — ${e.message || e}`);
        button.disabled = false;
        button.setText(label);
      }
    });
    return button;
  },
};
