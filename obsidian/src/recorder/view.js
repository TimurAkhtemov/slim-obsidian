// One recording's controller: its draft, its editor, its lifecycle. The rest of its methods are
// in this folder (capture, live, filing, render, review) and attached to its prototype at the bottom of this file.
"use strict";

const { MarkdownView, Notice } = require("obsidian");
const nodePath = require("path");
const { STAGING_DIR, RECORDER_ATTACHMENTS, DRAFT_SUFFIX, EMPTY_MEETING_BLOCK, recordingIdFromStaged,
        recordingId, splitMeetingNote, frontmatterScalar, hasMeetingBlock,
        embedLegacyMeetingBlock } = require("./meeting-note");

class RecorderView {
  constructor(leaf, plugin) {
    this.leaf = leaf || null;
    this.plugin = plugin;
    this.app = (leaf && leaf.app) || plugin.app;
    this.contentEl = null;
    this.editorView = null;
    this.activeTab = "notes";
    this.noteParts = { summary: "", notes: "", transcript: "" };
    this.summaryEditing = false;
    this.processingProgress = null;
    this.stopProgressWatch = null;
    this.liveTranscript = "";
    // What earlier SEGMENTS of this same recording said. Resume used to show an empty pane and
    // a clock back at zero, so the first forty minutes read as lost while the server had them.
    this.liveCarry = "";
    this.elapsedCarry = 0;
    this.liveTranscriptStatus = "Listening…";
    this.liveTranscriptQueue = Promise.resolve();
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveProcessor = null;
    this.silenceTap = null;       // one analyser per recorded channel; see captureSilence
    this.silenceSamples = null;
    this.silenceWatch = null;
    this.liveTranscriptActive = false;
    this.liveTranscriptPaused = false;
    this.state = "idle";          // idle | recording | processing | paused | card | finished | error
    this.chunks = [];             // ONLY used when the disk lane is unavailable
    this.pending = [];            // written buffers that a failed append still owes the file
    this.writeQueue = Promise.resolve();
    this.diskWrite = false;
    this.audioRel = "";
    this.audioAbs = "";
    this.recorder = null;
    this.streams = [];            // every raw input stream, so all get stopped
    this.audioCtx = null;
    this.elapsed = 0;
    this.timer = null;
    this.clock = null;           // the recording clock; see clockSeconds
    this.recordingId = "";
    this.draftRel = "";
    this.editorFrozen = false;
    this.editorLeaf = null;
    this.title = "";
    this.type = plugin.settings.lastType || "meeting-note";
    this.recordVoice = plugin.settings.recordVoice !== false;
    this.result = null;           // the server's response
    this.edited = false;          // has they touched the card?
    this.filedNotePath = "";
    this.filingFrom = "";
    this.filing = false;
    this.cardTitle = null;
    this.reviewCloseNotified = false;
  }

  async onOpen() {
    this.attachEditorObject();
    this.render();
  }

  async onClose() {
    // The final note is already safe. Closing or reloading the plugin must not impersonate the
    // explicit approval action by applying half-reviewed edits.
    this.teardownStream();
    if (this.stopProgressWatch) this.stopProgressWatch();
    this.detachEditorObject();
  }

  ownsEditor(info) {
    return info === (this.editorLeaf && this.editorLeaf.view) && !this.editorFrozen &&
      !!this.draftRel && !!info.file && info.file.path === this.draftRel;
  }

  async open() {
    if (this.state === "idle" && !this.draftRel) {
      this.recordingId = recordingId();
      await this.createDraft();
    } else if (this.filedNotePath) {
      await this.loadEditorFile(this.filedNotePath);
    } else if (this.draftRel) {
      await this.loadEditorFile(this.draftRel);
    }
    this.attachEditorObject();
    this.render();
  }

  attachEditorObject() {
    const view = this.editorLeaf && this.editorLeaf.view;
    if (!(view instanceof MarkdownView) || !view.file) return false;
    const object = typeof this.plugin.meetingBlockFor === "function"
      ? this.plugin.meetingBlockFor(view.file.path) : null;
    if (!object) return false;
    if (this.editorView === view && this.contentEl && this.contentEl.isConnected !== false) {
      return true;
    }
    this.detachEditorObject();
    this.editorView = view;
    object.empty();
    object.addClass("slim-meeting-object");
    object.addClass("slim-recorder");
    this.contentEl = object;
    return true;
  }

  detachEditorObject() {
    if (this.editorView && this.editorView.contentEl) {
      this.editorView.contentEl.removeClass("slim-recorder-draft-frozen");
    }
    if (this.contentEl && typeof this.contentEl.empty === "function") this.contentEl.empty();
    this.contentEl = null;
    this.editorView = null;
  }

  async ensureVaultDirs(path) {
    const parts = path.split("/").filter(Boolean);
    let current = "";
    for (const part of parts) {
      current = current ? `${current}/${part}` : part;
      if (!(await this.app.vault.adapter.exists(current))) {
        try { await this.app.vault.createFolder(current); } catch (e) { /* raced */ }
      }
    }
  }

  async loadEditorFile(path) {
    const file = this.app.vault.getAbstractFileByPath(path);
    if (!file) throw new Error(`no note at ${path}`);
    if (!this.isOpenMarkdownLeaf(this.editorLeaf)) {
      this.detachEditorObject();
      const prior = this.plugin.editorLeaf;
      const priorOwner = this.manager && typeof this.manager.sessionForLeaf === "function"
        ? this.manager.sessionForLeaf(prior) : null;
      this.editorLeaf = this.isOpenMarkdownLeaf(prior) && (!priorOwner || priorOwner === this)
        ? prior : this.app.workspace.getLeaf("split");
    }
    await this.editorLeaf.openFile(file, { active: true, state: { mode: "source", source: false } });
    this.plugin.editorLeaf = this.editorLeaf;
    // Live Preview exposes the source of the block containing the cursor. A brand-new note
    // otherwise opens with its caret inside the empty slim-meeting fence and looks broken until
    // the user clicks away. Keep the object rendered while leaving the native editor ready for
    // ordinary Markdown immediately below it.
    if (path === this.draftRel && this.state === "idle") {
      const editor = this.editorLeaf.view && this.editorLeaf.view.editor;
      if (editor && typeof editor.lastLine === "function" &&
          typeof editor.setCursor === "function") {
        const moveBelowObject = () => {
          editor.setCursor({ line: editor.lastLine(), ch: 0 });
          if (typeof editor.focus === "function") editor.focus();
        };
        moveBelowObject();
        // `openFile()` resolves before Obsidian has finished restoring the leaf's cursor. Run
        // once more after that render cycle or its restoration puts the caret back inside the
        // fence and exposes the raw block on first open.
        if (typeof requestAnimationFrame === "function") {
          await new Promise((resolve) => requestAnimationFrame(() => {
            moveBelowObject();
            resolve();
          }));
        }
      }
    }
    this.attachEditorObject();
  }

  isOpenMarkdownLeaf(leaf) {
    if (!leaf || !(leaf.view instanceof MarkdownView)) return false;
    const workspace = this.app && this.app.workspace;
    if (!workspace || typeof workspace.getLeavesOfType !== "function") return true;
    return workspace.getLeavesOfType("markdown").includes(leaf);
  }

  async loadNoteParts() {
    if (!this.filedNotePath || !this.app || !this.app.vault) return this.noteParts;
    const file = this.app.vault.getAbstractFileByPath(this.filedNotePath);
    if (!file) return this.noteParts;
    const text = await this.app.vault.read(file);
    this.noteParts = splitMeetingNote(text);
    return this.noteParts;
  }

  /* A cancelled recording parks its segments, their caches and the draft on DISK, but the
   * session lives only in this view. After an Obsidian restart the draft rendered as a passive
   * block: no Resume, no Finish, and no way back to the audio except by hand.
   * Adopting it turns the draft back into the paused object it was. */
  async adoptPausedDraft(draftPath) {
    const name = nodePath.posix.basename(draftPath);
    if (!name.endsWith(DRAFT_SUFFIX) || !name.includes("--recording--")) return false;
    const id = name.slice(0, -DRAFT_SUFFIX.length).split("--recording--").pop();
    if (!id) return false;
    const adapter = this.app && this.app.vault && this.app.vault.adapter;
    if (!adapter || typeof adapter.list !== "function") return false;
    let listing;
    try { listing = await adapter.list(STAGING_DIR); } catch (e) { return false; }
    const segments = (listing.files || [])
      .filter((f) => f.toLowerCase().endsWith(".webm") && recordingIdFromStaged(f) === id)
      .sort();
    if (!segments.length) return false;      // a draft with no audio is just a draft

    this.resetSession();
    this.recordingId = id;
    this.draftRel = draftPath;
    this.segments = segments;
    this.audioRel = segments[segments.length - 1];
    this.type = draftPath.startsWith("Journal/") ? "journal" : this.type;
    this.pausedTranscript = null;
    this.state = "paused";
    return true;
  }

  async openExisting(file, leaf) {
    if (!file || !leaf || !(leaf.view instanceof MarkdownView)) return false;
    const cache = this.app.metadataCache && this.app.metadataCache.getFileCache(file);
    const fm = cache && cache.frontmatter;
    if (!fm || fm.origin !== "recorded") return false;
    let text = await this.app.vault.read(file);
    if (!hasMeetingBlock(text)) {
      text = embedLegacyMeetingBlock(text);
      await this.app.vault.modify(file, text);
    }
    const parts = splitMeetingNote(text);
    if (!parts.transcript) return false;
    this.resetSession();
    this.editorLeaf = leaf;
    this.filedNotePath = file.path;
    this.noteParts = parts;
    const parent = file.path.includes("/") ? file.path.slice(0, file.path.lastIndexOf("/")) : "";
    const topics = Array.isArray(fm.topics) ? fm.topics.slice() : [];
    const type = String(fm.type || "meeting-note");
    this.result = {
      note: file.path,
      title: String(fm.title || file.basename || "Meeting note"),
      summary: parts.summary,
      notes_md: parts.notes,
      card: { dest_dir: parent, type_tag: type, topics, folders: [] },
    };
    this.cardTitle = this.result.title;
    // Absence means complete for backwards compatibility: hundreds of historical recordings
    // predate durable review state and must not suddenly become an inbox. New recordings write
    // `pending`, which makes review survive a closed tab or Obsidian restart.
    this.state = frontmatterScalar(text, "review_status") === "pending" ? "card" : "finished";
    this.activeTab = "summary";
    this.attachEditorObject();
    this.render();
    // The folder tree and the tag counts come only with a fresh recording's response, so a card
    // reopened after a reload browsed an empty box and called every topic "new". Not awaited:
    // the server may still be starting, and the note must open now.
    const card = this.result.card;
    this.cardListsLoad = Promise.resolve()
      .then(() => this.plugin.getJSON("/api/record/folders"))
      .then((lists) => {
        if (!this.result || this.result.card !== card) return;
        card.folders = lists.folders || [];
        card.tag_counts = lists.tag_counts || {};
        this.render();
      })
      .catch((e) => console.error("[slim] card folders", e));
    return true;
  }

  async saveDraft() {
    const view = this.editorLeaf && this.editorLeaf.view;
    if (view instanceof MarkdownView && typeof view.save === "function") await view.save();
  }

  resetSession() {
    if (this.manager) this.manager.releaseCapture(this);
    this.teardownStream();
    this.detachEditorObject();
    this.state = "idle";
    this.activeTab = "notes";
    this.noteParts = { summary: "", notes: "", transcript: "" };
    this.summaryEditing = false;
    this.processingProgress = null;
    if (this.stopProgressWatch) this.stopProgressWatch();
    this.stopProgressWatch = null;
    this.liveTranscript = "";
    this.liveCarry = "";
    this.elapsedCarry = 0;
    this.liveTranscriptStatus = "Listening…";
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveTranscriptActive = false;
    this.liveTranscriptPaused = false;
    this.recordingId = "";
    this.draftRel = "";
    this.audioRel = "";
    this.segments = [];
    // ⚠ Everything a retry or a resume leaves behind. Left set, an in-flight retry on note A
    // resolved into note B's view — its summary written over B's — because `openExisting`
    // repoints the view but used to leave `retrying` true.
    this.appendTo = "";
    this.resumeReturnState = "";
    this.retrying = false;
    this.retryStatus = "";
    this.retryInstructions = "";
    this.pausedTranscript = null;
    this.jobId = "";
    this.attempt = 0;
    this.audioAbs = "";
    this.title = "";
    this.result = null;
    this.edited = false;
    this.filedNotePath = "";
    this.filingFrom = "";
    this.filing = false;
    this.cardTitle = null;
    this.reviewCloseNotified = false;
    this.errorText = "";
    this.freezeEditor(false);
  }

  async reconcileMissingSessionFiles() {
    const missingDraft = this.draftRel && !this.app.vault.getAbstractFileByPath(this.draftRel);
    const missingNote = this.filedNotePath &&
      !this.app.vault.getAbstractFileByPath(this.filedNotePath);
    if (!missingDraft && !missingNote) return false;
    const audio = this.audioRel;
    this.resetSession();
    if (audio) {
      new Notice(`SLIM: the deleted note was released. Its staged audio remains at ${audio}.`);
    }
    return true;
  }

  /* A deletion the server made on this session's behalf is not their instruction to let go:
   * filing removes the draft before its response arrives, and `apply` unlinks the old path
   * while it moves the note — Obsidian reports that unlink as a delete, usually mid-request.
   * Read as their deletion it reset the session under the Approve button: "the deleted note was
   * released", then "Cannot set properties of null (setting 'note')" (2026-09-04). */
  expectsDeletion(path) {
    return this.state === "processing" || this.state === "recording" || path === this.filingFrom;
  }

  handleVaultDelete(file) {
    if (!file || (file.path !== this.draftRel && file.path !== this.filedNotePath)) return;
    if (this.expectsDeletion(file.path)) return;
    this.reconcileMissingSessionFiles().catch((e) => console.error("[slim] delete reconcile", e));
  }

  async trashManagedPath(path) {
    if (!path) return;
    const file = this.app.vault.getAbstractFileByPath(path);
    if (!file) return;
    if (this.app.fileManager && typeof this.app.fileManager.trashFile === "function") {
      await this.app.fileManager.trashFile(file);
    } else if (typeof this.app.vault.trash === "function") {
      await this.app.vault.trash(file, true);
    } else {
      await this.app.vault.delete(file);
    }
  }

  /* A staged segment and the `<segment>.transcript.json` the server cached beside it go
   * together. ⚠ NOT THROUGH THE INDEX: the server writes the cache outside the Vault API, often
   * a moment before the response that leads here, so Obsidian may not hold it yet and
   * `trashManagedPath` skipped it without a word — eleven orphans by 2026-09-28. */
  async removeStagedSegment(seg) {
    await this.trashManagedPath(seg);
    const cache = `${seg}.transcript.json`;
    const adapter = this.app.vault.adapter;
    if (await adapter.exists(cache)) await adapter.remove(cache);
  }

  async discardSession(restart) {
    const id = this.recordingId;
    const draft = this.draftRel;
    // Every segment of this recording, not just the one that was open — a resumed session
    // leaves several, and half a deleted recording is the worst of both outcomes.
    const audios = [...new Set([...(this.segments || []), this.audioRel].filter(Boolean))];
    const attachmentFolder = id ? `${RECORDER_ATTACHMENTS}/${id}` : "";
    const draftIsManaged = draft && draft.endsWith(DRAFT_SUFFIX) &&
      (draft.startsWith("Capture/_unfiled/") || draft.startsWith("Journal/"));
    const managed = (audio) => Boolean(audio) && Boolean(id) &&
      audio.startsWith(`${STAGING_DIR}/`) && recordingIdFromStaged(audio) === id;
    if (draftIsManaged) await this.trashManagedPath(draft);
    for (const audio of audios) {
      if (!managed(audio)) continue;
      await this.removeStagedSegment(audio);
    }
    if (attachmentFolder) await this.trashManagedPath(attachmentFolder);
    this.resetSession();
    new Notice("SLIM: discarded the failed recording and its draft.");
    if (restart) {
      if (this.manager) this.manager.claimCapture(this);
      await this.open();
      if (this.manager) this.manager.indexSession(this);
    }
  }

  closeSession() {
    this.resetSession();
    if (this.manager) this.manager.removeSession(this);
    new Notice("SLIM: closed the recorder. Saved draft and audio were kept.");
  }

  async createDraft() {
    if (!this.recordingId) this.recordingId = recordingId();
    const digits = this.recordingId.slice(0, 8);
    const date = `${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}`;
    const folder = this.type === "journal" ? "Journal" : "Capture/_unfiled";
    await this.ensureVaultDirs(folder);
    this.draftRel = `${folder}/${date}--recording--${this.recordingId}${DRAFT_SUFFIX}`;
    await this.app.vault.create(this.draftRel, EMPTY_MEETING_BLOCK);
    if (this.manager) this.manager.indexSession(this);
    await this.loadEditorFile(this.draftRel);
    return this.draftRel;
  }

  async ensureDraftLocation() {
    if (!this.draftRel) return this.createDraft();
    const folder = this.type === "journal" ? "Journal" : "Capture/_unfiled";
    if (this.draftRel.startsWith(`${folder}/`)) return this.draftRel;
    const file = this.app.vault.getAbstractFileByPath(this.draftRel);
    if (!file) { this.draftRel = ""; return this.createDraft(); }
    await this.ensureVaultDirs(folder);
    const next = `${folder}/${nodePath.posix.basename(this.draftRel)}`;
    if (this.app.fileManager && typeof this.app.fileManager.renameFile === "function") {
      await this.app.fileManager.renameFile(file, next);
    } else if (typeof this.app.vault.rename === "function") {
      await this.app.vault.rename(file, next);
    } else {
      throw new Error("Obsidian cannot move the setup draft into its privacy-safe folder");
    }
    this.draftRel = next;
    if (this.manager) this.manager.indexSession(this);
    await this.loadEditorFile(next);
    return next;
  }

  async removeUntouchedDraft() {
    if (!this.draftRel) return false;
    const draft = this.app.vault.getAbstractFileByPath(this.draftRel);
    if (!draft) { this.draftRel = ""; return false; }
    const contents = await this.app.vault.read(draft);
    if (contents !== "" && contents !== EMPTY_MEETING_BLOCK) return false;
    await this.app.vault.delete(draft);
    this.draftRel = "";
    return true;
  }

  freezeEditor(frozen) {
    this.editorFrozen = !!frozen;
    const view = this.editorLeaf && this.editorLeaf.view;
    if (view && view.contentEl && typeof view.contentEl.toggleClass === "function") {
      view.contentEl.toggleClass("slim-recorder-draft-frozen", this.editorFrozen);
    }
    if (frozen && typeof document !== "undefined" && document.activeElement &&
        typeof document.activeElement.blur === "function") document.activeElement.blur();
  }

  async handleEditorPaste(event, editor) {
    const files = Array.from((event.clipboardData && event.clipboardData.files) || []);
    if (!files.length || files.some((f) => !String(f.type || "").startsWith("image/"))) {
      return false;
    }
    const extensions = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };
    if (files.some((f) => !extensions[f.type])) return false;

    event.preventDefault();
    const folder = `${RECORDER_ATTACHMENTS}/${this.recordingId}`;
    await this.ensureVaultDirs(folder);
    const embeds = [];
    for (const file of files) {
      let n = 1;
      let path;
      do {
        path = `${folder}/paste-${String(n).padStart(3, "0")}.${extensions[file.type]}`;
        n += 1;
      } while (await this.app.vault.adapter.exists(path));
      await this.app.vault.createBinary(path, await file.arrayBuffer());
      embeds.push(`![[${path}]]`);
    }
    editor.replaceSelection(embeds.join("\n"));
    return true;
  }

  // ---- rendering -----------------------------------------------------------------------

  render() {
    const root = this.contentEl;
    if (!root) return;
    root.empty();
    root.addClass("slim-recorder");
    if (typeof root.setAttr === "function") root.setAttr("data-state", this.state);
    if (this.state === "idle") this.renderIdle(root);
    else if (this.state === "recording") this.renderRecording(root);
    else if (this.state === "processing") this.renderProcessing(root);
    else if (this.state === "paused") this.renderPaused(root);
    else if (this.state === "card") this.renderCard(root);
    else if (this.state === "finished") this.renderFinished(root);
    else this.renderError(root);
  }
}

Object.assign(RecorderView.prototype,
  require("./capture"),
  require("./live"),
  require("./filing"),
  require("./render"),
  require("./review"));

module.exports = { RecorderView };
