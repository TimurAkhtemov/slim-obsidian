// The plugin: views, commands, events, the live editor's host, and the HTTP calls to `slim chat`.
"use strict";

const { Plugin, MarkdownView, MarkdownRenderer, MarkdownRenderChild, Notice, requestUrl, setIcon,
        editorInfoField, editorLivePreviewField } = require("obsidian");
const nodePath = require("path");
const { VIEW_TYPE_COPILOT, CopilotView } = require("./chat/view");
const { makeLiveEditor } = require("./live/editor");
const { STAGING_DIR, DRAFT_SUFFIX, recordingIdFromStaged,
        renderPassiveMeetingBlock } = require("./recorder/meeting-note");
const { serverMethods } = require("./server");
const { MeetingSessionManager } = require("./recorder/sessions");
const { DEFAULTS, SlimSettingTab } = require("./settings-tab");
const { streamRequest } = require("./sse");

class SlimRecorderPlugin extends Plugin {
  async onload() {
    this.settings = Object.assign({}, DEFAULTS, await this.loadData());
    this.server = null;
    this._ensuring = null;
    this.meetingBlocks = new Map();
    this.meetings = new MeetingSessionManager(this);
    this.setupLive();
    this.registerMarkdownCodeBlockProcessor("slim-meeting", (_source, el, ctx) => {
      const path = ctx.sourcePath;
      el.addClass("slim-meeting-block");
      this.meetingBlocks.set(path, el);
      if (ctx && typeof ctx.addChild === "function" && MarkdownRenderChild) {
        const child = new MarkdownRenderChild(el);
        child.onunload = () => {
          if (this.meetingBlocks.get(path) === el) this.meetingBlocks.delete(path);
          const session = this.meetings.sessionForPath(path);
          if (session && session.contentEl === el) {
            if (typeof session.flushNotes === "function") session.flushNotes();
            session.contentEl = null;
            session.editorView = null;
          }
        };
        ctx.addChild(child);
      }
      const session = this.meetings.sessionForPath(path);
      if (session) {
        session.attachEditorObject();
        session.render();
      } else {
        // Adoption follows from the real Markdown leaf below. Until then the durable block is
        // still readable; no session is allowed to paint another meeting's state into it.
        renderPassiveMeetingBlock(this.app, this, el, _source, path);
      }
    });
    const activate = () => this.activate().catch((e) => {
      console.error("[slim] activate", e);
      new Notice(`SLIM: could not open the recorder — ${e.message || e}`);
    });
    this.addRibbonIcon("mic", "SLIM: record", activate);
    this.addCommand({ id: "start-recording", name: "Start recording",
                      callback: activate });
    this.registerView(VIEW_TYPE_COPILOT, (leaf) => new CopilotView(leaf, this));
    this.addRibbonIcon("message-square-text", "SLIM: open-note copilot",
                       () => this.activateCopilot());
    this.addCommand({ id: "open-note-copilot", name: "Open note copilot",
                      callback: () => this.activateCopilot() });
    if (this.app.workspace && typeof this.app.workspace.on === "function") {
      this.registerEvent(this.app.workspace.on("active-leaf-change", () => {
        const file = this.app.workspace.getActiveFile();
        for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE_COPILOT)) {
          if (leaf.view && typeof leaf.view.setActiveFile === "function") {
            leaf.view.setActiveFile(file);
          }
        }
        queueMicrotask(() => this.noticeClosedPendingReviews());
      }));
      this.registerEvent(this.app.workspace.on("layout-change", () => this.noticeClosedPendingReviews()));
      // Quitting with the caret in `My notes` fires no blur. Obsidian awaits quit tasks, and
      // the server is killed only at unload, after them.
      this.registerEvent(this.app.workspace.on("quit", (tasks) => {
        tasks.add(() => this.meetings.flushNotes());
      }));
    }
    this.addSettingTab(new SlimSettingTab(this.app, this));

    if (this.app.workspace && typeof this.app.workspace.on === "function" &&
        typeof this.registerEvent === "function") {
      this.registerEvent(this.app.workspace.on("editor-paste", (event, editor, info) => {
        const session = this.meetings.ownsEditor(info);
        if (!session) return;
        session.handleEditorPaste(event, editor)
          .catch((e) => { console.error("[slim] image paste", e); new Notice(`SLIM: ${e.message || e}`); });
      }));
      this.registerEvent(this.app.workspace.on("file-open", (file) => {
        const leaf = this.app.workspace.getMostRecentLeaf && this.app.workspace.getMostRecentLeaf();
        if (!file || !leaf || !(leaf.view instanceof MarkdownView)) return;
        for (const session of this.meetings.sessions.values()) {
          if (session.editorLeaf === leaf && file.path !== session.draftRel &&
              file.path !== session.filedNotePath) session.detachEditorObject();
        }
        const tracked = this.meetings.sessionForPath(file.path);
        if (tracked) {
          tracked.editorLeaf = leaf;
          tracked.attachEditorObject();
          tracked.render();
          return;
        }
        this.meetings.adoptFile(file, leaf).catch((e) => console.error("[slim] open note", e));
      }));
      if (this.app.vault && typeof this.app.vault.on === "function") {
        this.registerEvent(this.app.vault.on("delete", (file) => this.meetings.handleVaultDelete(file)));
        this.registerEvent(this.app.vault.on("rename", (file, oldPath) =>
          this.meetings.handleVaultRename(file, oldPath)));
      }
    }

    // Not awaited: the server takes seconds to bind and onload blocks Obsidian's startup.
    // `post()` waits on it before the first request, which is the moment it actually matters.
    this.ensureServer().catch((e) => console.error("[slim] ensureServer", e));
    if (this.app.workspace && typeof this.app.workspace.onLayoutReady === "function") {
      this.app.workspace.onLayoutReady(() => {
        this.warnAboutStrandedAudio();
        // `file-open` does not fire for leaves already visible when a plugin reloads. Adopt all
        // of them, not whichever one happened to be most recent.
        const leaves = this.app.workspace.getLeavesOfType("markdown") || [];
        for (const leaf of leaves) {
          const file = leaf && leaf.view instanceof MarkdownView && leaf.view.file;
          if (file) this.meetings.adoptFile(file, leaf)
            .catch((e) => console.error("[slim] open visible note", e));
        }
      });
    }
  }

  onunload() {
    if (this.meetings) this.meetings.onClose();
    this.stopServer();
  }

  async saveSettings() { await this.saveData(this.settings); }

  meetingBlockFor(path) {
    const el = this.meetingBlocks && this.meetingBlocks.get(path);
    // A code-block processor is invoked while Obsidian is still constructing the Live Preview
    // DOM, so its element can legitimately report `isConnected === false` at the exact moment
    // the already-open note is adopted after a plugin reload. The MarkdownRenderChild unload
    // hook removes genuinely stale entries; rejecting the construction-phase element here
    // loses the only mount opportunity and leaves an empty block until the user changes files.
    return el || null;
  }

  async openMeetingPath(path) {
    const file = this.app.vault.getAbstractFileByPath(path);
    if (!file) throw new Error(`no note at ${path}`);
    const workspace = this.app.workspace;
    const markdownLeaves = typeof workspace.getLeavesOfType === "function"
      ? workspace.getLeavesOfType("markdown") : [];
    let leaf = markdownLeaves.find((candidate) =>
      candidate.view && candidate.view.file && candidate.view.file.path === path);
    if (!leaf) {
      leaf = workspace.getLeaf(true);
      await leaf.openFile(file, { active: true, state: { mode: "source", source: false } });
    }
    let session = this.meetings.sessionForPath(path);
    if (session) {
      session.editorLeaf = leaf;
      await session.open();
    } else {
      session = await this.meetings.adoptFile(file, leaf);
    }
    if (typeof workspace.revealLeaf === "function") workspace.revealLeaf(leaf);
    return { leaf, session };
  }

  async revealMeetingPath(path) {
    await this.openMeetingPath(path);
    if (this.app.commands && typeof this.app.commands.executeCommandById === "function") {
      this.app.commands.executeCommandById("file-explorer:reveal-active-file");
    }
  }

  showPendingReviewNotice(session) {
    const path = session && session.filedNotePath;
    if (!path) return;
    const notice = new Notice(
      `SLIM: recording saved at ${path}. Review is still pending.`, 20000);
    const el = notice.noticeEl;
    if (!el || typeof el.empty !== "function") return;
    el.empty();
    el.createDiv({ text: "SLIM: recording saved. Review is still pending." });
    el.createDiv({ cls: "slim-notice-path", text: path });
    const actions = el.createDiv({ cls: "slim-notice-actions" });
    const reopen = actions.createEl("button", { cls: "mod-cta", text: "Reopen" });
    reopen.addEventListener("click", () => {
      this.openMeetingPath(path)
        .then(() => { if (typeof notice.hide === "function") notice.hide(); })
        .catch((e) => new Notice(`SLIM: could not reopen the note — ${e.message || e}`));
    });
    const reveal = actions.createEl("button", { text: "Reveal" });
    reveal.addEventListener("click", () => {
      this.revealMeetingPath(path)
        .then(() => { if (typeof notice.hide === "function") notice.hide(); })
        .catch((e) => new Notice(`SLIM: could not reveal the note — ${e.message || e}`));
    });
  }

  noticeClosedPendingReviews() {
    if (!this.meetings || !this.app.workspace ||
        typeof this.app.workspace.getLeavesOfType !== "function") return;
    const live = new Set(this.app.workspace.getLeavesOfType("markdown"));
    for (const session of this.meetings.sessions.values()) {
      const shown = session.editorLeaf && session.editorLeaf.view && session.editorLeaf.view.file;
      const stillShowingMeeting = shown &&
        (shown.path === session.filedNotePath || shown.path === session.draftRel);
      if (live.has(session.editorLeaf) && stillShowingMeeting) {
        session.reviewCloseNotified = false;
      } else if (session.state === "card" && !session.filing && session.editorLeaf &&
                 !session.reviewCloseNotified) {
        session.reviewCloseNotified = true;
        this.showPendingReviewNotice(session);
      }
    }
  }

  // ---- SLIM's changes in the note --------------------------------------------------------------
  // One extension in every Markdown editor: an editing run's changes land in the editor showing
  // the note, drawn where they are. Without CodeMirror (node tests) there is no live editor.
  setupLive() {
    this.liveViews = new Set();
    let cm;
    try {
      cm = { state: require("@codemirror/state"), view: require("@codemirror/view") };
    } catch (e) {
      return;
    }
    try { cm.commands = require("@codemirror/commands"); } catch (e) { /* no isolated history */ }
    this.live = makeLiveEditor(cm, {
      views: this.liveViews,
      infoField: editorInfoField,
      livePreviewField: editorLivePreviewField,
      render: (el, markdown, sourcePath) => MarkdownRenderer.render(this.app, markdown, el, sourcePath, this),
      // A drawing soon replaced (a draft's) is drawn under a component of its own, removed with it.
      renderFor: (el, markdown, sourcePath) => {
        if (!MarkdownRenderChild) return MarkdownRenderer.render(this.app, markdown, el, sourcePath, this).then(() => null);
        const child = this.addChild(new MarkdownRenderChild(el));
        return MarkdownRenderer.render(this.app, markdown, el, sourcePath, child)
          .then(() => () => this.removeChild(child), (error) => { this.removeChild(child); throw error; });
      },
      icon: (el, name) => setIcon?.(el, name),
      changed: (view, runId) => this.liveChanged(view, runId),
      refused: () => new Notice("SLIM: you've edited this change since, so it stays as it is."),
    });
    this.registerEditorExtension(this.live.extension);
  }

  // The editor showing a note: the one focused last.
  liveViewFor(path) {
    const live = this.live;
    let best = null;
    for (const view of (live ? this.liveViews : [])) {
      if (live.pathOf(view) === path && (!best || live.focusedAt(view) > live.focusedAt(best))) best = view;
    }
    return best;
  }

  // The editor holding a run, while one does: its marks, ↺ and Redo live there.
  liveRunView(runId) {
    for (const view of (this.live ? this.liveViews : [])) if (this.live.state(view, runId)) return view;
    return null;
  }

  // Changes land in Live Preview or source; reading view would hide them as they land.
  async showEditing(path) {
    for (const leaf of this.app.workspace?.getLeavesOfType?.("markdown") || []) {
      if (leaf.view?.file?.path !== path || leaf.view?.getMode?.() !== "preview") continue;
      const state = leaf.getViewState();
      await leaf.setViewState({ ...state, state: { ...state.state, mode: "source" } });
    }
  }

  // The note's editor, opening the note if no editor shows it (See before and Restore after the
  // note was closed).
  async noteView(path) {
    await this.showEditing(path);
    if (this.liveViewFor(path) || !this.live) return this.liveViewFor(path);
    const file = this.app.vault.getAbstractFileByPath(path);
    if (!file) return null;
    await this.app.workspace.getLeaf(false).openFile(file);
    return this.liveViewFor(path);
  }

  liveChanged(view, runId) {
    for (const leaf of this.app.workspace?.getLeavesOfType?.(VIEW_TYPE_COPILOT) || []) leaf.view?.runChanged?.(view, runId);
  }

  /* Writing chunks to disk is only half a guarantee if the survivor is never mentioned again.
   * `record.write_recording` unlinks the staged file once the audio is archived, so anything
   * non-empty still sitting here is a recording that never finished being filed. */
  async warnAboutStrandedAudio() {
    try {
      const adapter = this.app.vault.adapter;
      if (!(await adapter.exists(STAGING_DIR))) return;
      const listing = await adapter.list(STAGING_DIR);
      const webm = (listing.files || []).filter((f) => f.toLowerCase().endsWith(".webm"));
      const stranded = [];
      for (const f of webm) {
        // A zero-byte file is a recording that failed before its first chunk — nothing to
        // recover, and warning about it would train them to ignore this notice.
        try {
          const st = await adapter.stat(f);
          if (!st || st.size > 0) stranded.push(f);
        } catch (e) { stranded.push(f); }
      }
      if (!stranded.length) return;

      const draftsById = new Map();
      for (const folder of ["Capture/_unfiled", "Journal"]) {
        if (!(await adapter.exists(folder))) continue;
        const drafts = await adapter.list(folder);
        for (const path of drafts.files || []) {
          const name = nodePath.posix.basename(path);
          if (!name.endsWith(DRAFT_SUFFIX) || !name.includes("--recording--")) continue;
          const id = name.slice(0, -DRAFT_SUFFIX.length).split("--recording--").pop();
          if (id) draftsById.set(id, path);
        }
      }

      // ⚠ A resumed recording stages `<id>-001.webm`, `-002.webm`. Without stripping that
      // suffix the segments look stranded and their draft looks orphaned — the notice whose
      // whole job is to say "nothing was lost" would say the opposite.
      const paired = [...new Set(stranded.map(recordingIdFromStaged))].filter((id) => draftsById.has(id));
      if (paired.length) {
        new Notice(
          `SLIM: ${paired.length} recoverable recording${paired.length > 1 ? "s" : ""} ` +
          `have both a saved Markdown draft and audio (${paired.join(", ")}). Open the draft ` +
          "to review it; SLIM will not delete either source automatically.", 30000);
      }
      const unpaired = stranded.filter((path) => !draftsById.has(recordingIdFromStaged(path)));
      if (!unpaired.length) return;
      new Notice(
        `SLIM: ${unpaired.length} recording${unpaired.length > 1 ? "s" : ""} in ${STAGING_DIR} ` +
        "never finished filing. The audio is intact and playable — file it by hand.", 30000);
    } catch (e) {
      console.error("[slim] stranded audio check", e);
    }
  }

  async activate() {
    const { workspace } = this.app;
    const recent = typeof workspace.getMostRecentLeaf === "function"
      ? workspace.getMostRecentLeaf() : null;
    const leaf = recent && recent.view instanceof MarkdownView ? recent : null;
    const session = await this.meetings.activate(leaf);
    if (session.editorLeaf && typeof workspace.revealLeaf === "function") {
      workspace.revealLeaf(session.editorLeaf);
    }
  }

  /* The recording POST stays synchronous so its durable-source guarantees remain simple.
   * This lightweight companion request exposes only progress for that POST. A first 404 is
   * normal: the poll can win the race with the request handler creating its job record. */
  watchRecording(jobId, onProgress) {
    let stopped = false;
    let timer = null;
    const tick = async () => {
      if (stopped) return;
      try {
        const res = await requestUrl({
          url: `http://127.0.0.1:${this.settings.port}/api/record/progress?job_id=${encodeURIComponent(jobId)}`,
          method: "GET",
          throw: false,
        });
        if (res.status === 200 && res.json) {
          onProgress(res.json);
          if (res.json.done) { stopped = true; return; }
        }
      } catch (e) {
        // Progress is supplementary. The POST below remains the authoritative success/error
        // path, so a transient poll failure must never turn a recoverable recording into an
        // apparent failure.
        console.debug("[slim] progress poll", e);
      }
      if (!stopped) timer = window.setTimeout(tick, 600);
    };
    tick();
    return () => {
      stopped = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }

  async activateCopilot() {
    const { workspace } = this.app;
    const existing = workspace.getLeavesOfType(VIEW_TYPE_COPILOT);
    const leaf = existing.length ? existing[0] : workspace.getRightLeaf(false);
    await leaf.setViewState({ type: VIEW_TYPE_COPILOT, active: true });
    workspace.revealLeaf(leaf);
    if (leaf.view && typeof leaf.view.setActiveFile === "function") {
      await leaf.view.setActiveFile(workspace.getActiveFile());
    }
  }

  async saveActiveNote(file) {
    const view = this.app.workspace.getActiveViewOfType
      ? this.app.workspace.getActiveViewOfType(MarkdownView) : null;
    if (view && view.file?.path === file?.path && typeof view.save === "function") {
      await view.save();
    }
  }

  // `requestUrl` rather than fetch: it bypasses CORS, which a plugin origin otherwise trips on.
  async post(path, body) {
    // Cheap when the server is up (one loopback connect) and the whole point when it is not:
    // a recording made minutes after launch, or after the server died, still lands.
    await this.ensureServer();
    const res = await requestUrl({
      url: `http://127.0.0.1:${this.settings.port}${path}`,
      method: "POST",
      contentType: "application/json",
      body: JSON.stringify(body),
      throw: false,
    });
    if (res.status !== 200) {
      let detail = res.text;
      let cancelled = false;
      let silent = false;
      try {
        detail = (res.json && res.json.error) || detail;
        cancelled = !!(res.json && res.json.cancelled);
        silent = !!(res.json && res.json.silent);
      } catch (e) { /* not json */ }
      // A cancel is THEIR instruction, not a failure, and the caller has to be able to tell the
      // two apart: one parks the session, the other shows an error screen.
      const err = new Error(cancelled || silent
        ? detail : `${path} returned ${res.status}: ${detail}`);
      err.status = res.status;
      err.detail = detail;
      err.cancelled = cancelled;
      err.silent = silent;
      throw err;
    }
    return res.json;
  }

  postJSON(path, body) { return this.post(path, body); }

  async getJSON(path) {
    await this.ensureServer();
    const res = await requestUrl({
      url: `http://127.0.0.1:${this.settings.port}${path}`,
      method: "GET", throw: false,
    });
    if (res.status !== 200) {
      const detail = (res.json && res.json.error) || res.text;
      throw new Error(`${path} returned ${res.status}: ${detail}`);
    }
    return res.json;
  }

  async streamCopilot(body, handlers, signal) {
    await this.ensureServer();
    return streamRequest(this.settings.port, "/api/copilot/stream", body, handlers, signal);
  }

  postRecording(body) { return this.post("/api/record", body); }
  postLiveTranscript(body) { return this.post("/api/record/live", body); }
  applyCard(body) { return this.post("/api/record/apply", body); }
  completeReview(body) { return this.post("/api/record/review", body); }
  saveNotes(body) { return this.post("/api/record/notes", body); }
  retrySummary(body) { return this.post("/api/record/summary", body); }
  cancelRecording(body) { return this.post("/api/record/cancel", body); }
  recordTranscript(body) { return this.post("/api/record/transcript", body); }
  appendRecording(body) { return this.post("/api/record/append", body); }
}

Object.assign(SlimRecorderPlugin.prototype, serverMethods);

module.exports = SlimRecorderPlugin;
