// The open-note copilot sidebar. Its rendering and editing-run methods are in render.js and
// edit-run.js, attached to its prototype at the bottom of this file.
"use strict";

const { ItemView, Modal } = require("obsidian");
const { MAX_COPILOT_IMAGES, resizeCopilotImage } = require("./images");
const { sha256Hex, editorSelection, MAX_EDIT_CHARS, quickOrDeep, titleFromQuestion, threadId,
        cancelledError, historyText } = require("./text");
const { clear, create } = require("./dom");

const VIEW_TYPE_COPILOT = "slim-copilot";

// window.prompt is not an option: Electron defines it as a function that THROWS
// ("prompt() is not supported."), so a `!window.prompt` guard passes and the click rejects
// silently. `onSubmit` receives the trimmed title, or null when the modal is dismissed.
class RenameModal extends Modal {
  constructor(app, current, onSubmit) {
    super(app);
    this.current = current;
    this.onSubmit = onSubmit;
    this.done = false;
  }

  finish(value) {
    if (this.done) return;
    this.done = true;
    this.onSubmit(value);
  }

  onOpen() {
    const root = this.contentEl;
    create(root, "h3", "", "Rename chat");
    const input = create(root, "input", "slim-rename-input");
    input.type = "text";
    input.value = this.current;
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); submit(); }
    });
    const row = create(root, "div", "slim-row");
    const cancel = create(row, "button", "", "Cancel");
    cancel.onclick = () => this.close();
    const ok = create(row, "button", "mod-cta", "Rename");
    const submit = () => {
      const title = String(input.value || "").trim();
      if (title) this.finish(title);
      this.close();
    };
    ok.onclick = submit;
    if (typeof input.focus === "function") { input.focus(); input.select?.(); }
  }

  onClose() {
    clear(this.contentEl);
    this.finish(null);
  }
}

class CopilotView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.app = plugin.app;
    this.context = null;
    this.threads = [];
    this.activeThread = null;
    this.turns = [];
    this.mode = "quick";
    this.stage = "";
    this.draft = "";
    this.error = "";
    this.loading = false;
    this.sending = false;
    this.streamingText = "";
    this.abort = null;
    this.requestVersion = 0;
    this.pendingThreadId = null;
    this.deleteArmed = null;
    this.retryQuestion = "";
    this.pendingImages = [];
    this.retryImages = [];
    this.loadingPath = null;
    this.streamingThread = null;
    this.streamingTurns = null;
    this.el = null;                 // live handles from the last full render()
    this.composerFocused = false;
    this.groupOpen = new Map();     // sourceId -> open, where they toggled a group by hand
    this.showThreads = false;       // the history button: the chat list in the conversation's slot
    this.nearbyOpen = false;        // the context chip under the title
    this.liveRenderTimer = null;    // the throttled Markdown render of the streaming answer
    this.liveRenderVersion = 0;
    this.liveRenderedAt = 0;
    this.skills = [];               // /commands, from GET /api/copilot/skills
    this.deletedThreads = new Set(); // a late save must not resurrect these
    this.editRun = null;            // the editing run streaming now: its plan, steps and changes
    this.editClock = null;          // its seconds, in the chat while it runs
    this.stageSince = 0;            // when the call running now started, by the stage it named
    this.stageClock = null;         // the stage line's seconds, while a turn runs
    this.runTurns = new Map();      // before_ref -> the thread and turns its turn lives in
    this.savingRuns = null;         // a run's turn saves one save after another
    this.beforeTexts = new Map();   // before_ref -> the note as it was before that run
    this.restoreArmed = null;       // the before_ref whose Restore waits for "Restore anyway"
    this.restoreRefused = null;     // {ref, message}: why the server would not restore that run
    this.restoredFrom = new Map();  // before_ref -> the note as it was just before Restore
  }

  // Re-read whenever they start a command: a skill saved in `Skills/` a moment ago is there.
  async loadSkills() {
    try {
      this.skills = (await this.plugin.getJSON("/api/copilot/skills")).skills || [];
    } catch (error) {
      console.error("[slim] skills", error);
    }
    this.renderSlash();
  }

  pickSkill(skill) {
    this.draft = `/${skill.name} `;
    this.composerFocused = true;
    this.render();
  }

  getViewType() { return VIEW_TYPE_COPILOT; }
  getDisplayText() { return "SLIM Copilot"; }
  getIcon() { return "message-square-text"; }

  async onOpen() {
    // On the root, which outlives every render(): answers are rebuilt, the listener is not.
    const root = this.contentEl;
    if (typeof this.registerDomEvent === "function") {
      this.registerDomEvent(root, "click", (event) => this.onLinkClick(event));
    } else if (root && typeof root.addEventListener === "function") {
      root.addEventListener("click", (event) => this.onLinkClick(event));
    }
    const file = this.app.workspace.getActiveFile ? this.app.workspace.getActiveFile() : null;
    this.loadSkills();
    await this.setActiveFile(file);
  }

  async onClose() { this.cancel(); }

  // MarkdownRenderer turns an answer's [[wikilink]] into <a class="internal-link">, but Obsidian
  // wires clicks on those only inside its own note views; in this sidebar they were dead.
  // ⚠ Resolve BEFORE opening: openLinkText creates a missing target (see openNote), and the
  // model can name a note that does not exist.
  onLinkClick(event) {
    const link = event?.target?.closest?.("a.internal-link");
    if (!link) return;
    event.preventDefault();
    const target = link.getAttribute("data-href") || link.getAttribute("href") || "";
    const source = this.context?.source?.path || "";
    const linkpath = target.split("#")[0].split("|")[0].trim();
    const file = this.app.metadataCache?.getFirstLinkpathDest?.(linkpath, source);
    if (!file) {
      this.error = `Note not found: ${linkpath}`;
      this.render();
      return;
    }
    this.app.workspace.openLinkText(target, source, Boolean(event.metaKey || event.ctrlKey));
  }

  async setActiveFile(file) {
    const path = file && String(file.extension || "").toLowerCase() === "md" ? file.path : null;
    // Same note, loaded or loading: nothing to do. Focus moving between the editor and this
    // sidebar fires active-leaf-change with the SAME file; re-syncing here cancelled the
    // answer, erased the question, and — with onOpen, the event and activateCopilot all
    // racing — ran ingest_note twice into an IntegrityError. Edits since the open are picked
    // up by send() (refreshNote), not by focus changes.
    if (path && (this.context?.source.path === path || this.loadingPath === path)) return;
    const version = ++this.requestVersion;
    this.cancel();
    this.error = "";
    this.context = null;
    this.activeThread = null;
    this.turns = [];
    this.retryQuestion = "";
    this.pendingImages = [];
    this.retryImages = [];
    if (!file || String(file.extension || "").toLowerCase() !== "md") {
      this.error = "Open a saved Markdown note to use SLIM Copilot.";
      this.render();
      return;
    }
    this.loading = true;
    this.loadingPath = path;
    this.render();
    try {
      if (typeof this.plugin.saveActiveNote === "function") {
        await this.plugin.saveActiveNote(file);
      }
      if (version !== this.requestVersion) return;
      const [context, listing] = await Promise.all([
        this.plugin.postJSON("/api/copilot/context", { path: file.path }),
        this.plugin.getJSON("/api/copilot/threads"),
      ]);
      if (version !== this.requestVersion) return;
      this.context = context;
      this.threads = listing.threads || [];
      const wanted = (this.pendingThreadId
        ? this.threads.find((item) => item.id === this.pendingThreadId
          && item.source_id === context.source.id)
        : null) || this.threads.find((item) => item.source_id === context.source.id);
      if (wanted) {
        this.pendingThreadId = null;
        await this.openThread(wanted.id, version, true);
      }
    } catch (error) {
      // A 400 is the server refusing THIS note in a sentence meant for them; the route and
      // status in front of it are for a debugger.
      if (version === this.requestVersion) {
        this.error = (error.status === 400 && error.detail) || error.message || String(error);
      }
    } finally {
      if (version === this.requestVersion) {
        this.loading = false;
        this.loadingPath = null;
        this.render();
      }
    }
  }

  // The note as it is NOW: save the editor, re-sync the row (ingest_note is hash-keyed, so an
  // unchanged note costs one stat), skip the nearby-notes search. Called by send(), so every
  // question is answered from what they have typed, not from the version indexed at open.
  async refreshNote() {
    if (!this.context) return;
    const path = this.context.source.path;
    const file = this.app.workspace.getActiveFile ? this.app.workspace.getActiveFile() : null;
    if (file && file.path === path && typeof this.plugin.saveActiveNote === "function") {
      await this.plugin.saveActiveNote(file);
    }
    await this.saveOpenCopies(path);
    const fresh = await this.plugin.postJSON("/api/copilot/context", { path, related: false });
    this.context = { ...this.context, source: fresh.source };
  }

  newThread() {
    if (!this.context) return;
    this.activeThread = {
      id: threadId(), title: "New chat", source_id: this.context.source.id,
      source_path: this.context.source.path, reasoning_mode: this.mode,
    };
    this.turns = [];
    this.stage = "";
    this.error = "";
    this.retryQuestion = "";
    this.pendingImages = [];
    this.retryImages = [];
    this.showThreads = false;
    this.render();
  }

  async openThread(id, version = this.requestVersion, afterNavigation = false) {
    const data = await this.plugin.getJSON(`/api/copilot/thread?id=${encodeURIComponent(id)}`);
    if (version !== this.requestVersion) return;
    if (!afterNavigation && this.context && data.source_id !== this.context.source.id) {
      this.pendingThreadId = id;
      this.activeThread = null;
      this.turns = [];
      this.openNote(data.source_path);
      return;
    }
    this.activeThread = data;
    this.turns = data.turns || [];
    this.pendingImages = [];
    this.retryImages = [];
    // A legacy "auto" on disk must not reach the mode selector; `saveRecord` normalizes the
    // record itself, so nothing has to correct `data` here.
    this.mode = quickOrDeep(data.reasoning_mode);
    this.showThreads = false;
    this.render();
  }

  async saveRecord(thread, turns) {
    const body = {
      id: thread.id, title: thread.title,
      // A question's images by the refs the server stored them under. One pasted for a question
      // that failed before a run began was never stored: it has no id, and is not saved.
      turns: turns.map(({ attachments, ...turn }) => {
        const refs = (attachments || []).filter((image) => image.id)
          .map(({ id, name, mime, bytes }) => ({ id, name, mime, bytes }));
        return refs.length ? { ...turn, attachments: refs } : turn;
      }),
      source_id: thread.source_id, source_path: thread.source_path,
      // ⚠ Normalized HERE, not at the point a thread is opened: `renameThread` fetches a
      // thread it never opened and hands it straight to this method, so a pre-2026-09-03
      // conversation would post "auto" and be refused with nothing catching it.
      reasoning_mode: quickOrDeep(thread.reasoning_mode || this.mode),
    };
    const saved = await this.plugin.postJSON("/api/copilot/thread/save", body);
    Object.assign(thread, saved);
    const index = this.threads.findIndex((item) => item.id === thread.id);
    if (index >= 0) this.threads[index] = { ...thread };
    else this.threads.unshift({ ...thread });
    return saved;
  }

  async addImages(files) {
    if (this.sending) return;
    const images = [...(files || [])].filter((file) => String(file?.type || "").startsWith("image/"));
    if (!images.length) return;
    if (this.pendingImages.length + images.length > MAX_COPILOT_IMAGES) {
      this.error = `Attach at most ${MAX_COPILOT_IMAGES} images to one message.`;
      this.render();
      return;
    }
    try {
      for (const file of images) this.pendingImages.push(await resizeCopilotImage(file));
      this.error = "";
    } catch (error) {
      this.error = error.message || String(error);
    }
    this.render();
  }

  async send(text = this.draft) {
    const outgoingImages = this.pendingImages;
    const question = String(text || "").trim() || (outgoingImages.length ? "Explain this image." : "");
    if (!question || !this.context || this.sending) return;
    if (!this.activeThread) this.newThread();
    // A failed question waits under its error for Try again. Asking something else instead
    // takes it off the thread: it would be saved with the next answer and sent as history, and
    // a refused /command is sent as its whole skill prompt.
    const failed = this.turns.at(-1);
    if (this.retryQuestion && failed?.role === "you" && failed.text === this.retryQuestion) this.turns.pop();
    this.retryQuestion = "";
    this.retryImages = [];
    // Snapshot the thread that ASKED. openThread/newThread/deleteThread stay clickable while
    // an answer streams, and the answer belongs to this thread whatever is active when it
    // lands — otherwise B was saved with A's answer and A's question was never saved.
    const thread = this.activeThread;
    const turns = this.turns;
    const path = this.context.source.path;
    if (!turns.length) thread.title = titleFromQuestion(question);
    const asked = { role: "you", text: question,
      ...(outgoingImages.length ? { attachments: outgoingImages } : {}) };
    turns.push(asked);
    this.draft = "";
    this.pendingImages = [];
    this.showThreads = false;
    this.sending = true;
    this.stage = "Starting";
    this.stageStarted();
    this.error = "";
    let streamed = "";
    this.streamingText = "";
    this.streamingThread = thread;
    this.streamingTurns = turns;
    let completed = null;
    let run = null;                       // an editing run, from its `edit` event on
    let local = null, tracked = null;     // the editor tracking it from Send, and its name there
    const started = Date.now();
    const abort = new AbortController();
    this.abort = abort;
    // An editing run's events land in the note. One that throws must not end the stream (the
    // changes before it are in the note already), and one that arrives after Stop is dropped.
    const on = (handle) => (data) => {
      if (abort.signal.aborted) return;
      try { handle(data); } catch (error) { console.error("[slim] edit event", error); }
    };
    // A change's text while the model writes it is a draft in the note: on the editor tracking
    // this question until the server names the run, then on the run's, while it shows the note.
    const drafting = () => (run ? (this.inView(run) ? [run.view, run.ref] : []) : local ? [tracked, local] : []);
    this.render();
    try {
      await this.refreshNote();
      if (abort.signal.aborted) throw cancelledError();
      // The editor's text, not the file: Obsidian saves a moment after typing, and an editing
      // run works on exactly what the owner sees. The editor tracks the owner's typing from
      // here, under a name of the chat's own until the server names the run.
      const shown = this.plugin.liveViewFor?.(path) || null;
      const text = shown ? shown.state.doc.toString() : null;
      const doc = text !== null && text.length <= MAX_EDIT_CHARS ? text : null;
      const view = doc === null ? null : shown;
      if (view && this.plugin.live) {
        local = `send-${threadId()}`;
        tracked = view;
        this.plugin.live.startRun(view, { id: local, base: doc, pending: true });
      }
      // Only where it is visibly the input: a /command. A selection left in the note rode
      // along with every question, unseen (review, 2026-09-23).
      const picked = question.startsWith("/") ? editorSelection(this.app, path) : { text: "" };
      await this.plugin.streamCopilot({
        thread_id: thread.id, source_id: this.context.source.id, question,
        // Role, text and small image refs only, every turn whole: the server fits the
        // conversation to the model's window (copilot.fit_window). Slim turns also carry
        // evidence and stats; re-uploading those on every question grew with the square of the
        // thread. Image bytes stay server-side. A question with no answer after it (a chat
        // saved before a failed one came off) is not history.
        history: turns.slice(0, -1).filter((turn, i, all) => turn.role !== "you" || all[i + 1]?.role === "slim")
          .map((turn) => ({ ...turn, text: historyText(turn) }))
          .map(({ role, text, attachments }) => ({ role, text,
            ...(attachments?.length ? { attachments: attachments.map(
              ({ id, name, mime, bytes }) => ({ id, name, mime, bytes })) } : {}) })),
        // A regenerated question re-sends its stored refs; the server has the bytes.
        images: outgoingImages.map(({ id, name, mime, data, bytes }) =>
          data ? { name, mime, data } : { id, name, mime, bytes }),
        reasoning_mode: this.mode,
        ...(doc === null ? {} : { doc, doc_hash: sha256Hex(doc) }),
        selection: picked.text,
        ...(picked.from === undefined ? {} : { selection_from: picked.from }),
      }, {
        stage: (data) => this.setStage(data.stage),
        delta: (data) => {
          streamed += data.text || "";
          this.streamingText = streamed;
          this.renderDelta();
        },
        draft: on((data) => {
          const [view, id] = drafting();
          if (view) this.plugin.live?.startDraft?.(view, id, data);
        }),
        draft_text: on((data) => {
          const [view] = drafting();
          if (view) this.plugin.live?.growDraft?.(view, String(data.id), String(data.text || ""));
        }),
        draft_end: on((data) => {
          const [view] = drafting();
          if (view) this.plugin.live?.endDraft?.(view, String(data.id));
        }),
        edit: on((data) => {
          // A run's turn is kept however it ends, and one that stops or fails never gets the
          // `turn` with its question's image refs: the run's start names them.
          if (Array.isArray(data.attachments) && data.attachments.length) asked.attachments = data.attachments;
          const ref = String(data.before_ref || "");
          const claimed = Boolean(local && this.plugin.live?.claimRun(tracked, local, ref));
          local = null;
          run = { ref, hash_before: data.hash_before, plan: null, steps: [], changes: [], pending: [],
                  view: claimed ? tracked : null, path, started, thread, turns };
          this.editRun = run;
          Promise.resolve(this.plugin.showEditing?.(path)).catch((e) => console.warn("[slim] editing view", e));
          this.startEditClock();
          this.renderEdit();
        }),
        plan: on((data) => {
          if (!run) return;
          run.plan = { intro: String(data.intro || ""), steps: (data.steps || []).map(String) };
          this.renderEdit();
        }),
        step: on((data) => {
          if (data.state === "doing") this.stageStarted();      // a step is a call of its own
          if (!run || !(data.index >= 1)) return;
          run.steps[data.index - 1] = data.state;
          this.renderEdit();
        }),
        change: on((data) => run && this.landChange(run, data)),
        turn: (data) => { completed = data; },
        error: (data) => { throw new Error(data.error || "Copilot failed"); },
      }, abort.signal);
      if (!completed) throw new Error("The server ended without a completed turn.");
      asked.attachments = completed.attachments || [];
      delete completed.attachments;          // refs belong to the user turn, not the answer
      if (run) completed.edit_run = await this.settleRun(run, completed.edit_run || {});
      turns.push({ role: "slim", text: completed.answer || streamed, turn: completed });
      this.retryQuestion = "";
      this.retryImages = [];
      thread.reasoning_mode = this.mode;
      await this.saveRecord(thread, turns);
    } catch (error) {
      const stopped = error.name === "AbortError";
      if (run && turns.at(-1) === asked) {
        // What landed is in the note: it stays, and the turn keeps how far the run got and the
        // copy from before, so See before and Restore work after a reload too.
        turns.push(await this.partialTurn(run, stopped ? "stopped" : `error: ${error.message || error}`));
        if (!stopped) this.error = error.message || String(error);
        thread.reasoning_mode = this.mode;
        if (!this.deletedThreads.has(thread.id)) {
          await this.saveRecord(thread, turns).catch((e) => { this.error = e.message || String(e); });
        }
      } else if (stopped) {
        // Stopped before an answer: the question is theirs — hand it back rather than leave a
        // dangling "you" turn to ride along as history and be persisted with the next answer.
        if (turns.at(-1) === asked) {
          turns.pop(); this.draft = question; this.pendingImages = outgoingImages;
        }
      } else if (turns.at(-1) === asked) {
        // The model never answered: show the error under the question and offer a retry.
        this.error = error.message || String(error);
        this.retryQuestion = question;
        this.retryImages = outgoingImages;
      } else {
        // Answered but not saved (e.g. the thread is over its cap): the answer stays visible
        // with the reason. No retry — it would append a second you/slim pair every click.
        this.error = error.message || String(error);
      }
    } finally {
      if (local) this.plugin.live?.dropRun(tracked, local);
      if (run) this.finishRun(run);
      clearInterval(this.stageClock);
      this.stageClock = null;
      this.abort = null;
      this.sending = false;
      this.stopLiveRender();
      this.streamingText = "";
      this.streamingThread = null;
      this.streamingTurns = null;
      this.stage = "";
      this.render();
    }
  }

  // A call started: its seconds count from now, ticking on the second.
  stageStarted() {
    this.stageSince = Date.now();
    clearInterval(this.stageClock);
    this.stageClock = setInterval(() => { if (this.el?.stage) this.renderStage(); }, 1000);
    this.stageClock.unref?.();
  }

  // The server names a stage as each model call starts. The same one again is that call still
  // running (Deep's heartbeat while it thinks); a new one, or a new step, starts the count again.
  setStage(name) {
    const stage = name || "Working";
    if (stage !== this.stage) {
      this.stage = stage;
      this.stageStarted();
    }
    this.renderStage();
  }

  cancel() {
    if (this.abort) this.abort.abort();
    this.abort = null;
    this.sending = false;
    this.stopLiveRender();
    this.streamingText = "";
    this.stage = "";
  }

  // Ask the last question again: the pair comes off the thread and the question goes back
  // through send() with the image refs it had, so the answer lands where the first one did.
  async regenerate() {
    if (this.sending || !this.activeThread) return;
    const answer = this.turns.at(-1);
    const asked = this.turns.at(-2);
    if (answer?.role !== "slim" || asked?.role !== "you") return;
    // Asking again after an edit would apply its changes a second time.
    if (answer.turn?.edit_run) return;
    this.turns.splice(-2, 2);
    this.pendingImages = (asked.attachments || []).map((ref) => ({ ...ref }));
    await this.send(asked.text);
  }

  async retry() {
    const question = this.retryQuestion;
    if (!question || this.sending) return;
    const last = this.turns.at(-1);
    if (last?.role === "you" && last.text === question) this.turns.pop();
    this.retryQuestion = "";
    this.pendingImages = this.retryImages;
    this.retryImages = [];
    await this.send(question);
  }

  async deleteThread(id) {
    if (this.deleteArmed !== id) {
      this.deleteArmed = id;
      this.render();
      return;
    }
    this.deletedThreads.add(id);
    // A late answer would save into — and so resurrect — the thread being deleted.
    if (this.sending && this.streamingThread?.id === id) this.cancel();
    await this.plugin.postJSON("/api/copilot/thread/delete", { id });
    this.threads = this.threads.filter((item) => item.id !== id);
    if (this.activeThread?.id === id) this.newThread();
    this.deleteArmed = null;
    this.render();
  }

  async renameThread(id) {
    let thread = this.activeThread?.id === id ? this.activeThread : null;
    let turns = this.activeThread?.id === id ? this.turns : null;
    if (!thread) {
      thread = await this.plugin.getJSON(`/api/copilot/thread?id=${encodeURIComponent(id)}`);
      turns = thread.turns || [];
    }
    const title = await this.promptTitle(thread.title);
    if (!title) return;
    thread.title = title.slice(0, 120);
    if (this.activeThread?.id === id) this.activeThread.title = thread.title;
    await this.saveRecord(thread, turns);
    this.render();
  }

  // Guarded like the recorder's "Open note" button: openLinkText CREATES a missing target,
  // and citation / thread paths are only as fresh as the last ingest — a note dragged since
  // then would become an empty phantom, which the next active-leaf-change indexes and embeds.
  promptTitle(current) {
    return new Promise((resolve) => new RenameModal(this.app, current, resolve).open());
  }

  async openNote(path) {
    const file = this.app.vault && typeof this.app.vault.getAbstractFileByPath === "function"
      ? this.app.vault.getAbstractFileByPath(path) : null;
    if (!file) {
      this.error = `Note not found: ${path} — it may have moved since the last index.`;
      this.render();
      return;
    }
    await this.app.workspace.getLeaf(false).openFile(file);
  }

  reindex(path) {
    this.saveOpenCopies(path)
      .then(() => this.plugin.postJSON("/api/copilot/context", { path, related: false }))
      .catch((e) => console.warn("[slim] index after restore", path, e.message));
  }

  // Every pane showing the note writes its buffer first, so what reads the file next (a sync,
  // an index) sees what the owner typed, not the copy from before Obsidian's autosave.
  async saveOpenCopies(path) {
    for (const leaf of this.app.workspace?.getLeavesOfType?.("markdown") || []) {
      if (leaf.view?.file?.path === path && typeof leaf.view.save === "function") await leaf.view.save();
    }
  }
}

Object.assign(CopilotView.prototype,
  require("./render"),
  require("./edit-run"));

module.exports = { VIEW_TYPE_COPILOT, RenameModal, CopilotView };
