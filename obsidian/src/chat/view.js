// The vault assistant's sidebar: one chat, whatever note is open. Its rendering and editing-run methods are in render.js and
// edit-run.js, attached to its prototype at the bottom of this file.
"use strict";

const { ItemView, Modal } = require("obsidian");
const { MAX_COPILOT_IMAGES, resizeCopilotImage } = require("./images");
const { sha256Hex, editorSelection, MAX_EDIT_CHARS, effortOf, titleFromQuestion, threadId,
        cancelledError, historyText, threadRows, mentionQuery, mentionMatches } = require("./text");
const { clear, create, composing } = require("./dom");
const MAX_PINS = 12;     // the server refuses more (copilot.turn)

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
      if (composing(event)) return;
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
    this.openNote = null;           // {path, id|null}: the Markdown note open now, sent with each question
    this.context = null;            // the open note's nearby-notes payload ({source, related})
    this.threads = [];
    this.threadQuery = "";
    this.threadSearch = null;       // the debounce timer of the history search
    this.threadVersion = 0;         // the chat being opened: a later open wins
    this.searchVersion = 0;         // the history listing: a later search wins
    this.noteVersion = 0;
    this.activeThread = null;
    this.turns = [];
    this.effort = effortOf(plugin.settings?.effort);   // kept across restarts; medium unless changed
    this.stage = "";
    this.draft = "";
    this.error = "";
    this.sending = false;
    this.streamingText = "";
    this.liveTools = [];            // the lookups of the turn streaming now, a line each
    this.abort = null;
    this.deleteArmed = null;
    this.retryQuestion = "";
    this.retryImages = [];
    this.retryFor = null;           // the chat a failed question belongs to: only there is it retried
    this.retryError = "";           // its error, shown again when that chat is opened again
    this.pendingImages = [];
    this.loadingPath = null;
    this.streamingThread = null;
    this.streamingTurns = null;
    this.el = null;                 // live handles from the last full render()
    this.composerFocused = false;
    this.searchFocused = false;     // the history search had focus: a redraw gives it back
    this.showThreads = false;       // the history button: the chat list in the conversation's slot
    this.stuck = true;              // the transcript follows its bottom until they scroll up
    this.nearbyOpen = false;        // the nearby-notes popover over the composer
    this.effortOpen = false;        // the effort menu over the composer's toolbar
    this.caretAt = null;            // where the next render puts the composer's caret, once
    this.seenTurns = new WeakSet(); // turns already drawn: only a new one fades in
    this.liveFresh = false;         // the streaming answer is about to be drawn for the first time
    this.toolsOpen = new WeakSet(); // turns whose folded lookups are open
    this.sourcesOpen = new WeakSet(); // turns whose sources are open
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

  // The notes and folders `@` offers: Markdown files, and every folder but the root and the
  // app's own (.obsidian, .trash).
  mentionSources() {
    const vault = this.app?.vault;
    const files = (vault?.getMarkdownFiles?.() || []).map((file) => file.path)
      .filter((path) => !path.split("/").some((part) => part.startsWith(".")));
    const folders = (vault?.getAllLoadedFiles?.() || []).filter((item) => item.children && item.path
      && item.path !== "/" && !item.path.split("/").some((part) => part.startsWith(".")))
      .map((item) => item.path);
    return { files, folders };
  }

  mentionHits() {
    const found = mentionQuery(this.draft, this.caret);
    if (!found) return [];
    const { files, folders } = this.mentionSources();
    return mentionMatches(found.query, files, folders);
  }

  // Choosing an `@` match: the `@query` leaves the draft and a chip takes its place. Never a
  // duplicate, never a 13th (the server refuses more than MAX_PINS).
  pin(item) {
    // The first pin of a sidebar with no chat yet starts one, and keeps what was pasted for it.
    const thread = this.activeThread || (this.activeThread = this.freshThread());
    const found = mentionQuery(this.draft, this.caret);
    if (found) {
      const end = found.from + 1 + found.query.length;
      this.draft = this.draft.slice(0, found.from) + this.draft.slice(end).replace(/^ /, "");
      this.caret = found.from;
    }
    thread.pins = thread.pins || [];
    this.error = "";
    if (thread.pins.some((pin) => pin.path === item.path)) { this.composerFocused = true; this.render(); return; }
    if (thread.pins.length >= MAX_PINS) {
      this.error = `A chat holds at most ${MAX_PINS} pinned notes and folders.`;
    } else {
      thread.pins = [...thread.pins, { kind: item.kind, path: item.path }];
    }
    this.composerFocused = true;
    this.render();
    return this.savePins(thread);
  }

  unpin(path) {
    const thread = this.activeThread;
    if (!thread) return;
    thread.pins = (thread.pins || []).filter((pin) => pin.path !== path);
    this.render();
    return this.savePins(thread);
  }

  // Only a chat with turns is on disk; an empty one carries its pins until it has an answer. The
  // chat answering now is saved with its answer (or when it ends without one: `dirty`).
  async savePins(thread) {
    if (this.sending && thread === this.streamingThread) { thread.dirty = true; return; }
    if (thread !== this.activeThread || !this.turns.length) return;
    try { await this.saveRecord(thread, this.turns); } catch (error) { this.error = error.message || String(error); }
    this.render();
  }

  // ✕ on the open-note chip: this chat leaves the open note out, until clicked again. Not saved,
  // and an answer already streaming keeps the note it was asked on.
  toggleOpenNote() {
    if (!this.activeThread) this.activeThread = this.freshThread();
    this.activeThread.openNoteOff = !this.activeThread.openNoteOff;
    this.render();
  }

  pickSkill(skill) {
    this.draft = `/${skill.name} `;
    this.composerFocused = true;
    this.render();
  }

  // The toolbar's @: an `@` at the caret (after a space, where it would touch a word), which
  // opens the same picker as typing it.
  openMention() {
    const draft = String(this.draft || "");
    const at = Number.isInteger(this.caret) && this.caret <= draft.length ? this.caret : draft.length;
    const before = draft.slice(0, at);
    const lead = before && !/\s$/.test(before) ? " " : "";
    this.draft = `${before}${lead}@${draft.slice(at)}`;
    this.caret = this.caretAt = at + lead.length + 1;
    this.mentionDismissed = false;
    this.mentionIndex = 0;
    this.composerFocused = true;
    this.render();
  }

  // An empty chat's starter: its words in the composer, the caret after them, nothing sent.
  useStarter(text) {
    this.draft = text;
    this.caret = this.caretAt = text.length;
    this.composerFocused = true;
    this.render();
  }

  getViewType() { return VIEW_TYPE_COPILOT; }
  getDisplayText() { return "SLIM chat"; }
  getIcon() { return "message-square-text"; }

  async onOpen() {
    // On the root, which outlives every render(): answers are rebuilt, the listener is not.
    const root = this.contentEl;
    if (typeof this.registerDomEvent === "function") {
      this.registerDomEvent(root, "click", (event) => this.onRootClick(event));
    } else if (root && typeof root.addEventListener === "function") {
      root.addEventListener("click", (event) => this.onRootClick(event));
    }
    const file = this.app.workspace.getActiveFile ? this.app.workspace.getActiveFile() : null;
    this.loadSkills();
    await Promise.all([this.loadThreads(), this.setActiveFile(file)]);
    // Reopen the most recent chat, if there is one and nothing has been started meanwhile.
    const latest = threadRows(this.threads)[0];
    if (latest && !this.activeThread && !this.sending) await this.openThread(latest.id);
  }

  async onClose() { this.cancel(); }

  // A click anywhere else closes the composer's popovers (the effort menu, nearby notes).
  onRootClick(event) {
    const target = event?.target;
    const inside = (selector) => Boolean(target?.closest?.(selector));
    if (this.effortOpen && !inside(".slim-effort, .slim-effort-menu")) this.showEffortMenu(false);
    if (this.nearbyOpen && !inside(".slim-copilot-nearby, .slim-copilot-nearby-count")) this.showNearby(false);
    this.onLinkClick(event);
  }

  // MarkdownRenderer turns an answer's [[wikilink]] into <a class="internal-link">, but Obsidian
  // wires clicks on those only inside its own note views; in this sidebar they were dead.
  // ⚠ Resolve BEFORE opening: openLinkText creates a missing target (see openFile), and the
  // model can name a note that does not exist.
  onLinkClick(event) {
    const link = event?.target?.closest?.("a.internal-link");
    if (!link) return;
    event.preventDefault();
    const target = link.getAttribute("data-href") || link.getAttribute("href") || "";
    const source = this.openNote?.path || "";
    const linkpath = target.split("#")[0].split("|")[0].trim();
    const file = this.app.metadataCache?.getFirstLinkpathDest?.(linkpath, source);
    if (!file) {
      this.error = `Note not found: ${linkpath}`;
      this.render();
      return;
    }
    this.app.workspace.openLinkText(target, source, Boolean(event.metaKey || event.ctrlKey));
  }

  // The note open now. It only moves the chip and loads nearby notes: the chat, its answer
  // streaming and its error belong to the sidebar, not to a note.
  async setActiveFile(file) {
    const path = file && String(file.extension || "").toLowerCase() === "md" ? file.path : null;
    // Same note, loaded or loading: nothing to do. Focus moving between the editor and this
    // sidebar fires active-leaf-change with the SAME file; with onOpen, the event and
    // activateCopilot all racing, re-syncing ran ingest_note twice into an IntegrityError.
    // Edits since the open are picked up by send() (refreshNote), not by focus changes.
    if (path && (this.openNote?.path === path || this.loadingPath === path)) return;
    const version = ++this.noteVersion;
    this.context = null;
    if (!path) {
      this.openNote = null;
      this.loadingPath = null;
      this.renderNoteContext();
      return;
    }
    this.openNote = { path, id: null };
    this.loadingPath = path;
    this.renderNoteContext();
    try {
      if (typeof this.plugin.saveActiveNote === "function") {
        await this.plugin.saveActiveNote(file);
      }
      if (version !== this.noteVersion) return;
      const context = await this.plugin.postJSON("/api/copilot/context", { path });
      if (version !== this.noteVersion) return;
      this.context = context;
      this.openNote = { path, id: context.source?.id ?? null };
    } catch (error) {
      // Nearby notes are a convenience. A 400 is a note the server does not index (Skills/,
      // Profile/, _Reflections/): it still opens, and the question is sent with its path.
      if (version === this.noteVersion) console.warn("[slim] nearby notes", path, error.message || error);
    } finally {
      if (version === this.noteVersion) {
        this.loadingPath = null;
        this.renderNoteContext();
      }
    }
  }

  // The chat history: every thread, newest first, or those that match the search box.
  async loadThreads() {
    const version = ++this.searchVersion;
    const q = this.threadQuery.trim();
    try {
      const listing = await this.plugin.getJSON(
        `/api/copilot/threads${q ? `?q=${encodeURIComponent(q)}` : ""}`);
      if (version !== this.searchVersion) return;
      this.threads = listing.threads || [];
    } catch (error) {
      console.error("[slim] threads", error);
      return;
    }
    this.renderThreadRows();
  }

  searchThreads(value) {
    this.threadQuery = String(value || "");
    clearTimeout(this.threadSearch);
    this.threadSearch = setTimeout(() => this.loadThreads(), 200);
    this.threadSearch.unref?.();
  }

  // The note as it is NOW: save the editor, re-sync the row (ingest_note is hash-keyed, so an
  // unchanged note costs one stat), skip the nearby-notes search. Called by send(), so every
  // question is answered from what they have typed, not from the version indexed at open.
  async refreshNote() {
    if (!this.openNote) return;
    const path = this.openNote.path;
    const file = this.app.workspace.getActiveFile ? this.app.workspace.getActiveFile() : null;
    if (file && file.path === path && typeof this.plugin.saveActiveNote === "function") {
      await this.plugin.saveActiveNote(file);
    }
    await this.saveOpenCopies(path);
    let fresh;
    try {
      fresh = await this.plugin.postJSON("/api/copilot/context", { path, related: false });
    } catch (error) {
      // 400: the server does not index this note; it reads it from disk. Anything else surfaces,
      // in the server's own sentence where it gave one.
      if (error.status === 400) return;
      if (error.detail) throw new Error(error.detail);
      throw error;
    }
    if (this.openNote?.path === path) this.openNote = { path, id: fresh.source?.id ?? this.openNote.id };
    if (this.context?.source?.path === path) this.context = { ...this.context, source: fresh.source };
  }

  // A new chat's record, on no disk yet: it is saved with its first answer.
  freshThread() {
    return {
      id: threadId(), title: "New chat", pins: [],
      ...(this.openNote ? { source_id: this.openNote.id, source_path: this.openNote.path } : {}),
    };
  }

  newThread() {
    this.enterThread(this.freshThread(), []);
  }

  // The chat on screen from now on. What belonged to the one before goes with it: its error, an
  // armed delete or restore, images pasted for it. The chat a question failed in shows its error
  // and Retry again when it is opened again, and only there.
  enterThread(thread, turns) {
    this.activeThread = thread;
    this.turns = turns;
    this.error = this.retryFor === thread.id ? this.retryError : "";
    this.deleteArmed = null;
    this.restoreArmed = null;
    this.restoreRefused = null;
    this.beforeTexts.clear();             // fetched again when asked for: they are whole notes
    this.pendingImages = [];
    this.showThreads = false;
    this.stuck = true;                    // a chat opens at its newest turn
    this.render();
  }

  // The effort every question runs at from now on (a skill may name its own), kept in the
  // plugin's settings: the selector shows the last one chosen, in every chat and after a restart.
  setEffort(value) {
    this.effort = effortOf(value);
    if (!this.plugin.settings) return;
    this.plugin.settings.effort = this.effort;
    Promise.resolve(this.plugin.saveSettings?.()).catch((error) => { this.error = error.message || String(error); });
  }

  // Any chat opens as it is: it never moves the note the owner has open.
  // The chat answering now is opened as it is in memory: the copy on disk has neither its
  // question nor its answer, and a question asked in that copy would save over both.
  async openThread(id) {
    const version = ++this.threadVersion;
    const streaming = () => this.sending && this.streamingThread?.id === id;
    if (streaming()) {
      this.enterThread(this.streamingThread, this.streamingTurns);
      return;
    }
    const data = await this.plugin.getJSON(`/api/copilot/thread?id=${encodeURIComponent(id)}`);
    if (version !== this.threadVersion) return;
    if (streaming()) {
      this.enterThread(this.streamingThread, this.streamingTurns);
      return;
    }
    const turns = data.turns || [];
    this.keepRuns(data, turns);
    this.enterThread(data, turns);
  }

  // The runs kept for ↺ and Cmd-Z (runTurns) follow their chat's newest copy: one saved from an
  // older copy would replace the chat on disk and lose every turn after it.
  keepRuns(thread, turns) {
    for (const [ref, kept] of this.runTurns) {
      if (kept.thread.id === thread.id && kept.turns !== turns) this.runTurns.set(ref, { thread, turns });
    }
  }

  // `live` is the chat's own array when `turns` is a part of it (saveRun leaves out a question
  // still waiting for its answer).
  async saveRecord(thread, turns, live = turns) {
    delete thread.dirty;                  // this save carries its title and pins as they are now
    const body = {
      id: thread.id, title: thread.title,
      // A question's images by the refs the server stored them under. One pasted for a question
      // that failed before a run began was never stored: it has no id, and is not saved.
      turns: turns.map(({ attachments, ...turn }) => {
        const refs = (attachments || []).filter((image) => image.id)
          .map(({ id, name, mime, bytes }) => ({ id, name, mime, bytes }));
        return refs.length ? { ...turn, attachments: refs } : turn;
      }),
      ...(thread.source_id ? { source_id: thread.source_id } : {}),
      ...(thread.source_path ? { source_path: thread.source_path } : {}),
      ...(thread.pins ? { pins: thread.pins } : {}),
    };
    const saved = await this.plugin.postJSON("/api/copilot/thread/save", body);
    Object.assign(thread, saved);
    this.keepRuns(thread, live);
    // The history list's row: what it shows, not the conversation.
    const { turns: _turns, ...row } = thread;
    const index = this.threads.findIndex((item) => item.id === thread.id);
    if (index >= 0) this.threads[index] = row;
    else this.threads.unshift(row);
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
    if (!question || this.sending) return;
    this.clearBefore();
    if (!this.activeThread) this.activeThread = this.freshThread();
    // A failed question waits under its error for Try again. Asking something else instead
    // takes it off the thread: it would be saved with the next answer and sent as history, and
    // a refused /command is sent as its whole skill prompt. One failed in another chat waits there.
    if (this.retryFor === this.activeThread.id) {
      const failed = this.turns.at(-1);
      if (this.retryQuestion && failed?.role === "you" && failed.text === this.retryQuestion) this.turns.pop();
      this.clearRetry();
    }
    // Snapshot the thread that ASKED. openThread/newThread/deleteThread stay clickable while
    // an answer streams, and the answer belongs to this thread whatever is active when it
    // lands — otherwise B was saved with A's answer and A's question was never saved.
    const thread = this.activeThread;
    const turns = this.turns;
    // The note this question is asked on; null when the chat leaves it out (✕ on its chip).
    const path = thread.openNoteOff ? null : this.openNote?.path ?? null;
    if (!turns.length) thread.title = titleFromQuestion(question);
    const asked = { role: "you", text: question, ...(path ? { note: path } : {}),
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
    this.liveTools = [];
    this.streamingThread = thread;
    this.streamingTurns = turns;
    this.liveFresh = true;
    this.stuck = true;                    // sending always brings the conversation to its bottom
    let completed = null;
    let run = null;                       // an editing run, from its `edit` event on
    let local = null, tracked = null;     // the editor tracking it from Send, and its name there
    const started = Date.now();
    const effort = this.effort;           // what this turn runs at, whatever the selector says later
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
      if (path) await this.refreshNote();
      if (abort.signal.aborted) throw cancelledError();
      // The editor's text, not the file: Obsidian saves a moment after typing, and an editing
      // run works on exactly what the owner sees. The editor tracks the owner's typing from
      // here, under a name of the chat's own until the server names the run.
      const shown = path ? this.plugin.liveViewFor?.(path) || null : null;
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
      const picked = question.startsWith("/") && path ? editorSelection(this.app, path) : { text: "" };
      await this.plugin.streamCopilot({
        thread_id: thread.id, path, pins: thread.pins ?? [], question,
        // Role, text, small image refs and each answer's transcript, every turn whole: the server
        // fits the conversation to the model's window (copilot.turn.fit_window). Evidence and
        // stats stay out. Image bytes stay server-side. A question with no answer after it (a chat
        // saved before a failed one came off) is not history.
        history: turns.slice(0, -1).filter((turn, i, all) => turn.role !== "you" || all[i + 1]?.role === "slim")
          .map((turn) => ({ ...turn, text: historyText(turn), transcript: turn.turn?.transcript }))
          // An answer's transcript is the turn as the model had it (the question as sent, its
          // thinking, tool calls and results), replayed as Pi replays a session; the server falls
          // back to the text for an answer saved without one.
          .map(({ role, text, attachments, transcript }) => ({ role, text,
            ...(role === "slim" && Array.isArray(transcript) && transcript.length ? { transcript } : {}),
            ...(attachments?.length ? { attachments: attachments.map(
              ({ id, name, mime, bytes }) => ({ id, name, mime, bytes })) } : {}) })),
        // A regenerated question re-sends its stored refs; the server has the bytes.
        images: outgoingImages.map(({ id, name, mime, data, bytes }) =>
          data ? { name, mime, data } : { id, name, mime, bytes }),
        effort,
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
        // A new model call in the turn: what streamed before it (a lookup's preamble, a step's
        // chatter, a raw handle) was that call's, never the answer.
        delta_reset: () => {
          streamed = "";
          this.streamingText = "";
          this.renderDelta();
        },
        // A lookup once it ran: a line above the answer, kept with the turn (`turn.tools`).
        tool: on((data) => {
          this.liveTools.push({ name: String(data.name || ""), label: String(data.label || ""),
                                refused: Boolean(data.refused) });
          this.renderTools();
        }),
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
                  view: claimed ? tracked : null, path, started, effort, thread, turns };
          this.editRun = run;
          if (path) Promise.resolve(this.plugin.showEditing?.(path)).catch((e) => console.warn("[slim] editing view", e));
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
      // The answer takes the streaming one's place: it is not new on screen, so it does not fade in.
      const answer = { role: "slim", text: completed.answer || streamed, turn: completed };
      this.seenTurns.add(answer);
      turns.push(answer);
      await this.saveRecord(thread, turns);
    } catch (error) {
      const stopped = error.name === "AbortError";
      const message = error.message || String(error);
      // An error is shown in the chat it happened in, not in whichever chat is on screen now.
      const here = () => this.activeThread === thread;
      if (run && turns.at(-1) === asked) {
        // What landed is in the note: it stays, and the turn keeps how far the run got and the
        // copy from before, so See before and Restore work after a reload too.
        const partial = await this.partialTurn(run, stopped ? "stopped" : `error: ${message}`);
        this.seenTurns.add(partial);
        turns.push(partial);
        if (!stopped && here()) this.error = message;
        if (!this.deletedThreads.has(thread.id)) {
          await this.saveRecord(thread, turns).catch((e) => { if (here()) this.error = e.message || String(e); });
        }
      } else if (stopped) {
        // Stopped before an answer: the question is theirs — hand it back rather than leave a
        // dangling "you" turn to ride along as history and be persisted with the next answer.
        // A deleted chat's question is not put over whatever they are typing now.
        if (turns.at(-1) === asked) {
          turns.pop();
          if (!this.deletedThreads.has(thread.id)) { this.draft = question; this.pendingImages = outgoingImages; }
        }
      } else if (turns.at(-1) === asked) {
        // The model never answered: the error under the question, and a retry, in this chat.
        this.retryFor = thread.id;
        this.retryError = message;
        this.retryQuestion = question;
        this.retryImages = outgoingImages;
        if (here()) this.error = message;
      } else if (here()) {
        // Answered but not saved (e.g. the thread is over its cap): the answer stays visible
        // with the reason. No retry — it would append a second you/slim pair every click.
        this.error = message;
      }
    } finally {
      if (local) this.plugin.live?.dropRun(tracked, local);
      if (run) this.finishRun(run);
      // Renamed or re-pinned while it ran, and not saved since (no answer): saved now, without
      // the question that has none.
      if (thread.dirty && !this.deletedThreads.has(thread.id)) {
        const settled = turns.at(-1) === asked ? turns.slice(0, -1) : turns;
        if (settled.length) {
          this.savingRuns = (this.savingRuns || Promise.resolve())
            .then(() => this.saveRecord(thread, settled, turns))
            .catch((e) => console.error("[slim] save chat", e));
        }
      }
      if (this.abort === abort) {
        clearInterval(this.stageClock);
        this.stageClock = null;
        this.abort = null;
        this.sending = false;
        this.stopLiveRender();
        this.streamingText = "";
        this.liveTools = [];
        this.streamingThread = null;
        this.streamingTurns = null;
        this.stage = "";
      }
      this.render();
    }
  }

  clearRetry() {
    this.retryFor = null;
    this.retryError = "";
    this.retryQuestion = "";
    this.retryImages = [];
  }

  // A call started: its seconds count from now, ticking on the second.
  stageStarted() {
    this.stageSince = Date.now();
    clearInterval(this.stageClock);
    this.stageClock = setInterval(() => { if (this.el?.stage) this.renderStage(); }, 1000);
    this.stageClock.unref?.();
  }

  // The server names a stage as each model call starts. The same one again is that call still
  // running (its heartbeat while it thinks); a new one, or a new step, starts the count again.
  setStage(name) {
    const stage = name || "Working";
    if (stage !== this.stage) {
      this.stage = stage;
      this.stageStarted();
    }
    this.renderStage();
  }

  clearBefore() {
    for (const view of this.plugin.liveViews || []) {
      const ref = this.plugin.live?.showing?.(view);
      if (ref) this.plugin.live.showBefore(view, ref, "", false);
    }
  }

  // Stop: the stream is aborted, and the turn ends in send()'s own `finally`, which alone puts
  // the composer back to Send. Ending it here let a second click send while the first turn was
  // still unwinding, and that turn's cleanup then wiped the new one's state.
  cancel() {
    this.clearBefore();
    if (this.abort) this.abort.abort();
    this.stopLiveRender();
    this.streamingText = "";
    this.stage = this.sending ? "Stopping" : "";
    if (this.sending) this.renderStage();
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
    if (!question || this.sending || this.retryFor !== this.activeThread?.id) return;
    const last = this.turns.at(-1);
    if (last?.role === "you" && last.text === question) this.turns.pop();
    this.pendingImages = this.retryImages;
    this.clearRetry();
    await this.send(question);
  }

  async deleteThread(id) {
    if (this.deleteArmed !== id) {
      this.deleteArmed = id;
      this.render();
      return;
    }
    this.deletedThreads.add(id);
    if (this.retryFor === id) this.clearRetry();
    // A late answer would save into — and so resurrect — the thread being deleted.
    if (this.sending && this.streamingThread?.id === id) this.cancel();
    await this.plugin.postJSON("/api/copilot/thread/delete", { id });
    this.threads = this.threads.filter((item) => item.id !== id);
    if (this.activeThread?.id === id) this.newThread();
    this.deleteArmed = null;
    this.render();
  }

  // The chat answering now takes the title in memory, and its answer's save carries it (or, if
  // there is no answer, the save when it ends): a copy saved now would have the question with no
  // answer, and the answer's save would put the old title back.
  async renameThread(id) {
    const fetch = () => this.plugin.getJSON(`/api/copilot/thread?id=${encodeURIComponent(id)}`);
    const known = [this.streamingThread, this.activeThread, ...this.threads].find((item) => item?.id === id)
      || await fetch();
    const title = await this.promptTitle(known.title);
    if (!title) return;
    const name = title.slice(0, 120);
    const row = this.threads.find((item) => item.id === id);
    if (row) row.title = name;
    if (this.sending && this.streamingThread?.id === id) {
      this.streamingThread.title = name;
      this.streamingThread.dirty = true;
    } else {
      let thread = this.activeThread?.id === id ? this.activeThread : null;
      let turns = thread ? this.turns : null;
      if (!thread) {                      // read after the title is typed: a save meanwhile is in it
        thread = await fetch();
        turns = thread.turns || [];
      }
      thread.title = name;
      await this.saveRecord(thread, turns);
    }
    if (this.activeThread?.id === id) this.activeThread.title = name;
    this.render();
  }

  // Guarded like the recorder's "Open note" button: openLinkText CREATES a missing target,
  // and citation / thread paths are only as fresh as the last ingest — a note dragged since
  // then would become an empty phantom, which the next active-leaf-change indexes and embeds.
  promptTitle(current) {
    return new Promise((resolve) => new RenameModal(this.app, current, resolve).open());
  }

  async openFile(path) {
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
