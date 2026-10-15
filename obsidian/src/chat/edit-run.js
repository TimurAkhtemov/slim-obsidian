// CopilotView: an editing run's changes, landed in the note, and what the chat shows of them.
"use strict";

const { Notice } = require("obsidian");
const { sha256Hex, metaText, runCounts, editOutcome } = require("./text");
const { clear, create, atBottom } = require("./dom");

module.exports = {
  // ---- an editing run, as it lands ------------------------------------------------------------

  // A change to the note goes in through its editor; a new note is created at once. Either way
  // the chat keeps its id, label and what became of it, never its text.
  landChange(run, change) {
    const entry = { id: String(change.id), step: change.step, op: change.op, label: change.label || "",
                    needs: change.needs || [], ...(change.part_of != null ? { part_of: String(change.part_of) } : {}) };
    run.changes.push(entry);
    if (change.op === "create") {
      entry.path = change.path;
      entry.state = "creating";
      run.pending.push(this.createNote(change).then((made) => { Object.assign(entry, made); this.renderEdit(); }));
    } else if (this.inView(run)) {
      entry.state = "skipped";            // if the editor throws, the change did not land
      if (change.draft != null) this.plugin.live.endDraft?.(run.view, String(change.draft));   // it takes the draft's place
      entry.state = this.plugin.live.applyChange(run.view, run.ref, change);
    } else {
      // The note's editor is gone or shows another note: only the note in view is edited.
      entry.state = "skipped";
      this.cancel();
    }
    this.renderEdit();
  },

  // The run's editor is still open and still shows its note.
  inView(run) {
    const live = this.plugin.live;
    if (!run.view || !live || (this.plugin.liveViews && !this.plugin.liveViews.has(run.view))) return false;
    const shows = live.pathOf(run.view);
    return shows === null || shows === run.path;
  },

  // {state: "created"}, or {state: "failed", reason} in words the owner can act on.
  async createNote(change) {
    const vault = this.app.vault;
    try {
      // The adapter's check is the filesystem's, case-insensitive on macOS: `summary.md` must
      // not be written over an existing `Summary.md` that the path index spells differently.
      if (vault.getAbstractFileByPath(change.path) || await vault.adapter?.exists?.(change.path)) {
        return { state: "failed", reason: "a note is already there" };
      }
      await vault.create(change.path, change.text);
    } catch (error) {
      console.error("[slim] create note", change.path, error);
      return { state: "failed", reason: String(error?.message || error).split("\n")[0].slice(0, 200) };
    }
    // A note is not findable until it is indexed.
    this.plugin.postJSON("/api/copilot/context", { path: change.path, related: false })
      .catch((e) => console.warn("[slim] index new note", change.path, e.message));
    return { state: "created" };
  },

  async trashCreated(turn, entry) {
    const file = this.app.vault.getAbstractFileByPath(entry.path);
    try {
      if (file) await this.app.vault.trash(file, false);
      entry.state = "trashed";
    } catch (error) {
      this.error = error.message || String(error);
    }
    const found = this.findRunTurn(turn.turn.edit_run.before_ref);
    if (found) this.saveRun(found.thread, found.turns);
    this.render();
  },

  // What the turn keeps of a run: the plan, how each step and change went, and the hashes and
  // copy that See before and Restore need. `hash_after` is the editor's own: a change skipped
  // because the owner typed there never reached the server's draft.
  async settleRun(run, server) {
    await Promise.all(run.pending);
    const live = run.view && this.plugin.live?.state(run.view, run.ref);
    const states = new Map((live?.changes || []).map((change) => [change.id, change.state]));
    const doc = run.view?.state?.doc?.toString();
    return {
      plan: run.plan || server.plan || { intro: "", steps: [] },
      steps_done: server.steps_done ?? run.steps.filter((state) => state && state !== "doing").length,
      step_states: run.steps.map((state) => state || null),
      changes: run.changes.map(({ id, step, op, label, needs, part_of, path, state, reason }) => ({
        id, step, op, label, needs, ...(part_of ? { part_of } : {}), ...(path ? { path } : {}),
        state: states.get(id) || state, ...(reason ? { reason } : {}) })),
      stopped: server.stopped ?? null,
      outline_met: server.outline_met ?? null,
      summary: server.summary || "",
      before_ref: run.ref,
      hash_before: run.hash_before,
      hash_after: doc === undefined ? server.hash_after || null : sha256Hex(doc),
      typed: Boolean(live?.touched),
    };
  },

  async partialTurn(run, stopped) {
    const edit_run = await this.settleRun(run, { stopped });
    const text = run.plan?.intro || "";
    return { role: "slim", text, turn: { answer: text, mode: this.mode, citations: [],
                                         timing: { total_ms: Date.now() - run.started }, edit_run } };
  },

  // The run is over: ↺ appears in the note, and the note is saved and re-indexed once.
  finishRun(run) {
    clearInterval(this.editClock);
    this.editClock = null;
    if (this.editRun === run) this.editRun = null;
    if (run.view) this.plugin.live?.endRun(run.view, run.ref);
    this.runTurns.set(run.ref, { thread: run.thread, turns: run.turns });
    if (run.changes.some((change) => change.state === "applied")) {
      this.saveOpenCopies(run.path)
        .then(() => this.plugin.postJSON("/api/copilot/context", { path: run.path, related: false }))
        .catch((e) => console.warn("[slim] index after edit", run.path, e.message));
    }
  },

  startEditClock() {
    clearInterval(this.editClock);
    this.editClock = setInterval(() => { if (this.el?.liveEdit) this.renderEdit(); }, 1000);
    this.editClock.unref?.();
  },

  // ---- an edit turn in the chat: the intro, a line per step, what came of it ---------------

  // The run streaming now, redrawn in place on each event: the transcript is not rebuilt.
  renderEdit() {
    if (!this.el?.liveEdit) { this.renderSoon(); return; }
    const follow = atBottom(this.el.transcript);
    clear(this.el.liveEdit);
    this.drawLiveEdit(this.el.liveEdit);
    if (follow) this.el.transcript.scrollTop = this.el.transcript.scrollHeight;
  },

  drawLiveEdit(parent) {
    const run = this.editRun;
    if (!run) return;
    this.renderEditRun(parent, { plan: run.plan, step_states: run.steps, changes: run.changes }, null);
    const meta = create(parent, "div", "slim-edit-meta");
    create(meta, "span", "", `${this.mode === "deep" ? "Deep" : "Quick"} · ${Math.max(0, Math.round((Date.now() - run.started) / 1000))} s`);
  },

  renderEditRun(message, run, pending) {
    const box = create(message, "div", "slim-edit");
    const intro = run.plan?.intro || "";
    if (intro) {
      const el = create(box, "div", "slim-edit-intro slim-rendered-markdown");
      if (pending) pending.push(this.renderMarkdown(el, intro).catch((e) => console.error("[slim] render intro", e)));
      else el.textContent = intro;
    }
    const steps = run.plan?.steps || [];
    const states = run.step_states || [];
    const shown = steps.map((text, i) => [text, states[i] || (i < (run.steps_done || 0) ? "done" : null)])
      .filter(([, state]) => state);
    if (shown.length) {
      const list = create(box, "ul", "slim-steps");
      for (const [text, state] of shown) {
        const item = create(list, "li", `slim-step is-${state}`);
        create(item, "span", "slim-step-icon", { doing: "", done: "✓", partial: "◐", skipped: "–" }[state] ?? "");
        create(item, "span", state === "doing" ? "slim-step-text slim-shimmer" : "slim-step-text", text);
      }
    }
    if (pending && "stopped" in run) create(box, "p", "slim-edit-outcome", editOutcome(run));
    for (const change of run.changes || []) {
      // A change made inside another was SLIM's own rework: it is never named on its own.
      if (change.state === "skipped" && !change.part_of) {
        create(box, "small", "slim-edit-note", `Skipped “${change.label || change.id}”: you edited this here.`);
      }
      if (change.op !== "create") continue;
      const row = create(box, "div", "slim-edit-note slim-edit-created");
      const what = { creating: "Creating", created: "Created", trashed: "Moved to trash:",
                     failed: `Not created (${change.reason || "it failed"}):` }[change.state] || "";
      create(row, "span", "", `${what} `);
      const name = create(row, "button", "slim-link", change.path);
      name.onclick = () => this.openNote(change.path);
      if (change.state === "created" && pending) {
        create(row, "span", "", " · ");
        const undo = create(row, "button", "slim-link", "Undo");
        const turn = this.turns.find((t) => t.turn?.edit_run === run);
        undo.onclick = () => turn && this.trashCreated(turn, change);
      }
    }
  },

  // N changes · Undo all · Redo · See before · Hide marks · Quick · T s, while the note's editor
  // holds the run; once it has closed, or after a reload: N changes · See before · Restore.
  renderEditMeta(message, run, turn) {
    const meta = create(message, "div", "slim-edit-meta");
    const ref = run.before_ref;
    const parts = [];
    const text = (value) => parts.push(["span", value]);
    const link = (label, act, disabled = false) => parts.push(["button", label, act, disabled]);
    const { landed, undone } = runCounts(run);
    const inNote = landed.some((change) => change.op !== "create");
    const view = ref && this.plugin.liveRunView?.(ref);
    const live = view && this.plugin.live?.state(view, ref);
    if (this.restoreRefused?.ref === ref) {
      create(meta, "div", "slim-edit-warning", this.restoreRefused.message).setAttribute("role", "alert");
    }
    if (this.restoreArmed === ref) {
      const warning = create(meta, "div", "slim-edit-warning",
        "You've edited this note since this change; restoring would discard those edits.");
      warning.setAttribute("role", "alert");
      link("Restore anyway", () => this.restore(turn, true));
      link("Cancel", () => { this.restoreArmed = null; this.render(); });
    } else if (run.restored) {
      text("Restored the version from before this edit");
      if (this.restoredFrom.has(ref)) link("Undo", () => this.unrestore(turn));
    } else {
      if (landed.length) text(`${landed.length} change${landed.length === 1 ? "" : "s"}${undone ? `, ${undone} undone` : ""}`);
      const shownIn = view || this.plugin.liveViewFor?.(this.context?.source?.path);
      const showing = Boolean(ref && shownIn && this.plugin.live?.showing(shownIn) === ref);
      if (live) {
        link("Undo all", () => this.undoAllRun(ref), !live.changes.some((change) => change.state === "applied"));
        if (live.redoable) link("Redo", () => this.redoRun(ref));
        link(showing ? "Back to now" : "See before", () => this.toggleBefore(turn, !showing));
        link(live.marks ? "Hide marks" : "Show marks", () => this.plugin.live.setMarks(view, ref, !live.marks));
      } else if (inNote && ref) {
        link(showing ? "Back to now" : "See before", () => this.toggleBefore(turn, !showing));
        link("Restore", () => this.restore(turn));
      }
    }
    if (turn?.turn) text(metaText(turn.turn));
    parts.forEach(([tag, label, act, disabled], i) => {
      if (i) create(meta, "span", "slim-edit-sep", "·");
      const el = create(meta, tag, tag === "button" ? "slim-link" : "", label);
      if (tag !== "button") return;
      el.disabled = Boolean(disabled);
      el.onclick = async () => {
        try { await act(); } catch (error) { this.error = error.message || String(error); }
        this.render();
      };
    });
  },

  // The note's editor moved a run's changes (↺, Cmd-Z, Undo all): the turn follows and is saved,
  // with the hash Restore compares against.
  runChanged(view, runId) {
    const live = this.plugin.live?.state(view, runId);
    if (this.editRun?.ref === runId) {
      for (const change of this.editRun.changes) {
        const now = live?.changes.find((item) => item.id === change.id);
        if (now && change.op !== "create") change.state = now.state;
      }
      this.renderEdit();
      return;
    }
    const found = this.findRunTurn(runId);
    if (found && live) {
      const run = found.turn.turn.edit_run;
      for (const change of run.changes || []) {
        const now = live.changes.find((item) => item.id === change.id);
        if (now && change.op !== "create") change.state = now.state;
      }
      run.hash_after = sha256Hex(view.state.doc.toString());
      if (live.touched) run.typed = true;
      this.saveRun(found.thread, found.turns);
    }
    this.renderSoon();
  },

  // The turn a run belongs to, in the chat on screen or in the one it was asked in.
  findRunTurn(ref) {
    const at = (turns) => turns?.find((turn) => turn.role === "slim" && turn.turn?.edit_run?.before_ref === ref);
    if (at(this.turns)) return { thread: this.activeThread, turns: this.turns, turn: at(this.turns) };
    const kept = this.runTurns.get(ref);
    const turn = at(kept?.turns);
    if (!turn) return null;
    // The same chat reopened is a fresh copy of it: save that one.
    if (this.activeThread?.id === kept.thread.id) return null;
    return { ...kept, turn };
  },

  // A question still waiting for its answer is not saved with it: the answer's save brings it.
  saveRun(thread, turns) {
    if (!thread || this.deletedThreads.has(thread.id)) return this.savingRuns;
    const settled = this.sending && turns === this.streamingTurns && turns.at(-1)?.role === "you"
      ? turns.slice(0, -1) : turns;
    this.savingRuns = (this.savingRuns || Promise.resolve())
      .then(() => this.saveRecord(thread, settled))
      .catch((error) => { this.error = error.message || String(error); this.render(); });
    return this.savingRuns;
  },

  async beforeText(ref) {
    if (!this.beforeTexts.has(ref)) {
      const data = await this.plugin.getJSON(`/api/copilot/before?thread=${encodeURIComponent(this.activeThread.id)}`
        + `&ref=${encodeURIComponent(ref)}`);
      this.beforeTexts.set(ref, String(data.text ?? ""));
    }
    return this.beforeTexts.get(ref);
  },

  async undoAllRun(ref) {
    const view = this.plugin.liveRunView?.(ref);
    if (!view) return;
    const { kept } = this.plugin.live.undoAll(view, ref);
    if (kept) new Notice(`SLIM: ${kept} change${kept === 1 ? "" : "s"} you've edited since stay as they are.`);
  },

  redoRun(ref) {
    const view = this.plugin.liveRunView?.(ref);
    if (view && this.plugin.live.redo(view, ref) === "refused") {
      new Notice("SLIM: you've edited that part of the note since, so it stays as it is.");
    }
  },

  // The note as it was before the run, over the note, read-only; the file is not touched.
  async toggleBefore(turn, on) {
    const run = turn.turn.edit_run;
    const view = this.plugin.liveRunView?.(run.before_ref) || await this.plugin.noteView?.(this.context.source.path);
    if (!view) throw new Error("Open the note in an editor to see it as it was.");
    this.plugin.live.showBefore(view, run.before_ref, on ? await this.beforeText(run.before_ref) : "", on);
  },

  // The note as it was before the run, as one editor change (Cmd-Z takes it back). If the note
  // has been edited since, it says so first and waits for a second click. The server says what
  // may be put back: never a recording's frontmatter, block or transcript as they were, which a
  // Resume may have added to since; then it refuses, and the chat says why.
  async restore(turn, force = false) {
    const run = turn.turn.edit_run;
    const ref = run.before_ref;
    const view = await this.plugin.noteView?.(this.context.source.path);
    if (!view || !this.plugin.live) throw new Error("Open the note in an editor to restore it.");
    const doc = view.state.doc.toString();
    this.restoreRefused = null;
    if (!force && (run.typed || sha256Hex(doc) !== run.hash_after)) {
      this.restoreArmed = ref;
      this.render();
      return;
    }
    this.restoreArmed = null;
    const answer = await this.plugin.postJSON("/api/copilot/restore", { thread_id: this.activeThread.id, ref, doc });
    if (answer?.refused || typeof answer?.text !== "string") {
      this.restoreRefused = { ref, message: answer?.refused || "SLIM could not restore this note." };
      this.render();
      return;
    }
    this.restoredFrom.set(ref, doc);
    this.plugin.live.restore(view, answer.text);
    Object.assign(run, { restored: true, typed: false, hash_after: sha256Hex(view.state.doc.toString()) });
    this.saveRun(this.activeThread, this.turns);
    this.reindex(this.context.source.path);
    this.render();
  },

  // Undo of a Restore, while the note is still exactly the restored copy.
  async unrestore(turn) {
    const run = turn.turn.edit_run;
    const was = this.restoredFrom.get(run.before_ref);
    const view = this.plugin.liveViewFor?.(this.context.source.path);
    if (was === undefined || !view) return;
    if (sha256Hex(view.state.doc.toString()) !== run.hash_after) {
      throw new Error("The note has changed since it was restored; use Cmd-Z in the note.");
    }
    this.plugin.live.restore(view, was);
    this.restoredFrom.delete(run.before_ref);
    Object.assign(run, { restored: false, hash_after: sha256Hex(view.state.doc.toString()) });
    this.saveRun(this.activeThread, this.turns);
    this.reindex(this.context.source.path);
  },
};
