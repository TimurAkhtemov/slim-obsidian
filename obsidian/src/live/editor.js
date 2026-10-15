// SLIM's changes in the note: one CodeMirror extension that lands an editing run's changes in the
// editor showing the note, draws them, and undoes and redoes them.
"use strict";

const { changeMarks, textDiff } = require("./diff");
const { inside, starts, place, mapMarks, dependents, holdable, same, moved, fromHistory,
        tipFor } = require("./runs");
const { makeWidgets } = require("./widgets");

/* The live editor: one CodeMirror extension, the same in every Markdown editor. `cm` is the
 * CodeMirror modules (state, view and commands: Obsidian's at runtime, real ones in a browser
 * test); `host` is the plugin's side: the live editors, Obsidian's editor fields, a Markdown
 * renderer, an icon setter, and `changed(view, runId)`, told whenever a run's changes move.
 *
 * Per run, code keeps:
 *   base    the note as it was sent: the server's first copy;
 *   agent   the server's copy now (every change, whether it landed here or not);
 *   kept    the changes now in the note; a skipped change is never kept;
 *   doc     base with the kept changes: SLIM's version of the note;
 *   toDoc   agent → doc; owner  doc → the editor's text (everything else that changed the note).
 * Undo and redo change `kept` and replay the changes from base, so a change and the changes made
 * inside it come and go together, exactly: an inverse mapped through a later change made inside
 * its text leaves that change's words behind. A change the server says is `part_of` another
 * reworked only SLIM's own text from this run: it is drawn, counted and undone as that one.
 */
function makeLiveEditor(cm, host) {
  const { StateField, StateEffect, ChangeSet, Text, Compartment, EditorState } = cm.state;
  const { EditorView, Decoration, WidgetType, ViewPlugin } = cm.view;
  const isolate = cm.commands?.isolateHistory;
  const AGENT = "slim.agent", UNDO = "slim.undo", REDO = "slim.redo", RESTORE = "slim.restore";
  const setRun = StateEffect.define();      // {id, run}: this transaction is that run's own
  const clearRuns = StateEffect.define();
  const showBefore = StateEffect.define();  // {runId, text} or null
  const setDraft = StateEffect.define();    // {start: draft} | {grow: {id, text}} | {end: id} | {endRun: runId} | {rename: {from, to}}
  const lock = new Compartment();
  const focused = new WeakMap();
  const leaving = new WeakMap();            // view → the timer that lets go of its hovered change
  const LEAVE_MS = 500;

  const pathOf = (state) => (host.infoField && state.field(host.infoField, false)?.file?.path) || null;
  const livePreview = (state) => !host.livePreviewField || state.field(host.livePreviewField, false) !== false;
  const own = (spec) => ({ ...spec, annotations: isolate ? isolate.of("full") : undefined });

  const runsField = StateField.define({
    create: () => new Map(),
    update(runs, tr) {
      const mine = new Map();
      for (const effect of tr.effects) {
        if (effect.is(clearRuns)) return new Map();
        if (effect.is(setRun)) mine.set(effect.value.id, effect.value.run);
      }
      if (!mine.size && !tr.docChanged) return runs;
      const path = pathOf(tr.state);
      const history = tr.isUserEvent("undo") || tr.isUserEvent("redo");
      const next = new Map();
      for (const [id, run] of runs) {
        if (mine.has(id)) continue;
        if (!tr.docChanged) { next.set(id, run); continue; }
        if (run.path && path && run.path !== path) continue;  // the editor shows another note now
        next.set(id, (history && fromHistory(ChangeSet, run, tr))
          || { ...run, owner: run.owner.compose(tr.changes) });
      }
      for (const [id, run] of mine) if (run) next.set(id, run);
      return next;
    },
  });

  const beforeField = StateField.define({
    create: () => null,
    update(value, tr) {
      for (const effect of tr.effects) if (effect.is(showBefore)) value = effect.value;
      return value;
    },
  });

  const runOf = (state, runId) => state.field(runsField, false)?.get(runId) || null;

  // ---- drawing -----------------------------------------------------------------------------

  const { Gone, Undo, Folded, Before, Draft } = makeWidgets(WidgetType, host, {
    showHover: (view, runId, id) => showHover(view, runId, id),
    leaveHover: (view) => leaveHover(view),
    undo: (view, runId, ids) => api.undo(view, runId, ids),
    reveal: (view, runId, id) => api.reveal(view, runId, id),
    toggleOpen: (view, runId, id) => api.toggleOpen(view, runId, id),
    showBefore: (view, runId, text, on) => api.showBefore(view, runId, text, on),
  });

  function draw(state) {
    const runs = state.field(runsField, false);
    if (!runs?.size) return Decoration.none;
    const { doc } = state;
    const raw = !livePreview(state);
    const path = pathOf(state) || "";
    const clamp = (pos) => Math.max(0, Math.min(doc.length, pos));
    const ranges = [];
    for (const run of runs.values()) {
      if (!run.marks || !run.display.marks || run.owner.newLength !== doc.length) continue;
      const settled = !run.active;
      // A change and the changes `part_of` it are drawn as one, under its id: one rail, one
      // hover, one ↺ on the first line any of them covers.
      const drawn = [], lines = new Map();
      for (const step of run.steps) {
        const root = step.partOf || step.id;
        if (!run.kept.has(step.id) || !run.kept.has(root) || !run.marks.has(step.id)) continue;
        const at = mapMarks(run.marks.get(step.id), run.owner);
        drawn.push({ step, root, at });
        if (!lines.has(root)) lines.set(root, new Set());
        for (const [f, t] of at.rails) {
          if (t <= f) continue;
          let end = t;
          while (end > f + 1 && doc.sliceString(end - 1, end) === "\n") end -= 1;
          for (let pos = clamp(f); pos <= clamp(end - 1) && pos < doc.length + 1;) {
            const line = doc.lineAt(pos);
            lines.get(root).add(line.from);
            if (line.to >= doc.length) break;
            pos = line.to + 1;
          }
        }
      }
      const byId = new Map(run.steps.map((step) => [step.id, step]));
      for (const [root, starts] of lines) {
        const hovered = run.display.hover === root;
        const cls = `slim-change${hovered ? " is-hovered" : ""}${run.latest === root ? " slim-arrive" : ""}`;
        const attributes = { "data-slim-run": run.id, "data-slim-change": root };
        for (const from of starts) ranges.push(Decoration.line({ class: cls, attributes }).range(from));
        if (settled && starts.size) {
          const widget = new Undo(run.id, root, tipFor(run, byId.get(root)), hovered);
          ranges.push(Decoration.widget({ widget, side: -1 }).range(Math.min(...starts)));
        }
      }
      for (const { step, root, at } of drawn) {
        // A change made inside another is measured against SLIM's own draft, not the owner's
        // text: it draws its rails and nothing else.
        if (step.id !== root) continue;
        for (const [f, t] of at.ins) if (t > f) ranges.push(Decoration.mark({ class: "slim-ins" }).range(clamp(f), clamp(t)));
        for (const [f, t] of at.formats) if (t > f) ranges.push(Decoration.mark({ class: "slim-format" }).range(clamp(f), clamp(t)));
        for (const [p, text] of at.dels) {
          ranges.push(Decoration.widget({ widget: new Gone(text, raw, path), side: -1 }).range(clamp(p)));
        }
        at.folds.forEach(([p, text, isMove], k) => {
          const widget = new Folded({ runId: run.id, id: root, text, moved: isMove, path,
            open: run.display.open.has(root), hovered: run.display.hover === root,
            undo: settled && !lines.get(root).size && k === 0 ? tipFor(run, step) : null });
          ranges.push(Decoration.widget({ widget, block: true, side: -1 }).range(doc.lineAt(clamp(p)).from));
        });
      }
    }
    return Decoration.set(ranges, true);
  }

  const marksField = StateField.define({
    create: draw,
    update(value, tr) {
      const was = tr.startState, now = tr.state;
      if (tr.docChanged || was.field(runsField, false) !== now.field(runsField, false)
          || livePreview(was) !== livePreview(now)) return draw(now);
      return value;
    },
    provide: (field) => EditorView.decorations.from(field),
  });

  const beforeMarks = StateField.define({
    create: () => Decoration.none,
    update(value, tr) {
      const shown = tr.state.field(beforeField, false);
      if (!shown) return Decoration.none;
      if (!tr.docChanged && shown === tr.startState.field(beforeField, false)) return value;
      const widget = new Before(shown.runId, shown.text, pathOf(tr.state) || "");
      return Decoration.set([Decoration.replace({ widget, block: true }).range(0, tr.state.doc.length)]);
    },
    provide: (field) => EditorView.decorations.from(field),
  });

  // ---- drafts: a change's text while the model writes it ------------------------------------

  // Shown where the change will go and never in the note: no history, no save, nothing for a
  // change to skip. A replace dims the text it will replace. Each draft's place is in this
  // editor's own positions, so every edit, the owner's or SLIM's, maps it; the change that
  // lands, the server, or the run's end takes it away.
  const draftsField = StateField.define({
    create: () => new Map(),
    update(drafts, tr) {
      let next = drafts;
      const path = pathOf(tr.state);                // the editor shows another note now: none of its drafts
      if (drafts.size && (tr.docChanged || path !== pathOf(tr.startState))) {
        next = new Map([...drafts].filter(([, d]) => !(d.path && path && d.path !== path)).map(([id, d]) => {
          if (!tr.docChanged) return [id, d];
          if (d.from === d.to) { const at = tr.changes.mapPos(d.from, -1); return [id, { ...d, from: at, to: at }]; }
          const from = tr.changes.mapPos(d.from, 1);
          return [id, { ...d, from, to: Math.max(from, tr.changes.mapPos(d.to, -1)) }];
        }));
      }
      for (const effect of tr.effects) {
        if (!effect.is(setDraft)) continue;
        const { start, grow, end, endRun, rename } = effect.value;
        next = new Map(next);
        if (start) next.set(start.id, start);
        if (grow && next.has(grow.id)) next.set(grow.id, { ...next.get(grow.id), text: next.get(grow.id).text + grow.text });
        if (end != null) next.delete(end);
        for (const [id, d] of next) {
          if (endRun != null && d.runId === endRun) next.delete(id);
          if (rename && d.runId === rename.from) next.set(id, { ...d, runId: rename.to });
        }
      }
      return next;
    },
  });
  const draftsOf = (state, runId) => [...(state.field(draftsField, false)?.values() || [])].some((d) => d.runId === runId);

  // Under the text it will replace, or at the start of the line its insert goes before; at the
  // end of a line, under it.
  const draftMarks = StateField.define({
    create: () => Decoration.none,
    update(value, tr) {
      const drafts = tr.state.field(draftsField, false);
      if (drafts === tr.startState.field(draftsField, false) && !tr.docChanged) return value;
      if (!drafts?.size) return Decoration.none;
      const { doc } = tr.state;
      const path = pathOf(tr.state) || "";
      const ranges = [];
      for (const d of drafts.values()) {
        const from = Math.min(d.from, doc.length), to = Math.min(Math.max(d.to, from), doc.length);
        for (let pos = from; pos < to;) {
          const line = doc.lineAt(pos);
          ranges.push(Decoration.line({ class: "slim-drafting" }).range(line.from));
          if (line.to >= doc.length) break;
          pos = line.to + 1;
        }
        const line = doc.lineAt(to), widget = new Draft(d.id, d.text, path);
        ranges.push(to === line.from ? Decoration.widget({ widget, block: true, side: -1 }).range(to)
          : Decoration.widget({ widget, block: true, side: 1 }).range(line.to));
      }
      return Decoration.set(ranges, true);
    },
    provide: (field) => EditorView.decorations.from(field),
  });

  // Every live editor, so the plugin finds the one showing a note; and each run that moved, so
  // the chat and its saved turn follow what happened in the note (↺, Cmd-Z, See before).
  const tracker = ViewPlugin.fromClass(class {
    constructor(view) {
      this.view = view;
      host.views.add(view);
      focused.set(view, view.hasFocus ? Date.now() : 0);
    }
    update(update) {
      if (update.focusChanged && update.view.hasFocus) focused.set(update.view, Date.now());
      const was = update.startState.field(runsField, false), now = update.state.field(runsField, false);
      const told = new Set();
      if (was !== now) {
        for (const [id, run] of was) if (!run.pending && now.get(id)?.version !== run.version) told.add(id);
        for (const [id, run] of now) if (!run.pending && !was.has(id)) told.add(id);
      }
      const before = update.startState.field(beforeField, false), after = update.state.field(beforeField, false);
      if (before !== after) for (const shown of [before, after]) if (shown) told.add(shown.runId);
      if (told.size) queueMicrotask(() => { for (const id of told) host.changed?.(update.view, id); });
    }
    destroy() {
      host.views.delete(this.view);
      clearTimeout(leaving.get(this.view));          // never let go of a hover in a closed editor
      leaving.delete(this.view);
    }
  });

  // A change under the pointer is marked and shows its ↺ at once. Leaving waits a moment before
  // letting go, so the pointer can cross the margin to ↺.
  const showHover = (view, runId, id) => {
    clearTimeout(leaving.get(view));
    leaving.delete(view);
    const runs = view.state.field(runsField, false);
    if (!runs) return;
    const effects = [];
    for (const run of runs.values()) {
      const want = run.id === runId ? id : null;
      if (run.display.hover !== want) {
        effects.push(setRun.of({ id: run.id, run: { ...run, display: { ...run.display, hover: want } } }));
      }
    }
    if (effects.length) view.dispatch({ effects });
  };
  const leaveHover = (view) => {
    const runs = view.state.field(runsField, false);
    if (leaving.has(view) || ![...(runs?.values() || [])].some((run) => run.display.hover)) return;
    leaving.set(view, setTimeout(() => { leaving.delete(view); showHover(view, null, null); }, LEAVE_MS));
  };
  const hover = (view, target) => {
    const el = target?.closest?.("[data-slim-change]");
    if (el) showHover(view, el.dataset.slimRun, el.dataset.slimChange);
    else leaveHover(view);
    return false;
  };

  const api = {
    extension: [runsField, marksField, beforeField, beforeMarks, draftsField, draftMarks, tracker, lock.of([]),
      EditorView.domEventHandlers({
        mouseover: (event, view) => hover(view, event.target),
        mouseleave: (_event, view) => hover(view, null),
      })],
    pathOf: (view) => pathOf(view.state),
    focusedAt: (view) => focused.get(view) || 0,

    // A run starts on the text that was sent, at Send: every keystroke from then on is the
    // owner's, mapped exactly (anything already typed counts as one replace). Until the server
    // names the run it is `pending`, under the chat's own name, and the chat is not told of it.
    startRun(view, { id, base, pending = false }) {
      const sent = Text.of(String(base).split("\n"));
      const now = view.state.doc.toString();
      const run = {
        id, path: pathOf(view.state), base: sent, agent: sent, doc: sent, steps: [], kept: new Set(),
        toDoc: ChangeSet.empty(sent.length), owner: ChangeSet.of(textDiff(String(base), now), sent.length),
        marks: new Map(), redo: [], active: true, latest: null, version: 1, pending,
        display: { marks: true, hover: null, open: new Set() },
      };
      view.dispatch({ effects: setRun.of({ id, run }) });
    },

    // The server's name for a run started at Send; false when the editor no longer has it.
    claimRun(view, from, to) {
      const run = runOf(view.state, from);
      if (!run) return false;
      view.dispatch({ effects: [setRun.of({ id: from, run: null }), setDraft.of({ rename: { from, to } }),
        setRun.of({ id: to, run: { ...run, id: to, pending: false, version: run.version + 1 } })] });
      return true;
    },

    // A question answered, or a request that never became a run: nothing to keep.
    dropRun(view, id) {
      const effects = draftsOf(view.state, id) ? [setDraft.of({ endRun: id })] : [];
      if (runOf(view.state, id)) effects.push(setRun.of({ id, run: null }));
      if (effects.length) view.dispatch({ effects });
    },

    // A change's text while the model writes it, at its place in the server's copy (UTF-16 units
    // of that copy, as splices are), shown where that place is now. False when the editor has no
    // such run or the place is not in its copy.
    startDraft(view, runId, { id, from, to, text = "" }) {
      const run = runOf(view.state, runId), path = pathOf(view.state);
      if (!run || !Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < from || to > run.agent.length) return false;
      if (run.path && path && run.path !== path) return false;      // the editor shows another note now
      const point = (pos, side) => run.owner.mapPos(run.toDoc.mapPos(pos, side), side);
      const a = point(from, from === to ? -1 : 1), b = from === to ? a : Math.max(a, point(to, -1));
      const effects = [setDraft.of({ start: { id: String(id), runId, path: run.path, from: a, to: b, text: String(text) } })];
      if (!view.hasFocus) effects.push(EditorView.scrollIntoView(b, { y: "nearest", yMargin: 80 }));
      view.dispatch({ effects });
      return true;
    },

    growDraft(view, id, text) {
      if (text && view.state.field(draftsField, false)?.has(id)) view.dispatch({ effects: setDraft.of({ grow: { id, text } }) });
    },

    endDraft(view, id) {
      if (view.state.field(draftsField, false)?.has(id)) view.dispatch({ effects: setDraft.of({ end: id }) });
    },

    // One change as the server made it: splices on its copy, in order. It lands where those
    // splices sit now, unless the owner has edited there (or undone what it changes), or a change
    // it needs is not in the note: then it is skipped, and later changes still land where they
    // belong.
    applyChange(view, runId, change) {
      const run = runOf(view.state, runId);
      if (!run || !run.active) return "skipped";
      const splices = (change.splices || []).map(({ from, to, insert }) => ({ from, to, insert: String(insert ?? "") }));
      let end = 0;
      for (const s of splices) {
        if (!Number.isInteger(s.from) || !Number.isInteger(s.to) || s.from < end || s.to < s.from || s.to > run.agent.length) {
          return "skipped";
        }
        end = s.to;
      }
      const changes = ChangeSet.of(splices, run.agent.length);
      const pieces = splices.map((s) => ({ old: run.agent.sliceString(s.from, s.to), now: s.insert }));
      // What it needs: what the server says, and every earlier change whose text its splices
      // cut into (a line diff can reach past the blocks a tool call named).
      const needs = new Set((change.needs || []).map(String));
      for (const earlier of run.steps) {
        const cut = earlier.regions.some(([a, b]) => splices.some(({ from, to }) =>
          from < to ? from < b && to > a : a < from && from < b));
        if (cut) needs.add(earlier.id);
      }
      // A change `part_of` an earlier one reworked only SLIM's own text from this run: it is
      // drawn and undone as part of that change (its root), and never stands without it.
      const named = change.part_of == null ? null : run.steps.find((earlier) => earlier.id === String(change.part_of));
      const partOf = named ? named.partOf || named.id : null;
      if (partOf) needs.add(partOf);
      let shift = 0;
      const regions = [];
      for (const { from, to, insert } of splices) {
        if (insert) regions.push([from + shift, from + shift + insert.length]);
        shift += insert.length - (to - from);
      }
      const remap = (step) => ({ ...step, regions: step.regions.map(([a, b]) => {
        const f = changes.mapPos(a, 1);
        return [f, Math.max(f, changes.mapPos(b, -1))];
      }) });
      const step = { id: String(change.id), label: change.label || "", op: change.op || "", step: change.step, partOf,
                     needs: [...needs], splices, changes, regions, rel: changeMarks(change.op, pieces) };
      const at = splices.map(({ from, to }) => {
        const a = run.toDoc.mapPos(from, 1);
        return [a, from === to ? a : Math.max(a, run.toDoc.mapPos(to, -1))];
      });
      const touched = run.steps.some((earlier) => needs.has(earlier.id) && !run.kept.has(earlier.id))
        || splices.some(({ from, to }) => run.toDoc.touchesRange(from, to))
        || at.some(([from, to]) => run.owner.touchesRange(from, to));
      const agent = changes.apply(run.agent);
      const steps = run.steps.map(remap);
      if (touched) {
        const next = { ...run, steps: [...steps, { ...step, skipped: true }], agent,
                       toDoc: changes.invert(run.agent).compose(run.toDoc), version: run.version + 1 };
        view.dispatch({ effects: setRun.of({ id: runId, run: next }) });
        return "skipped";
      }
      const onDoc = changes.map(run.toDoc);
      const marks = new Map([...run.marks].map(([id, m]) => [id, mapMarks(m, onDoc)]));
      marks.set(step.id, place(step.rel, starts(splices, run.toDoc)));
      const next = { ...run, steps: [...steps, step], agent, kept: new Set([...run.kept, step.id]),
                     doc: onDoc.apply(run.doc), toDoc: run.toDoc.map(changes, true),
                     owner: run.owner.map(onDoc, true), marks, latest: partOf || step.id, version: run.version + 1 };
      const landing = onDoc.map(run.owner);
      const effects = [setRun.of({ id: runId, run: next })];
      // Where it landed comes into view, unless the owner is typing in this editor.
      if (!view.hasFocus) {
        let first = null;
        landing.iterChangedRanges((_fromA, _toA, fromB) => { if (first === null) first = fromB; });
        if (first !== null) effects.push(EditorView.scrollIntoView(first, { y: "nearest", yMargin: 80 }));
      }
      view.dispatch(own({ changes: landing, effects, userEvent: AGENT }));
      return "applied";
    },

    // The run is over: no more changes come, and ↺ appears.
    endRun(view, runId) {
      const run = runOf(view.state, runId);
      const effects = draftsOf(view.state, runId) ? [setDraft.of({ endRun: runId })] : [];
      if (run?.active) effects.push(setRun.of({ id: runId, run: { ...run, active: false, latest: null, version: run.version + 1 } }));
      if (effects.length) view.dispatch({ effects });
    },

    // ↺: these changes and everything made inside them, as one step. Refused, not forced, when
    // the owner has edited that text since.
    undo(view, runId, ids) {
      const run = runOf(view.state, runId);
      if (!run) return "gone";
      const gone = dependents(run, ids);
      const kept = new Set([...run.kept].filter((id) => !gone.has(id)));
      if (same(kept, run.kept)) return "nothing";
      const next = moved(ChangeSet, run, kept, [...run.kept].filter((id) => gone.has(id)));
      if (!next) return "refused";
      view.dispatch(own({ changes: next.changes, effects: setRun.of({ id: runId, run: next.run }), userEvent: UNDO }));
      return "undone";
    },

    // Puts back what the last ↺ (or Undo all) took out.
    redo(view, runId) {
      const run = runOf(view.state, runId);
      const group = run?.redo.at(-1);
      if (!group) return "nothing";
      const kept = holdable(run, [...run.kept, ...group]);
      const next = moved(ChangeSet, run, kept, null);
      if (!next) return "refused";
      next.run.redo = run.redo.slice(0, -1).filter((g) => g.some((id) => !kept.has(id)));
      view.dispatch(own({ changes: next.changes, effects: setRun.of({ id: runId, run: next.run }), userEvent: REDO }));
      return "redone";
    },

    // Every change the owner has not edited since, and what those need, comes out as one step.
    // A change and those `part_of` it go or stay together, and are counted once.
    undoAll(view, runId) {
      const run = runOf(view.state, runId);
      if (!run || !run.kept.size) return { undone: 0, kept: 0 };
      const rootOf = new Map(run.steps.map((step) => [step.id, step.partOf || step.id]));
      const edited = new Set([...run.kept]
        .filter((id) => (run.marks.get(id)?.rails || []).some(([f, t]) => inside(run.owner, f, t))).map((id) => rootOf.get(id)));
      // What stays: the changes the owner edited, what they need, and every part of a root that
      // stays, until nothing more comes in.
      let kept = new Set([...run.kept].filter((id) => edited.has(rootOf.get(id))));
      for (let size = -1; size !== kept.size;) {
        size = kept.size;
        kept = holdable(run, kept);
        for (const id of run.kept) if (kept.has(rootOf.get(id))) kept.add(id);
      }
      const gone = [...run.kept].filter((id) => !kept.has(id));
      const roots = (ids) => [...ids].filter((id) => rootOf.get(id) === id).length;
      const next = gone.length ? moved(ChangeSet, run, kept, gone) : null;
      if (!next) return { undone: 0, kept: roots(run.kept) };
      view.dispatch(own({ changes: next.changes, effects: setRun.of({ id: runId, run: next.run }), userEvent: UNDO }));
      return { undone: roots(gone), kept: roots(kept) };
    },

    setMarks(view, runId, on) {
      const run = runOf(view.state, runId);
      if (run && run.display.marks !== on) {
        view.dispatch({ effects: setRun.of({ id: runId, run: { ...run, display: { ...run.display, marks: on }, version: run.version + 1 } }) });
      }
    },

    toggleOpen(view, runId, id) {
      const run = runOf(view.state, runId);
      if (!run) return;
      const open = new Set(run.display.open);
      if (open.has(id)) open.delete(id); else open.add(id);
      view.dispatch({ effects: setRun.of({ id: runId, run: { ...run, display: { ...run.display, open } } }) });
    },

    // A moved block: show where it went.
    reveal(view, runId, id) {
      const run = runOf(view.state, runId);
      const at = run?.marks.get(id) && mapMarks(run.marks.get(id), run.owner);
      const pos = at?.rails[0]?.[0] ?? at?.folds[0]?.[0];
      if (pos != null && pos <= view.state.doc.length) {
        view.dispatch({ effects: EditorView.scrollIntoView(pos, { y: "center" }) });
      }
    },

    // The note as it was before the run, drawn over the note: read-only, the file untouched.
    showBefore(view, runId, beforeText, on) {
      const shown = view.state.field(beforeField, false);
      if (!on && shown?.runId !== runId) return;
      view.dispatch({
        effects: [showBefore.of(on ? { runId, text: String(beforeText) } : null),
                  lock.reconfigure(on ? [EditorView.editable.of(false), EditorState.readOnly.of(true)] : [])],
      });
    },
    showing: (view) => view.state.field(beforeField, false)?.runId || null,

    // The note as it was before the run, as one editor change: Cmd-Z takes it back. The runs'
    // marks go with it.
    restore(view, beforeText) {
      view.dispatch(own({
        changes: textDiff(view.state.doc.toString(), String(beforeText)),
        effects: [clearRuns.of(null), showBefore.of(null), lock.reconfigure([])],
        userEvent: RESTORE,
      }));
    },

    state(view, runId) {
      const run = runOf(view.state, runId);
      if (!run) return null;
      const changes = run.steps.map((step) => ({
        id: step.id, label: step.label,
        state: step.skipped ? "skipped" : run.kept.has(step.id) ? "applied" : "undone",
      }));
      return {
        changes, undone: changes.filter((change) => change.state === "undone").length,
        redoable: run.redo.length > 0, active: run.active, marks: run.display.marks,
        before: api.showing(view) === runId, touched: !run.owner.empty,
      };
    },
  };
  return api;
}

module.exports = { makeLiveEditor };
