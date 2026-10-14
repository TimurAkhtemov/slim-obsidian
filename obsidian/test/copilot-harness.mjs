// The copilot's test harness: a CopilotView over a fake plugin, and a fake live editor for
// editing runs. Import it before requiring anything from src/; it installs the stubs
// (fakes.mjs) and draws Markdown into the copilot's fake DOM as one <p>.
import { createRequire } from "node:module";
import { fakeEl, find, setMarkdownRender } from "./fakes.mjs";

const require = createRequire(import.meta.url);
const { CopilotView } = require("../src/chat-view.js");

// The real renderer appends the parsed tree before its promise resolves; one <p> stands in.
setMarkdownRender(async (_app, markdown, el) => { el.createEl("p", { text: markdown }); });

function makeView(plugin = {}) {
  return new CopilotView({ contentEl: { empty() {} } }, {
    app: { workspace: { openLinkText() {} } },
    postJSON: async () => ({ threads: [] }),
    getJSON: async () => ({ threads: [] }),
    streamCopilot: async () => {},
    ...plugin,
  });
}

// --- rendering: the view must not be torn down on every token -------------------------------

function renderedView(plugin = {}) {
  const root = fakeEl("div");
  let rebuilds = 0;
  const originalEmpty = root.empty;
  root.empty = () => { rebuilds += 1; originalEmpty(); };
  const view = new CopilotView({ contentEl: root }, {
    app: { workspace: {} },
    postJSON: async () => ({}), getJSON: async () => ({ threads: [] }), streamCopilot: async () => {},
    ...plugin,
  });
  view.context = { source: { id: "s1", path: "Notes/one.md", title: "One" }, related: [] };
  return { view, root, rebuilds: () => rebuilds };
}

const text = (el) => find(el, (node) => node.cls === "slim-message-text" && node.parent?.cls.includes("is-streaming"));

// ---- one chat that answers or edits: the run lands in the note, the chat shows its steps -------

// A stand-in for makeLiveEditor: it records what the chat asks of the note's editor.
function fakeLive(text = "# A\n\nold\n") {
  const calls = [];
  const runs = new Map();
  const editor = { path: "Notes/one.md", text, state: { doc: { toString: () => editor.text } } };
  const live = {
    calls,
    pathOf: (view) => view.path,
    startRun(view, { id, base }) { calls.push(["start", id, base]); runs.set(id, { changes: [], active: true, marks: true, redoable: false, touched: false }); },
    applyChange(view, id, change) {
      calls.push(["apply", change.id]);
      const state = change.blocked ? "skipped" : "applied";
      if (state === "applied") view.text += `[${change.id}]`;
      runs.get(id).changes.push({ id: change.id, label: change.label, state });
      return state;
    },
    claimRun(_view, from, to) {
      calls.push(["claim", from, to]);
      if (!runs.has(from)) return false;
      runs.set(to, runs.get(from));
      runs.delete(from);
      return true;
    },
    dropRun(_view, id) { calls.push(["drop", id]); runs.delete(id); },
    endRun(_view, id) { calls.push(["end", id]); runs.get(id).active = false; },
    state(_view, id) {
      const run = runs.get(id);
      if (!run) return null;
      return { changes: run.changes, undone: run.changes.filter((c) => c.state === "undone").length,
               redoable: run.redoable, active: run.active, marks: run.marks, before: false, touched: run.touched };
    },
    undoAll(_view, id) { calls.push(["undoAll", id]); for (const c of runs.get(id).changes) c.state = "undone"; return { undone: 1, kept: 0 }; },
    redo(_view, id) { calls.push(["redo", id]); return "redone"; },
    setMarks(_view, id, on) { calls.push(["marks", id, on]); runs.get(id).marks = on; },
    showBefore(_view, id, text, on) { calls.push(["before", id, text, on]); },
    showing: () => null,
    restore(view, text) { calls.push(["restore", text]); view.text = text; runs.clear(); },
    startDraft(_view, id, draft) { calls.push(["draft", id, draft.id, draft.from, draft.to, draft.text]); return runs.has(id); },
    growDraft(_view, id, text) { calls.push(["grow", id, text]); },
    endDraft(_view, id) { calls.push(["drop", id]); },
  };
  return { live, editor, runs };
}

function editView({ events, live, editor, saved = [], indexed = [], before = "# A\n\nold\n", vault,
                   restored = (body) => ({ text: before }), posted = [] } = {}) {
  const view = makeView({
    live,
    liveViewFor: () => editor,
    liveRunView: (id) => (live.state(editor, id) ? editor : null),
    noteView: async () => editor,
    showEditing: async () => {},
    postJSON: async (route, body) => {
      if (route.endsWith("save")) saved.push(structuredClone(body));
      if (route.endsWith("context")) { indexed.push(body.path); return { source: { id: "s1", path: "Notes/one.md" } }; }
      if (route === "/api/copilot/restore") { posted.push(body); return restored(body); }
      return {};
    },
    getJSON: async (route) => (route.startsWith("/api/copilot/before") ? { text: before } : { threads: [] }),
    streamCopilot: events,
  });
  view.app = { vault: vault || {}, workspace: { openLinkText() {}, getLeavesOfType: () => [] } };
  view.render = () => {};
  view.saveOpenCopies = async () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  view.activeThread = { id: "chat-1", title: "t", source_id: "s1", source_path: "Notes/one.md" };
  return view;
}

const PLAN = { intro: "I'll rename the heading and tighten the intro.", steps: ["Rename the heading", "Tighten the intro"], outline: null };
const change = (id, step, extra = {}) => ({ id, step, op: "replace", label: `Replaced ${id}`, needs: [],
                                            splices: [{ from: 0, to: 1, insert: "x" }], ...extra });

export { makeView, renderedView, text, fakeLive, editView, PLAN, change };
