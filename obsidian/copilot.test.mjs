import assert from "node:assert/strict";
import Module from "node:module";
import test from "node:test";
import http from "node:http";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const originalLoad = Module._load;
// main.js is loaded whole — the copilot lives inside it (see the "ONE file" test below), so
// the stub must satisfy everything main.js destructures at load, not only what the copilot uses.
const obsidianStub = {
  Plugin: class {},
  ItemView: class { constructor(leaf) { this.leaf = leaf; this.contentEl = leaf.contentEl; } },
  MarkdownView: class {},
  PluginSettingTab: class {},
  Setting: class {},
  Notice: class {},
  Modal: class {
    constructor(app) { this.app = app; this.contentEl = fakeEl("div"); }
    open() { this.onOpen(); }
    close() { this.onClose(); }
  },
  requestUrl: async () => ({ status: 200, json: {} }),
  // The real renderer appends the parsed tree before its promise resolves; one <p> stands in.
  MarkdownRenderer: { async render(_app, markdown, el) { el.createEl("p", { text: markdown }); } },
};
Module._load = function (request) {
  if (request === "obsidian") return obsidianStub;
  return originalLoad.apply(this, arguments);
};

const { CopilotView, RenameModal, SSEParser, groupThreads, streamRequest,
        groupCitations, noteTitle, normalizeMath, SUGGESTED_PROMPTS, metaText,
        sha256Hex, slashQuery, matchSkills, inputsText, editorSelection,
        historyText, editOutcome, wordTokens, wordOps, markable, formatChange,
        changeMarks, beforeBlocks, textDiff, excerpt } =
  createRequire(import.meta.url)(path.join(HERE, "main.js"));

test("SSE parsing survives arbitrary chunk boundaries", () => {
  const parser = new SSEParser();
  assert.deepEqual(parser.push("event: sta"), []);
  assert.deepEqual(parser.push("ge\ndata: {\"stage\":\"Writing\"}\n\nevent: delta\n"), [
    { event: "stage", data: { stage: "Writing" } },
  ]);
  assert.deepEqual(parser.push("data: {\"text\":\"Hi\"}\n\n"), [
    { event: "delta", data: { text: "Hi" } },
  ]);
});

test("thread hierarchy puts the active note first and sorts chats by recency", () => {
  const groups = groupThreads([
    { id: "old", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-01-01" },
    { id: "new", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-02-01" },
    { id: "other", source_id: "s2", source_path: "Notes/two.md", updated_at: "2026-03-01" },
  ], "s1");
  assert.equal(groups[0].sourceId, "s1");
  assert.deepEqual(groups[0].threads.map((item) => item.id), ["new", "old"]);
  assert.equal(groups[0].expanded, true);
  assert.equal(groups[1].expanded, false);
});

// A DOM just large enough for render(): Obsidian's createEl/empty plus the handful of
// properties the copilot reads and writes. Tests walk `children`; nothing is queried by CSS.
function fakeEl(tag = "div", options = {}) {
  const el = {
    tag, cls: options.cls || "", textContent: options.text ?? "", children: [], attrs: {},
    listeners: {}, value: "", open: false, scrollTop: 0, scrollHeight: 0, clientHeight: 100,
    createEl(childTag, childOptions = {}) {
      const child = fakeEl(childTag, childOptions);
      child.parent = el;
      el.children.push(child);
      return child;
    },
    createDiv(childOptions = {}) { return el.createEl("div", childOptions); },
    empty() { el.children = []; },
    addClass() {}, removeClass() {},
    setAttribute(name, value) { el.attrs[name] = value; },
    addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
    focus() { el.focused = true; },
    select() {},
    remove() { if (el.parent) el.parent.children = el.parent.children.filter((child) => child !== el); },
  };
  return el;
}

function find(el, predicate) {
  if (predicate(el)) return el;
  for (const child of el.children) {
    const hit = find(child, predicate);
    if (hit) return hit;
  }
  return null;
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function makeView(plugin = {}) {
  return new CopilotView({ contentEl: { empty() {} } }, {
    app: { workspace: { openLinkText() {} } },
    postJSON: async () => ({ threads: [] }),
    getJSON: async () => ({ threads: [] }),
    streamCopilot: async () => {},
    ...plugin,
  });
}

test("a late context response cannot replace the newly active note", async () => {
  const first = deferred();
  const second = deferred();
  const view = makeView({
    postJSON: async (route, body) => route.endsWith("context")
      ? (body.path === "Notes/one.md" ? first.promise : second.promise)
      : { threads: [] },
  });
  view.render = () => {};
  const one = view.setActiveFile({ path: "Notes/one.md", extension: "md" });
  const two = view.setActiveFile({ path: "Notes/two.md", extension: "md" });
  second.resolve({ source: { id: "s2", path: "Notes/two.md" }, related: [] });
  await two;
  first.resolve({ source: { id: "s1", path: "Notes/one.md" }, related: [] });
  await one;
  assert.equal(view.context.source.id, "s2");
});

test("unsupported active files produce an unavailable state without an HTTP call", async () => {
  let called = false;
  const view = makeView({ postJSON: async () => { called = true; } });
  view.render = () => {};
  await view.setActiveFile({ path: "Board.canvas", extension: "canvas" });
  assert.equal(called, false);
  assert.match(view.error, /Markdown note/);
});

test("a note the server refuses shows the server's sentence, not the route and status", async () => {
  // 2026-09-29: "/api/copilot/context returned 400: note is not indexable", in red.
  const view = makeView({
    postJSON: async () => {
      const err = new Error("/api/copilot/context returned 400: SLIM leaves this note out");
      err.status = 400;
      err.detail = "SLIM leaves this note out";
      throw err;
    },
    getJSON: async () => ({ threads: [] }),
  });
  view.render = () => {};
  await view.setActiveFile({ path: "Skills/brief.md", extension: "md" });
  assert.equal(view.error, "SLIM leaves this note out");
});

test("the active Markdown view is saved before its note is synchronized", async () => {
  const order = [];
  const file = { path: "Notes/one.md", extension: "md" };
  const view = makeView({
    saveActiveNote: async (candidate) => { assert.equal(candidate, file); order.push("save"); },
    postJSON: async () => {
      order.push("context");
      return { source: { id: "s1", path: file.path }, related: [] };
    },
  });
  view.render = () => {};
  await view.setActiveFile(file);
  assert.deepEqual(order, ["save", "context"]);
});

test("sending streams deltas and stores the turn with the depth they picked", async () => {
  const saved = [];
  const view = makeView({
    postJSON: async (route, body) => {
      if (route.endsWith("save")) saved.push(body);
      if (route.endsWith("context")) return { source: { id: "source-1", path: "Notes/one.md" } };
      return route.endsWith("threads") ? { threads: [] } : {};
    },
    streamCopilot: async (_body, handlers) => {
      handlers.stage({ stage: "Thinking deeply" });
      handlers.delta({ text: "Grounded " });
      assert.equal(view.streamingText, "Grounded ");
      handlers.delta({ text: "answer" });
      handlers.turn({ answer: "Grounded answer", mode: "deep", citations: [] });
    },
  });
  view.render = () => {};
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  view.mode = "deep";
  await view.send("Compare these");
  assert.equal(view.turns.at(-1).text, "Grounded answer");
  assert.equal(view.turns.at(-1).turn.mode, "deep");
  assert.equal(saved.length, 1);
  assert.equal(saved[0].source_id, "source-1");
});

test("cancel aborts the in-flight stream and returns the composer to idle", () => {
  const view = makeView();
  let aborted = false;
  view.abort = { abort() { aborted = true; } };
  view.sending = true;
  view.cancel();
  assert.equal(aborted, true);
  assert.equal(view.sending, false);
});

test("retry resends the failed question without duplicating its user turn", async () => {
  let attempts = 0;
  const view = makeView({
    streamCopilot: async (_body, handlers) => {
      attempts += 1;
      if (attempts === 1) throw new Error("model unavailable");
      handlers.turn({ answer: "Recovered", mode: "quick", citations: [] });
    },
    postJSON: async (route) => route.endsWith("context")
      ? { source: { id: "source-1", path: "Notes/one.md" } } : {},
  });
  view.render = () => {};
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  await view.send("Explain this");
  assert.equal(view.retryQuestion, "Explain this");
  await view.retry();
  assert.deepEqual(view.turns.map((turn) => turn.role), ["you", "slim"]);
  assert.equal(view.turns[0].text, "Explain this");
  assert.equal(view.turns[1].text, "Recovered");
});

test("a failed question leaves the chat when something else is asked, and is never sent as history", async () => {
  const bodies = [];
  const view = makeView({
    streamCopilot: async (body, handlers) => {
      bodies.push(body);
      if (bodies.length === 1) throw new Error("/tidy works on selected text: select some in the note first");
      handlers.turn({ answer: `Answer ${bodies.length}`, mode: "quick", citations: [] });
    },
    postJSON: async (route) => route.endsWith("context")
      ? { source: { id: "source-1", path: "Notes/one.md" } } : {},
  });
  view.render = () => {};
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  await view.send("/tidy");
  assert.equal(view.retryQuestion, "/tidy");
  await view.send("Any places to expand?");
  assert.deepEqual(bodies[1].history, []);
  assert.deepEqual(view.turns.map((turn) => turn.text), ["Any places to expand?", "Answer 2"]);
  assert.equal(view.retryQuestion, "");
  // A chat saved before this kept its unanswered question: it stays on screen, and is not sent.
  view.turns.unshift({ role: "you", text: "/tidy" });
  await view.send("Ok make the revisions");
  assert.deepEqual(bodies[2].history.map((turn) => turn.text), ["Any places to expand?", "Answer 2"]);
});

test("an SSE error handler rejects streamRequest instead of escaping the callback", async () => {
  const server = http.createServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "text/event-stream" });
    response.end('event: error\ndata: {"error":"offline"}\n\n');
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    await assert.rejects(
      streamRequest(server.address().port, "/copilot", {}, {
        error(data) { throw new Error(data.error); },
      }),
      /offline/,
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("opening a chat owned by another note navigates before loading it", async () => {
  const opened = [];
  const view = makeView({
    app: {
      workspace: { getLeaf: () => ({ async openFile(file) { opened.push(file.path); } }) },
      vault: { getAbstractFileByPath: (path) => ({ path }) },
    },
    getJSON: async () => ({
      id: "other", source_id: "source-2", source_path: "Notes/two.md", turns: [],
      reasoning_mode: "quick",
    }),
  });
  view.render = () => {};
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  await view.openThread("other");
  assert.deepEqual(opened, ["Notes/two.md"]);
  assert.equal(view.pendingThreadId, "other");
  assert.equal(view.activeThread, null);
});

test("a thread saved as Auto still opens, with Quick selected", async () => {
  // Auto went on 2026-09-03, but threads written before it say "auto" on disk forever, and
  // the server refuses that word now. Opening one must not put a mode in the selector that
  // the next save cannot post.
  const view = makeView({
    getJSON: async () => ({
      id: "old", source_id: "source-1", source_path: "Notes/one.md", turns: [],
      reasoning_mode: "auto",
    }),
  });
  view.render = () => {};
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  await view.openThread("old");
  assert.equal(view.mode, "quick");
});

test("renaming a thread saved as Auto posts Quick, not the word the server refuses", async () => {
  // ⚠ `renameThread` fetches a thread it never OPENED, so a coercion that lived only in
  // openThread missed this path entirely: the save was rejected and the click handler has no
  // catch, so renaming an old conversation failed silently.
  const saves = [];
  const view = makeView({
    getJSON: async () => ({
      id: "old", title: "Old", source_id: "source-2", source_path: "Notes/two.md",
      reasoning_mode: "auto", turns: [{ role: "you", text: "Keep me" }],
    }),
    postJSON: async (_route, body) => { saves.push(body); return body; },
  });
  view.render = () => {};
  view.promptTitle = async () => "Renamed";
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  view.mode = "deep";                       // the selector must not leak into an old thread
  await view.renameThread("old");
  assert.equal(saves.length, 1);
  assert.equal(saves[0].reasoning_mode, "quick");
  assert.equal(saves[0].title, "Renamed");
});

test("the main plugin registers the copilot view and follows active notes", () => {
  const source = fs.readFileSync(path.join(HERE, "main.js"), "utf8");
  assert.match(source, /registerView\(VIEW_TYPE_COPILOT/);
  assert.match(source, /active-leaf-change/);
  assert.match(source, /activateCopilot/);
  const manifest = JSON.parse(fs.readFileSync(path.join(HERE, "manifest.json"), "utf8"));
  assert.equal(manifest.name, "SLIM");
});

test("the plugin is ONE file: Obsidian's loader cannot resolve a relative require", () => {
  // Obsidian evals main.js with a shim `require` that serves `obsidian` and CodeMirror and
  // hands everything else to the renderer's global require — bound to Obsidian's bundle,
  // not the plugin folder. `require("./copilot")` throws before the plugin class exists,
  // which takes the RECORDER down with it. Node resolves relative paths, so only this
  // assertion sees it.
  const source = fs.readFileSync(path.join(HERE, "main.js"), "utf8");
  assert.doesNotMatch(source, /require\(\s*["']\.\.?\//);
  assert.equal(fs.existsSync(path.join(HERE, "copilot.js")), false);
});

test("re-selecting the note that is already active neither cancels the answer nor re-syncs", async () => {
  // Obsidian fires active-leaf-change when focus moves between the editor and the sidebar,
  // and getActiveFile() still names the same note. That click must not abort the stream,
  // erase the question, or flash "Indexing this note...".
  let contextCalls = 0;
  const view = makeView({
    postJSON: async (route) => { if (route.endsWith("context")) contextCalls += 1; return { threads: [] }; },
  });
  view.render = () => {};
  const file = { path: "Notes/one.md", extension: "md" };
  view.context = { source: { id: "s1", path: file.path }, related: [] };
  view.activeThread = { id: "t1", source_id: "s1" };
  view.turns = [{ role: "you", text: "Why?" }];
  let aborted = false;
  view.abort = { abort() { aborted = true; } };
  view.sending = true;
  await view.setActiveFile(file);
  assert.equal(aborted, false);
  assert.equal(contextCalls, 0);
  assert.equal(view.turns.length, 1);
  assert.equal(view.activeThread.id, "t1");
});

test("activation syncs a new note once, however many callers race to it", async () => {
  // onOpen, active-leaf-change and activateCopilot all call setActiveFile for the same
  // file; two concurrent context POSTs race ingest_note into an IntegrityError (500).
  const gate = deferred();
  let contextCalls = 0;
  const view = makeView({
    postJSON: async (route) => {
      if (!route.endsWith("context")) return { threads: [] };
      contextCalls += 1;
      return gate.promise;
    },
  });
  view.render = () => {};
  const file = { path: "Notes/one.md", extension: "md" };
  const first = view.setActiveFile(file);
  const second = view.setActiveFile(file);
  gate.resolve({ source: { id: "s1", path: file.path }, related: [] });
  await Promise.all([first, second]);
  assert.equal(contextCalls, 1);
  assert.equal(view.context.source.id, "s1");
});

test("send re-syncs the open note first, without the nearby-notes search", async () => {
  // The index must reflect what they typed since the note was opened; the sidebar no longer
  // re-syncs on every focus change, so the question itself does it — cheaply.
  const calls = [];
  const view = makeView({
    saveActiveNote: async () => { calls.push("save"); },
    postJSON: async (route, body) => {
      calls.push(route.split("/").pop());
      if (route.endsWith("context")) {
        assert.equal(body.related, false);
        return { source: { id: "s1", path: "Notes/one.md", title: "Fresh" }, related: [] };
      }
      return {};
    },
    streamCopilot: async (body, handlers) => {
      calls.push("stream");
      assert.equal(body.source_id, "s1");
      handlers.turn({ answer: "ok", mode: "quick", citations: [] });
    },
  });
  view.render = () => {};
  view.app.workspace.getActiveFile = () => ({ path: "Notes/one.md", extension: "md" });
  view.context = { source: { id: "s1", path: "Notes/one.md", title: "Stale" }, related: [{ path: "Notes/two.md" }] };
  await view.send("What changed?");
  assert.deepEqual(calls, ["save", "context", "stream", "save"]);
  assert.equal(view.context.source.title, "Fresh");
  assert.equal(view.context.related.length, 1);     // nearby notes kept from the open
});

function contextFor(route) {
  return route.endsWith("context") ? { source: { id: "s1", path: "Notes/one.md" } } : null;
}

test("the answer lands in the thread that asked, even after switching threads mid-stream", async () => {
  const saved = [];
  const gate = deferred();
  const view = makeView({
    postJSON: async (route, body) => {
      if (route.endsWith("save")) saved.push(body);
      return contextFor(route) || { threads: [] };
    },
    getJSON: async () => ({
      id: "B", title: "B", source_id: "s1", source_path: "Notes/one.md", reasoning_mode: "quick",
      turns: [{ role: "you", text: "old" }, { role: "slim", text: "older" }],
    }),
    streamCopilot: async (_body, handlers) => {
      await gate.promise;
      handlers.turn({ answer: "For A", mode: "quick", citations: [] });
    },
  });
  view.render = () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  view.activeThread = { id: "A", title: "A", source_id: "s1", source_path: "Notes/one.md" };
  const sending = view.send("Question for A");
  await view.openThread("B");
  gate.resolve();
  await sending;
  assert.equal(saved.length, 1);
  assert.equal(saved[0].id, "A");
  assert.deepEqual(saved[0].turns.map((turn) => turn.text), ["Question for A", "For A"]);
  assert.deepEqual(view.turns.map((turn) => turn.text), ["old", "older"]);   // B untouched
});

test("stop hands the unanswered question back to the composer", async () => {
  // A dangling "you" turn would ride along as history ending in a user turn and be
  // persisted with the next answer; the question is theirs — give it back to them.
  const view = makeView({
    postJSON: async (route) => contextFor(route) || {},
    streamCopilot: (_body, _handlers, signal) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => {
        const error = new Error("Cancelled");
        error.name = "AbortError";
        reject(error);
      });
    }),
  });
  view.render = () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  const sending = view.send("Half-asked");
  await new Promise((resolve) => setImmediate(resolve));
  view.cancel();
  await sending;
  assert.deepEqual(view.turns, []);
  assert.equal(view.draft, "Half-asked");
});

test("a failed thread save shows the answer once and offers no retry that would duplicate it", async () => {
  const view = makeView({
    postJSON: async (route) => {
      if (route.endsWith("save")) throw new Error("thread is over the cap — start a new conversation");
      return contextFor(route) || {};
    },
    streamCopilot: async (_body, handlers) => {
      handlers.turn({ answer: "A1", mode: "quick", citations: [] });
    },
  });
  view.render = () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  await view.send("Q");
  assert.deepEqual(view.turns.map((turn) => turn.text), ["Q", "A1"]);
  assert.match(view.error, /over the cap/);
  assert.equal(view.retryQuestion, "");
});

test("deleting the thread that is streaming stops the stream first", async () => {
  const gate = deferred();
  const saved = [];
  const view = makeView({
    postJSON: async (route, body) => {
      if (route.endsWith("save")) saved.push(body);
      return contextFor(route) || {};
    },
    streamCopilot: (_body, handlers, signal) => new Promise((resolve, reject) => {
      signal.addEventListener("abort", () => {
        const error = new Error("Cancelled");
        error.name = "AbortError";
        reject(error);
      });
      gate.promise.then(() => { handlers.turn({ answer: "late", citations: [] }); resolve(); });
    }),
  });
  view.render = () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  view.activeThread = { id: "A", title: "A", source_id: "s1", source_path: "Notes/one.md" };
  const sending = view.send("Q");
  await new Promise((resolve) => setImmediate(resolve));
  await view.deleteThread("A");      // arms
  await view.deleteThread("A");      // confirms
  gate.resolve();
  await sending;
  assert.equal(saved.length, 0);     // a deleted thread is never resurrected by a late answer
  assert.equal(view.sending, false);
});

test("text-only history carries role and text only, the whole conversation", async () => {
  // Each stored slim turn carries its evidence pack and stats; re-uploading all of it on
  // every question grew with the square of the thread. Every turn goes, each whole: the
  // server fits the conversation to the model's window.
  let history;
  const view = makeView({
    postJSON: async (route) => contextFor(route) || {},
    streamCopilot: async (body, handlers) => {
      history = body.history;
      handlers.turn({ answer: "ok", mode: "quick", citations: [] });
    },
  });
  view.render = () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  view.activeThread = { id: "A", title: "A", source_id: "s1", source_path: "Notes/one.md" };
  view.turns = Array.from({ length: 16 }, (_v, i) => ({
    role: i % 2 ? "slim" : "you", text: i === 1 ? "a".repeat(3123) : `t${i}`,
    turn: { evidence: ["big"], citations: [] },
  }));
  await view.send("Q");
  assert.equal(history.length, 16);
  assert.deepEqual(Object.keys(history[0]), ["role", "text"]);
  assert.equal(history[0].text, "t0");
  assert.equal(history[1].text.length, 3123);
  assert.equal(history.at(-1).text, "t15");
});

test("an attached image is sent once, saved as a ref, and included in later history", async () => {
  const sent = [];
  const saved = [];
  const pending = { name: "equation.png", mime: "image/png", data: "BASE64",
                    preview: "data:image/png;base64,BASE64", bytes: 6 };
  const ref = { id: "a".repeat(64), name: "equation.png", mime: "image/png", bytes: 6 };
  const view = makeView({
    postJSON: async (route, body) => {
      if (route.endsWith("context")) return { source: { id: "s1", path: "Notes/one.md" } };
      if (route.endsWith("save")) saved.push(body);
      return {};
    },
    streamCopilot: async (body, handlers) => {
      sent.push(body);
      handlers.turn({ answer: "Explained", mode: "quick", citations: [], attachments: [ref] });
    },
  });
  view.render = () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  view.pendingImages = [pending];
  await view.send("Explain this equation");
  assert.deepEqual(sent[0].images, [{ name: "equation.png", mime: "image/png", data: "BASE64" }]);
  assert.deepEqual(view.turns[0].attachments, [ref]);
  assert.deepEqual(saved[0].turns[0].attachments, [ref]);

  await view.send("Why is that term squared?");
  assert.deepEqual(sent[1].history[0].attachments, [ref]);
  assert.deepEqual(sent[1].images, []);
});

test("an image can be sent without typed text and a failed request keeps it for retry", async () => {
  let attempts = 0;
  const image = { name: "equation.png", mime: "image/png", data: "BASE64",
                  preview: "data:image/png;base64,BASE64", bytes: 6 };
  const view = makeView({
    postJSON: async (route) => route.endsWith("context")
      ? { source: { id: "s1", path: "Notes/one.md" } } : {},
    streamCopilot: async (body, handlers) => {
      attempts += 1;
      assert.equal(body.question, "Explain this image.");
      if (attempts === 1) throw new Error("offline");
      handlers.turn({ answer: "Explained", citations: [], attachments: [] });
    },
  });
  view.render = () => {};
  view.context = { source: { id: "s1", path: "Notes/one.md" }, related: [] };
  view.pendingImages = [image];
  await view.send("");
  assert.deepEqual(view.retryImages, [image]);
  await view.retry();
  assert.equal(attempts, 2);
  assert.equal(view.turns.at(-1).role, "slim");
});

test("the composer renders pending previews and an attach control; a sent image stays a thumbnail", () => {
  const { view, root } = renderedView();
  view.activeThread = { id: "chat-1", source_id: "s1" };
  view.pendingImages = [{ name: "pending.png", preview: "data:image/png;base64,x" }];
  view.turns = [{ role: "you", text: "Explain", attachments: [
    { id: "a".repeat(64), name: "stored.png", mime: "image/png", bytes: 4 },
  ] }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-copilot-image-previews") !== null, true);
  assert.equal(find(root, (node) => node.attrs["aria-label"] === "Attach images") !== null, true);
  const image = find(root, (node) => node.cls === "slim-message-image");
  assert.equal(image.src, `http://127.0.0.1:7546/api/copilot/image?thread=chat-1&id=${"a".repeat(64)}`);
  assert.equal(find(root, (node) => node.cls === "slim-image-chip"), null);
  image.onerror();                                                  // the store no longer has it
  assert.equal(find(root, (node) => node.cls === "slim-message-image"), null);
  assert.equal(find(root, (node) => node.cls === "slim-image-chip").textContent, "stored.png");
});

test("copy and regenerate sit under an answer; regenerate re-asks with the stored image refs", async () => {
  const bodies = [];
  const { view, root } = renderedView({
    postJSON: async (route) => route.endsWith("context") ? { source: { id: "s1", path: "Notes/one.md" } } : {},
    streamCopilot: async (body, handlers) => {
      bodies.push(body);
      handlers.turn({ answer: "Second", mode: "quick", citations: [], timing: { total_ms: 8200 } });
    },
  });
  const ref = { id: "b".repeat(64), name: "eq.png", mime: "image/png", bytes: 4 };
  view.activeThread = { id: "chat-1", source_id: "s1", source_path: "Notes/one.md" };
  view.turns = [
    { role: "you", text: "First?" },
    { role: "slim", text: "One", turn: { mode: "deep", timing: { total_ms: 42000 } } },
    { role: "you", text: "Explain", attachments: [ref] },
    { role: "slim", text: "Two", turn: { mode: "quick", timing: { total_ms: 8200 } } },
  ];
  view.render();
  const metas = [];
  find(root, (node) => { if (node.cls === "slim-message-meta") metas.push(node.textContent); return false; });
  assert.deepEqual(metas, ["Deep · 42 s", "Quick · 8 s"]);
  const regenerates = [];
  find(root, (node) => { if (node.attrs["aria-label"] === "Regenerate") regenerates.push(node); return false; });
  assert.equal(regenerates.length, 1);                                // the last answer only
  assert.equal(find(root, (node) => node.attrs["aria-label"] === "Copy answer") !== null, true);
  await view.regenerate();
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].question, "Explain");
  assert.deepEqual(bodies[0].images, [ref]);                          // refs, no bytes
  assert.deepEqual(view.turns.map((turn) => turn.text), ["First?", "One", "Explain", "Second"]);
  assert.equal(metaText({ mode: "quick" }), "Quick");
});

test("the streaming answer renders as Markdown in place, and a stale render never lands", async () => {
  const { view, root } = renderedView();
  view.sending = true;
  view.streamingTurns = view.turns;
  view.streamingText = "**bold** start";
  view.render();
  const liveText = text(root);
  await view.renderLive();
  assert.equal(liveText.hidden, true);
  const rendered = () => find(root, (node) => node.cls.includes("slim-rendered-markdown") && node.parent?.cls.includes("is-streaming") && !node.hidden);
  assert.deepEqual(rendered().children.map((child) => child.textContent), ["**bold** start"]);
  // An older render resolving after a newer one is dropped, not swapped in.
  const { MarkdownRenderer } = obsidianStub;
  const slow = deferred();
  const original = MarkdownRenderer.render;
  MarkdownRenderer.render = async (_app, markdown, el) => { await slow.promise; el.createEl("p", { text: markdown }); };
  view.streamingText = "old";
  const older = view.renderLive();
  MarkdownRenderer.render = original;
  view.streamingText = "newer";
  await view.renderLive();
  slow.resolve();
  await older;
  assert.deepEqual(rendered().children.map((child) => child.textContent), ["newer"]);
  const streamingArticle = find(root, (node) => node.cls.includes("is-streaming"));
  assert.equal(streamingArticle.children.filter((child) => child.cls.includes("slim-rendered-markdown")).length, 1);
  // Stopping invalidates anything still in flight and clears the timer.
  view.renderDelta();
  assert.notEqual(view.liveRenderTimer, null);
  view.cancel();
  assert.equal(view.liveRenderTimer, null);
});

test("plain-text paste replaces the composer selection and keeps the caret editable", () => {
  const { view, root } = renderedView();
  view.draft = "before old after";
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  input.selectionStart = 7;
  input.selectionEnd = 10;
  let prevented = 0;
  input.onpaste({
    clipboardData: { files: [], getData: (type) => type === "text/plain" ? "new" : "" },
    preventDefault() { prevented += 1; },
  });
  assert.equal(input.value, "before new after");
  assert.equal(view.draft, "before new after");
  assert.equal(input.selectionStart, 10);
  assert.equal(input.selectionEnd, 10);
  assert.equal(prevented, 1);
});

test("backspace and delete edit the composer selection even when Obsidian owns the sidebar keymap", () => {
  const { view, root } = renderedView();
  view.draft = "abcde";
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  let prevented = 0;
  const key = (value) => input.onkeydown({
    key: value, shiftKey: false, metaKey: false, altKey: false, ctrlKey: false,
    preventDefault() { prevented += 1; },
  });

  input.selectionStart = input.selectionEnd = 3;
  key("Backspace");
  assert.equal(input.value, "abde");
  assert.equal(input.selectionStart, 2);

  input.selectionStart = 1;
  input.selectionEnd = 3;
  key("Delete");
  assert.equal(input.value, "ae");
  assert.equal(view.draft, "ae");
  assert.equal(prevented, 2);
});

test("a cited path that no longer exists is reported, never created as an empty note", async () => {
  // openLinkText creates the target when it is missing. Citation paths come from DB rows
  // and thread records that are only as fresh as the last ingest; a note dragged since then
  // would become a phantom — which the next active-leaf-change then indexes and embeds.
  const opened = [];
  let linkText = 0;
  const view = makeView({
    app: {
      workspace: {
        openLinkText() { linkText += 1; },
        getLeaf: () => ({ async openFile(file) { opened.push(file.path); } }),
      },
      vault: { getAbstractFileByPath: (path) => (path === "Notes/there.md" ? { path } : null) },
    },
  });
  view.render = () => {};
  await view.openNote("Notes/gone.md");
  assert.equal(linkText, 0);
  assert.deepEqual(opened, []);
  assert.match(view.error, /Notes\/gone\.md/);
  await view.openNote("Notes/there.md");
  assert.deepEqual(opened, ["Notes/there.md"]);
});

test("rename asks through an Obsidian modal, because Electron's window.prompt throws", async () => {
  // Electron 39 defines window.prompt as `function(){ throw new Error("prompt() is not
  // supported.") }`, so a `!window.prompt` guard passes and every rename click rejected.
  globalThis.window = { prompt() { throw new Error("prompt() is not supported."); } };
  const saves = [];
  const view = makeView({
    getJSON: async () => ({
      id: "other", title: "Old", source_id: "source-2", source_path: "Notes/two.md",
      reasoning_mode: "deep", turns: [{ role: "you", text: "Keep me" }],
    }),
    postJSON: async (_route, body) => { saves.push(body); return body; },
  });
  view.render = () => {};
  view.promptTitle = async (current) => `${current} renamed`;
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  await view.renameThread("other");
  assert.equal(saves[0].title, "Old renamed");
  assert.equal(saves[0].turns[0].text, "Keep me");
});

test("the rename modal resolves the typed title on submit and null when dismissed", async () => {
  const results = [];
  const modal = new RenameModal({}, "Old", (value) => results.push(value));
  modal.open();
  const input = find(modal.contentEl, (el) => el.tag === "input");
  assert.equal(input.value, "Old");
  input.value = "  New title  ";
  const submit = find(modal.contentEl, (el) => el.tag === "button" && el.cls.includes("mod-cta"));
  submit.onclick();
  modal.close();
  assert.deepEqual(results, ["New title"]);

  const dismissed = [];
  const second = new RenameModal({}, "Old", (value) => dismissed.push(value));
  second.open();
  second.close();
  assert.deepEqual(dismissed, [null]);
});

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
const transcriptOf = (root) => find(root, (node) => node.cls === "slim-copilot-transcript");

test("a delta updates the streaming answer in place instead of rebuilding the view", () => {
  const { view, root, rebuilds } = renderedView();
  view.sending = true;
  view.streamingTurns = view.turns;
  view.streamingText = "Gro";
  view.render();
  const before = rebuilds();
  view.streamingText = "Grounded";
  view.renderDelta();
  assert.equal(rebuilds(), before);                       // ~40 tokens/s must not mean 40 teardowns/s
  assert.equal(text(root).textContent, "Grounded");
  view.stage = "Writing";
  view.renderStage();
  assert.equal(rebuilds(), before);
  assert.equal(find(root, (node) => node.cls === "slim-copilot-stage").textContent, "Writing");
});

test("the transcript follows the streaming answer when they were reading the bottom", () => {
  const { view, root } = renderedView();
  view.sending = true;
  view.streamingTurns = view.turns;
  view.streamingText = "Gro";
  view.render();
  const transcript = transcriptOf(root);
  transcript.scrollHeight = 500;                          // the answer grew past the pane
  transcript.scrollTop = 400;                             // and they are reading its end
  view.streamingText = "Grounded, at length";
  view.renderDelta();
  assert.equal(transcript.scrollTop, 500);
  transcript.scrollTop = 100;                             // they scrolled up to read something
  transcript.scrollHeight = 900;
  view.streamingText = "Grounded, at even more length";
  view.renderDelta();
  assert.equal(transcript.scrollTop, 100);                // and is not yanked back down
});

test("the composer keeps focus across a full render", () => {
  const { view, root } = renderedView();
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  input.onfocus();
  view.render();
  assert.equal(find(root, (node) => node.tag === "textarea").focused, true);
});

test("a note group they expanded by hand stays expanded across renders", () => {
  const { view, root } = renderedView();
  view.showThreads = true;
  view.threads = [
    { id: "a", title: "A", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-02-01" },
    { id: "b", title: "B", source_id: "s2", source_path: "Notes/two.md", updated_at: "2026-01-01" },
  ];
  view.render();
  const other = () => find(root, (node) => node.tag === "details"
    && node.children.some((child) => child.tag === "summary" && child.attrs.title === "Notes/two.md"));
  assert.equal(other().children.find((child) => child.tag === "summary").textContent, "two");
  assert.equal(other().open, false);
  other().children.find((child) => child.tag === "summary").onclick();
  view.render();
  assert.equal(other().open, true);
});

test("a turn shows when its answer was cut off; a bracket in prose is plain text", () => {
  // Sources come from the server's follow-up call, never from markers, so [9] in prose is
  // text and nothing flags it. Truncation is still their to know about.
  const { view, root } = renderedView();
  view.turns = [{ role: "you", text: "Q" }, {
    role: "slim", text: "Half [1] and [9]",
    turn: { citations: [], verification: { truncated: true } },
  }];
  view.render();
  const notes = [];
  find(root, (node) => { if (node.cls === "slim-verification") notes.push(node.textContent); return false; });
  assert.equal(notes.length, 1);
  assert.match(notes[0], /cut off/i);
});

// --- the redesign: rendered Markdown, titles not paths, one source per note ------------------

test("LaTeX delimiters qwen writes become the dollar forms Obsidian renders", () => {
  assert.equal(normalizeMath("Let \\(x\\) be \\[ y = 2 \\]."), "Let $x$ be $$ y = 2 $$.");
  assert.equal(normalizeMath("multi \\[\na\n\\] line"), "multi $$\na\n$$ line");
  assert.equal(normalizeMath("keep `\\[1, 3\\]` and ```\n\\(code\\)\n``` alone"),
               "keep `\\[1, 3\\]` and ```\n\\(code\\)\n``` alone");
  assert.equal(normalizeMath("already $x$ and $$y$$"), "already $x$ and $$y$$");
  assert.equal(normalizeMath(""), "");
  // Markdown-escaped brackets that are not math stay as written.
  const prose = "The list is \\[1, 3\\] and a link \\[here\\](x). Also arr\\[0\\] and A: \\( \\).";
  assert.equal(normalizeMath(prose), prose);
  assert.equal(normalizeMath("\\(\\theta\\) and \\(x\\) and \\(a+b\\)"), "$\\theta$ and $x$ and $a+b$");
});

test("citations group by note in order of first appearance", () => {
  const intro = { source_id: "a", path: "Notes/nlp/Introduction to NLP.md", title: "Introduction to NLP" };
  const groups = groupCitations([
    { n: 1, ...intro }, { n: 2, ...intro },
    { n: 3, source_id: "b", path: "Notes/nlp/Word embeddings.md", title: "" },
    { n: 4, ...intro },
  ]);
  assert.deepEqual(groups.map((group) => [group.title, group.ns]),
    [["Introduction to NLP", [1, 2, 4]], ["Word embeddings", [3]]]);
});

test("a note title comes from the frontmatter, then the file name, then the path", () => {
  const file = { path: "Notes/nlp/Word embeddings.md", basename: "Word embeddings", extension: "md" };
  const folder = { path: "Notes/nlp", name: "nlp" };
  const app = {
    vault: { getAbstractFileByPath: (p) => (p === file.path ? file : p === folder.path ? folder : null) },
    metadataCache: { getFileCache: () => ({ frontmatter: { title: "Embeddings, week 3" } }) },
  };
  assert.equal(noteTitle(app, file.path), "Embeddings, week 3");
  app.metadataCache.getFileCache = () => null;
  assert.equal(noteTitle(app, file.path), "Word embeddings");
  assert.equal(noteTitle(app, folder.path), "nlp");                // a folder has no cache
  assert.equal(noteTitle({ workspace: {} }, "Notes/gone/Old note.md"), "Old note");
});

test("the header shows the note title and keeps the path as a tooltip", () => {
  const { view, root } = renderedView();
  view.context.source = { id: "s1", path: "Notes/nlp/Introduction to NLP.md", title: "Introduction to NLP" };
  view.render();
  const note = find(root, (node) => node.cls === "slim-copilot-note");
  assert.equal(note.textContent, "Introduction to NLP");
  assert.equal(note.attrs.title, "Notes/nlp/Introduction to NLP.md");
  assert.equal(find(root, (node) => node.cls === "slim-copilot-path"), null);
  assert.equal(find(root, (node) => node.cls === "slim-copilot-eyebrow").textContent, "SLIM");
});

test("nearby notes fold under the title, closed until they open them, and stay open across renders", () => {
  const { view, root } = renderedView();
  view.context.related = [
    { source_id: "s2", path: "Notes/nlp/Word embeddings.md", title: "Word embeddings", scope: "course" },
    { source_id: "s3", path: "Notes/nlp/Tokens.md", title: "Tokens", scope: "folder" },
  ];
  view.render();
  const chip = () => find(root, (node) => node.cls === "slim-copilot-context");
  assert.equal(chip().open, false);
  const summary = chip().children.find((child) => child.tag === "summary");
  assert.deepEqual(summary.children.map((child) => child.textContent), ["One", "2 nearby"]);
  assert.equal(summary.children[0].attrs.title, "Notes/one.md");
  summary.onclick();
  view.render();
  assert.equal(chip().open, true);
  view.context.related = [];
  view.render();
  assert.equal(chip(), null);                                       // no chip without nearby notes
  assert.equal(find(root, (node) => node.cls === "slim-copilot-note").textContent, "One");
});

test("the history button swaps the chat list into the conversation's slot; opening a chat swaps back", async () => {
  const { view, root } = renderedView({
    getJSON: async () => ({ id: "a", source_id: "s1", source_path: "Notes/one.md", turns: [], reasoning_mode: "quick" }),
  });
  view.threads = [{ id: "a", title: "A", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-02-01" }];
  view.render();
  assert.notEqual(find(root, (node) => node.cls === "slim-copilot-transcript"), null);
  assert.equal(find(root, (node) => node.cls === "slim-copilot-threads"), null);
  find(root, (node) => node.attrs["aria-label"] === "Chats").onclick();
  assert.equal(find(root, (node) => node.cls === "slim-copilot-transcript"), null);
  const list = find(root, (node) => node.cls === "slim-copilot-threads");
  assert.notEqual(list, null);
  // A delta while the list shows must not rebuild the view per token.
  view.sending = true; view.streamingTurns = view.turns; view.streamingText = "x";
  const built = view.render.bind(view); let rebuilt = 0; view.render = () => { rebuilt += 1; built(); };
  view.renderDelta(); view.renderStage();
  assert.equal(rebuilt, 0);
  view.sending = false; view.render = built;
  find(root, (node) => node.cls === "slim-chat-open").onclick();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(view.showThreads, false);
  assert.notEqual(find(root, (node) => node.cls === "slim-copilot-transcript"), null);
});

test("an empty thread offers suggested prompts, and one click sends it", () => {
  const { view, root } = renderedView();
  const sent = [];
  view.send = async (text) => { sent.push(text); };
  view.render();
  const chips = [];
  find(root, (node) => { if (node.cls === "slim-suggest") chips.push(node); return false; });
  assert.deepEqual(chips.map((chip) => chip.textContent), SUGGESTED_PROMPTS);
  chips[1].onclick();
  assert.deepEqual(sent, ["Quiz me on this note"]);
  view.turns = [{ role: "you", text: "Q" }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-suggest"), null);
});

test("their turn is a bubble on the right and the composer starts as one line", () => {
  const styles = fs.readFileSync(path.join(HERE, "styles.css"), "utf8");
  assert.match(styles, /\.slim-message\.is-you\s*\{[^}]*align-self:\s*flex-end/s);
  assert.match(styles, /\.slim-message\.is-you\s*\{[^}]*border-radius/s);
  assert.match(styles, /\.slim-copilot-input\s*\{[^}]*min-height:\s*36px/s);
  const { view, root } = renderedView();
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  assert.equal(input.placeholder, "Ask SLIM, or / for skills");
});

test("nearby notes and chat groups read as titles, with the path on hover", () => {
  const { view, root } = renderedView();
  view.context.related = [
    { source_id: "s2", path: "Notes/nlp/Word embeddings.md", title: "Word embeddings", scope: "course" },
  ];
  view.threads = [
    { id: "b", title: "B", source_id: "s2", source_path: "Notes/nlp/Word embeddings.md", updated_at: "2026-01-01" },
  ];
  view.render();
  const nearby = find(root, (node) => node.cls === "slim-related-note");
  assert.equal(nearby.attrs.title, "Notes/nlp/Word embeddings.md");
  assert.deepEqual(nearby.children.map((child) => child.textContent), ["Word embeddings", "course"]);
  view.showThreads = true;
  view.render();
  const summary = find(root, (node) => node.tag === "summary" && node.attrs.title === "Notes/nlp/Word embeddings.md");
  assert.equal(summary.textContent, "Word embeddings");
});

test("an answer is rendered as Markdown while a question stays plain text", () => {
  const { view, root } = renderedView();
  view.turns = [{ role: "you", text: "**Q**" }, { role: "slim", text: "**A**", turn: { citations: [] } }];
  view.render();
  const question = find(root, (node) => node.cls.includes("slim-message-text") && node.parent.cls.includes("is-you"));
  assert.equal(question.textContent, "**Q**");
  assert.deepEqual(question.children, []);
  const answer = find(root, (node) => node.cls.includes("slim-message-text") && node.parent.cls.includes("is-slim"));
  assert.equal(answer.textContent, "");                             // nothing set as raw text
  assert.deepEqual(answer.children.map((child) => [child.tag, child.textContent]), [["p", "**A**"]]);
});

const citedTurn = () => {
  const intro = { source_id: "a", path: "Notes/nlp/Introduction to NLP.md", title: "Introduction to NLP" };
  const embeddings = { source_id: "b", path: "Notes/nlp/Word embeddings.md", title: "Word embeddings" };
  return { role: "slim", text: "A [1] B [6] C [7]", turn: { citations: [
    ...[1, 2, 3, 4, 5].map((n) => ({ n, ...intro })), { n: 6, ...embeddings }, { n: 7, ...intro },
  ] } };
};

test("sources are one chip per note under the answer, and open that note", () => {
  const { view, root } = renderedView();
  const opened = [];
  view.openNote = async (path) => { opened.push(path); };
  view.turns = [citedTurn()];
  view.render();
  const rows = [];
  find(root, (node) => { if (node.cls === "slim-source") rows.push(node); return false; });
  assert.equal(rows.length, 2);                                     // seven fragments, two notes
  assert.deepEqual(rows.map((row) => row.textContent), ["Introduction to NLP", "Word embeddings"]);
  assert.equal(rows[0].attrs.title, "Notes/nlp/Introduction to NLP.md");
  assert.equal(find(root, (node) => node.cls === "slim-sources").children[0].textContent, "From your notes");
  rows[0].onclick();
  assert.deepEqual(opened, ["Notes/nlp/Introduction to NLP.md"]);
  view.turns = [{ role: "slim", text: "General knowledge.", turn: { citations: [] } }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-sources"), null);   // nothing when no note was used
});

test("the rendered-answer chain settles without a throw on a DOM that has no text nodes", async () => {
  // Marker linking and the re-scroll run after the Markdown promise. Under this fake DOM
  // there are no childNodes and no document; a walk without guards rejects unhandled here
  // and, in Obsidian, would leave the answer rendered but the markers dead.
  const rejections = [];
  const onRejection = (error) => rejections.push(error);
  process.on("unhandledRejection", onRejection);
  try {
    const { view } = renderedView();
    view.turns = [citedTurn()];
    view.render();
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.off("unhandledRejection", onRejection);
  }
  assert.deepEqual(rejections, []);
});

test("completed chat text can be selected and copied in Obsidian's ItemView", () => {
  // Obsidian gives custom ItemViews `user-select: none`. Textareas remain editable because
  // Chromium treats form controls specially, but ordinary question and answer text inherits
  // the prohibition unless the plugin explicitly restores selection on the transcript.
  const styles = fs.readFileSync(path.join(HERE, "styles.css"), "utf8");
  assert.match(styles, /\.slim-copilot-transcript\s*\{[^}]*user-select:\s*text\s*;/s);
});

// A [[wikilink]] in an answer is rendered by MarkdownRenderer as <a class="internal-link">, and
// Obsidian wires clicks on those only inside its own note views — in this sidebar it was dead.
function linkView(existing) {
  const opened = [];
  const view = makeView({
    app: {
      metadataCache: {
        getFirstLinkpathDest(linkpath, source) {
          return existing.includes(linkpath) ? { path: `Notes/${linkpath}.md`, source } : null;
        },
      },
      workspace: { openLinkText(...args) { opened.push(args); } },
    },
  });
  view.context = { source: { path: "Notes/open.md" } };
  view.render = () => {};
  return { view, opened };
}

function linkClick(href, mods = {}) {
  const link = { getAttribute: (name) => (name === "data-href" ? href : null) };
  const event = {
    ...mods,
    target: { closest: (sel) => (sel === "a.internal-link" ? link : null) },
    prevented: false,
    preventDefault() { this.prevented = true; },
  };
  return event;
}

test("a wikilink in an answer opens the note it names", () => {
  const { view, opened } = linkView(["Bayes rule"]);
  const event = linkClick("Bayes rule#Evidence");
  view.onLinkClick(event);
  assert.equal(event.prevented, true);
  assert.deepEqual(opened, [["Bayes rule#Evidence", "Notes/open.md", false]]);
});

test("cmd-click opens the linked note in a new tab", () => {
  const { view, opened } = linkView(["Bayes rule"]);
  view.onLinkClick(linkClick("Bayes rule", { metaKey: true }));
  assert.deepEqual(opened, [["Bayes rule", "Notes/open.md", true]]);
});

test("a wikilink to a note that does not exist never creates one", () => {
  // openLinkText CREATES a missing target — an empty phantom the next ingest indexes.
  const { view, opened } = linkView([]);
  const event = linkClick("Invented title");
  view.onLinkClick(event);
  assert.equal(event.prevented, true);
  assert.deepEqual(opened, []);
  assert.match(view.error, /Note not found: Invented title/);
});

test("a click that is not on a wikilink is left alone", () => {
  const { view, opened } = linkView(["Bayes rule"]);
  const event = { target: { closest: () => null }, preventDefault() { this.prevented = true; } };
  view.onLinkClick(event);
  assert.equal(event.prevented, undefined);
  assert.deepEqual(opened, []);
});


// ---- skills ----------------------------------------------------------------------------------

test("a slash command completes only while its first word is being typed", () => {
  const skills = [{ name: "actions" }, { name: "consolidate-my-notes" }, { name: "quiz" }];
  assert.equal(slashQuery("/con"), "con");
  assert.equal(slashQuery("/"), "");
  assert.equal(slashQuery("/con focus"), null);
  assert.equal(slashQuery("what /con"), null);
  assert.deepEqual(matchSkills(skills, "con").map((s) => s.name), ["consolidate-my-notes"]);
  assert.equal(matchSkills(skills, "").length, 3);
  assert.deepEqual(matchSkills(skills, null), []);
});

test("the inputs line says what code handed the model, including what it left out", () => {
  assert.equal(inputsText({ used: ["a", "b"], selection: false, dropped: ["c"], cut: [],
                            skipped: ["d"], missing: ["Nope"] }),
    "Read 2 notes · 1 left out (too long together) · 1 without that section · [[Nope]] not found");
  assert.equal(inputsText({ used: [], selection: true }), "Read your selection");
  assert.equal(inputsText(undefined), "");
});

test("the selection comes from the pane showing the open note, not the focused one", () => {
  const leaf = (path, text) => ({ view: { file: { path }, editor: { getSelection: () => text } } });
  const app = { workspace: { getLeavesOfType: () => [leaf("Notes/other.md", "wrong"), leaf("Notes/a.md", "right")] } };
  assert.equal(editorSelection(app, "Notes/a.md"), "right");
  assert.equal(editorSelection(app, "Notes/none.md"), "");
});

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

test("a question sends the open note's text and its hash; there is no toggle and no edit flag", async () => {
  const { live, editor } = fakeLive();
  let body;
  const view = editView({ live, editor, events: async (sent, handlers) => {
    body = sent;
    handlers.delta({ text: "An answer." });
    handlers.turn({ answer: "An answer.", mode: "quick", citations: [] });
  } });
  await view.send("What is this?");
  assert.equal(body.doc, "# A\n\nold\n");
  assert.equal(body.doc_hash, sha256Hex("# A\n\nold\n"));
  assert.equal("edit" in body, false);
  assert.equal(body.selection, "");
  assert.deepEqual(live.calls, []);                            // an answer touches no editor
  assert.equal(view.turns.at(-1).text, "An answer.");
  assert.equal(view.turns.at(-1).turn.edit_run, undefined);
  const { view: shown, root } = renderedView();
  shown.render();
  assert.equal(find(root, (node) => String(node.cls).includes("slim-edit-toggle")), null);
  assert.equal(find(root, (node) => node.tag === "textarea").placeholder, "Ask SLIM, or / for skills");
});

test("only a /command carries the note's selection", async () => {
  const { live, editor } = fakeLive();
  const bodies = [];
  const view = editView({ live, editor, events: async (sent, handlers) => { bodies.push(sent); handlers.turn({ answer: "ok", citations: [] }); } });
  const leaf = { view: { file: { path: "Notes/one.md" }, getMode: () => "source", editor: { getSelection: () => "picked" } } };
  view.app.workspace = { getLeavesOfType: () => [leaf], getMostRecentLeaf: () => leaf };
  await view.send("tidy it");
  await view.send("/tidy");
  assert.deepEqual(bodies.map((b) => b.selection), ["", "picked"]);
});

test("an editing run's events land in the note in order and the turn keeps what happened", async () => {
  const { live, editor } = fakeLive();
  const saved = [], indexed = [];
  let midway = null;
  const view = editView({ live, editor, saved, indexed, events: async (_body, handlers) => {
    handlers.stage({ stage: "Thinking" });
    handlers.edit({ before_ref: "a".repeat(32), hash_before: sha256Hex("# A\n\nold\n") });
    handlers.plan(PLAN);
    handlers.step({ index: 1, state: "doing" });
    handlers.change(change("c1", 1));
    handlers.step({ index: 1, state: "done" });
    handlers.step({ index: 2, state: "doing" });
    midway = { steps: [...view.editRun.steps], changes: view.editRun.changes.map((c) => c.state) };
    handlers.change(change("c2", 2, { blocked: true }));
    handlers.step({ index: 2, state: "partial" });
    handlers.turn({ answer: PLAN.intro, mode: "quick", citations: [], timing: { total_ms: 41000 }, edit_run: {
      plan: PLAN, steps_done: 2, changes: [change("c1", 1), change("c2", 2)], stopped: null, outline_met: true,
      fixups: 0, hash_before: "h0", hash_after: "server-draft", summary: PLAN.intro, before_ref: "a".repeat(32) } });
  } });
  await view.send("Ok make the revisions");
  assert.deepEqual(live.calls.map((c) => c.slice(0, 2)), [["start", "a".repeat(32)], ["apply", "c1"], ["apply", "c2"], ["end", "a".repeat(32)]]);
  assert.equal(live.calls[0][2], "# A\n\nold\n");                     // the run starts on the text that was sent
  assert.deepEqual(midway, { steps: ["done", "doing"], changes: ["applied"] });
  const run = view.turns.at(-1).turn.edit_run;
  assert.deepEqual(run.changes.map(({ id, state }) => [id, state]), [["c1", "applied"], ["c2", "skipped"]]);
  assert.equal(run.changes.some((c) => "splices" in c), false);      // the thread keeps no note text
  assert.deepEqual(run.step_states, ["done", "partial"]);
  assert.equal(run.hash_after, sha256Hex(editor.text));                // the editor's own hash: c2 never landed
  assert.equal(run.before_ref, "a".repeat(32));
  assert.equal(view.editRun, null);
  assert.equal(saved.at(-1).turns.at(-1).turn.edit_run.changes.length, 2);
  assert.deepEqual(indexed.slice(-1), ["Notes/one.md"]);              // saved and re-indexed once, at the end
});

test("the chat shows the intro, a line per step, the outcome and the meta line", () => {
  const { live, editor, runs } = fakeLive();
  const { view, root } = renderedView({ live, liveRunView: (id) => (live.state(editor, id) ? editor : null) });
  const run = { plan: PLAN, steps_done: 2, step_states: ["done", "partial"], stopped: null, summary: PLAN.intro,
                before_ref: "b".repeat(32), hash_before: "h0", hash_after: "h1",
                changes: [{ id: "c1", label: "Replaced s1", state: "applied" }, { id: "c2", label: "Replaced b4", state: "undone" },
                          { id: "c3", label: "Inserted after b9", state: "skipped" }] };
  view.turns = [{ role: "you", text: "Ok make the revisions" },
                { role: "slim", text: PLAN.intro, turn: { mode: "quick", timing: { total_ms: 41000 }, edit_run: run } }];
  runs.set("b".repeat(32), { changes: run.changes, active: false, marks: true, redoable: true, touched: false });
  view.render();
  const texts = [];
  (function walk(el) { if (el.textContent) texts.push(el.textContent); el.children.forEach(walk); })(root);
  for (const want of [PLAN.intro, "Rename the heading", "Tighten the intro", "Done.", "2 changes, 1 undone",
                      "Undo all", "Redo", "See before", "Hide marks", "Quick · 41 s"]) {
    assert.ok(texts.includes(want), `${want} in ${JSON.stringify(texts)}`);
  }
  assert.ok(texts.some((t) => /Inserted after b9/.test(t) && /you edited this here/.test(t)), texts.join(" | "));
  assert.equal(find(root, (node) => node.cls === "slim-step is-partial") !== null, true);
  // Once the note's editor has closed (or after a reload), the line offers See before and Restore.
  runs.clear();
  view.render();
  const later = [];
  (function walk(el) { if (el.textContent) later.push(el.textContent); el.children.forEach(walk); })(root);
  assert.ok(later.includes("See before") && later.includes("Restore") && !later.includes("Undo all"), later.join(" | "));
});

test("the meta line's actions reach the note's editor", async () => {
  const { live, editor, runs } = fakeLive();
  const { view, root } = renderedView({ live, liveRunView: (id) => (live.state(editor, id) ? editor : null),
    getJSON: async () => ({ text: "# A\n\nold\n" }), postJSON: async () => ({}) });
  view.activeThread = { id: "chat-1", source_id: "s1", source_path: "Notes/one.md" };
  const ref = "c".repeat(32);
  const run = { plan: PLAN, step_states: ["done", "done"], steps_done: 2, stopped: null, before_ref: ref, hash_after: "h",
                changes: [{ id: "c1", label: "x", state: "applied" }] };
  view.turns = [{ role: "you", text: "q" }, { role: "slim", text: "a", turn: { mode: "quick", edit_run: run } }];
  runs.set(ref, { changes: [{ id: "c1", state: "applied" }], active: false, marks: true, redoable: true, touched: false });
  view.render();
  const click = async (label) => { await find(root, (node) => node.tag === "button" && node.textContent === label).onclick(); };
  await click("Hide marks");
  await click("Undo all");
  await click("Redo");
  await click("See before");
  assert.deepEqual(live.calls.map((c) => c[0] === "before" ? [c[0], c[2], c[3]] : c.slice(0, 1)),
                   [["marks"], ["undoAll"], ["redo"], ["before", "# A\n\nold\n", true]]);
});

test("stop keeps a partial turn: what landed stays, with its copy from before", async () => {
  const { live, editor } = fakeLive();
  const saved = [], indexed = [];
  const view = editView({ live, editor, saved, indexed, events: (_body, handlers, signal) => new Promise((_resolve, reject) => {
    handlers.edit({ before_ref: "d".repeat(32), hash_before: "h0" });
    handlers.plan(PLAN);
    handlers.step({ index: 1, state: "doing" });
    handlers.change(change("c1", 1));
    signal.addEventListener("abort", () => {
      handlers.change(change("late", 1));                 // same chunk, after Stop: dropped
      const error = new Error("Cancelled");
      error.name = "AbortError";
      reject(error);
    });
  }) });
  const sending = view.send("Ok make the revisions");
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
  view.cancel();
  await sending;
  assert.deepEqual(view.turns.map((t) => t.role), ["you", "slim"]);  // the question is not handed back
  assert.equal(view.draft, "");
  const run = view.turns.at(-1).turn.edit_run;
  assert.equal(run.stopped, "stopped");
  assert.equal(run.before_ref, "d".repeat(32));
  assert.equal(run.hash_before, "h0");
  assert.equal(run.hash_after, sha256Hex(editor.text));
  assert.deepEqual(run.changes.map((c) => c.id), ["c1"]);
  assert.equal(live.calls.some((c) => c[1] === "late"), false);
  assert.equal(saved.at(-1).turns.at(-1).turn.edit_run.stopped, "stopped");
  assert.deepEqual(indexed.slice(-1), ["Notes/one.md"]);
  assert.match(editOutcome(run), /^Stopped after 0 of 2 steps\. Not done: Rename the heading and Tighten the intro\.$/);
});

test("when the note's editor closes or shows another note, the run stops and what landed stays", async () => {
  const { live, editor } = fakeLive();
  const view = editView({ live, editor, events: (_body, handlers, signal) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => { const error = new Error("Cancelled"); error.name = "AbortError"; reject(error); });
    handlers.edit({ before_ref: "5".repeat(32), hash_before: "h0" });
    handlers.plan(PLAN);
    handlers.change(change("c1", 1));
    editor.path = "Notes/other.md";                                   // the leaf now shows another note
    handlers.change(change("c2", 2));
  }) });
  view.plugin.liveViews = new Set([editor]);
  await view.send("Ok make the revisions");
  const run = view.turns.at(-1).turn.edit_run;
  assert.deepEqual(run.changes.map(({ id, state }) => [id, state]), [["c1", "applied"], ["c2", "skipped"]]);
  assert.equal(run.stopped, "stopped");
  assert.deepEqual(live.calls.filter((c) => c[0] === "apply").map((c) => c[1]), ["c1"]);
});

test("an error after changes landed keeps them and says why it stopped", async () => {
  const { live, editor } = fakeLive();
  const view = editView({ live, editor, events: async (_body, handlers) => {
    handlers.edit({ before_ref: "e".repeat(32), hash_before: "h0" });
    handlers.plan(PLAN);
    handlers.change(change("c1", 1));
    handlers.error({ error: "the model failed" });
  } });
  await view.send("Ok make the revisions");
  assert.equal(view.turns.at(-1).turn.edit_run.stopped, "error: the model failed");
  assert.match(view.error, /the model failed/);
  assert.equal(view.retryQuestion, "");                                // no retry: it would apply them again
});

test("a handler that throws does not end the run", async () => {
  const { live, editor } = fakeLive();
  live.applyChange = () => { throw new Error("boom"); };
  const view = editView({ live, editor, events: async (_body, handlers) => {
    handlers.edit({ before_ref: "f".repeat(32), hash_before: "h0" });
    handlers.change(change("c1", 1));
    handlers.turn({ answer: "Done", mode: "quick", citations: [], edit_run: { plan: { intro: "", steps: [] }, steps_done: 0,
      changes: [], stopped: null, summary: "Done: 1 change", before_ref: "f".repeat(32) } });
  } });
  await view.send("rename it");
  assert.equal(view.turns.at(-1).role, "slim");
  assert.equal(view.error, "");
});

test("restore warns when the note was edited since, and needs a second click", async () => {
  const { live, editor } = fakeLive("# A\n\nnew\n");
  const saved = [];
  const view = editView({ live, editor, saved, before: "# A\n\nold\n" });
  const run = { plan: PLAN, step_states: [], steps_done: 2, stopped: null, before_ref: "9".repeat(32),
                hash_after: sha256Hex("# A\n\nnew\n"), changes: [{ id: "c1", label: "x", state: "applied" }] };
  view.turns = [{ role: "you", text: "q" }, { role: "slim", text: "a", turn: { edit_run: run } }];
  editor.text = "# A\n\nnew, and mine\n";                              // edited since
  await view.restore(view.turns[1]);
  assert.equal(view.restoreArmed, "9".repeat(32));
  assert.deepEqual(live.calls, []);
  const { view: shown, root } = renderedView({ live });
  shown.turns = view.turns;
  shown.restoreArmed = view.restoreArmed;
  shown.render();
  const texts = [];
  (function walk(el) { if (el.textContent) texts.push(el.textContent); el.children.forEach(walk); })(root);
  assert.ok(texts.includes("You've edited this note since this change; restoring would discard those edits."));
  assert.ok(texts.includes("Restore anyway") && texts.includes("Cancel"));
  await view.restore(view.turns[1], true);                             // Restore anyway still asks the server
  assert.deepEqual(live.calls, [["restore", "# A\n\nold\n"]]);
  assert.equal(editor.text, "# A\n\nold\n");
  assert.equal(run.restored, true);
  assert.equal(saved.at(-1).turns[1].turn.edit_run.restored, true);
  // Unedited since: one click. A run that was typed into while it ran warns as well.
  editor.text = "# A\n\nnew\n";
  run.restored = false;
  run.hash_after = sha256Hex(editor.text);
  run.typed = true;
  await view.restore(view.turns[1]);
  assert.equal(live.calls.length, 1);
  run.typed = false;
  await view.restore(view.turns[1]);
  assert.equal(live.calls.length, 2);
});

test("Restore applies what the server returns, and shows its refusal instead when a recording changed", async () => {
  const { live, editor } = fakeLive("# A\n\nnew\n");
  const posted = [];
  let reply = { refused: "The recording has new transcript since this change; restoring would drop it." };
  const view = editView({ live, editor, posted, restored: () => reply });
  const ref = "4".repeat(32);
  const run = { plan: PLAN, step_states: [], steps_done: 1, stopped: null, before_ref: ref,
                hash_after: sha256Hex("# A\n\nnew\n"), changes: [{ id: "c1", label: "x", state: "applied" }] };
  view.turns = [{ role: "you", text: "q" }, { role: "slim", text: "a", turn: { edit_run: run } }];
  await view.restore(view.turns[1]);
  assert.deepEqual(posted, [{ thread_id: "chat-1", ref, doc: "# A\n\nnew\n" }]);
  assert.deepEqual(live.calls, []);                                    // refused: the note is untouched
  assert.equal(run.restored, undefined);
  const { view: shown, root } = renderedView({ live });
  Object.assign(shown, { turns: view.turns, restoreRefused: view.restoreRefused });
  shown.render();
  const texts = [];
  (function walk(el) { if (el.textContent) texts.push(el.textContent); el.children.forEach(walk); })(root);
  assert.ok(texts.includes(reply.refused), texts.join(" | "));
  reply = { text: "# A\n\nold, with the transcript as it is now\n" };
  await view.restore(view.turns[1]);
  assert.deepEqual(live.calls, [["restore", "# A\n\nold, with the transcript as it is now\n"]]);
  assert.equal(run.restored, true);
  assert.equal(view.restoreRefused, null);
});

test("the note's editor moving a change (↺, Cmd-Z) updates and saves the turn", async () => {
  const { live, editor, runs } = fakeLive();
  const saved = [];
  const view = editView({ live, editor, saved });
  const ref = "7".repeat(32);
  const run = { plan: PLAN, step_states: [], steps_done: 1, stopped: null, before_ref: ref, hash_after: "old",
                changes: [{ id: "c1", label: "x", state: "applied" }, { id: "c2", label: "y", state: "applied" }] };
  view.turns = [{ role: "you", text: "q" }, { role: "slim", text: "a", turn: { edit_run: run } }];
  runs.set(ref, { changes: [{ id: "c1", state: "applied" }, { id: "c2", state: "undone" }], active: false, marks: true, redoable: true, touched: false });
  editor.text = "after the undo";
  view.runChanged(editor, ref);
  await view.savingRuns;
  assert.deepEqual(run.changes.map((c) => c.state), ["applied", "undone"]);
  assert.equal(run.hash_after, sha256Hex("after the undo"));
  assert.deepEqual(saved.at(-1).turns[1].turn.edit_run.changes.map((c) => c.state), ["applied", "undone"]);
  assert.equal(historyText(view.turns[1]), "[Changed the note: I'll rename the heading and tighten the intro.; steps: Rename the heading; Tighten the intro; 2 changes, 1 undone]");
});

test("a new note from a skill is created at once, and its ↺ moves it to Obsidian's trash", async () => {
  const { live, editor } = fakeLive();
  const files = {};
  const trashed = [];
  const vault = {
    adapter: { exists: async (p) => p in files },
    getAbstractFileByPath: (p) => (p in files ? { path: p } : null),
    create: async (p, text) => { files[p] = text; return { path: p }; },
    trash: async (file, system) => { trashed.push([file.path, system]); delete files[file.path]; },
  };
  files["Notes/taken.md"] = "theirs";
  const view = editView({ live, editor, vault, events: async (_body, handlers) => {
    handlers.edit({ before_ref: "8".repeat(32), hash_before: "h0" });
    handlers.plan({ intro: "", steps: [] });
    handlers.change({ id: "c1", step: 1, op: "create", label: "Created Notes/brief.md", needs: [], path: "Notes/brief.md", text: "# Brief\n" });
    handlers.change({ id: "c2", step: 1, op: "create", label: "Created Notes/taken.md", needs: [], path: "Notes/taken.md", text: "mine" });
    handlers.turn({ answer: "Done", mode: "quick", citations: [], edit_run: { plan: { intro: "", steps: [] }, steps_done: 0,
      changes: [], stopped: null, summary: "Done: 2 changes", before_ref: "8".repeat(32) } });
  } });
  await view.send("/brief");
  assert.equal(files["Notes/brief.md"], "# Brief\n");
  assert.equal(files["Notes/taken.md"], "theirs");                        // never written over
  assert.equal(live.calls.some((c) => c[0] === "apply"), false);          // a new note is not an editor change
  const run = view.turns.at(-1).turn.edit_run;
  assert.deepEqual(run.changes.map(({ path, state }) => [path, state]), [["Notes/brief.md", "created"], ["Notes/taken.md", "failed"]]);
  await view.trashCreated(view.turns.at(-1), run.changes[0]);
  assert.deepEqual(trashed, [["Notes/brief.md", false]]);
  assert.equal(run.changes[0].state, "trashed");
});

test("history names an edit turn's plan and what became of its changes", () => {
  const turn = (run) => ({ role: "slim", text: "x", turn: { edit_run: run } });
  assert.equal(historyText(turn({ plan: { intro: "Rename it.", steps: [] }, changes: [{ state: "applied" }] })),
               "[Changed the note: Rename it.; 1 change, 0 undone]");
  assert.equal(historyText(turn({ plan: PLAN, steps_done: 1, stopped: "stopped",
                                  changes: [{ state: "applied" }, { state: "skipped" }] })),
               "[Changed the note: I'll rename the heading and tighten the intro.; steps: Rename the heading; Tighten the intro; 1 change, 0 undone; stopped after 1 of 2 steps]");
  assert.equal(historyText(turn({ plan: PLAN, restored: true, changes: [{ state: "applied" }, { state: "applied" }] })),
               "[Changed the note: I'll rename the heading and tighten the intro.; steps: Rename the heading; Tighten the intro; 2 changes, 2 undone]");
  assert.equal(historyText({ role: "slim", text: "An answer." }), "An answer.");
  assert.equal(historyText({ role: "you", text: "q" }), "q");
});

test("the outcome says done, or how far it got and why", () => {
  const run = (extra) => ({ plan: PLAN, changes: [{ state: "applied" }], ...extra });
  assert.equal(editOutcome(run({ stopped: null, summary: PLAN.intro })), "Done.");
  assert.equal(editOutcome(run({ stopped: null, summary: "The note now follows the five parts." })), "Done: The note now follows the five parts.");
  assert.equal(editOutcome(run({ stopped: null, summary: "Done: 3 changes" })), "Done: 3 changes");
  assert.equal(editOutcome(run({ stopped: "call limit", steps_done: 1, step_states: ["partial"] })),
               "Stopped after 1 of 2 steps: it reached its limit of calls. Not done: the rest of Rename the heading and Tighten the intro.");
  assert.equal(editOutcome({ plan: PLAN, changes: [], stopped: "stopped" }), "Stopped before changing anything.");
});

test("no regenerate after an edit turn: asking again would apply its changes twice", async () => {
  const view = makeView();
  view.render = () => {};
  view.turns = [{ role: "you", text: "q" }, { role: "slim", text: "x", turn: { edit_run: { changes: [{ state: "applied" }] } } }];
  let streamed = false;
  view.plugin.streamCopilot = async () => { streamed = true; };
  view.activeThread = { id: "t" };
  await view.regenerate();
  assert.equal(streamed, false);
  assert.equal(view.turns.length, 2);
});

test("words keep math, bold, code and links whole, so a mark never splits their syntax", () => {
  assert.deepEqual(wordTokens("a $x + y$ **b c** `d e` [[f g]] h"),
    ["a ", "$x + y$ ", "**b c** ", "`d e` ", "[[f g]] ", "h"]);
  assert.deepEqual(wordOps("For a vector, the softmax is:", "For a vector:"), [
    ["=", "For ", "For "], ["=", "a ", "a "], ["-", "vector, "], ["-", "the "], ["-", "softmax "],
    ["-", "is:"], ["+", "vector:"]]);
});

test("a change gets word marks only when its lines keep their kind and half their words", () => {
  assert.equal(markable("## Properties\n", "## Key Properties\n"), true);
  assert.equal(markable("For a vector $z$ of $K$ real numbers, defined as:\n", "For a vector $z$ of $K$ real numbers:\n"), true);
  assert.equal(markable("1. **Non-negativity**: positive.\n", "*   **Output**: positive.\n"), false);   // marker changed
  assert.equal(markable("one two three four\n", "five six seven eight\n"), false);                      // rewritten
  assert.equal(markable("- Syntactic ambiguity: scope.\n", "- **Syntactic ambiguity**: scope.\n"), true); // every word kept
  assert.equal(markable("$$ a + b $$\n", "$$ a + c $$\n"), false);                                     // math stays whole
  assert.equal(markable("a\nb\n", "a\n"), false);
  assert.equal(markable("", "new\n"), false);
});

test("a change that keeps every word is named for the formatting it moves", () => {
  assert.equal(formatChange("## Why is NLP Hard?\n", "### Why is NLP Hard?\n"), "Heading level 2 → 3");
  assert.equal(formatChange("- Lexical ambiguity: many senses.\n- Syntactic ambiguity: scope.\n",
                            "- **Lexical ambiguity**: many senses.\n- **Syntactic ambiguity**: scope.\n"), "Bold added");
  assert.equal(formatChange("- Open vocabulary: new words.\n", "- **Open vocabulary**: new words.\n\n"), "Bold added");
  assert.equal(formatChange("**Note**: x\n", "Note: x\n"), "Bold removed");
  assert.equal(formatChange("a\nb\n", "a\n\nb\n"), "Spacing only");
  assert.equal(formatChange("- a\n", "  - a\n"), "Indentation changed");
  assert.equal(formatChange("- a\n", "1. a\n"), "Formatting only");
  // Words that change, math that changes, and additions and removals are not formatting.
  assert.equal(formatChange("## Properties\n", "## Key Properties\n"), "");
  assert.equal(formatChange("$w_1$\n", "$w1$\n"), "");
  assert.equal(formatChange("", "new\n"), "");
  assert.equal(formatChange("old\n", ""), "");
});

// ---- the live editor's pure parts (the editor itself runs in the real-CodeMirror harness) ----

test("a change's marks: added words, removed words where they were, formatting, and folds", () => {
  const [words] = changeMarks("replace", [{ old: "The quick brown fox jumps.\n\n", now: "The quick red fox jumps.\n\n" }]);
  const now = "The quick red fox jumps.\n\n";
  assert.deepEqual(words.rail, [0, now.length]);
  assert.deepEqual(words.ins.map(([f, t]) => now.slice(f, t)), ["red"]);
  assert.deepEqual(words.dels, [[now.indexOf("red"), "brown"]]);         // struck before what replaced it
  assert.equal(words.fold, null);
  const [format] = changeMarks("replace", [{ old: "## Why\n", now: "### Why\n" }]);
  assert.deepEqual([format.formats, format.ins, format.dels], [[[0, 7]], [], []]);
  const [rewrite] = changeMarks("replace", [{ old: "one two three four\n", now: "five six seven eight\n" }]);
  assert.deepEqual(rewrite.fold, { text: "one two three four\n", moved: false });  // old text folds above the new
  const [gone, landed] = changeMarks("move", [{ old: "## A\n\nx\n\n", now: "" }, { old: "", now: "## A\n\nx\n\n" }]);
  assert.deepEqual([gone.rail, gone.fold.moved, landed.fold, landed.rail], [null, true, null, [0, 9]]);
});

test("a word fixed with edit_text and a heading renamed are marked word by word, not folded", () => {
  const [word] = changeMarks("edit_text", [{ old: "Fine.\n", now: "Good.\n" }]);
  assert.deepEqual([word.fold, word.dels, word.ins], [null, [[0, "Fine."]], [[0, 5]]]);
  const [renamed] = changeMarks("replace", [{ old: "## Middle\n", now: "## Centre\n" }]);
  assert.deepEqual([renamed.fold, renamed.dels, renamed.ins], [null, [[3, "Middle"]], [[3, 9]]]);
  // A paragraph rewritten whole still folds above the new one, and math stays whole.
  assert.equal(changeMarks("replace", [{ old: "Fine.\n", now: "Good.\n" }])[0].fold.text, "Fine.\n");
  assert.equal(changeMarks("edit_text", [{ old: "$$ a + b $$\n", now: "$$ a + c $$\n" }])[0].fold.text, "$$ a + b $$\n");
  // A rename that also changes the level keeps its words and marks only that.
  assert.equal(changeMarks("replace", [{ old: "## Middle\n", now: "### Centre\n" }])[0].fold.text, "## Middle\n");
});

test("the change between two texts is one replace from the common prefix to the common suffix", () => {
  assert.deepEqual(textDiff("abc\ndef\n", "abc\nXYZ\ndef\n"), [{ from: 4, to: 4, insert: "XYZ\n" }]);
  assert.deepEqual(textDiff("same", "same"), []);
  assert.deepEqual(textDiff("🎙️ a", "🎙️ b"), [{ from: 4, to: 5, insert: "b" }]);   // UTF-16, as the editor counts
});

test("a folded removal shows its first line without markers", () => {
  assert.equal(excerpt("\n## **Key** Ideas\n\nbody"), "Key Ideas");
  assert.equal(excerpt("- `item` one\n"), "item one");
});

test("See before marks what is gone, what changed, what moved and where something was added", () => {
  const before = "# T\n\nThe quick brown fox.\n\n## Gone\n\nM block.\n\nKeep one.\n\nKeep two.\n";
  const now = "# T\n\nThe quick red fox.\n\nA brand new paragraph.\n\nKeep one.\n\nKeep two.\n\nM block.\n";
  const blocks = beforeBlocks(before, now);
  assert.deepEqual(blocks.map((b) => b.kind), ["same", "changed", "removed", "moved", "added", "same", "same"]);
  assert.equal(blocks[1].markdown, "The quick ~~brown~~ fox.");
  assert.equal(blocks[4].markdown, "A brand new paragraph.");
  // A fence with blank lines inside is one block.
  assert.deepEqual(beforeBlocks("```\na\n\nb\n```\n", "```\na\n\nb\n```\n").map((b) => b.kind), ["same"]);
});

const SlimRecorderPlugin = createRequire(import.meta.url)(path.join(HERE, "main.js"));

test("a run lands in the editor focused last that shows the note, and is found again by its id", () => {
  const plugin = new SlimRecorderPlugin();
  const editor = (path, at, runs = []) => ({ path, at, runs });
  const older = editor("Notes/a.md", 1), newer = editor("Notes/a.md", 2, ["run-1"]), other = editor("Notes/b.md", 3);
  plugin.liveViews = new Set([older, newer, other]);
  plugin.live = { pathOf: (view) => view.path, focusedAt: (view) => view.at,
                  state: (view, runId) => (view.runs.includes(runId) ? {} : null) };
  assert.equal(plugin.liveViewFor("Notes/a.md"), newer);
  assert.equal(plugin.liveViewFor("Notes/none.md"), null);
  assert.equal(plugin.liveRunView("run-1"), newer);
  assert.equal(plugin.liveRunView("run-2"), null);
});

