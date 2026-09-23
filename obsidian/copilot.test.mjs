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
        groupCitations, noteTitle, normalizeMath, SUGGESTED_PROMPTS, metaText } =
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

test("text-only history carries role and text only, last twelve turns", async () => {
  // Each stored slim turn carries its evidence pack and stats; re-uploading all of it on
  // every question grew with the square of the thread, and the server keeps 12 turns anyway.
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
  view.turns = Array.from({ length: 15 }, (_v, i) => ({
    role: i % 2 ? "slim" : "you", text: `t${i}`, turn: { evidence: ["big"], citations: [] },
  }));
  await view.send("Q");
  assert.equal(history.length, 12);
  assert.deepEqual(Object.keys(history[0]), ["role", "text"]);
  assert.equal(history.at(-1).text, "t14");
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
  assert.equal(input.placeholder, "Ask SLIM");
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
