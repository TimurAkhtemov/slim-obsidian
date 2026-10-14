// The copilot sidebar: the active note, send, cancel, retry, threads, images, rename.
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { find, obsidianStub, srcText } from "./fakes.mjs";
import { makeView, renderedView, text } from "./copilot-harness.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { metaText } = require("../src/chat-text.js");
const { RenameModal } = require("../src/chat-view.js");

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
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
  const source = srcText("plugin.js");
  assert.match(source, /registerView\(VIEW_TYPE_COPILOT/);
  assert.match(source, /active-leaf-change/);
  assert.match(source, /activateCopilot/);
  const manifest = JSON.parse(fs.readFileSync(path.join(HERE, "..", "manifest.json"), "utf8"));
  assert.equal(manifest.name, "SLIM");
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
