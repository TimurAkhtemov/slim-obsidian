// The copilot sidebar: the active note, send, cancel, retry, threads, images, rename.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { find, focusState, obsidianStub } from "../fakes.mjs";
import { makeView, renderedView, text } from "../copilot-harness.mjs";

const require = createRequire(import.meta.url);
const { metaText } = require("../../src/chat/text.js");
const { RenameModal } = require("../../src/chat/view.js");

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
  assert.equal(view.openNote.id, "s2");
  assert.equal(view.context.source.id, "s2");
});

test("a canvas or other non-Markdown file clears the open note without an HTTP call", async () => {
  let called = false;
  const view = makeView({ postJSON: async () => { called = true; } });
  view.render = () => {};
  await view.setActiveFile({ path: "Board.canvas", extension: "canvas" });
  assert.equal(called, false);
  assert.equal(view.openNote, null);
  assert.equal(view.error, "");
});

test("a note the server will not index still opens, and no error is put on the chat", async () => {
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
  assert.equal(view.openNote.path, "Skills/brief.md");
  assert.equal(view.error, "");
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

test("sending streams deltas and stores the turn with the effort they picked", async () => {
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
      handlers.turn({ answer: "Grounded answer", effort: "high", citations: [] });
    },
  });
  view.render = () => {};
  view.openNote = { path: "Notes/one.md", id: "source-1" };
  view.effort = "high";
  await view.send("Compare these");
  assert.equal(view.turns.at(-1).text, "Grounded answer");
  assert.equal(view.turns.at(-1).turn.effort, "high");
  assert.equal(saved.length, 1);
  assert.equal(saved[0].source_id, "source-1");
  assert.equal(saved[0].turns[0].note, "Notes/one.md");
});

test("history sent with a question names the note each earlier one was on and what its answer looked at", async () => {
  const bodies = [];
  const view = makeView({
    postJSON: async (route) => route.endsWith("context") ? { source: { id: "source-1", path: "Notes/one.md" } } : {},
    streamCopilot: async (body, handlers) => {
      bodies.push(body);
      handlers.turn({ answer: `Answer ${bodies.length}`, mode: "quick",
                      citations: [{ n: 1, path: "Notes/two.md", title: "two" }] });
    },
  });
  view.render = () => {};
  view.openNote = { path: "Notes/one.md", id: "source-1" };
  await view.send("What is A?");
  await view.send("And B?");
  assert.deepEqual(bodies[1].history.map((turn) => turn.text),
                   ["(on Notes/one.md) What is A?", "Answer 1\nlooked at: Notes/two.md"]);
});

test("a new model call in the turn clears what streamed before it", async () => {
  const view = makeView({
    postJSON: async (route) => route.endsWith("context") ? { source: { id: "source-1", path: "Notes/one.md" } } : {},
    streamCopilot: async (_body, handlers) => {
      handlers.delta({ text: "Let me search [b12]…" });
      handlers.delta_reset({});
      assert.equal(view.streamingText, "");
      handlers.delta({ text: "Found " });
      handlers.delta({ text: "it." });
      assert.equal(view.streamingText, "Found it.");
      handlers.turn({ mode: "quick", citations: [] });
    },
  });
  view.render = () => {};
  await view.send("Where is it?");
  assert.equal(view.turns.at(-1).text, "Found it.");      // a turn with no answer keeps what streamed last
});

test("cancel aborts the in-flight stream; the turn ends, and the composer goes idle, when the stream does", () => {
  const view = makeView();
  view.render = () => {};
  let aborted = false;
  view.abort = { abort() { aborted = true; } };
  view.sending = true;
  view.cancel();
  assert.equal(aborted, true);
  assert.equal(view.sending, true);                                 // send()'s finally ends it
  assert.equal(view.stage, "Stopping");
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
  assert.equal(view.turns.at(-2).note, undefined);           // asked with no note open: no "(on …)"
});

test("opening an old thread does not navigate", async () => {
  const opened = [];
  const view = makeView({
    app: {
      workspace: { openLinkText(...args) { opened.push(args); }, getLeaf: () => ({ async openFile(file) { opened.push(file.path); } }) },
      vault: { getAbstractFileByPath: (path) => ({ path }) },
    },
    getJSON: async () => ({
      id: "other", source_id: "s1", source_path: "A.md", reasoning_mode: "quick", pins: [{ kind: "note", path: "C.md" }],
      turns: [{ role: "you", text: "Q" }, { role: "slim", text: "A" }],
    }),
  });
  view.render = () => {};
  view.openNote = { path: "B.md", id: "s2" };
  await view.openThread("other");
  assert.deepEqual(opened, []);
  assert.equal(view.activeThread.id, "other");
  assert.deepEqual(view.turns.map((turn) => turn.text), ["Q", "A"]);
  assert.deepEqual(view.openNote, { path: "B.md", id: "s2" });
});

test("an old thread opens and renames without touching the effort", async () => {
  // A chat saved before effort says "quick", "deep" or even "auto" on disk. A chat no longer has
  // an effort of its own: the selector stays as chosen, and a save posts none.
  const saves = [];
  const view = makeView({
    settings: { effort: "low" }, saveSettings: async () => {},
    getJSON: async () => ({
      id: "old", title: "Old", source_id: "source-1", source_path: "Notes/one.md",
      reasoning_mode: "auto", turns: [{ role: "you", text: "Keep me" }],
    }),
    postJSON: async (_route, body) => { saves.push(body); return body; },
  });
  view.render = () => {};
  view.promptTitle = async () => "Renamed";
  view.context = { source: { id: "source-1", path: "Notes/one.md" }, related: [] };
  await view.openThread("old");
  assert.equal(view.effort, "low");
  await view.renameThread("old");
  assert.equal(saves.length, 1);
  assert.equal(saves[0].title, "Renamed");
  assert.equal("reasoning_mode" in saves[0] || "effort" in saves[0], false);
});

test("the effort menu offers Pi's four levels with a hint each, starts at the saved one and keeps the last chosen", async () => {
  const settings = { port: 7546, effort: "high" };
  let writes = 0;
  const bodies = [];
  const { view, root } = renderedView({
    settings, saveSettings: async () => { writes += 1; },
    streamCopilot: async (body, handlers) => { bodies.push(body); handlers.turn({ answer: "ok", effort: body.effort, citations: [] }); },
  });
  view.render();
  const button = () => find(root, (node) => node.cls === "slim-effort");
  const menu = () => find(root, (node) => node.cls === "slim-effort-menu");
  assert.equal(find(button(), (node) => node.cls === "slim-effort-label").textContent, "High");
  assert.equal(menu().hidden, true);
  button().onclick();
  assert.equal(menu().hidden, false);
  assert.equal(button().attrs["aria-expanded"], "true");
  const items = menu().children;
  assert.deepEqual(items.map((item) => item.children.map((part) => part.textContent)),
                   [["Off", "fastest"], ["Low", "quick thinking"], ["Medium", "default"], ["High", "most thorough"]]);
  assert.deepEqual(items.filter((item) => item.cls.includes("is-selected")).map((item) => item.children[0].textContent), ["High"]);
  items[0].onclick();
  assert.equal(menu().hidden, true);                                // picking closes the menu
  assert.equal(find(button(), (node) => node.cls === "slim-effort-label").textContent, "Off");
  assert.equal(view.effort, "off");
  assert.equal(settings.effort, "off");
  assert.equal(writes, 1);
  await view.send("Quick one");
  assert.equal(bodies[0].effort, "off");
  assert.equal("reasoning_mode" in bodies[0], false);
  assert.equal(view.turns.at(-1).turn.effort, "off");
});

test("with no effort saved, or one that is not a level, the selector starts at Medium", () => {
  assert.equal(makeView().effort, "medium");
  assert.equal(makeView({ settings: {} }).effort, "medium");
  assert.equal(makeView({ settings: { effort: "deep" } }).effort, "medium");
  assert.equal(makeView({ settings: { effort: "low" } }).effort, "low");
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
  view.openNote = { path: file.path, id: "s1" };
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
      assert.equal(body.path, "Notes/one.md");
      handlers.turn({ answer: "ok", mode: "quick", citations: [] });
    },
  });
  view.render = () => {};
  view.app.workspace.getActiveFile = () => ({ path: "Notes/one.md", extension: "md" });
  view.openNote = { path: "Notes/one.md", id: "s1" };
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
    { role: "slim", text: "One", turn: { mode: "deep", timing: { total_ms: 42000 } } },      // saved as Deep
    { role: "you", text: "Explain", attachments: [ref] },
    { role: "slim", text: "Two", turn: { mode: "quick", timing: { total_ms: 8200 } } },
  ];
  view.render();
  const metas = [];
  find(root, (node) => { if (node.cls === "slim-message-meta") metas.push(node.textContent); return false; });
  assert.deepEqual(metas, ["High · 42 s", "Medium · 8 s"]);
  const regenerates = [];
  find(root, (node) => { if (node.attrs["aria-label"] === "Regenerate") regenerates.push(node); return false; });
  assert.equal(regenerates.length, 1);                                // the last answer only
  assert.equal(find(root, (node) => node.attrs["aria-label"] === "Copy answer") !== null, true);
  await view.regenerate();
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].question, "Explain");
  assert.deepEqual(bodies[0].images, [ref]);                          // refs, no bytes
  assert.deepEqual(view.turns.map((turn) => turn.text), ["First?", "One", "Explain", "Second"]);
  assert.equal(metaText({ mode: "quick" }), "Medium");
  assert.equal(metaText({ effort: "low", mode: "deep", timing: { total_ms: 46000 } }), "Low · 46 s");
  assert.equal(metaText({ effort: "off" }), "Off");
  assert.equal(metaText({}), "Medium");
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
  await view.openFile("Notes/gone.md");
  assert.equal(linkText, 0);
  assert.deepEqual(opened, []);
  assert.match(view.error, /Notes\/gone\.md/);
  await view.openFile("Notes/there.md");
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

test("slash menu navigates with arrows, dismisses on Escape, and skips broken skills", () => {
  const { view, root } = renderedView();
  view.skills = [
    { name: "broken", error: "missing prompt" },
    { name: "actions", description: "Extract action items" },
    { name: "quiz", description: "Quiz me" },
  ];
  view.draft = "/";
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  const key = (k) => input.onkeydown({
    key: k, shiftKey: false, metaKey: false, altKey: false, ctrlKey: false,
    preventDefault() {},
  });
  // First item is broken; Tab when index 0 is selected falls back to the first valid skill ("actions"),
  // or ArrowDown selects "actions" then "quiz".
  key("ArrowDown");
  key("ArrowDown");
  assert.equal(view.slashIndex, 2);
  key("Escape");
  const menu = find(root, (node) => node.cls === "slim-slash-menu");
  assert.equal(menu.hidden, true);
  // Typing again reopens the menu, and Tab picks the selected valid skill.
  input.value = "/q";
  input.oninput();
  assert.equal(menu.hidden, false);
  key("Tab");
  assert.equal(view.draft, "/quiz ");
});

test("Enter sends a skill named in full, or picks the one the arrows chose", () => {
  const { view, root } = renderedView();
  view.skills = [{ name: "quiz", description: "Quiz me" }, { name: "quiz-hard", description: "Harder" }];
  const sent = [];
  view.send = () => sent.push(view.draft);
  view.draft = "/quiz";
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  const key = (k) => input.onkeydown({
    key: k, shiftKey: false, metaKey: false, altKey: false, ctrlKey: false,
    preventDefault() {},
  });
  key("Enter");
  assert.deepEqual(sent, ["/quiz"]);
  key("ArrowDown");
  key("Enter");
  assert.equal(view.draft, "/quiz-hard ");
  assert.deepEqual(sent, ["/quiz"]);
});

// ---- the sidebar belongs to the owner, not to a note ------------------------------------------

function chatView(extra = {}) {
  const bodies = [];
  const saved = [];
  const view = makeView({
    postJSON: async (route, body) => {
      if (route.endsWith("save")) { saved.push(structuredClone(body)); return {}; }
      if (route.endsWith("context")) return { source: { id: `id-${body.path}`, path: body.path }, related: [] };
      return { threads: [] };
    },
    streamCopilot: async (body, handlers) => {
      bodies.push(body);
      handlers.turn({ answer: "ok", mode: "quick", citations: [] });
    },
    ...extra,
  });
  view.render = () => {};
  return { view, bodies, saved };
}

const noteFile = (path) => ({ path, extension: "md" });

test("switching notes keeps the chat", async () => {
  const { view, bodies } = chatView();
  await view.setActiveFile(noteFile("A.md"));
  await view.send("About A");
  const thread = view.activeThread;
  const turns = view.turns;
  await view.setActiveFile(noteFile("B.md"));
  assert.equal(view.activeThread, thread);
  assert.equal(view.turns, turns);
  assert.equal(view.turns.length, 2);
  await view.send("About B");
  assert.equal(bodies[0].path, "A.md");
  assert.equal(bodies[1].path, "B.md");
  assert.equal(bodies[1].thread_id, thread.id);
  assert.deepEqual(bodies[1].pins, []);
  assert.equal(view.turns[0].note, "A.md");
  assert.equal(view.turns[2].note, "B.md");
});

test("a note switch leaves an error and a streaming answer alone", async () => {
  const gate = deferred();
  const { view } = chatView({
    streamCopilot: async (_body, handlers) => { handlers.delta({ text: "part" }); await gate.promise; handlers.turn({ answer: "part", mode: "quick", citations: [] }); },
  });
  await view.setActiveFile(noteFile("A.md"));
  const sending = view.send("Q");
  await new Promise((resolve) => setImmediate(resolve));
  view.error = "kept";
  await view.setActiveFile(noteFile("B.md"));
  assert.equal(view.sending, true);
  assert.equal(view.streamingText, "part");
  assert.equal(view.error, "kept");
  gate.resolve();
  await sending;
});

test("switching notes mid-answer keeps the answer, saved to the thread that asked", async () => {
  const gate = deferred();
  const { view, saved } = chatView({
    streamCopilot: async (_body, handlers) => { handlers.delta({ text: "Half" }); await gate.promise; handlers.turn({ answer: "Whole", mode: "quick", citations: [] }); },
  });
  await view.setActiveFile(noteFile("A.md"));
  const sending = view.send("Q");
  await new Promise((resolve) => setImmediate(resolve));
  const asked = view.activeThread;
  await view.setActiveFile(noteFile("B.md"));
  gate.resolve();
  await sending;
  assert.equal(saved.length, 1);
  assert.equal(saved[0].id, asked.id);
  assert.deepEqual(saved[0].turns.map((turn) => turn.text), ["Q", "Whole"]);
  assert.equal(view.turns.at(-1).text, "Whole");
});

test("no note open still sends, with path null", async () => {
  const { view, bodies, saved } = chatView();
  await view.setActiveFile(null);
  await view.send("Anything");
  assert.equal(bodies[0].path, null);
  assert.equal("note" in view.turns[0], false);
  assert.equal(saved.length, 1);
  assert.equal("source_id" in saved[0], false);
  assert.equal(view.error, "");
});

test("a new chat takes the open note's id and path when there is one", async () => {
  const { view } = chatView();
  view.newThread();
  assert.equal(view.activeThread.source_path, undefined);
  await view.setActiveFile(noteFile("A.md"));
  view.newThread();
  assert.deepEqual([view.activeThread.source_id, view.activeThread.source_path], ["id-A.md", "A.md"]);
});

test("history is flat and searchable", async () => {
  const urls = [];
  const all = [
    { id: "t1", title: "One", source_path: "Notes/a.md", updated_at: "2026-01-01T00:00:00Z" },
    { id: "t3", title: "Three", source_path: null, updated_at: "2026-03-01T00:00:00Z" },
    { id: "t2", title: "Two", source_path: "Notes/b.md", updated_at: "2026-02-01T00:00:00Z" },
  ];
  const { view } = renderedView({
    getJSON: async (url) => { urls.push(url); return { threads: url.includes("q=") ? [all[0]] : all }; },
  });
  await view.loadThreads();
  view.showThreads = true;
  view.render();
  const root = view.contentEl;
  const rows = () => { const out = []; find(root, (n) => { if (n.cls.startsWith("slim-chat-row")) out.push(n); return false; }); return out; };
  const titles = () => rows().map((row) => find(row, (n) => n.cls === "slim-chat-title").textContent);
  assert.deepEqual(titles(), ["Three", "Two", "One"]);
  const faint = find(root, (n) => n.cls === "slim-chat-note");
  assert.equal(faint.textContent.trim(), "b");
  const search = find(root, (n) => n.tag === "input" && n.cls === "slim-thread-search");
  search.value = "ab"; search.oninput();
  search.value = "abc"; search.oninput();
  assert.deepEqual(urls, ["/api/copilot/threads"]);              // debounced: nothing yet
  await new Promise((resolve) => setTimeout(resolve, 260));
  assert.deepEqual(urls, ["/api/copilot/threads", "/api/copilot/threads?q=abc"]);
  assert.equal(rows().length, 1);
  assert.equal(find(root, (n) => n.cls === "slim-thread-search"), search);   // the box survives the redraw
});

function refusing(status, detail) {
  return async (route, body) => {
    if (route.endsWith("context")) { const e = new Error(`${route} returned ${status}: ${detail}`); e.status = status; e.detail = detail; throw e; }
    return route.endsWith("save") ? {} : { threads: [] };
  };
}

test("a note the server does not index still sends its path, with no error", async () => {
  const bodies = [];
  const view = makeView({
    postJSON: refusing(400, "SLIM does not index this note"),
    streamCopilot: async (body, handlers) => { bodies.push(body); handlers.turn({ answer: "ok", mode: "quick", citations: [] }); },
  });
  view.render = () => {};
  await view.setActiveFile(noteFile("Skills/brief.md"));
  await view.send("What does this do?");
  assert.equal(bodies[0].path, "Skills/brief.md");
  assert.equal(view.error, "");
  assert.equal(view.turns.at(-1).text, "ok");
});

test("a server failure while syncing the note surfaces in the server's sentence", async () => {
  const view = makeView({ postJSON: refusing(500, "index is busy") });
  view.render = () => {};
  view.openNote = { path: "A.md", id: null };
  await view.send("Q");
  assert.equal(view.error, "index is busy");
});

test("opening the sidebar reopens the most recently updated chat and navigates nowhere", async () => {
  let linked = 0;
  const threads = [
    { id: "old", updated_at: "2026-01-01" }, { id: "new", updated_at: "2026-02-01" },
  ];
  const view = makeView({
    getJSON: async (url) => url.startsWith("/api/copilot/threads") ? { threads }
      : url.startsWith("/api/copilot/thread?") ? { id: url.split("=")[1], title: url.split("=")[1], reasoning_mode: "quick",
          turns: [{ role: "you", text: `Q ${url.split("=")[1]}` }] } : { skills: [] },
  });
  view.app.workspace.openLinkText = () => { linked += 1; };
  view.app.workspace.getActiveFile = () => null;
  view.render = () => {};
  await view.onOpen();
  assert.equal(view.activeThread.id, "new");
  assert.deepEqual(view.turns.map((t) => t.text), ["Q new"]);
  assert.equal(linked, 0);
});

// --- @ pins and the open-note chip ---------------------------------------------------------

function pinView(extra = {}) {
  const bodies = [], saved = [];
  const view = makeView({
    postJSON: async (route, body) => { if (route.endsWith("save")) { saved.push(body); return body; } return { source: { id: "s1", path: "Notes/one.md" } }; },
    streamCopilot: async (body, handlers) => { bodies.push(body); handlers.turn({ answer: "ok", mode: "quick", citations: [] }); },
    ...extra,
  });
  view.app.vault = {
    getMarkdownFiles: () => [{ path: "Notes/x.md" }, { path: ".trash/old.md" }],
    getAllLoadedFiles: () => [{ path: "/", children: [] }, { path: "Notes", children: [] }, { path: ".obsidian", children: [] }, { path: "Notes/x.md" }],
  };
  view.render = () => {};
  view.openNote = { path: "Notes/one.md", id: "s1" };
  return { view, bodies, saved };
}

test("picking an @ match pins it, clears the @query, and the next send carries the pin", async () => {
  const { view, bodies } = pinView();
  view.draft = "ask @wee";
  view.pin({ kind: "note", path: "Notes/x.md" });
  assert.equal(view.draft, "ask ");
  view.pin({ kind: "note", path: "Notes/x.md" });                 // no duplicate
  assert.deepEqual(view.activeThread.pins, [{ kind: "note", path: "Notes/x.md" }]);
  await view.send("what is it");
  assert.deepEqual(bodies[0].pins, [{ kind: "note", path: "Notes/x.md" }]);
});

test("the picker lists notes and folders, never the root or hidden folders", () => {
  const { view } = pinView();
  view.draft = "@";
  assert.deepEqual(view.mentionHits(), [{ kind: "folder", path: "Notes" }, { kind: "note", path: "Notes/x.md" }]);
});

test("a chat holds at most 12 pins; the 13th is refused with a sentence", () => {
  const { view } = pinView();
  for (let i = 0; i < 13; i++) { view.draft = "@"; view.pin({ kind: "note", path: `n${i}.md` }); }
  assert.equal(view.activeThread.pins.length, 12);
  assert.match(view.error, /at most 12/);
});

test("unpinning removes the chip and saves the thread without it", async () => {
  const { view, saved } = pinView();
  view.newThread();
  view.activeThread.pins = [{ kind: "folder", path: "Capture" }, { kind: "note", path: "Notes/x.md" }];
  view.turns = [{ role: "you", text: "q" }, { role: "slim", text: "a" }];
  await view.unpin("Capture");
  assert.deepEqual(view.activeThread.pins, [{ kind: "note", path: "Notes/x.md" }]);
  assert.deepEqual(saved.at(-1).pins, [{ kind: "note", path: "Notes/x.md" }]);
});

test("the open-note chip's ✕ leaves the note out (path and doc) until clicked again", async () => {
  const { view, bodies } = pinView();
  view.plugin.liveViewFor = () => ({ state: { doc: { toString: () => "# one" } } });
  view.toggleOpenNote();
  await view.send("first");
  assert.equal(bodies[0].path, null);
  assert.equal("doc" in bodies[0], false);
  view.toggleOpenNote();
  await view.send("second");
  assert.equal(bodies[1].path, "Notes/one.md");
  assert.equal(bodies[1].doc, "# one");
});

test("the sidebar draws pin chips with a remove button and an ✕ on the open-note chip", () => {
  const { view, root } = renderedView();
  view.newThread();
  view.activeThread.pins = [{ kind: "note", path: "Notes/x.md" }];
  view.render();
  const chip = find(root, (n) => n.cls?.includes("slim-pin "));
  assert.ok(chip);
  find(chip, (n) => n.cls === "slim-chip-x").onclick();
  assert.deepEqual(view.activeThread.pins, []);
  find(root, (n) => n.cls?.includes("slim-copilot-note-off")).onclick({ preventDefault() {}, stopPropagation() {} });
  assert.equal(view.activeThread.openNoteOff, true);
});

// --- the pickers: the arrows move the highlight and keep it in view ------------------------------

function pickerView() {
  const { view, root } = renderedView();
  view.app.vault = {
    getMarkdownFiles: () => [{ path: "Notes/school/Graphs and shortest paths.md" }, { path: "Notes/x.md" },
                             { path: "top.md" }],
    getAllLoadedFiles: () => [{ path: "Notes", children: [] }, { path: "Notes/school", children: [] }],
  };
  const sent = [];
  view.send = () => sent.push(view.draft);
  const input = () => find(root, (node) => node.tag === "textarea");
  const key = (k) => input().onkeydown({ key: k, shiftKey: false, metaKey: false, altKey: false, ctrlKey: false,
                                         preventDefault() {} });
  const type = (text) => { input().value = text; input().selectionStart = text.length; input().oninput(); };
  const rows = (cls) => {
    const found = [];
    find(root, (node) => { if (node.cls?.split(" ").includes(cls)) found.push(node); return false; });
    return found;
  };
  view.render();
  return { view, root, sent, key, type, rows };
}

test("the @ picker: arrows move the highlight into view; Enter and Tab pick; Escape closes; Enter never sends", () => {
  const { view, sent, key, type, rows } = pickerView();
  type("about @");
  const items = () => rows("slim-mention-item");
  assert.equal(items().length, 5);
  assert.ok(items()[0].cls.includes("is-selected"));
  key("ArrowDown");
  assert.equal(view.mentionIndex, 1);
  const selected = items().find((row) => row.cls.includes("is-selected"));
  assert.equal(selected, items()[1]);
  assert.deepEqual(selected.scrolledIntoView, [{ block: "nearest" }]);
  key("ArrowUp");
  assert.deepEqual(items()[0].scrolledIntoView, [{ block: "nearest" }]);
  key("Enter");                                                   // picks, never sends
  assert.deepEqual(sent, []);
  assert.deepEqual(view.activeThread.pins, [{ kind: "folder", path: "Notes" }]);
  type("about @");
  key("ArrowDown");
  key("Tab");
  assert.deepEqual(view.activeThread.pins.map((pin) => pin.path), ["Notes", "Notes/school"]);
  type("about @x");
  key("Escape");
  assert.equal(find(view.contentEl, (node) => node.cls === "slim-mention-menu").hidden, true);
  assert.deepEqual(sent, []);
  key("Enter");                                                   // closed: Enter sends again
  assert.deepEqual(sent, ["about @x"]);
});

test("an @ row is a title or a folder name on the left and its parent folder faint on the right", () => {
  const { type, rows } = pickerView();
  type("@");
  const [notes, school, top, x, graphs] = rows("slim-mention-item");      // folders, then notes, shortest first
  const part = (row, cls) => find(row, (node) => node.cls?.split(" ").includes(cls));
  assert.ok(part(notes, "slim-mention-icon"));                    // a folder has the folder icon
  assert.equal(part(notes, "slim-mention-name").textContent, "Notes");
  assert.equal(part(notes, "slim-mention-path"), null);           // at the root: no parent
  assert.equal(part(school, "slim-mention-name").textContent, "school");
  assert.equal(part(school, "slim-mention-path").textContent, "Notes");
  assert.equal(part(top, "slim-mention-name").textContent, "top");
  assert.equal(part(top, "slim-mention-icon"), null);
  assert.equal(part(x, "slim-mention-name").textContent, "x");
  assert.equal(part(graphs, "slim-mention-name").textContent, "Graphs and shortest paths");
  assert.equal(part(graphs, "slim-mention-path").textContent, "Notes/school");
  assert.equal(graphs.attrs.title, "Notes/school/Graphs and shortest paths.md");
  assert.equal(rows("slim-slash-name").length, 0);                // not the slash menu's monospace
});

test("the / menu: arrows keep the highlighted skill in view", () => {
  const { view, key, type, rows } = pickerView();
  view.skills = [{ name: "actions", description: "a" }, { name: "quiz", description: "q" }];
  type("/");
  view.renderSlash();
  key("ArrowDown");
  const selected = rows("slim-slash-item").find((row) => row.cls.includes("is-selected"));
  assert.equal(view.slashIndex, 1);
  assert.deepEqual(selected.scrolledIntoView, [{ block: "nearest" }]);
});

// --- the composer: one box, its chips, its toolbar ------------------------------------------------

const composerOf = (root) => find(root, (node) => node.cls === "slim-copilot-composer");

test("the @ button puts an @ at the caret and opens the same picker as typing it", () => {
  const { view, root, key, rows } = pickerView();
  const at = () => find(root, (node) => node.attrs["aria-label"] === "Add a note or folder");
  view.draft = "about";
  view.render();
  at().onclick();
  const input = find(root, (node) => node.tag === "textarea");
  assert.equal(input.value, "about @");                             // a space first: an @ touching a word is no mention
  assert.equal(input.focused, true);
  assert.equal(input.selectionStart, 7);
  assert.equal(find(root, (node) => node.cls === "slim-mention-menu").hidden, false);
  assert.equal(rows("slim-mention-item").length, 5);
  key("Enter");                                                     // the picker's keys, as when typed
  assert.deepEqual(view.activeThread.pins, [{ kind: "folder", path: "Notes" }]);
  view.draft = "a b";
  view.caret = 1;
  at().onclick();
  assert.equal(view.draft, "a @ b");                                // at the caret, not the end
  assert.equal(view.mentionHits().length, 5);
});

test("the send button is a stop button while a turn runs, and stopping ends the turn", async () => {
  let started;
  const running = new Promise((resolve) => { started = resolve; });
  const { view, root } = renderedView({
    streamCopilot: (_body, _handlers, signal) => new Promise((_resolve, reject) => {
      started();
      signal.addEventListener("abort", () => reject(Object.assign(new Error("Cancelled"), { name: "AbortError" })));
    }),
  });
  view.render();
  const send = () => find(root, (node) => node.cls?.split(" ").includes("slim-send"));
  assert.equal(send().attrs["aria-label"], "Send");
  assert.ok(send().cls.includes("is-empty"));                       // nothing to send yet
  view.draft = "a long question";
  view.render();
  assert.ok(!send().cls.includes("is-empty"));
  const turn = send().onclick();
  await running;
  assert.equal(send().attrs["aria-label"], "Stop response");
  assert.ok(send().cls.includes("is-stop"));
  send().onclick();                                                 // stop
  await turn;
  assert.equal(view.sending, false);
  assert.equal(send().attrs["aria-label"], "Send");
  assert.equal(view.draft, "a long question");                      // the unanswered question comes back
});

test("the composer's chips are the open note and the @ pins, inside the composer", () => {
  const { view, root } = renderedView();
  view.newThread();
  view.activeThread.pins = [{ kind: "note", path: "Notes/x.md" }, { kind: "folder", path: "Capture/weekly" }];
  view.render();
  const chips = find(composerOf(root), (node) => node.cls === "slim-composer-chips");
  const names = chips.children.map((chip) => find(chip, (node) => ["slim-copilot-note", "slim-pin-name"].includes(node.cls)).textContent);
  assert.deepEqual(names, ["One", "x", "weekly/"]);
  assert.deepEqual(chips.children.map((chip) => chip.cls),
                   ["slim-ctx-chip slim-copilot-note-chip", "slim-pin is-note", "slim-pin is-folder"]);
  assert.equal(chips.children[2].attrs.title, "Capture/weekly");
  view.toggleOpenNote();                                            // left out: struck through, ↩ to bring it back
  const off = find(composerOf(root), (node) => node.cls?.includes("slim-copilot-note-off"));
  assert.ok(off.cls.includes("is-off"));
  assert.equal(off.attrs["aria-label"], "Include the open note again");
  view.openNote = null;
  view.activeThread.pins = [];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-composer-chips"), null);   // nothing to show: no row
});

test("the effort menu closes on a click elsewhere, not on one inside it", () => {
  const { view, root } = renderedView();
  view.render();
  const menu = () => find(root, (node) => node.cls === "slim-effort-menu");
  const button = find(root, (node) => node.cls === "slim-effort");
  button.onclick();
  assert.equal(menu().hidden, false);
  view.onRootClick({ target: { closest: (selector) => (selector.includes(".slim-effort-menu") ? {} : null) } });
  assert.equal(menu().hidden, false);                               // a click inside keeps it
  view.onRootClick({ target: { closest: () => null } });
  assert.equal(menu().hidden, true);
  assert.equal(view.effortOpen, false);
});

test("the effort menu works from the keyboard: focus moves in, the arrows move, Enter and Space pick, Escape and Tab close to the button", () => {
  const settings = { effort: "medium" };
  const { view, root } = renderedView({ settings, saveSettings: async () => {} });
  view.render();
  const button = () => find(root, (node) => node.cls === "slim-effort");
  const menu = () => find(root, (node) => node.cls === "slim-effort-menu");
  const focused = () => focusState.active;
  const name = () => focused().children[0]?.textContent;
  // Each key goes to whatever has focus, as the browser sends it.
  const press = (key, extra = {}) => {
    const event = { key, shiftKey: false, prevented: false, preventDefault() { this.prevented = true; },
                    stopPropagation() {}, ...extra };
    focused().onkeydown?.(event);
    return event;
  };
  button().focus();
  press("ArrowDown");                                               // opens it, like a select
  assert.equal(menu().hidden, false);
  assert.equal(name(), "Medium");                                   // focus on the level in use
  press("ArrowDown");
  assert.equal(name(), "High");
  press("ArrowDown");
  assert.equal(name(), "Off");                                      // wraps
  press("ArrowUp");
  assert.equal(name(), "High");
  assert.ok(press("Escape").prevented);
  assert.equal(menu().hidden, true);
  assert.equal(focused(), button());                                // back on the trigger
  assert.equal(view.effort, "medium");                              // nothing picked
  button().onclick();                                               // Enter on a focused button clicks it
  assert.equal(name(), "Medium");
  press("ArrowUp");
  press("Enter");
  assert.equal(view.effort, "low");
  assert.equal(settings.effort, "low");
  assert.equal(menu().hidden, true);
  assert.equal(focused(), button());
  assert.equal(find(button(), (node) => node.cls === "slim-effort-label").textContent, "Low");
  button().onclick();
  press("End");
  press(" ");
  assert.equal(view.effort, "high");
  button().onclick();
  const tab = press("Tab");
  assert.equal(tab.prevented, false);                               // the Tab goes on from the button
  assert.equal(menu().hidden, true);
  assert.equal(focused(), button());
  assert.equal(view.effort, "high");
});

test("history rows show a faint relative date and the note the chat was on", () => {
  const { view, root } = renderedView();
  view.threads = [{ id: "a", title: "A", source_path: "Notes/one.md", updated_at: new Date().toISOString() },
                  { id: "b", title: "B", updated_at: "2026-01-02T12:00:00Z" }];
  view.showThreads = true;
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-copilot-title").textContent, "Chats");
  const subs = [];
  find(root, (node) => { if (node.cls === "slim-chat-sub") subs.push(node.children.map((part) => part.textContent)); return false; });
  assert.deepEqual(subs, [["Today", "one"], ["Jan 2"]]);
});

// ---- one chat streams while the owner moves between chats --------------------------------------

// A view over a thread store: saves land in `disk`, opening a chat reads a copy of it.
function storeView({ stream, ...extra } = {}) {
  const disk = new Map();
  const bodies = [];
  const view = makeView({
    postJSON: async (route, body) => {
      if (route.endsWith("thread/save")) { disk.set(body.id, structuredClone(body)); return {}; }
      return contextFor(route) || { threads: [] };
    },
    getJSON: async (route) => {
      const id = decodeURIComponent(route.split("id=")[1] || "");
      return structuredClone(disk.get(id) || { id, title: id, turns: [] });
    },
    streamCopilot: async (body, handlers, signal) => { bodies.push(body); return stream(body, handlers, signal, bodies.length); },
    ...extra,
  });
  view.render = () => {};
  return { view, disk, bodies };
}

const texts = (turns) => turns.map((turn) => turn.text);

test("reopening the chat that is streaming shows the live answer, and the next question keeps it on disk", async () => {
  const gate = deferred();
  const { view, disk } = storeView({ stream: async (_body, handlers, _signal, n) => {
    if (n === 1) await gate.promise;
    handlers.turn({ answer: `answer ${n}`, citations: [] });
  } });
  disk.set("A", { id: "A", title: "A", turns: [{ role: "you", text: "old" }, { role: "slim", text: "old answer" }] });
  await view.openThread("A");
  const sending = view.send("new q");
  await view.openThread("B");
  await view.openThread("A");                                       // the history list's way back
  assert.equal(view.turns, view.streamingTurns);                    // the live chat, not a copy from disk
  gate.resolve();
  await sending;
  assert.deepEqual(texts(view.turns), ["old", "old answer", "new q", "answer 1"]);
  await view.send("follow-up");
  assert.deepEqual(texts(disk.get("A").turns), ["old", "old answer", "new q", "answer 1", "follow-up", "answer 2"]);
});

test("an error and its Retry stay with the chat that failed; Retry there re-asks with its images", async () => {
  const gate = deferred();
  const { view, bodies } = storeView({ stream: async (_body, handlers, _signal, n) => {
    if (n === 1) { await gate.promise; throw new Error("boom"); }
    handlers.turn({ answer: "ok", citations: [] });
  } });
  view.activeThread = { id: "A", title: "A" };
  view.pendingImages = [{ name: "x.png", mime: "image/png", data: "AAAA", preview: "data:image/png;base64,AAAA" }];
  const sending = view.send("question for A");
  await view.openThread("B");                                       // it fails while B is on screen
  gate.resolve();
  await sending;
  assert.equal(view.error, "");
  await view.retry();                                               // nothing to retry in B
  assert.equal(bodies.length, 1);
  await view.openThread("C");
  assert.equal(view.error, "");
  await view.openThread("A");
  assert.equal(view.error, "boom");
  assert.equal(view.retryQuestion, "question for A");
  await view.retry();
  assert.equal(bodies[1].thread_id, "A");
  assert.equal(bodies[1].question, "question for A");
  assert.deepEqual(bodies[1].images.map((image) => image.name), ["x.png"]);
});

test("Retry is drawn only in the chat that failed", async () => {
  const { view, root } = renderedView({
    getJSON: async (route) => ({ id: route.split("id=")[1], title: "B", turns: [] }),
    streamCopilot: async () => { throw new Error("boom"); },
  });
  view.activeThread = { id: "A", title: "A" };
  await view.send("question for A");
  const retry = () => find(root, (node) => node.attrs?.["aria-label"] === "Retry response");
  assert.ok(retry());
  await view.openThread("B");
  assert.equal(retry(), null);
  assert.equal(find(root, (node) => node.cls === "slim-copilot-error"), null);
  view.newThread();
  assert.equal(retry(), null);
});

test("a history row clicked as the search fires still opens", async () => {
  const opening = deferred();
  const view = makeView({
    getJSON: async (route) => (route.startsWith("/api/copilot/thread?") ? opening.promise : { threads: [] }),
  });
  view.render = () => {};
  const open = view.openThread("A");
  await view.loadThreads();                                         // the search's debounce fires meanwhile
  opening.resolve({ id: "A", title: "A", turns: [{ role: "you", text: "q" }] });
  await open;
  assert.equal(view.activeThread?.id, "A");
});

test("a pin or the open note's ✕ on a sidebar with no chat yet keeps the pasted images", () => {
  const { view } = pinView();
  const image = { name: "x.png", mime: "image/png", data: "AAAA" };
  view.pendingImages = [image];
  view.draft = "@x";
  view.pin({ kind: "note", path: "Notes/x.md" });
  assert.deepEqual(view.pendingImages, [image]);
  view.activeThread = null;
  view.toggleOpenNote();
  assert.deepEqual(view.pendingImages, [image]);
  assert.equal(view.activeThread.openNoteOff, true);
});

test("a pin changed in another chat while one streams is saved", async () => {
  const gate = deferred();
  const { view, disk } = storeView({ stream: async (_body, handlers) => {
    await gate.promise;
    handlers.turn({ answer: "ok", citations: [] });
  } });
  disk.set("B", { id: "B", title: "B", pins: [], turns: [{ role: "you", text: "q" }, { role: "slim", text: "a" }] });
  view.activeThread = { id: "A", title: "A" };
  const sending = view.send("question for A");
  await view.openThread("B");
  await view.pin({ kind: "note", path: "Notes/x.md" });
  assert.deepEqual(disk.get("B").pins, [{ kind: "note", path: "Notes/x.md" }]);
  gate.resolve();
  await sending;
});

test("renaming the chat that streams, from another chat, survives the answer's save", async () => {
  const gate = deferred();
  const { view, disk } = storeView({ stream: async (_body, handlers) => {
    await gate.promise;
    handlers.turn({ answer: "answer", citations: [] });
  } });
  disk.set("A", { id: "A", title: "A", turns: [{ role: "you", text: "old" }, { role: "slim", text: "old answer" }] });
  view.promptTitle = async () => "Renamed";
  await view.openThread("A");
  const sending = view.send("new q");
  await view.openThread("B");
  await view.renameThread("A");
  gate.resolve();
  await sending;
  assert.equal(disk.get("A").title, "Renamed");
  assert.deepEqual(texts(disk.get("A").turns), ["old", "old answer", "new q", "answer"]);
});

test("renaming the chat that streams never saves its unanswered question, and a failed answer still keeps the title", async () => {
  const gate = deferred();
  const { view, disk } = storeView({ stream: async () => { await gate.promise; throw new Error("boom"); } });
  disk.set("A", { id: "A", title: "A", turns: [{ role: "you", text: "old" }, { role: "slim", text: "old answer" }] });
  view.promptTitle = async () => "Renamed";
  await view.openThread("A");
  const sending = view.send("new q");
  await view.renameThread("A");
  assert.deepEqual(texts(disk.get("A").turns), ["old", "old answer"]);
  assert.equal(view.activeThread.title, "Renamed");
  gate.resolve();
  await sending;
  await view.savingRuns;
  assert.equal(disk.get("A").title, "Renamed");
  assert.deepEqual(texts(disk.get("A").turns), ["old", "old answer"]);
  assert.equal(view.retryQuestion, "new q");                       // still offered under its error
});

test("Stop, then a quick second click, neither sends nor breaks the next turn", async () => {
  const bodies = [];
  const { view, root } = renderedView({
    streamCopilot: (body, handlers, signal) => new Promise((resolve, reject) => {
      bodies.push(body);
      if (bodies.length > 1) { handlers.turn({ answer: "second", citations: [] }); resolve(); return; }
      // The stream notices the stop a moment later, as a socket does.
      signal.addEventListener("abort", () => setTimeout(() => reject(Object.assign(new Error("Cancelled"), { name: "AbortError" })), 5));
    }),
  });
  const send = () => find(root, (node) => node.cls?.split(" ").includes("slim-send"));
  view.draft = "first";
  view.render();
  const first = send().onclick();
  await new Promise((resolve) => setImmediate(resolve));
  send().onclick();                                                 // Stop
  view.draft = "typed meanwhile";
  send().onclick();                                                 // a second click before the stream ended
  await first;
  assert.equal(bodies.length, 1);
  assert.equal(view.sending, false);
  assert.equal(view.abort, null);
  view.draft = "next";
  await view.send();
  assert.equal(bodies.length, 2);
  assert.equal(view.turns.at(-1).text, "second");
  assert.equal(view.sending, false);
});

test("Enter while an input method is composing never sends or picks", () => {
  const { view, sent, type, rows } = pickerView();
  const input = find(view.contentEl, (node) => node.tag === "textarea");
  const compose = (extra) => input.onkeydown({ key: "Enter", shiftKey: false, metaKey: false, altKey: false,
                                                ctrlKey: false, preventDefault() {}, ...extra });
  type("こんにちは");
  compose({ isComposing: true });
  compose({ keyCode: 229 });
  assert.deepEqual(sent, []);
  type("about @");
  compose({ isComposing: true });
  assert.equal(view.activeThread, null);                            // nothing pinned
  assert.ok(rows("slim-mention-item").length);
  let renamed = null;
  const modal = new RenameModal({}, "Old", (value) => { renamed = value; });
  modal.open();
  const field = find(modal.contentEl, (el) => el.tag === "input");
  field.value = "Half";
  field.listeners.keydown[0]({ key: "Enter", isComposing: true, preventDefault() {} });
  assert.equal(renamed, null);
});

test("\"/\" then Enter with the command menu open never sends a bare slash", () => {
  const { view, root } = renderedView();
  view.skills = [{ name: "quiz", description: "Quiz me" }];
  const sent = [];
  view.send = () => sent.push(view.draft);
  view.render();
  const input = () => find(root, (node) => node.tag === "textarea");
  input().value = "/";
  input().oninput();
  input().onkeydown({ key: "Enter", shiftKey: false, metaKey: false, altKey: false, ctrlKey: false, preventDefault() {} });
  assert.deepEqual(sent, []);
  view.skills = [];
  view.draft = "/";
  view.render();
  input().onkeydown({ key: "Enter", shiftKey: false, metaKey: false, altKey: false, ctrlKey: false, preventDefault() {} });
  assert.deepEqual(sent, []);
});

test("while one chat streams, another chat's composer waits with a hint instead of a Stop", async () => {
  let started;
  const running = new Promise((resolve) => { started = resolve; });
  const { view, root } = renderedView({
    getJSON: async () => ({ id: "B", title: "B", turns: [{ role: "you", text: "q" }, { role: "slim", text: "a" }] }),
    streamCopilot: (_body, _handlers, signal) => new Promise((_resolve, reject) => {
      started();
      signal.addEventListener("abort", () => reject(Object.assign(new Error("Cancelled"), { name: "AbortError" })));
    }),
  });
  const send = () => find(root, (node) => node.cls?.split(" ").includes("slim-send"));
  const hint = () => find(root, (node) => node.cls === "slim-composer-hint");
  view.activeThread = { id: "A", title: "A" };
  view.draft = "question for A";
  view.render();
  const turn = send().onclick();
  await running;
  await view.openThread("B");
  assert.equal(send().disabled, true);
  assert.notEqual(send().attrs["aria-label"], "Stop response");
  assert.equal(hint().textContent, "SLIM is answering in another chat");
  send().onclick?.();
  assert.equal(view.sending, true);                                 // A's answer is not stopped from B
  await view.openThread("A");
  assert.equal(send().attrs["aria-label"], "Stop response");
  assert.equal(hint(), null);
  send().onclick();
  await turn;
  await view.openThread("B");
  assert.equal(send().disabled, false);
});
