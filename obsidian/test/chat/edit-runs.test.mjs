// Editing runs: changes landing in the note, the chat's account of them, stop, restore, drafts.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { find } from "../fakes.mjs";
import { PLAN, change, editView, fakeLive, makeView, renderedView } from "../copilot-harness.mjs";

const require = createRequire(import.meta.url);
const { editOutcome, historyText, sha256Hex } = require("../../src/chat/text.js");
const SlimRecorderPlugin = require("../../src/plugin.js");

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
  assert.deepEqual(live.calls.map((c) => c[0]), ["start", "drop"]);  // an answer leaves nothing in the editor
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
  const leaf = { view: { file: { path: "Notes/one.md" }, getMode: () => "source", editor: { getSelection: () => "picked",
    getCursor: () => ({ line: 2, ch: 0 }), posToOffset: () => 5 } } };
  view.app.workspace = { getLeavesOfType: () => [leaf], getMostRecentLeaf: () => leaf };
  await view.send("tidy it");
  await view.send("/tidy");
  assert.deepEqual(bodies.map((b) => [b.selection, b.selection_from]), [["", undefined], ["picked", 5]]);
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
      hash_before: "h0", hash_after: "server-draft", summary: PLAN.intro, before_ref: "a".repeat(32) } });
  } });
  await view.send("Ok make the revisions");
  const local = live.calls[0][1];
  assert.deepEqual(live.calls.map((c) => c.slice(0, 3)), [["start", local, "# A\n\nold\n"], ["claim", local, "a".repeat(32)],
    ["apply", "c1"], ["apply", "c2"], ["end", "a".repeat(32)]]);      // the run starts at Send, on the text sent
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

test("a skipped change made inside another is not named: it was SLIM's own rework", () => {
  const { live } = fakeLive();
  const { view, root } = renderedView({ live });
  const run = { plan: { intro: "Adding a section.", steps: [] }, stopped: null, summary: "", before_ref: "2".repeat(32),
                changes: [{ id: "c1", label: "Added “Condensed”", state: "applied" },
                          { id: "c2", label: "Rewrote “SLIM's draft”", state: "skipped", part_of: "c1" },
                          { id: "c3", label: "Changed “a” to “b”", state: "skipped" }] };
  view.turns = [{ role: "you", text: "q" }, { role: "slim", text: "a", turn: { mode: "quick", edit_run: run } }];
  view.render();
  const texts = [];
  (function walk(el) { if (el.textContent) texts.push(el.textContent); el.children.forEach(walk); })(root);
  assert.ok(texts.includes("Skipped “Changed “a” to “b””: you edited this here."), texts.join(" | "));
  assert.equal(texts.some((t) => t.includes("Rewrote")), false, texts.join(" | "));
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

test("a run that stops or fails keeps its question's pasted images as the server stored them", async () => {
  // Its turn is kept and saved, but no `turn` came to swap in the refs: the run's start names them.
  const pending = { name: "eq.png", mime: "image/png", data: "BASE64", preview: "data:image/png;base64,BASE64", bytes: 6 };
  const ref = { id: "a".repeat(64), name: "eq.png", mime: "image/png", bytes: 6 };
  for (const ending of ["stop", "error"]) {
    const { live, editor } = fakeLive();
    const saved = [];
    const view = editView({ live, editor, saved, events: (_body, handlers, signal) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => { const error = new Error("Cancelled"); error.name = "AbortError"; reject(error); });
      handlers.edit({ before_ref: "d".repeat(32), hash_before: "h0", attachments: [ref] });
      handlers.plan(PLAN);
      handlers.change(change("c1", 1));
      if (ending === "error") handlers.error({ error: "the model failed" });
    }) });
    view.pendingImages = [pending];
    const sending = view.send("Turn the screenshot into a table");
    await new Promise((resolve) => setImmediate(resolve));
    await new Promise((resolve) => setImmediate(resolve));
    if (ending === "stop") view.cancel();
    await sending;
    assert.deepEqual(view.turns.map((t) => t.role), ["you", "slim"], ending);
    assert.deepEqual(saved.at(-1).turns[0].attachments, [ref], ending);
    await view.saveRun(view.activeThread, view.turns);                 // a later save: ↺ on its change
    assert.deepEqual(saved.at(-1).turns[0].attachments, [ref], ending);
  }
});

test("a question that failed before an answer never saves its pasted images: the server kept none", async () => {
  const pending = { name: "eq.png", mime: "image/png", data: "BASE64", preview: "data:image/png;base64,BASE64", bytes: 6 };
  const saved = [];
  const view = editView({ live: fakeLive().live, editor: fakeLive().editor, saved,
                          events: async () => { throw new Error("offline"); } });
  view.turns = [{ role: "you", text: "Fix the title" },
                { role: "slim", text: "Done", turn: { mode: "quick", edit_run: { before_ref: "e".repeat(32), changes: [] } } }];
  view.pendingImages = [pending];
  await view.send("Explain this");
  assert.deepEqual(view.retryImages, [pending]);                      // Try again sends them again
  assert.deepEqual(view.turns.at(-1).attachments, [pending]);          // and they still show under it
  await view.saveRun(view.activeThread, view.turns);                 // ↺ on the earlier run saves the chat
  assert.equal(saved.at(-1).turns.at(-1).text, "Explain this");
  assert.equal("attachments" in saved.at(-1).turns.at(-1), false);
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

test("a note over the edit size is still asked about: its text is not sent", async () => {
  const { live, editor } = fakeLive("x".repeat(400_001));
  let body;
  const view = editView({ live, editor, events: async (sent, handlers) => {
    body = sent;
    handlers.turn({ answer: "It is long.", mode: "quick", citations: [] });
  } });
  await view.send("What is this?");
  assert.equal("doc" in body, false);
  assert.equal("doc_hash" in body, false);
  assert.deepEqual(live.calls, []);                                    // nothing to track: no run can land
  assert.equal(view.turns.at(-1).text, "It is long.");
});

test("a run whose text was not sent lands nothing in the editor", async () => {
  const { live, editor } = fakeLive("x".repeat(400_001));
  const view = editView({ live, editor, events: (_body, handlers, signal) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => { const error = new Error("Cancelled"); error.name = "AbortError"; reject(error); });
    handlers.edit({ before_ref: "3".repeat(32), hash_before: "h0" });
    handlers.change(change("c1", 1));
  }) });
  await view.send("rename it");
  assert.equal(live.calls.some((c) => c[0] === "apply"), false);
  assert.deepEqual(view.turns.at(-1).turn.edit_run.changes.map((c) => c.state), ["skipped"]);
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
  vault.create = async (p, text) => {
    if (p.includes("?")) throw new Error("File name cannot contain any of the following characters: * \" \\ / < > : | ?");
    files[p] = text;
    return { path: p };
  };
  const view = editView({ live, editor, vault, events: async (_body, handlers) => {
    handlers.edit({ before_ref: "8".repeat(32), hash_before: "h0" });
    handlers.plan({ intro: "", steps: [] });
    handlers.change({ id: "c1", step: 1, op: "create", label: "Created Notes/brief.md", needs: [], path: "Notes/brief.md", text: "# Brief\n" });
    handlers.change({ id: "c2", step: 1, op: "create", label: "Created Notes/taken.md", needs: [], path: "Notes/taken.md", text: "mine" });
    handlers.change({ id: "c3", step: 1, op: "create", label: "Created Notes/why?.md", needs: [], path: "Notes/why?.md", text: "x" });
    handlers.turn({ answer: "Done", mode: "quick", citations: [], edit_run: { plan: { intro: "", steps: [] }, steps_done: 0,
      changes: [], stopped: null, summary: "Done: 2 changes", before_ref: "8".repeat(32) } });
  } });
  await view.send("/brief");
  assert.equal(files["Notes/brief.md"], "# Brief\n");
  assert.equal(files["Notes/taken.md"], "theirs");                        // never written over
  assert.equal(live.calls.some((c) => c[0] === "apply"), false);          // a new note is not an editor change
  const run = view.turns.at(-1).turn.edit_run;
  assert.deepEqual(run.changes.map(({ path, state, reason }) => [path, state, reason]), [
    ["Notes/brief.md", "created", undefined],
    ["Notes/taken.md", "failed", "a note is already there"],
    ["Notes/why?.md", "failed", "File name cannot contain any of the following characters: * \" \\ / < > : | ?"]]);
  const { view: shown, root } = renderedView({ live });
  shown.turns = view.turns;
  shown.render();
  const texts = [];
  (function walk(el) { if (el.textContent) texts.push(el.textContent); el.children.forEach(walk); })(root);
  assert.ok(texts.some((t) => t.includes("Not created (a note is already there):")), texts.join(" | "));
  assert.ok(texts.some((t) => t.includes("Not created (File name cannot contain")), texts.join(" | "));
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

test("a change made inside another counts as part of it, in the chat and in history", async () => {
  // The owner's run: c1 added a section; c2, c3 and c4 only rewrote what c1 and c2 wrote.
  const inside = (id, partOf, needs) => change(id, 1, { op: "replace", label: `Rewrote ${id}`, part_of: partOf, needs });
  const { live, editor, runs } = fakeLive();
  const saved = [];
  const view = editView({ live, editor, saved, events: async (_body, handlers) => {
    handlers.edit({ before_ref: "1".repeat(32), hash_before: "h0" });
    handlers.plan({ intro: "Adding an explanation.", steps: [] });
    handlers.change(change("c1", 1, { op: "insert", label: "Added “Condensed”" }));
    handlers.change(inside("c2", "c1", ["c1"]));
    handlers.change(inside("c3", "c1", ["c1"]));
    handlers.change(inside("c4", "c1", ["c2"]));
    handlers.change(change("c5", 1, { label: "Rewrote “Intro”" }));
    handlers.turn({ answer: "Adding an explanation.", mode: "quick", citations: [], edit_run: {
      plan: { intro: "Adding an explanation.", steps: [] }, steps_done: 0, changes: [], stopped: null,
      summary: "Adding an explanation.", before_ref: "1".repeat(32) } });
  } });
  await view.send("add a condensed explanation under the screenshot");
  const turn = view.turns.at(-1);
  const kept = saved.at(-1).turns.at(-1).turn.edit_run;
  assert.deepEqual(kept.changes.map((c) => [c.id, c.part_of ?? null]),
                   [["c1", null], ["c2", "c1"], ["c3", "c1"], ["c4", "c1"], ["c5", null]]);
  assert.equal("fixups" in kept, false);
  assert.equal(historyText(turn), "[Changed the note: Adding an explanation.; 2 changes, 0 undone]");
  // ↺ on the section takes what was made inside it with it: one change undone, not four.
  for (const c of runs.get("1".repeat(32)).changes) if (c.id !== "c5") c.state = "undone";
  view.runChanged(editor, "1".repeat(32));
  await view.savingRuns;
  assert.equal(historyText(turn), "[Changed the note: Adding an explanation.; 2 changes, 1 undone]");
  const { view: shown, root } = renderedView({ live, liveRunView: (id) => (live.state(editor, id) ? editor : null) });
  shown.turns = view.turns;
  shown.render();
  const texts = [];
  (function walk(el) { if (el.textContent) texts.push(el.textContent); el.children.forEach(walk); })(root);
  assert.ok(texts.includes("2 changes, 1 undone"), texts.join(" | "));
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

test("a change's text is a draft in the note while it is written, and the change takes its place", async () => {
  const { live, editor } = fakeLive();
  const view = editView({ live, editor, events: async (_body, handlers) => {
    handlers.draft({ id: "d1.0", from: 3, to: 3, text: "" });          // the first call: before the run is named
    handlers.draft_text({ id: "d1.0", text: "New" });
    handlers.edit({ before_ref: "a".repeat(32), hash_before: sha256Hex("# A\n\nold\n") });
    handlers.plan({ intro: "Add a line.", steps: [], outline: null });
    handlers.change(change("c1", 1, { draft: "d1.0" }));
    handlers.draft({ id: "d2.0", from: 5, to: 8, text: "x" });
    handlers.draft_end({ id: "d2.0" });
    handlers.turn({ answer: "Add a line.", mode: "quick", citations: [], edit_run: {
      plan: { intro: "Add a line.", steps: [] }, steps_done: 1, changes: [change("c1", 1)], stopped: null } });
  } });
  await view.send("add a line");
  const local = live.calls[0][1], run = "a".repeat(32);
  assert.deepEqual(live.calls.filter((c) => ["draft", "grow", "drop", "apply"].includes(c[0])), [
    ["draft", local, "d1.0", 3, 3, ""], ["grow", "d1.0", "New"],           // on the editor tracking Send
    ["drop", "d1.0"], ["apply", "c1"],                                     // the change takes its draft's place
    ["draft", run, "d2.0", 5, 8, "x"], ["drop", "d2.0"]]);
  assert.equal("draft" in view.turns.at(-1).turn.edit_run.changes[0], false);
});
