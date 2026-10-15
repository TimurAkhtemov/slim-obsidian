// The plugin class: stranded audio, reload adoption, progress polling, pending-review notices.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { fakeContentEl, noticeInstances, notices, obsidianStub,
         setRequestUrl } from "./fakes.mjs";
import { findButton, plugin, sessionManagerHarness, view } from "./recorder-harness.mjs";

const require = createRequire(import.meta.url);
const PluginClass = require("../src/plugin.js");
const { VIEW_TYPE_COPILOT, CopilotView } = require("../src/chat/view.js");
const { MeetingSessionManager } = require("../src/recorder/sessions.js");
const manifest = require("../manifest.json");

// The plugin after `onload`, against a workspace that records what was registered. No server
// is spawned and no live editor is built: neither is what these tests are about.
async function loadedPlugin(leavesOfType = () => []) {
  const p = Object.create(PluginClass.prototype);
  const got = { views: {}, commands: [], handlers: {}, blocks: {}, layoutReady: [] };
  const on = (name, fn) => { got.handlers[name] = fn; return name; };
  p.app = {
    workspace: { on, getActiveFile: () => null, getLeavesOfType: (type) => leavesOfType(type),
                 onLayoutReady: (fn) => got.layoutReady.push(fn) },
    vault: { on },
  };
  Object.assign(p, {
    loadData: async () => ({}), setupLive() {}, ensureServer: async () => {},
    registerMarkdownCodeBlockProcessor: (lang, fn) => { got.blocks[lang] = fn; },
    registerView: (type, factory) => { got.views[type] = factory; },
    addCommand: (command) => got.commands.push(command.id),
    addRibbonIcon() {}, registerEvent() {}, addSettingTab() {},
  });
  await p.onload();
  return { p, got };
}

test("startup reports a recoverable draft and audio pair by recording ID", async () => {
  const id = "20260823T143000-abcd1234";
  const audio = `Attachments/_incoming/${id}.webm`;
  const draft = `Capture/_unfiled/2026-08-23--recording--${id}.slim-draft.md`;
  const listings = {
    "Attachments/_incoming": { files: [audio] },
    "Capture/_unfiled": { files: [draft] },
    Journal: { files: [] },
  };
  const p = plugin();
  p.app = { vault: { adapter: {
    exists: async (path) => path in listings,
    list: async (path) => listings[path],
    stat: async () => ({ size: 42 }),
  } } };

  await p.warnAboutStrandedAudio();

  assert.equal(notices.length, 1);
  assert.match(notices[0], /recoverable/i);
  assert.match(notices[0], new RegExp(id));
  assert.match(notices[0], /draft.*audio|audio.*draft/i);
});

test("startup reports paired and unpaired stranded recordings separately", async () => {
  const pairedId = "20260823T143000-abcd1234";
  const unpairedId = "20260823T150000-deadbeef";
  const listings = {
    "Attachments/_incoming": { files: [
      `Attachments/_incoming/${pairedId}.webm`,
      `Attachments/_incoming/${unpairedId}.webm`,
    ] },
    "Capture/_unfiled": { files: [
      `Capture/_unfiled/2026-08-23--recording--${pairedId}.slim-draft.md`,
    ] },
    Journal: { files: [] },
  };
  const p = plugin();
  p.app = { vault: { adapter: {
    exists: async (path) => path in listings,
    list: async (path) => listings[path],
    stat: async () => ({ size: 42 }),
  } } };

  await p.warnAboutStrandedAudio();

  assert.equal(notices.length, 2);
  assert.match(notices[0], new RegExp(pairedId));
  assert.match(notices[1], /1 recording/);
  assert.match(notices[1], /never finished filing/);
});

test("recording progress tolerates the initial race and streams the live state", async () => {
  const plugin = Object.create(PluginClass.prototype);
  plugin.settings = { port: 7546 };
  const states = [];
  let calls = 0;
  setRequestUrl(async () => {
    calls += 1;
    if (calls === 1) return { status: 404, json: { error: "not started" } };
    return { status: 200, json: { stage: "summarizing", thinking: "actual token", done: true } };
  });
  globalThis.window = {
    setTimeout(fn) { queueMicrotask(fn); return 1; },
    clearTimeout() {},
  };

  const stop = plugin.watchRecording("safe-job", (state) => states.push(state));
  await new Promise((resolve) => setImmediate(resolve));
  stop();

  assert.equal(calls, 2);
  assert.equal(states.length, 1);
  assert.equal(states[0].thinking, "actual token");
});

test("plugin microphone activation delegates to the meeting manager", async () => {
  const plugin = Object.create(PluginClass.prototype);
  const leaf = { view: new obsidianStub.MarkdownView({}) };
  leaf.view.file = { path: "Notes/current.md" };
  let received = null;
  const session = { editorLeaf: leaf };
  plugin.app = { workspace: {
    getMostRecentLeaf: () => leaf,
    revealLeaf: (value) => { assert.equal(value, leaf); },
  } };
  plugin.meetings = { activate: async (value) => { received = value; return session; } };

  await plugin.activate();

  assert.equal(received, leaf);
});

test("closing a pending review says where it is and offers recovery actions once", () => {
  view(); // reset the notice harness
  const plugin = Object.create(PluginClass.prototype);
  const leaf = { view: { file: { path: "Notes/school/CS203 NLP/x.md" } } };
  const session = {
    state: "card", editorLeaf: leaf, filedNotePath: "Notes/school/CS203 NLP/x.md",
    reviewCloseNotified: false,
  };
  let liveLeaves = [];
  plugin.app = { workspace: { getLeavesOfType: () => liveLeaves } };
  plugin.meetings = { sessions: new Map([["x", session]]) };

  plugin.noticeClosedPendingReviews();
  plugin.noticeClosedPendingReviews();

  assert.equal(notices.length, 1);
  assert.match(notices[0], /Review is still pending/);
  assert.match(notices[0], /CS203 NLP/);
  assert.ok(findButton(noticeInstances[0].noticeEl, "Reopen"));
  assert.ok(findButton(noticeInstances[0].noticeEl, "Reveal"));

  liveLeaves = [leaf];
  plugin.noticeClosedPendingReviews();
  assert.equal(session.reviewCloseNotified, false, "reopening rearms the next real close");
});

test("a leaf change while Approve is in flight does not announce a pending review", async () => {
  // `apply` moves the note by create + unlink. Obsidian drops the deleted file from its leaf,
  // and that layout change ran `noticeClosedPendingReviews` while the request was still out:
  // state `card`, leaf no longer showing the meeting — "Review is still pending", with the
  // path from before the move (2026-09-09). The note was filed; the notice was wrong.
  const v = view();
  v.state = "card";
  v.edited = true;
  v.filedNotePath = "Capture/_unfiled/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {
    dest_dir: "Capture/school", type_tag: "lecture", topics: [], folders: [],
  } };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  const leaf = { view: { file: { path: "Capture/_unfiled/x.md" } } };
  v.editorLeaf = leaf;
  v.plugin.app = { workspace: { getLeavesOfType: () => [] } };
  v.plugin.meetings = { sessions: new Map([["x", v]]) };
  v.plugin.applyCard = async () => {
    leaf.view.file = null;                                  // Obsidian let go of the deleted file
    PluginClass.prototype.noticeClosedPendingReviews.call(v.plugin);
    return { note: "Capture/school/x.md" };
  };
  v.loadEditorFile = async () => {};
  v.loadNoteParts = async () => v.noteParts;
  const root = fakeContentEl();
  v.contentEl = root;
  v.render();

  await findButton(root, "Approve & finish").events.click();

  assert.equal(v.state, "finished");
  assert.ok(!notices.some((n) => /Review is still pending/.test(n)), notices.join(" | "));
});

test("plugin reload adopts every visible recorded block into its own session", async () => {
  const adopted = [];
  const { manager } = sessionManagerHarness();
  manager.sessionFactory = () => ({
    sessionId: `adopt-${adopted.length + 1}`,
    recordingId: "",
    state: "idle",
    draftRel: "",
    filedNotePath: "",
    async openExisting(file) {
      adopted.push(file.path);
      this.filedNotePath = file.path;
      this.state = "finished";
      return true;
    },
    async adoptPausedDraft() { return false; },
    render() {}, onClose() {}, ownsEditor() { return false; },
  });
  const a = { path: "Capture/project/a.md" };
  const b = { path: "Capture/project/b.md" };

  const sessionA = await manager.adoptFile(a, {});
  const sessionB = await manager.adoptFile(b, {});
  const againA = await manager.adoptFile(a, {});

  assert.notEqual(sessionA, sessionB);
  assert.equal(againA, sessionA);
  assert.deepEqual(adopted, [a.path, b.path]);
});

test("onload registers the copilot view and its command", async () => {
  const { got } = await loadedPlugin();
  assert.ok(got.views[VIEW_TYPE_COPILOT]({}) instanceof CopilotView);
  assert.ok(got.commands.includes("open-note-copilot"));
  assert.ok(got.commands.includes("start-recording"));
  assert.equal(manifest.name, "SLIM");
});

test("the copilot follows the active note", async () => {
  const seen = [];
  const copilot = { view: { setActiveFile: (file) => seen.push(file) } };
  const { p, got } = await loadedPlugin((type) => (type === VIEW_TYPE_COPILOT ? [copilot] : []));
  const note = { path: "Notes/a.md" };
  p.app.workspace.getActiveFile = () => note;

  got.handlers["active-leaf-change"]();

  assert.deepEqual(seen, [note]);
});

test("plugin reload adopts every already-visible meeting note", async () => {
  const leafFor = (path) => {
    const leaf = { view: new obsidianStub.MarkdownView({}) };
    leaf.view.file = { path };
    return leaf;
  };
  const leaves = [leafFor("Capture/a.md"), leafFor("Capture/b.md"), { view: {} }];
  const { p, got } = await loadedPlugin((type) => (type === "markdown" ? leaves : []));
  const adopted = [];
  p.meetings.adoptFile = async (file, leaf) => { adopted.push([file.path, leaf]); };
  p.warnAboutStrandedAudio = async () => {};

  got.layoutReady.forEach((fn) => fn());

  assert.deepEqual(adopted, [["Capture/a.md", leaves[0]], ["Capture/b.md", leaves[1]]]);
});

test("a meeting block is found while Live Preview is still building it", async () => {
  // At reload the code-block processor runs before its element is in the document; refusing
  // a disconnected element loses the only mount and leaves the block empty.
  const { p, got } = await loadedPlugin();
  const el = fakeContentEl();
  el.isConnected = false;

  got.blocks["slim-meeting"]("", el, { sourcePath: "Capture/a.md" });

  assert.equal(p.meetingBlockFor("Capture/a.md"), el);
  assert.equal(p.meetingBlockFor("Capture/other.md"), null);
});

test("quitting Obsidian flushes every meeting's notes", async () => {
  const { p, got } = await loadedPlugin();
  let flushed = 0;
  p.meetings.flushNotes = async () => { flushed += 1; };
  const tasks = [];

  got.handlers.quit({ add: (task) => tasks.push(task) });
  await Promise.all(tasks.map((task) => task()));

  assert.equal(flushed, 1);
});

test("the plugin owns a meeting-session manager, not one global recorder", async () => {
  const { p } = await loadedPlugin();
  assert.ok(p.meetings instanceof MeetingSessionManager);
  assert.equal(p.recorder, undefined);
});
