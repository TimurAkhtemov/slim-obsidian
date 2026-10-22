// Delete from every state, and the leaf filing empties.
import assert from "node:assert/strict";
import test from "node:test";
import { fakeContentEl, findClass } from "../fakes.mjs";
import { fileBackedView } from "../recorder-harness.mjs";

// --- the leaf filing empties -----------------------------------------------------------------

test("filing reuses the tab Obsidian emptied when the server removed the draft", async () => {
  // Obsidian empties (not closes) a lone tab whose file is deleted. Read as "no editor", that
  // split a second pane beside the empty tab on every recording (2026-09-30).
  const { v, files, editorLeaf } = fileBackedView();
  files.set("Capture/a/x.md", { path: "Capture/a/x.md", text: "" });
  editorLeaf.view = { getViewType: () => "empty" };
  v.editorLeaf = editorLeaf;
  v.app.workspace.getLeavesOfType = () => [];
  v.app.workspace.iterateAllLeaves = (fn) => fn(editorLeaf);
  v.app.workspace.getLeaf = () => { throw new Error("no new pane for a live tab"); };

  await v.loadEditorFile("Capture/a/x.md");

  assert.equal(v.editorLeaf, editorLeaf);
  assert.deepEqual(editorLeaf.opened, ["Capture/a/x.md"]);
});

test("a tab they pointed at something else is not taken back", async () => {
  const { v, files, editorLeaf } = fileBackedView();
  files.set("Capture/a/x.md", { path: "Capture/a/x.md", text: "" });
  const theirs = { app: v.app, view: { getViewType: () => "pdf" } };
  v.editorLeaf = theirs;
  v.app.workspace.getLeavesOfType = () => [];
  v.app.workspace.iterateAllLeaves = (fn) => fn(theirs);
  const asked = [];
  v.app.workspace.getLeaf = (how) => { asked.push(how); return editorLeaf; };

  await v.loadEditorFile("Capture/a/x.md");

  assert.deepEqual(asked, ["tab"]);
  assert.equal(v.editorLeaf, editorLeaf);
});

test("a tab Obsidian closed is replaced by a tab, never a split", async () => {
  const { v, files, editorLeaf } = fileBackedView();
  files.set("Capture/a/x.md", { path: "Capture/a/x.md", text: "" });
  v.editorLeaf = { app: v.app, view: null };            // detached with its siblings present
  v.app.workspace.getLeavesOfType = () => [];
  v.app.workspace.iterateAllLeaves = () => {};
  const asked = [];
  v.app.workspace.getLeaf = (how) => { asked.push(how); return editorLeaf; };

  await v.loadEditorFile("Capture/a/x.md");

  assert.deepEqual(asked, ["tab"]);
  assert.equal(v.editorLeaf, editorLeaf);
});

// --- Delete ----------------------------------------------------------------------------------

function deletable() {
  const made = fileBackedView();
  const removed = [];
  made.v.manager = {
    indexSession() {}, releaseCapture() {}, claimCapture() {},
    removeSession(s) { removed.push(s); },
  };
  return { ...made, removed };
}

test("Delete in setup trashes the draft and lets the session go", async () => {
  globalThis.window = globalThis.window || {};
  globalThis.window.setTimeout = () => 0;
  const { v, files, removed } = deletable();
  v.recordingId = "20260930T090000-abcd1234";
  await v.createDraft();
  const draft = v.draftRel;
  v.contentEl = fakeContentEl();
  v.render();

  const button = findClass(v.contentEl, "slim-delete");
  await button.events.click();
  assert.equal(files.has(draft), true, "one click only arms it");
  assert.equal(button.text, "Click again to delete");
  await button.events.click();

  assert.equal(files.has(draft), false);
  assert.deepEqual(removed, [v]);
  assert.equal(v.draftRel, "");
});

test("every state offers Delete", () => {
  const { v } = deletable();
  v.plugin.settings.port = 7546;
  v.result = { title: "x", card: { dest_dir: "Capture/a", type_tag: "lecture", topics: [], folders: [] } };
  v.filedNotePath = "Capture/a/x.md";
  v.renderSource = () => {};
  for (const state of ["idle", "recording", "processing", "paused", "error", "card", "finished"]) {
    v.state = state;
    v.contentEl = fakeContentEl();
    v.render();
    assert.ok(findClass(v.contentEl, "slim-delete"), `${state} has a Delete button`);
  }
});

test("Delete mid-recording stops capture without filing and trashes every segment", async () => {
  const { v, files, removed } = deletable();
  v.recordingId = "20260930T090000-abcd1234";
  await v.createDraft();
  const draft = v.draftRel;
  const segs = ["Attachments/_incoming/20260930T090000-abcd1234-001.webm",
                "Attachments/_incoming/20260930T090000-abcd1234-002.webm"];
  for (const s of segs) files.set(s, { path: s, bytes: Buffer.from([1]) });
  v.segments = segs.slice();
  v.audioRel = segs[1];
  v.state = "recording";
  let stopped = false;
  v.recorder = { state: "recording", stop() { stopped = true; this.state = "inactive"; this.onstop(); } };
  v.streams = [];
  v.plugin.postRecording = async () => { throw new Error("must not file"); };

  await v.deleteRecording();

  assert.equal(stopped, true);
  for (const s of segs) assert.equal(files.has(s), false);
  assert.equal(files.has(draft), false);
  assert.deepEqual(removed, [v]);
});

test("Delete on a filed note trashes it, its archived audio and pasted images, and forgets it", async () => {
  const { v, files, removed } = deletable();
  const note = "Capture/a/2026-09-30--standup--abc.md";
  const audio = "Attachments/Recordings/abc.webm";
  const shared = "Attachments/Recordings/shared.webm";
  const images = "Attachments/Recorder/20260930T090000-abcd1234";
  files.set(note, { path: note,
    text: `---\naudio:\n  - ${audio}\n  - ${shared}\n---\n![[${images}/paste-001.png]]\n` });
  files.set(audio, { path: audio, bytes: Buffer.from([1]) });
  files.set(shared, { path: shared, bytes: Buffer.from([1]) });
  files.set(images, { path: images, folder: true });
  files.set("Attachments/Recordings/other.webm", { path: "Attachments/Recordings/other.webm" });
  // Not in the metadata cache yet, as a note filed a moment ago is not; another note names
  // the same content-addressed audio.
  const other = { path: "Capture/a/other.md" };
  v.app.vault.getMarkdownFiles = () => [files.get(note), other];
  v.app.metadataCache = {
    getFileCache: (f) => (f === other ? { frontmatter: { audio: shared } } : null),
  };
  v.filedNotePath = note;
  v.state = "finished";
  const forgot = [];
  v.plugin.forgetRecording = async (body) => { forgot.push(body); return body; };

  await v.deleteRecording();

  assert.equal(files.has(note), false);
  assert.equal(files.has(audio), false);
  assert.equal(files.has(images), false);
  assert.equal(files.has(shared), true, "another note's source stays");
  assert.equal(files.has("Attachments/Recordings/other.webm"), true);
  assert.deepEqual(forgot, [{ note }]);
  assert.deepEqual(removed, [v]);
});

test("Delete while processing cancels, then deletes whatever the job left", async () => {
  const { v, files, removed } = deletable();
  v.recordingId = "20260930T090000-abcd1234";
  await v.createDraft();
  const draft = v.draftRel;
  const seg = "Attachments/_incoming/20260930T090000-abcd1234-001.webm";
  files.set(seg, { path: seg, bytes: Buffer.from([1]) });
  v.segments = [seg];
  v.audioRel = seg;
  v.saveDraft = async () => {};
  v.state = "processing";
  let release;
  v.plugin.postRecording = () => new Promise((_, reject) => { release = reject; });
  let cancelled = null;
  v.plugin.cancelRecording = async (body) => { cancelled = body; return { cancelling: true }; };

  const running = v.submitRecording([seg]);
  await v.deleteRecording();
  assert.ok(cancelled, "the job is asked to stop");
  assert.equal(files.has(draft), true, "nothing is deleted under a running job");

  release(Object.assign(new Error("Cancelled"), { cancelled: true }));
  await running;

  assert.equal(files.has(draft), false);
  assert.equal(files.has(seg), false);
  assert.deepEqual(removed, [v]);
});

test("Delete while processing still deletes a note the job filed before it could stop", async () => {
  const { v, files, removed } = deletable();
  const note = "Capture/a/x.md";
  v.recordingId = "20260930T090000-abcd1234";
  v.draftRel = "Capture/_unfiled/2026-09-30--recording--20260930T090000-abcd1234.slim-draft.md";
  v.saveDraft = async () => {};
  v.loadEditorFile = async () => { throw new Error("a deleted note is not reopened"); };
  v.state = "processing";
  v.plugin.cancelRecording = async () => ({ cancelling: true });
  v.plugin.forgetRecording = async (b) => b;
  let release;
  v.plugin.postRecording = () => new Promise((resolve) => { release = resolve; });

  const running = v.submitRecording(["Attachments/_incoming/20260930T090000-abcd1234-001.webm"]);
  await v.deleteRecording();
  files.set(note, { path: note, text: "" });
  release({ note, card: {}, title: "x" });
  await running;

  assert.equal(files.has(note), false);
  assert.deepEqual(removed, [v]);
});

test("Discarding a resumed take leaves the filed note as it was", async () => {
  const { v, files, removed } = deletable();
  const note = "Capture/a/x.md";
  const seg = "Attachments/_incoming/20260930T100000-ffff0000-001.webm";
  files.set(note, { path: note, text: "keep" });
  files.set(seg, { path: seg, bytes: Buffer.from([1]) });
  v.filedNotePath = note;
  v.appendTo = note;
  v.resumeReturnState = "finished";
  v.segments = [seg];
  v.state = "recording";

  await v.deleteRecording();

  assert.equal(files.has(note), true);
  assert.equal(files.has(seg), false);
  assert.equal(v.state, "finished");
  assert.equal(v.appendTo, "");
  assert.deepEqual(removed, []);
});
