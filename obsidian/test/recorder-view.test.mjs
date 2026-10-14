// The recorder's draft lifecycle, the native editor it freezes, and image paste.
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { obsidianStub, srcText } from "./fakes.mjs";
import { fileBackedView } from "./recorder-harness.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// --- a real Markdown draft is the notes field ------------------------------------------------

test("the recorder is a controller that mounts its object into a native MarkdownView", () => {
  const { v } = fileBackedView();
  assert.equal(v instanceof obsidianStub.ItemView, false);
  assert.equal(v instanceof obsidianStub.MarkdownView, false);
});

test("a non-journal recording creates its draft inside Capture/_unfiled", async () => {
  const { v, created, editorLeaf } = fileBackedView({ type: "lecture" });
  v.recordingId = "20260823T143000Z-abcd1234";
  await v.createDraft();
  assert.deepEqual(created, [[
    "Capture/_unfiled/2026-08-23--recording--20260823T143000Z-abcd1234.slim-draft.md",
    "````slim-meeting\n\n````\n\n",
  ]]);
  assert.equal(v.draftRel, created[0][0]);
  assert.ok(editorLeaf.view instanceof obsidianStub.MarkdownView);
  assert.deepEqual(editorLeaf.opened, [created[0][0]]);
  assert.deepEqual(editorLeaf.view.editor.cursor, { line: 4, ch: 0 });
  assert.equal(editorLeaf.view.editor.focused, 1);
});

test("closing the setup tab reacquires a live leaf on the next microphone click", async () => {
  const { v, files, editorLeaf } = fileBackedView({ type: "lecture" });
  v.recordingId = "20260825T182801-b1b875b4";
  await v.createDraft();
  const draft = v.draftRel;

  const replacement = {
    app: v.app,
    view: null,
    opened: [],
    async openFile(file) {
      this.view = new obsidianStub.MarkdownView(this);
      this.view.file = file;
      this.opened.push(file.path);
    },
  };
  v.plugin.editorLeaf = editorLeaf; // the cached leaf was closed with the setup tab
  v.app.workspace.getLeavesOfType = () => [];
  v.app.workspace.getLeaf = () => replacement;

  await v.open();

  assert.equal(files.has(draft), true);
  assert.equal(v.draftRel, draft);
  assert.equal(v.editorLeaf, replacement);
  assert.deepEqual(replacement.opened, [draft]);
  assert.deepEqual(replacement.view.editor.cursor, { line: 4, ch: 0 });
});

test("a new meeting does not fall back to another session's cached pane", async () => {
  const { v, editorLeaf } = fileBackedView({ type: "lecture" });
  editorLeaf.view = new obsidianStub.MarkdownView(editorLeaf);
  const replacement = {
    app: v.app,
    view: null,
    opened: [],
    async openFile(file) {
      this.view = new obsidianStub.MarkdownView(this);
      this.view.file = file;
      this.opened.push(file.path);
    },
  };
  v.plugin.editorLeaf = editorLeaf;
  v.manager = {
    sessionForLeaf: (leaf) => leaf === editorLeaf ? { sessionId: "A" } : null,
    indexSession() {},
  };
  v.app.workspace.getLeavesOfType = () => [editorLeaf, replacement];
  v.app.workspace.getLeaf = () => replacement;
  v.recordingId = "20260827T061500-abcd1234";

  await v.createDraft();

  assert.equal(v.editorLeaf, replacement);
  assert.deepEqual(editorLeaf.opened, []);
  assert.equal(replacement.opened.length, 1);
});

test("a journal draft begins and remains inside the privacy floor", async () => {
  const { v, created } = fileBackedView({ type: "journal" });
  v.recordingId = "20260823T143000Z-abcd1234";
  await v.createDraft();
  assert.equal(created[0][0],
               "Journal/2026-08-23--recording--20260823T143000Z-abcd1234.slim-draft.md");
});

test("a start failure removes only an untouched plugin-created draft", async () => {
  const { v, files } = fileBackedView();
  v.recordingId = "20260823T143000Z-abcd1234";
  await v.createDraft();
  const path = v.draftRel;

  assert.equal(await v.removeUntouchedDraft(), true);
  assert.equal(files.has(path), false);
  assert.equal(v.draftRel, "");
});

test("a start failure preserves a draft once it contains user input", async () => {
  const { v, files } = fileBackedView();
  v.recordingId = "20260823T143000Z-abcd1234";
  await v.createDraft();
  const path = v.draftRel;
  files.get(path).text = "Do not lose this";

  assert.equal(await v.removeUntouchedDraft(), false);
  assert.equal(files.has(path), true);
  assert.equal(v.draftRel, path);
});

test("deleting a failed draft releases the controller for the next microphone click", async () => {
  const { v, files } = fileBackedView();
  v.recordingId = "20260824T210000-deadbeef";
  await v.createDraft();
  const deleted = v.draftRel;
  v.audioRel = "Attachments/_incoming/20260824T210000-deadbeef.webm";
  v.state = "error";
  files.delete(deleted);                    // Obsidian file-explorer deletion

  assert.equal(await v.reconcileMissingSessionFiles(), true);
  assert.equal(v.state, "idle");
  assert.equal(v.draftRel, "");
  assert.equal(v.recordingId, "");

  v.recordingId = "20260824T210100-cafebabe";
  await v.createDraft();
  assert.match(v.draftRel, /cafebabe\.slim-draft\.md$/);
});

test("delete and start over removes only the managed failed-session sources", async () => {
  const { v, files } = fileBackedView();
  v.recordingId = "20260824T210000-deadbeef";
  await v.createDraft();
  const draft = v.draftRel;
  v.audioRel = "Attachments/_incoming/20260824T210000-deadbeef.webm";
  files.set(v.audioRel, { path: v.audioRel, bytes: Buffer.from([1]) });
  files.set("Capture/keep.md", { path: "Capture/keep.md", text: "do not delete" });
  v.state = "error";

  await v.discardSession(false);

  assert.equal(files.has(draft), false);
  assert.equal(files.has("Attachments/_incoming/20260824T210000-deadbeef.webm"), false);
  assert.equal(files.has("Capture/keep.md"), true);
  assert.equal(v.state, "idle");
});

test("the inline object has no raw-Markdown escape hatch", () => {
  const source = srcText();
  assert.doesNotMatch(source, /Open raw Markdown/);
});

test("processing freezes the native editor and an error unlocks it", () => {
  const { v, editorLeaf } = fileBackedView();
  editorLeaf.view = new obsidianStub.MarkdownView(editorLeaf);
  editorLeaf.view.contentEl = { classes: new Set(), toggleClass(name, on) {
    if (on) this.classes.add(name); else this.classes.delete(name);
  } };
  v.editorLeaf = editorLeaf;
  v.freezeEditor(true);
  assert.equal(v.editorFrozen, true);
  assert.equal(editorLeaf.view.contentEl.classes.has("slim-recorder-draft-frozen"), true);
  v.freezeEditor(false);
  assert.equal(v.editorFrozen, false);
  assert.equal(editorLeaf.view.contentEl.classes.has("slim-recorder-draft-frozen"), false);
});

test("processing freezes only the native editor, never the meeting controls", () => {
  const css = readFileSync(path.join(HERE, "..", "styles.css"), "utf8");
  assert.match(css, /\.slim-recorder-draft-frozen \.cm-line:not\(:has\(\.slim-meeting-block\)\)[^{]*\{[^}]*pointer-events:\s*none/s);
  assert.match(css, /\.slim-recorder-draft-frozen \.slim-meeting-block[^{]*\{[^}]*pointer-events:\s*auto/s);
});

test("review panes own their scroll instead of painting over filing controls", () => {
  const css = readFileSync(path.join(HERE, "..", "styles.css"), "utf8");
  assert.match(css, /\.slim-recorder\s*>\s*\*\s*\{[^}]*flex:\s*0 0 auto/s);
  assert.match(css, /\.slim-summary-editor,[^{]*\.slim-notes-editor\s*\{[^}]*width:\s*100%[^}]*overflow:\s*auto[^}]*resize:\s*none/s);
});

for (const [mime, ext] of [["image/png", "png"], ["image/jpeg", "jpg"],
                           ["image/webp", "webp"]]) {
  test(`pasted ${mime} becomes a stable vault-relative embed`, async () => {
    const { v, binaries } = fileBackedView();
    v.recordingId = "paste-id";
    v.draftRel = "Capture/_unfiled/x.slim-draft.md";
    let prevented = false;
    const image = { type: mime, name: `clipboard.${ext}`,
                    arrayBuffer: async () => Uint8Array.from([1, 2, 3]).buffer };
    const event = { clipboardData: { files: [image] }, preventDefault() { prevented = true; } };
    const editor = { inserted: [], replaceSelection(text) { this.inserted.push(text); } };

    assert.equal(await v.handleEditorPaste(event, editor), true);
    assert.equal(prevented, true);
    assert.equal(binaries[0][0], `Attachments/Recorder/paste-id/paste-001.${ext}`);
    assert.deepEqual(editor.inserted, [
      `![[Attachments/Recorder/paste-id/paste-001.${ext}]]`,
    ]);
  });
}

test("image paste dedupes without overwriting an earlier attachment", async () => {
  const existing = ["Attachments/Recorder/paste-id/paste-001.png"];
  const { v, binaries } = fileBackedView({ existing });
  v.recordingId = "paste-id";
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  const image = { type: "image/png", name: "image.png",
                  arrayBuffer: async () => Uint8Array.from([4]).buffer };
  const event = { clipboardData: { files: [image] }, preventDefault() {} };
  const editor = { replaceSelection() {} };

  await v.handleEditorPaste(event, editor);
  assert.equal(binaries[0][0], "Attachments/Recorder/paste-id/paste-002.png");
});

test("ordinary text paste stays entirely native to Obsidian", async () => {
  const { v, binaries } = fileBackedView();
  v.recordingId = "paste-id";
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  let prevented = false;
  const event = { clipboardData: { files: [] }, preventDefault() { prevented = true; } };
  const editor = { replaceSelection() {} };

  assert.equal(await v.handleEditorPaste(event, editor), false);
  assert.equal(prevented, false);
  assert.equal(binaries.length, 0);
});

test("a paused recording survives an Obsidian restart", async () => {
  // Found in review: cancel parks the segments, their caches and the draft on DISK, but the
  // session state lives only in the view. After a restart the startup notice said "open the
  // draft" and the draft rendered as a passive block — no Resume, no Finish, no way back to
  // the audio except `slim inbox` by hand.
  const { v, files } = fileBackedView();
  const id = "20260826T210000-deadbeef";
  const draft = `Capture/_unfiled/2026-08-26--recording--${id}.slim-draft.md`;
  files.set(draft, { path: draft, text: "````slim-meeting\n````\n\n- typed while listening\n" });
  for (const n of ["001", "002"]) {
    const p = `Attachments/_incoming/${id}-${n}.webm`;
    files.set(p, { path: p, bytes: Buffer.from([1]) });
  }

  const adopted = await v.adoptPausedDraft(draft);

  assert.equal(adopted, true);
  assert.equal(v.state, "paused");
  assert.equal(v.recordingId, id);
  assert.equal(v.draftRel, draft);
  assert.deepEqual(v.segments, [`Attachments/_incoming/${id}-001.webm`,
                                `Attachments/_incoming/${id}-002.webm`]);
});

test("a draft with no staged audio is not adopted as paused", async () => {
  const { v, files } = fileBackedView();
  const draft = "Capture/_unfiled/2026-08-26--recording--nothing.slim-draft.md";
  files.set(draft, { path: draft, text: "````slim-meeting\n````\n" });
  assert.equal(await v.adoptPausedDraft(draft), false);
});
