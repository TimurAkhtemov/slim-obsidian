// The meeting block's screens: processing, review, errors, the card.
import assert from "node:assert/strict";
import test from "node:test";
import { fakeContentEl, findClass } from "./fakes.mjs";
import { fileBackedView, findButton, view } from "./recorder-harness.mjs";

test("AI summary renders as Markdown until the user enters edit mode", () => {
  const v = view();
  v.activeTab = "summary";
  v.result = { summary: "## Key points\n\n- rendered", notes_md: "", card: {} };
  v.noteParts = { summary: v.result.summary, notes: "", transcript: "raw" };

  const previewRoot = fakeContentEl();
  v.renderSource(previewRoot, true);
  const preview = findClass(previewRoot, "slim-summary-preview");
  assert.ok(preview, "review starts with rendered Markdown");
  assert.equal(findClass(previewRoot, "slim-summary-editor"), null);

  preview.events.click({ target: {} });
  assert.equal(v.summaryEditing, true);

  const editRoot = fakeContentEl();
  v.renderSource(editRoot, true);
  const editor = findClass(editRoot, "slim-summary-editor");
  assert.ok(editor, "clicking the rendered summary swaps in Markdown text");
  assert.ok(findClass(editRoot, "slim-live-markdown-editor"),
            "formatting remains rendered while the caret is active");
  editor.events.input({ target: { value: "## Revised\n\n- exact Markdown" } });
  assert.equal(v.result.summary, "## Revised\n\n- exact Markdown");
  editor.events.blur();
  assert.equal(v.summaryEditing, false, "leaving the field returns to rendered mode");
});

test("recording and processing each render one clear interaction surface", () => {
  const v = view();
  v.recorder = { state: "recording" };
  const recording = fakeContentEl();
  v.renderRecording(recording);
  assert.ok(findClass(recording, "slim-recording-console"));
  assert.ok(findClass(recording, "slim-live-transcript"));
  assert.equal(findClass(recording, "slim-object-tabs"), null,
               "disabled review tabs do not crowd the recording controls");

  v.processingProgress = {
    label: "Writing the summary", detail: "Finding the important parts",
    thinking: "actual local reasoning", draft: '{"overview":"raw JSON is hidden"}',
    metrics: { transcript_words: 42, summary_output_tokens: 80 },
  };
  const processing = fakeContentEl();
  v.renderProcessing(processing);
  assert.ok(findClass(processing, "slim-thinking"));
  assert.equal(findClass(processing, "slim-thinking-copy"), null,
               "raw local-model reasoning is never rendered into the note");
  assert.equal(findClass(processing, "slim-progress-steps"), null);
  assert.equal(findClass(processing, "slim-model-stream"), null,
               "the normal UI never exposes the structured JSON console");
});

test("a meeting waiting on A names its queue position inside B's own block", () => {
  const v = view();
  v.title = "Meeting B";
  v.processingProgress = {
    stage: "queued",
    label: "Queued behind 1 recording",
    detail: "Waiting for the local recording pipeline",
    metrics: {},
  };

  const root = fakeContentEl();
  v.renderProcessing(root);
  const current = findClass(root, "slim-processing-current");
  assert.equal(current.children[1].children[0].options.text, "Queued behind 1 recording");
  assert.equal(current.children[1].children[1].options.text,
               "Waiting for the local recording pipeline");
});

test("an error block names the recording whose sources need recovery", () => {
  const v = view();
  v.title = "Client interview B";
  v.errorText = "ASR unavailable";
  v.draftRel = "Capture/_unfiled/b.slim-draft.md";
  v.audioRel = "Attachments/_incoming/b-001.webm";

  const root = fakeContentEl();
  v.renderError(root);

  const header = findClass(root, "slim-object-head");
  assert.equal(header.children[0].children[1].options.text, "Client interview B");
});

test("the notes field stays editable after the meeting is filed", () => {
  // Their words, and the meeting block hides them from Obsidian's own editor — so read-only
  // here means not editable anywhere, and reaching them meant reopening the review.
  const v = view();
  v.state = "finished";
  v.activeTab = "notes";
  v.result = { summary: "s", notes_md: "- typed live", card: null };
  v.noteParts = { summary: "s", notes: "- typed live", transcript: "raw" };

  const root = fakeContentEl();
  v.renderSource(root, false);
  assert.ok(findClass(root, "slim-notes-editor"), "notes are editable without entering review");
});

test("the AI summary stays behind the review gate", () => {
  const v = view();
  v.state = "finished";
  v.activeTab = "summary";
  v.result = { summary: "## Key points\n\n- one", notes_md: "", card: null };
  v.noteParts = { summary: v.result.summary, notes: "", transcript: "raw" };

  const root = fakeContentEl();
  v.renderSource(root, false);
  assert.equal(findClass(root, "slim-summary-editor"), null);
  const preview = findClass(root, "slim-summary-preview");
  assert.ok(preview && !preview.classes.has("is-editable"),
            "a derived summary is still edited deliberately, not by clicking into it");
});

test("the paused card shows what was captured, and pads its own content", async () => {
  const v = view();
  v.state = "paused";
  v.segments = ["Attachments/_incoming/x-001.webm"];
  v.plugin.recordTranscript = async () => ({ text: "the first part", pending: 0, words: 3 });

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  await Promise.resolve();
  await Promise.resolve();

  assert.ok(findClass(root, "slim-paused"), "paused content sits in its own padded body");
  assert.ok(findButton(root, "Resume recording"));
});

test("the review always says exactly where the durable note was saved", () => {
  const v = view();
  v.state = "card";
  v.filedNotePath = "Notes/school/CS203 NLP/Lecture Notes/Module 1/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {
    dest_dir: "Notes/school", type_tag: "lecture", topics: ["nlp"], folders: [],
  } };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();

  const saved = findClass(root, "slim-saved-location");
  assert.ok(saved);
  assert.equal(findClass(saved, "slim-saved-location-path").options.text, v.filedNotePath);
  assert.ok(findButton(saved, "Reveal"));
});

test("a reopened card fetches the vault's folders and tags, and redraws with them", async () => {
  // 2026-09-29: after a reload "browse the vault" was an empty box and every topic read "new":
  // only a fresh recording's response carried either list.
  const path = "Capture/job-search/pending.md";
  const { v, files, editorLeaf } = fileBackedView();
  const file = { path, basename: "pending", text: [
    "---", "origin: recorded", "review_status: pending", "---", "", "````slim-meeting",
    "## Transcript", "", "words", "````", "",
  ].join("\n") };
  files.set(path, file);
  await editorLeaf.openFile(file);
  v.app.metadataCache = { getFileCache: () => ({ frontmatter: { origin: "recorded" } }) };
  const asked = [];
  v.plugin.getJSON = async (route) => {
    asked.push(route);
    return { folders: [{ path: "Capture/job-search", count: 3 }], tag_counts: { "job-search": 3 } };
  };
  assert.equal(await v.openExisting(file, editorLeaf), true);
  await v.cardListsLoad;
  assert.deepEqual(asked, ["/api/record/folders"]);
  assert.deepEqual(v.result.card.folders, [{ path: "Capture/job-search", count: 3 }]);
  assert.deepEqual(v.result.card.tag_counts, { "job-search": 3 });
});
