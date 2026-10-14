// Filing: stop, retry, resume, cancel, notes and summary saves, and the server's moves.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { fakeContentEl, findClass, notices, resetNotices } from "./fakes.mjs";
import { fileBackedView, findButton, view } from "./recorder-harness.mjs";

const require = createRequire(import.meta.url);
const PluginClass = require("../src/plugin.js");
const { MeetingSessionManager } = require("../src/sessions.js");

// --- a failed transcript is refused, and retrying is one button ------------------------------
// The server used to file an empty note and return 200, so there was nothing to retry — it had
// already reported success (2026-08-23).

test("retry posts the SAME staged audio, not a new recording", async () => {
  const posted = [];
  // start() always creates a draft before recording begins, so a real submitRecording call
  // always carries `draft`, never `notes_md` (`submitRecording` in src/recorder-filing.js, D3).
  const { v } = fileBackedView();
  v.audioRel = "Attachments/_incoming/2026-08-23-100356.webm";
  v.draftRel = "Capture/_unfiled/2026-08-23-100356.slim-draft.md";
  v.saveDraft = async () => {};
  v.loadEditorFile = async () => {};
  v.type = "lecture";
  v.title = "  K-means  ";
  v.plugin.postRecording = async (body) => { posted.push(body); return { note: "Notes/x.md" }; };

  await v.submitRecording(v.audioRel);
  assert.deepEqual(posted, [{
    audio: "Attachments/_incoming/2026-08-23-100356.webm",
    draft: "Capture/_unfiled/2026-08-23-100356.slim-draft.md",
    type: "lecture",
    title: "K-means",
  }]);
  assert.equal(v.state, "card");
});

test("a failed post leaves the audio path intact so it can be retried", async () => {
  const v = view();
  v.audioRel = "Attachments/_incoming/x.webm";
  v.plugin.postRecording = async () => { throw new Error("transcription failed: ASR exploded"); };
  await assert.rejects(() => v.submitRecording(v.audioRel));
  assert.equal(v.audioRel, "Attachments/_incoming/x.webm");
});

test("filing saves the editor before posting the draft path", async () => {
  const { v } = fileBackedView();
  const order = [];
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  v.audioRel = "Attachments/_incoming/x.webm";
  v.title = "  K-means  ";
  v.saveDraft = async () => { order.push("save"); };
  v.loadEditorFile = async (p) => { order.push(`load:${p}`); };
  v.plugin.postRecording = async (body) => {
    order.push(["post", body]);
    return { note: "Notes/x.md", card: {}, title: "K-means" };
  };

  await v.submitRecording(v.audioRel);

  assert.deepEqual(order, [
    "save",
    ["post", { audio: "Attachments/_incoming/x.webm",
               draft: "Capture/_unfiled/x.slim-draft.md",
               type: "lecture", title: "K-means" }],
    "load:Notes/x.md",
  ]);
});

test("review approval sends editable user notes with the summary", async () => {
  const v = view();
  v.result = { summary: "edited summary", notes_md: "- added in review", card: {
    dest_dir: "Notes", type_tag: "lecture", topics: [],
  } };
  v.noteParts = { summary: "old", notes: "", transcript: "raw" };
  v.filedNotePath = "Notes/x.md";
  v.cardTitle = "Title";
  let body = null;
  v.plugin.applyCard = async (value) => { body = value; return {}; };

  await v.applyCard();

  assert.equal(body.summary_md, "edited summary");
  assert.equal(body.notes_md, "- added in review");
});

test("a failed draft post preserves both paths for the same retry", async () => {
  const { v } = fileBackedView();
  v.draftRel = "Capture/_unfiled/same.slim-draft.md";
  v.audioRel = "Attachments/_incoming/same.webm";
  v.plugin.postRecording = async () => { throw new Error("ASR failed"); };

  await assert.rejects(() => v.submitRecording(v.audioRel));
  assert.equal(v.draftRel, "Capture/_unfiled/same.slim-draft.md");
  assert.equal(v.audioRel, "Attachments/_incoming/same.webm");
});

test("a draft-save failure after Stop reaches the retry screen", async () => {
  const { v } = fileBackedView();
  v.audioRel = "Attachments/_incoming/same.webm";
  v.draftRel = "Capture/_unfiled/same.slim-draft.md";
  v.recorder = {
    stop() { queueMicrotask(() => this.onstop()); },
  };
  v.teardownStream = () => {};
  let audioSaved = false;
  v.writeAudio = async () => { audioSaved = true; return v.audioRel; };
  v.saveDraft = async () => { throw new Error("vault save failed"); };
  let renders = 0;
  v.render = () => { renders += 1; };

  await v.stop();

  assert.equal(v.state, "error");
  assert.match(v.errorText, /vault save failed/);
  assert.equal(v.editorFrozen, false);
  assert.equal(audioSaved, true);
  assert.equal(v.audioRel, "Attachments/_incoming/same.webm");
  assert.equal(v.draftRel, "Capture/_unfiled/same.slim-draft.md");
  assert.equal(renders, 1);
});

test("a draft-save failure during retry stays on the retry screen", async () => {
  const { v } = fileBackedView();
  v.audioRel = "Attachments/_incoming/same.webm";
  v.draftRel = "Capture/_unfiled/same.slim-draft.md";
  v.saveDraft = async () => { throw new Error("vault save failed again"); };
  let renders = 0;
  v.render = () => { renders += 1; };
  const buttons = {};
  const row = { createEl(tag, options) {
    return { addEventListener(event, callback) { buttons[options.text] = callback; } };
  } };
  const root = {
    createEl() { return {}; },
    createDiv() { return row; },
  };
  v.renderObjectHeader = () => {};
  v.renderError(root);

  await buttons["Try again"]();

  assert.equal(v.state, "error");
  assert.match(v.errorText, /vault save failed again/);
  assert.equal(v.editorFrozen, false);
  assert.equal(renders, 1);
});

test("leaving the notes field saves the notes and files nothing", async () => {
  const v = view();
  v.state = "finished";
  v.activeTab = "notes";
  v.filedNotePath = "Notes/x.md";
  v.result = { summary: "s", notes_md: "- one", card: null };
  v.noteParts = { summary: "s", notes: "- one", transcript: "raw" };
  let saved = null;
  v.plugin.saveNotes = async (body) => { saved = body; return { note: "Notes/x.md" }; };
  v.plugin.applyCard = async () => { throw new Error("typing a note is not a filing decision"); };

  const root = fakeContentEl();
  v.renderSource(root, false);
  const editor = findClass(root, "slim-notes-editor");
  editor.events.input({ target: { value: "- one\n- two" } });
  await editor.events.blur();

  assert.deepEqual(saved, { note: "Notes/x.md", notes_md: "- one\n- two" });
});

test("an unchanged notes field saves nothing on the way out", async () => {
  const v = view();
  v.state = "finished";
  v.activeTab = "notes";
  v.filedNotePath = "Notes/x.md";
  v.result = { summary: "s", notes_md: "- one", card: null };
  v.noteParts = { summary: "s", notes: "- one", transcript: "raw" };
  let calls = 0;
  v.plugin.saveNotes = async () => { calls += 1; return {}; };

  const root = fakeContentEl();
  v.renderSource(root, false);
  await findClass(root, "slim-notes-editor").events.blur();
  assert.equal(calls, 0, "every click away must not be a write to the vault");
});

test("retry summary re-runs the summarizer with their instructions", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "the bland one", notes_md: "", card: { dest_dir: "Notes" } };
  v.noteParts = { summary: "the bland one", notes: "", transcript: "raw" };
  let body = null;
  v.plugin.retrySummary = async (value) => {
    body = value;
    return { note: "Notes/x.md", summary: "## Key points\n\n### Kernels\n- the dual form" };
  };

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  const input = findClass(root, "slim-retry-instructions");
  assert.ok(input, "the steer is a field beside the button, not a hidden setting");
  input.events.input({ target: { value: "organise by concept" } });
  await findButton(root, "Retry summary").events.click();

  assert.equal(body.note, "Notes/x.md");
  assert.equal(body.instructions, "organise by concept");
  assert.match(v.result.summary, /### Kernels/);
  assert.equal(v.noteParts.transcript, "raw", "a retry never touches the record");
});

test("a failed retry keeps the summary that is already there", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "the bland one", notes_md: "", card: { dest_dir: "Notes" } };
  v.noteParts = { summary: "the bland one", notes: "", transcript: "raw" };
  v.plugin.retrySummary = async () => { throw new Error("ctx_saturated"); };

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  await findButton(root, "Retry summary").events.click();

  assert.equal(v.result.summary, "the bland one");
  assert.ok(notices.some((n) => /ctx_saturated/.test(n)), "the failure is said out loud");
});

test("the retry status says what the model is doing, and for how long", async () => {
  // "Live local-model activity" is the progress DETAIL and says nothing; the label does. A
  // wait of a minute with no moving number reads as a stall.
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "old", notes_md: "", card: { dest_dir: "Notes" } };
  v.noteParts = { summary: "old", notes: "", transcript: "raw" };
  const seen = [];
  v.plugin.watchRecording = (jobId, onProgress) => {
    onProgress({ stage: "summarizing", label: "Reasoning about the summary",
                 detail: "Live local-model activity" });
    seen.push(v.retryStatus);
    return () => {};
  };
  v.plugin.retrySummary = async () => ({ note: "Notes/x.md", summary: "new" });

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  await findButton(root, "Retry summary").events.click();

  assert.match(seen[0], /Reasoning about the summary/);
  assert.match(seen[0], /\d+s/, "the elapsed seconds move while they wait");
  assert.doesNotMatch(seen[0], /Live local-model activity/);
});

/* --- 2026-08-26: an accidental Stop is recoverable ------------------------------------ */

test("a resumed recording writes a new segment beside the first", async () => {
  const { v } = fileBackedView();
  v.recordingId = "20260826T090000-abcd1234";
  v.segments = ["Attachments/_incoming/20260826T090000-abcd1234-001.webm"];
  assert.equal(v.nextSegmentPath(), "Attachments/_incoming/20260826T090000-abcd1234-002.webm");
  v.segments = [];
  assert.equal(v.nextSegmentPath(), "Attachments/_incoming/20260826T090000-abcd1234-001.webm");
});

test("finishing a resumed recording posts every segment in order", async () => {
  const { v } = fileBackedView();
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  v.segments = ["Attachments/_incoming/x-001.webm", "Attachments/_incoming/x-002.webm"];
  v.saveDraft = async () => {};
  v.loadEditorFile = async () => {};
  let body = null;
  v.plugin.postRecording = async (value) => {
    body = value;
    return { note: "Notes/x.md", card: {}, title: "T" };
  };
  await v.submitRecording(v.segments);
  assert.deepEqual(body.audio, ["Attachments/_incoming/x-001.webm",
                                "Attachments/_incoming/x-002.webm"]);
});

test("one segment still posts a bare path", async () => {
  const { v } = fileBackedView();
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  v.saveDraft = async () => {};
  v.loadEditorFile = async () => {};
  let body = null;
  v.plugin.postRecording = async (value) => {
    body = value;
    return { note: "Notes/x.md", card: {}, title: "T" };
  };
  await v.submitRecording(["Attachments/_incoming/x-001.webm"]);
  assert.equal(body.audio, "Attachments/_incoming/x-001.webm");
});

test("a cancelled pass parks the session instead of failing it", async () => {
  // Their case: Stop hit by accident, 40 minutes in. A cancel is not an error — the draft and
  // every segment stay on disk and they carry on recording into the same note.
  const { v } = fileBackedView();
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  v.segments = ["Attachments/_incoming/x-001.webm"];
  v.state = "processing";
  v.saveDraft = async () => {};
  v.plugin.postRecording = async () => {
    const err = new Error("cancelled — nothing was written or deleted");
    err.cancelled = true;
    throw err;
  };

  await v.submitRecording(v.segments).catch(() => {});

  assert.equal(v.state, "paused");
  assert.equal(v.draftRel, "Capture/_unfiled/x.slim-draft.md", "the draft is not released");
  assert.deepEqual(v.segments, ["Attachments/_incoming/x-001.webm"]);
  assert.equal(v.errorText || "", "", "a cancel is not an error screen");
});

test("the paused object offers both ways out", () => {
  const v = view();
  v.state = "paused";
  v.segments = ["a-001.webm", "a-002.webm"];
  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  assert.ok(findButton(root, "Resume recording"));
  assert.ok(findButton(root, "Finish"), "a cancel never traps them in the paused state");
});

test("processing offers a cancel that asks the server to stop", async () => {
  const v = view();
  v.state = "processing";
  v.recordingId = "job-9";
  v.processingProgress = { label: "Transcribing", detail: "", steps: [], metrics: {} };
  let asked = null;
  v.plugin.cancelRecording = async (body) => { asked = body; return { cancelling: true }; };

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  await findButton(root, "Cancel").events.click();

  assert.deepEqual(asked, { job_id: "job-9" });
});

test("discarding a resumed session removes every segment", async () => {
  const { v, files } = fileBackedView();
  v.recordingId = "20260826T210000-deadbeef";
  await v.createDraft();
  const draft = v.draftRel;
  v.segments = ["Attachments/_incoming/20260826T210000-deadbeef-001.webm",
                "Attachments/_incoming/20260826T210000-deadbeef-002.webm"];
  v.segments.forEach((p) => files.set(p, { path: p, bytes: Buffer.from([1]) }));
  files.set("Capture/keep.md", { path: "Capture/keep.md", text: "do not delete" });
  v.state = "error";

  await v.discardSession(false);

  assert.equal(files.has(draft), false);
  assert.equal(files.has("Attachments/_incoming/20260826T210000-deadbeef-001.webm"), false);
  assert.equal(files.has("Attachments/_incoming/20260826T210000-deadbeef-002.webm"), false);
  assert.equal(files.has("Capture/keep.md"), true);
});

// The server writes `<segment>.transcript.json` outside the Vault API, so Obsidian's index may
// not hold it yet (or at all). Removing it through the index left it on disk with its audio gone:
// eleven such orphans had piled up in _incoming by 2026-09-28.
function unindexedCaches(files, vault) {
  vault.getAbstractFileByPath = (p) => (p.endsWith(".transcript.json") ? null : files.get(p) || null);
}

test("discarding a session removes each segment's cached transcript with its audio", async () => {
  const { v, files } = fileBackedView();
  unindexedCaches(files, v.app.vault);
  v.recordingId = "20260826T210000-deadbeef";
  await v.createDraft();
  const seg = "Attachments/_incoming/20260826T210000-deadbeef-001.webm";
  v.segments = [seg];
  files.set(seg, { path: seg, bytes: Buffer.from([1]) });
  files.set(`${seg}.transcript.json`, { path: `${seg}.transcript.json`, text: "{}" });
  v.state = "error";

  await v.discardSession(false);

  assert.equal(files.has(seg), false);
  assert.equal(files.has(`${seg}.transcript.json`), false);
});

test("a silent resumed segment is dropped together with its cached transcript", async () => {
  const { v, files } = fileBackedView();
  unindexedCaches(files, v.app.vault);
  const seg = "Attachments/_incoming/20260826T210000-deadbeef-002.webm";
  files.set(seg, { path: seg, bytes: Buffer.from([1]) });
  files.set(`${seg}.transcript.json`, { path: `${seg}.transcript.json`, text: "{}" });
  v.state = "processing";
  v.recordingId = "20260826T210000-deadbeef";
  v.appendTo = "Notes/x.md";
  v.segments = [seg];
  v.plugin.appendRecording = async () => {
    const err = new Error("no words were captured — nothing was added");
    err.silent = true;
    throw err;
  };

  await v.submitAppend(v.segments);

  assert.equal(files.has(seg), false);
  assert.equal(files.has(`${seg}.transcript.json`), false);
});

test("each attempt gets its own job, so a cancelled one cannot haunt the next", async () => {
  // Seen live: Stop after a resume showed "Cancelled" for the whole run. The job id was the
  // recording id, so the poll found the PREVIOUS attempt's finished-and-cancelled record and
  // stopped watching immediately.
  const { v } = fileBackedView();
  v.recordingId = "rec-1";
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  v.saveDraft = async () => {};
  v.loadEditorFile = async () => {};
  const jobs = [];
  v.plugin.watchRecording = (jobId) => { jobs.push(jobId); return () => {}; };
  v.plugin.postRecording = async (body) => {
    jobs.push(body.job_id);
    return { note: "Notes/x.md", card: {}, title: "T" };
  };

  await v.submitRecording(["Attachments/_incoming/rec-1-001.webm"]);
  await v.submitRecording(["Attachments/_incoming/rec-1-001.webm"]);

  assert.equal(jobs[0], jobs[1], "the watch and the post agree within one attempt");
  assert.notEqual(jobs[0], jobs[2], "a second attempt is a different job");
});

test("cancel names the job that is actually running", async () => {
  const v = view();
  v.state = "processing";
  v.recordingId = "rec-1";
  v.jobId = "rec-1-2";
  let asked = null;
  v.plugin.cancelRecording = async (body) => { asked = body; return { cancelling: true }; };
  await v.cancelProcessing();
  assert.deepEqual(asked, { job_id: "rec-1-2" });
});

test("notes typed during review save themselves too", async () => {
  // They asked what happens if they never clicks Approve & finish: the note is already filed, but
  // edits made IN the review were only written by that button. The one state the autosave did
  // not cover was the one they were most likely to walk away from.
  const v = view();
  v.state = "card";
  v.activeTab = "notes";
  v.filedNotePath = "Notes/x.md";
  v.result = { summary: "s", notes_md: "- one", card: {} };
  v.noteParts = { summary: "s", notes: "- one", transcript: "raw" };
  let saved = null;
  v.plugin.saveNotes = async (body) => { saved = body; return {}; };

  const root = fakeContentEl();
  v.renderSource(root, true);
  const editor = findClass(root, "slim-notes-editor");
  editor.events.input({ target: { value: "- one\n- two" } });
  await editor.events.blur();

  assert.deepEqual(saved, { note: "Notes/x.md", notes_md: "- one\n- two" });
});

test("approval without edits durably completes review", async () => {
  const v = view();
  v.state = "card";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {
    dest_dir: "Notes", type_tag: "lecture", topics: [], folders: [],
  } };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  let completed = null;
  v.plugin.completeReview = async (body) => { completed = body; return body; };
  v.loadNoteParts = async () => v.noteParts;
  const root = fakeContentEl();
  v.contentEl = root;
  v.render();

  await findButton(root, "Approve & finish").events.click();

  assert.deepEqual(completed, { note: "Notes/x.md" });
  assert.equal(v.state, "finished");
});

test("a filed note can be resumed from review and from finished", () => {
  const v = view();
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: { dest_dir: "Notes" } };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };

  for (const state of ["card", "finished"]) {
    v.state = state;
    const root = fakeContentEl();
    v.contentEl = root;
    v.render();
    assert.ok(findButton(root, "Resume recording"), `${state} offers resume`);
  }
});

test("resuming a filed note appends to it instead of making a second note", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {} };
  v.noteParts = { summary: "s", notes: "", transcript: "the first part" };
  v.startAppendRecording = async () => { v.appendTo = v.filedNotePath; };

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  await findButton(root, "Resume recording").events.click();
  assert.equal(v.appendTo, "Notes/x.md", "the note it will grow is remembered");

  let body = null;
  v.plugin.appendRecording = async (value) => { body = value; return { note: "Notes/x.md", words: 4 }; };
  v.loadNoteParts = async () => v.noteParts;
  v.loadEditorFile = async () => {};
  v.segments = ["Attachments/_incoming/y-001.webm"];
  await v.submitRecording(v.segments);

  assert.equal(body.note, "Notes/x.md");
  assert.deepEqual(body.audio, ["Attachments/_incoming/y-001.webm"]);
  assert.equal(v.state, "finished", "an append returns to the note, not to a review card");
});

test("resuming a filed note starts from the transcript it already has", () => {
  // Seen live: resume from a finished note showed an EMPTY live transcript, so the words
  // already in the note looked gone. The pane is the recording's transcript, not this
  // segment's.
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {} };
  v.noteParts = { summary: "s", notes: "", transcript: "everything said before" };
  v.start = async () => {};

  return v.startAppendRecording().then(() => {
    assert.equal(v.shownTranscript(), "everything said before");
  });
});

test("resuming from review returns to review, not past it", async () => {
  // Seen live: resume from the AI-review card and the note filed itself on Stop — the approval
  // step they were standing in was skipped entirely.
  const v = view();
  v.state = "card";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: { dest_dir: "Notes" } };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  v.start = async () => {};
  v.loadEditorFile = async () => {};
  v.loadNoteParts = async () => v.noteParts;
  v.plugin.appendRecording = async () => ({ note: "Notes/x.md", words: 4 });

  await v.startAppendRecording();
  v.segments = ["Attachments/_incoming/y-001.webm"];
  await v.submitRecording(v.segments);

  assert.equal(v.state, "card", "they still gets to approve the note they were approving");
});

test("resuming from a finished note returns to finished", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {} };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  v.start = async () => {};
  v.loadEditorFile = async () => {};
  v.loadNoteParts = async () => v.noteParts;
  v.plugin.appendRecording = async () => ({ note: "Notes/x.md", words: 4 });

  await v.startAppendRecording();
  v.segments = ["Attachments/_incoming/y-001.webm"];
  await v.submitRecording(v.segments);
  assert.equal(v.state, "finished");
});

test("a recording with no words says so quietly and offers no scary choice", async () => {
  // Their words: "it's really not an error... it shouldn't be so loud and in your face", and they
  // should never be asked whether to delete their own audio over a silent take.
  const v = view();
  v.state = "processing";
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  v.segments = ["Attachments/_incoming/x-001.webm"];
  v.saveDraft = async () => {};
  v.plugin.postRecording = async () => {
    const err = new Error("no words were captured — nothing was filed");
    err.silent = true;
    throw err;
  };

  await v.submitRecording(v.segments);

  assert.equal(v.state, "paused", "back to the ordinary way forward");
  assert.equal(v.errorText || "", "", "no error screen");
  assert.ok(notices.some((n) => /no words/i.test(n)));
});

test("try again re-posts the whole recording, not its last segment", async () => {
  // ⚠ A LOST-RECORDING BUG, found in review. `Try again` posted `this.audioRel`, which is the
  // CURRENT segment. A resumed recording that failed and was retried would have filed the
  // second half and dropped the first, with the error screen saying nothing was lost.
  const v = view();
  v.state = "error";
  v.errorText = "boom";
  v.draftRel = "Capture/_unfiled/x.slim-draft.md";
  v.segments = ["Attachments/_incoming/x-001.webm", "Attachments/_incoming/x-002.webm"];
  v.audioRel = "Attachments/_incoming/x-002.webm";
  v.saveDraft = async () => {};
  v.loadEditorFile = async () => {};
  let body = null;
  v.plugin.postRecording = async (value) => {
    body = value;
    return { note: "Notes/x.md", title: "T",
             card: { dest_dir: "Notes", topics: [], folders: [] } };
  };

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  await findButton(root, "Try again").events.click();

  assert.deepEqual(body.audio, ["Attachments/_incoming/x-001.webm",
                                "Attachments/_incoming/x-002.webm"]);
});

test("resuming a filed note does not mint a stray draft", async () => {
  // ⚠ DATA LOSS, found in review. `start()` calls `ensureDraftLocation()`, which creates a
  // draft whenever `draftRel` is empty — and it is empty for a filed note. So every resume
  // created an orphan in Capture/_unfiled, switched the editor to it, and anything typed while
  // resuming went there and was never merged into the note or cleaned up.
  const { v, files } = fileBackedView();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {} };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  let madeDraft = 0;
  v.createDraft = async () => { madeDraft += 1; return "Capture/_unfiled/stray.slim-draft.md"; };
  v.buildStream = async () => { throw new Error("stop here — the draft decision is made"); };

  await v.startAppendRecording().catch(() => {});

  assert.equal(madeDraft, 0, "a filed note IS the draft; a second one orphans their typing");
  assert.equal(v.draftRel, "");
  void files;
});

test("saving notes moves the baseline, so one edit is one write", async () => {
  // Found in review: the blur handler closed over a baseline captured at render time and
  // `saveNotes` never moved it, so every later click away re-posted the same notes — each one
  // a full note rewrite plus an index pass.
  const v = view();
  v.state = "finished";
  v.activeTab = "notes";
  v.filedNotePath = "Notes/x.md";
  v.result = { summary: "s", notes_md: "- one", card: null };
  v.noteParts = { summary: "s", notes: "- one", transcript: "raw" };
  let calls = 0;
  v.plugin.saveNotes = async () => { calls += 1; return {}; };

  const root = fakeContentEl();
  v.renderSource(root, false);
  const editor = findClass(root, "slim-notes-editor");
  editor.events.input({ target: { value: "- one\n- two" } });
  await editor.events.blur();
  await editor.events.blur();
  await editor.events.blur();

  assert.equal(calls, 1, "three click-aways after one edit is one write");
});

test("notes typed and never blurred are saved when the meeting is flushed", async () => {
  // Quitting Obsidian, or closing the note, with the caret in the field fires no blur; the
  // last edit was lost (BACKLOG). The manager's flush is what `quit` and the block unload call.
  const v = view();
  v.state = "finished";
  v.activeTab = "notes";
  v.filedNotePath = "Notes/x.md";
  v.result = { summary: "s", notes_md: "- one", card: null };
  v.noteParts = { summary: "s", notes: "- one", transcript: "raw" };
  const saved = [];
  v.plugin.saveNotes = async (body) => { saved.push(body.notes_md); return {}; };
  const manager = new MeetingSessionManager(v.plugin, () => v);
  manager.createSession();

  const root = fakeContentEl();
  v.renderSource(root, false);
  findClass(root, "slim-notes-editor").events.input({ target: { value: "- one\n- two" } });
  await manager.flushNotes();
  await manager.flushNotes();

  assert.deepEqual(saved, ["- one\n- two"]);
});

test("a blur and a flush in the same moment write the notes once", async () => {
  const v = view();
  v.state = "finished";
  v.activeTab = "notes";
  v.filedNotePath = "Notes/x.md";
  v.result = { summary: "s", notes_md: "- one", card: null };
  v.noteParts = { summary: "s", notes: "- one", transcript: "raw" };
  let calls = 0;
  let land;
  v.plugin.saveNotes = () => { calls += 1; return new Promise((r) => { land = r; }); };

  const root = fakeContentEl();
  v.renderSource(root, false);
  const editor = findClass(root, "slim-notes-editor");
  editor.events.input({ target: { value: "- one\n- two" } });
  const blurred = editor.events.blur();
  const flushed = v.flushNotes();
  land({});
  await Promise.all([blurred, flushed]);

  assert.equal(calls, 1);
});

test("a retry updates its status without rebuilding the card under the caret", async () => {
  // Found in review: the retry watcher called render() on every 600ms poll while the state is
  // `card`/`finished` — the states whose DOM holds the live notes editor and the instructions
  // field. A 60-300s retry destroyed and rebuilt them a few hundred times.
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "old", notes_md: "", card: { dest_dir: "Notes" } };
  v.noteParts = { summary: "old", notes: "", transcript: "raw" };
  let renders = 0;
  const realRender = v.render.bind(v);
  v.render = () => { renders += 1; realRender(); };
  v.plugin.watchRecording = (jobId, onProgress) => {
    for (let i = 0; i < 20; i += 1) onProgress({ label: "Reasoning about the summary" });
    return () => {};
  };
  v.plugin.retrySummary = async () => ({ note: "Notes/x.md", summary: "new" });

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  renders = 0;
  await findButton(root, "Retry summary").events.click();

  assert.ok(renders <= 3, `20 progress ticks caused ${renders} rebuilds`);
});

test("leaving a note mid-retry does not paint its summary onto the next one", async () => {
  // Found in review: `resetSession` cleared twenty fields but none of the retry/append ones,
  // so an in-flight retry on note A resolved into note B's view.
  const v = view();
  v.retrying = true;
  v.retryStatus = "Reasoning…";
  v.appendTo = "Notes/a.md";
  v.resumeReturnState = "card";
  v.pausedTranscript = "stale";
  v.jobId = "a-1";

  v.resetSession();

  assert.equal(v.retrying, false);
  assert.equal(v.appendTo, "");
  assert.equal(v.resumeReturnState, "");
  assert.equal(v.pausedTranscript, null);
  assert.equal(v.jobId, "");
});

test("a retry that finishes after they moved on writes nothing into the new note", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/a.md";
  v.result = { title: "A", summary: "a's summary", notes_md: "", card: { dest_dir: "Notes" } };
  v.noteParts = { summary: "a's summary", notes: "", transcript: "raw" };
  v.plugin.retrySummary = async () => {
    v.filedNotePath = "Notes/b.md";           // they clicked into another note meanwhile
    v.result = { title: "B", summary: "b's summary", notes_md: "", card: { dest_dir: "Notes" } };
    return { note: "Notes/a.md", summary: "a's NEW summary" };
  };

  const root = fakeContentEl();
  v.contentEl = root;
  v.render();
  await findButton(root, "Retry summary").events.click();

  assert.equal(v.result.summary, "b's summary", "B keeps its own summary");
});

test("cancel does not claim success when the job is not registered yet", async () => {
  // The click can beat the POST that registers the job. `/cancel` then answers
  // {cancelling:false}, the recording proceeds, and the note is filed — under a notice that
  // said nothing would be written.
  const v = view();
  v.state = "processing";
  v.jobId = "job-1";
  v.plugin.cancelRecording = async () => ({ cancelling: false });
  await v.cancelProcessing();
  assert.ok(notices.some((n) => /could not|not stopped|try again/i.test(n)),
            `notice was: ${notices.join(" | ")}`);
  assert.ok(!notices.some((n) => /Nothing is written or deleted/.test(n)));
});

test("a resume that cannot get the microphone does not leave a phantom recording", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {} };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  v.start = async () => { throw new Error("no audio source available"); };

  await v.startAppendRecording();

  assert.notEqual(v.state, "recording", "no recording UI without a recorder");
  assert.equal(v.appendTo, "", "and no half-started append left behind");
});

test("an append keeps its recovery state until the note is actually open", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Notes/x.md";
  v.appendTo = "Notes/x.md";
  v.resumeReturnState = "finished";
  v.segments = ["Attachments/_incoming/y-001.webm"];
  v.result = { title: "T", summary: "s", notes_md: "", card: {} };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  v.plugin.appendRecording = async () => ({ note: "Notes/x.md", words: 4 });
  v.loadEditorFile = async () => { throw new Error("could not open the note"); };

  await v.submitAppend(v.segments).catch(() => {});

  assert.equal(v.appendTo, "", "the append is committed — it must not be retried as one");
  assert.notEqual(v.state, "error", "a committed append is not a failure");
});

test("stopping releases the capture before its processing request begins", async () => {
  const v = view();
  let released = false;
  v.manager = { releaseCapture(session) { assert.equal(session, v); released = true; } };
  v.recorder = { stop() { queueMicrotask(() => this.onstop()); } };
  v.stopLiveTranscript = async () => {};
  v.teardownStream = () => {};
  v.segments = [];
  v.writeAudio = async () => "Attachments/_incoming/a-001.webm";
  v.saveDraft = async () => {};
  v.freezeEditor = () => {};
  v.render = () => {};
  v.submitRecording = async () => {
    assert.equal(released, true, "meeting B may reserve capture while A processes");
  };

  await v.stop();
  assert.equal(released, true);
});

test("filing reindexes only the session whose draft moved", async () => {
  const { v } = fileBackedView();
  v.draftRel = "Capture/_unfiled/a.slim-draft.md";
  v.audioRel = "Attachments/_incoming/a-001.webm";
  const indexed = [];
  v.manager = { indexSession(session) { indexed.push([session.draftRel, session.filedNotePath]); } };
  v.plugin.postRecording = async () => ({
    note: "Capture/project/a.md", title: "A", summary: "", notes_md: "",
    card: { dest_dir: "Capture/project", topics: [], folders: [] },
  });
  v.loadEditorFile = async () => {};
  v.loadNoteParts = async () => v.noteParts;

  await v.submitRecording(v.audioRel);

  assert.ok(indexed.some(([draft, note]) => draft === "" && note === "Capture/project/a.md"));
});

test("filing indexes the note before the editor opens it, so file-open finds the session", async () => {
  // Obsidian fires `file-open` while `openFile` runs. With the filed path still unindexed at
  // that moment the plugin adopted the note as a SECOND session: two cards on one block,
  // approved separately, and a "review still pending" notice on close for the one they never
  // saw (2026-09-05, three of five lectures approved twice).
  const { v } = fileBackedView();
  v.draftRel = "Capture/_unfiled/a.slim-draft.md";
  v.audioRel = "Attachments/_incoming/a-001.webm";
  const paths = new Map();
  v.manager = { indexSession(session) {
    paths.clear();
    for (const p of [session.draftRel, session.filedNotePath]) if (p) paths.set(p, session);
  } };
  v.plugin.postRecording = async () => ({
    note: "Capture/project/a.md", title: "A", summary: "", notes_md: "",
    card: { dest_dir: "Capture/project", topics: [], folders: [] },
  });
  let trackedAtOpen = null;
  v.loadEditorFile = async (path) => { trackedAtOpen = paths.get(path) || null; };
  v.loadNoteParts = async () => v.noteParts;

  await v.submitRecording(v.audioRel);

  assert.equal(trackedAtOpen, v, "the filed path is this session's before the leaf opens it");
  assert.equal(paths.get("Capture/_unfiled/a.slim-draft.md"), undefined);
});

test("the server's move of a note under review is not their deletion of it", async () => {
  // `apply` writes the new path and unlinks the old one; Obsidian reports the unlink as a
  // delete, usually before the response lands. Read as their deletion, it reset the session
  // mid-apply: "the deleted note was released" and then "Cannot set properties of null
  // (setting 'note')" on the Approve button (2026-09-04).
  const src = "Capture/project/a.md";
  const dest = "Capture/other/a.md";
  const { v, files } = fileBackedView({ existing: [src] });
  resetNotices();
  const manager = new MeetingSessionManager(v.plugin, () => v);
  manager.createSession();
  v.state = "card";
  v.filedNotePath = src;
  v.audioRel = "Attachments/_incoming/a-001.webm";
  v.result = { note: src, title: "A", summary: "", notes_md: "",
               card: { dest_dir: "Capture/other", type_tag: "lecture", topics: [] } };
  manager.indexSession(v);
  v.plugin.applyCard = async () => {
    files.delete(src);
    files.set(dest, { path: dest, text: "" });
    manager.handleVaultDelete({ path: src });
    return { note: dest };
  };

  await v.applyCard();

  assert.equal(v.filedNotePath, dest);
  assert.equal(v.state, "card");
  assert.ok(manager.sessions.has(v.sessionId), "the session outlives its own move");
  assert.equal(manager.sessionForPath(dest), v);
  assert.equal(manager.sessionForPath(src), null);
  assert.ok(!notices.some((n) => /released/.test(n)), notices.join(" | "));
});

test("approving a moved note does not reopen a leaf that already shows it", async () => {
  // A move is a RENAME, and Obsidian renames in place: the leaf keeps its file object and the
  // object's path changes. `applyCard` then reopened that same file in that same leaf — the one
  // step approve-with-edits took that approve-without did not, when "Cannot set properties of
  // null" surfaced after changing the suggested location.
  const moved = "Capture/school/x.md";
  const { v, files, editorLeaf } = fileBackedView({ existing: [moved] });
  await editorLeaf.openFile(files.get(moved));
  editorLeaf.opened = [];
  v.editorLeaf = editorLeaf;
  v.state = "card";
  v.edited = true;
  v.filedNotePath = "Capture/_unfiled/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {
    dest_dir: "Capture/school", type_tag: "lecture", topics: [], folders: [],
  } };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  v.plugin.applyCard = async () => ({ note: moved });
  v.plugin.meetingBlockFor = () => null;

  await v.applyCard();

  assert.deepEqual(editorLeaf.opened, [], "the note is already on screen");
  assert.equal(v.filedNotePath, moved);
});

test("a filing the server completed finishes the review even when the pane cannot be refreshed", async () => {
  // The server had moved the note; the editor refresh then threw, and the handler reported the
  // FILING as failed: a raw TypeError in a notice, Approve re-enabled, the session left at
  // `card` — so the next leaf change announced "Review is still pending" for a note that was
  // done. The server's answer is the truth; a pane that cannot be refreshed is a console matter.
  const v = view();
  v.state = "card";
  v.edited = true;
  v.filedNotePath = "Capture/_unfiled/x.md";
  v.result = { title: "T", summary: "s", notes_md: "", card: {
    dest_dir: "Capture/school", type_tag: "lecture", topics: [], folders: [],
  } };
  v.noteParts = { summary: "s", notes: "", transcript: "raw" };
  v.editorLeaf = { view: { file: { path: "Capture/_unfiled/x.md" } } };   // not yet on screen
  v.plugin.applyCard = async () => ({ note: "Capture/school/x.md" });
  v.loadEditorFile = async () => { throw new TypeError("Cannot set properties of null (setting 'cm')"); };
  v.loadNoteParts = async () => v.noteParts;
  const root = fakeContentEl();
  v.contentEl = root;
  v.render();

  await findButton(root, "Approve & finish").events.click();

  assert.equal(v.state, "finished");
  assert.equal(v.filedNotePath, "Capture/school/x.md");
  assert.ok(!notices.some((n) => /Cannot set properties/.test(n)), notices.join(" | "));

  // ...and closing that leaf later must not announce a review that is done.
  v.plugin.app = { workspace: { getLeavesOfType: () => [] } };
  v.plugin.meetings = { sessions: new Map([["x", v]]) };
  PluginClass.prototype.noticeClosedPendingReviews.call(v.plugin);
  assert.ok(!notices.some((n) => /Review is still pending/.test(n)), notices.join(" | "));
});
