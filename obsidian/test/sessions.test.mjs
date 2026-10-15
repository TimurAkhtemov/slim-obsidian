// Meeting sessions: one capture slot, stale setup drafts, paths that move.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { fileBackedView, sessionManagerHarness, view } from "./recorder-harness.mjs";

const require = createRequire(import.meta.url);
const { RecorderView } = require("../src/recorder/view.js");
const { MeetingSessionManager } = require("../src/recorder/sessions.js");

// --- several durable meeting sessions, one capture ------------------------------------------

test("a processing meeting does not hold the microphone hostage", async () => {
  const { manager, made } = sessionManagerHarness();
  const meetingA = manager.createSession();
  meetingA.state = "processing";
  meetingA.draftRel = "Capture/_unfiled/a.slim-draft.md";
  manager.indexSession(meetingA);

  const meetingB = await manager.activate();

  assert.notEqual(meetingB, meetingA);
  assert.equal(made.length, 2);
  assert.equal(manager.activeCaptureId, meetingB.sessionId);
  assert.equal(meetingB.opened, 1);
});

test("a new capture never reuses a pane owned by a processing meeting", async () => {
  const { manager } = sessionManagerHarness();
  const sharedLeaf = { name: "A pane" };
  const meetingA = manager.createSession(sharedLeaf);
  meetingA.editorLeaf = sharedLeaf;
  meetingA.state = "processing";

  const meetingB = await manager.activate(sharedLeaf);

  assert.equal(meetingA.editorLeaf, sharedLeaf);
  assert.notEqual(meetingB.editorLeaf, sharedLeaf,
    "A may finish and navigate its pane without replacing B");
});

test("reopening the microphone keeps an active capture in its own pane", async () => {
  const { manager } = sessionManagerHarness();
  const captureLeaf = { name: "capture pane" };
  const unrelatedLeaf = { name: "another pane" };
  const meeting = manager.createSession(captureLeaf);
  meeting.editorLeaf = captureLeaf;
  manager.claimCapture(meeting);

  const reopened = await manager.activate(unrelatedLeaf);

  assert.equal(reopened, meeting);
  assert.equal(meeting.editorLeaf, captureLeaf);
});

function setupDraftFrom(manager, recordingId, { untouched }) {
  const meeting = manager.createSession();
  meeting.recordingId = recordingId;
  meeting.draftRel = `Capture/_unfiled/x--recording--${recordingId}.slim-draft.md`;
  meeting.removed = 0;
  meeting.removeUntouchedDraft = async () => {
    if (!untouched) return false;
    meeting.removed += 1;
    meeting.draftRel = "";
    return true;
  };
  manager.indexSession(meeting);
  manager.claimCapture(meeting);
  return meeting;
}

function today() {
  const d = new Date();
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}` +
    `${String(d.getDate()).padStart(2, "0")}`;
}

test("an untouched setup draft from an earlier day is replaced, not reopened", async () => {
  const { manager, made } = sessionManagerHarness();
  const stale = setupDraftFrom(manager, "20260915T213136-d16525f8", { untouched: true });

  const fresh = await manager.activate();

  assert.notEqual(fresh, stale);
  assert.equal(made.length, 2);
  assert.equal(stale.removed, 1);
  assert.equal(manager.sessions.has(stale.sessionId), false);
  assert.equal(manager.activeCaptureId, fresh.sessionId);
});

test("a setup draft from today is reopened", async () => {
  const { manager, made } = sessionManagerHarness();
  const draft = setupDraftFrom(manager, `${today()}T090000-abcd1234`, { untouched: true });

  assert.equal(await manager.activate(), draft);
  assert.equal(draft.removed, 0);
  assert.equal(made.length, 1);
});

test("an earlier day's setup draft with their input in it is kept and reopened", async () => {
  const { manager, made } = sessionManagerHarness();
  const draft = setupDraftFrom(manager, "20260915T213136-d16525f8", { untouched: false });

  assert.equal(await manager.activate(), draft);
  assert.equal(made.length, 1);
});

test("a second microphone click reveals the active capture instead of creating C", async () => {
  const { manager, made } = sessionManagerHarness();
  const meetingA = manager.createSession();
  meetingA.state = "processing";

  const meetingB = await manager.activate();
  meetingB.state = "recording";
  const again = await manager.activate();

  assert.equal(again, meetingB);
  assert.equal(made.length, 2);
  assert.equal(meetingB.opened, 2, "the existing capture is revealed again");
});

test("paths and progress remain owned by the correct meeting", () => {
  const { manager } = sessionManagerHarness();
  const meetingA = manager.createSession();
  const meetingB = manager.createSession();
  meetingA.draftRel = "Capture/_unfiled/a.slim-draft.md";
  meetingB.draftRel = "Capture/_unfiled/b.slim-draft.md";
  meetingA.processingProgress = { label: "Transcribing A" };
  meetingB.processingProgress = { label: "Queued B" };
  manager.indexSession(meetingA);
  manager.indexSession(meetingB);

  assert.equal(manager.sessionForPath(meetingA.draftRel), meetingA);
  assert.equal(manager.sessionForPath(meetingB.draftRel), meetingB);
  assert.equal(manager.sessionForPath(meetingA.draftRel).processingProgress.label, "Transcribing A");
  assert.equal(manager.sessionForPath(meetingB.draftRel).processingProgress.label, "Queued B");

  const old = meetingA.draftRel;
  meetingA.draftRel = "";
  meetingA.filedNotePath = "Capture/project/a.md";
  manager.indexSession(meetingA);
  assert.equal(manager.sessionForPath(old), null);
  assert.equal(manager.sessionForPath("Capture/project/a.md"), meetingA);
  assert.equal(manager.sessionForPath(meetingB.draftRel), meetingB);
});

test("reopening and deleting one meeting cannot reset another", async () => {
  const { manager } = sessionManagerHarness();
  const meetingA = manager.createSession();
  const meetingB = manager.createSession();
  meetingA.state = "card";
  meetingA.filedNotePath = "Capture/project/a.md";
  meetingB.state = "idle";
  meetingB.draftRel = "Capture/_unfiled/b.slim-draft.md";
  manager.indexSession(meetingA);
  manager.indexSession(meetingB);
  manager.claimCapture(meetingB);

  const reopened = await manager.openPath(meetingA.filedNotePath);
  assert.equal(reopened, meetingA);
  assert.equal(meetingA.opened, 1);
  assert.equal(meetingB.state, "idle");

  manager.removeSession(meetingB);
  assert.equal(manager.activeCaptureId, null);
  assert.equal(manager.sessionForPath(meetingB.draftRel), null);
  assert.equal(manager.sessionForPath(meetingA.filedNotePath), meetingA);
  assert.equal(meetingA.state, "card");
});

test("only one meeting can own live capture resources", () => {
  const { manager } = sessionManagerHarness();
  const meetingA = manager.createSession();
  const meetingB = manager.createSession();

  manager.claimCapture(meetingA);
  meetingA.state = "recording";
  assert.throws(() => manager.claimCapture(meetingB), /already active/i);
  manager.releaseCapture(meetingA);
  assert.doesNotThrow(() => manager.claimCapture(meetingB));
});

test("an idle setup draft does not block Resume on another note", () => {
  // The ribbon's setup draft reserves the lease so a second click returns to it. It records
  // nothing, so a Resume elsewhere takes the lease over instead of reporting a recording
  // "already active" that they could not see (2026-09-05).
  const { manager } = sessionManagerHarness();
  const draft = manager.createSession();
  manager.claimCapture(draft);
  const finished = manager.createSession();
  finished.state = "finished";

  assert.doesNotThrow(() => manager.claimCapture(finished));
  assert.equal(manager.activeCaptureId, finished.sessionId);
});

test("recording from a displaced setup draft asks for the lease again", async () => {
  // Its lease taken by a Resume, the draft's Record button must not capture on top of it.
  const v = view();
  v.manager = { claimCapture() { throw new Error("another meeting capture is already active"); } };
  await assert.rejects(() => v.start(), /already active/);
});

test("a pending review survives plugin reload while historical notes stay finished", async () => {
  const path = "Capture/school/pending.md";
  const textFor = (status) => [
    "---", 'title: "Pending"', "type: lecture", "topics: [nlp]", "origin: recorded",
    ...(status ? [`review_status: ${status}`] : []), "---", "", "````slim-meeting",
    "## Transcript", "", "durable words", "````", "",
  ].join("\n");

  for (const [status, expected] of [["pending", "card"], ["", "finished"]]) {
    const { v, files, editorLeaf } = fileBackedView();
    const file = { path, basename: "pending", text: textFor(status) };
    files.set(path, file);
    await editorLeaf.openFile(file);
    v.app.metadataCache = { getFileCache: () => ({ frontmatter: {
      origin: "recorded", title: "Pending", type: "lecture", topics: ["nlp"],
    } }) };

    assert.equal(await v.openExisting(file, editorLeaf), true);
    assert.equal(v.state, expected);
    assert.equal(v.filedNotePath, path);
  }
});

test("a vault rename moves only the matching session path", () => {
  const { manager } = sessionManagerHarness();
  const meetingA = manager.createSession();
  const meetingB = manager.createSession();
  meetingA.filedNotePath = "Capture/_unfiled/a.md";
  meetingB.filedNotePath = "Capture/_unfiled/b.md";
  manager.indexSession(meetingA);
  manager.indexSession(meetingB);

  manager.handleVaultRename({ path: "Capture/project/a.md" }, "Capture/_unfiled/a.md");

  assert.equal(meetingA.filedNotePath, "Capture/project/a.md");
  assert.equal(manager.sessionForPath("Capture/_unfiled/a.md"), null);
  assert.equal(manager.sessionForPath("Capture/project/a.md"), meetingA);
  assert.equal(manager.sessionForPath("Capture/_unfiled/b.md"), meetingB);
});

test("deleting the active setup draft releases only that meeting", () => {
  const { manager } = sessionManagerHarness();
  const meetingA = manager.createSession();
  const meetingB = manager.createSession();
  meetingA.state = "processing";
  meetingA.draftRel = "Capture/_unfiled/a.slim-draft.md";
  meetingB.state = "idle";
  meetingB.draftRel = "Capture/_unfiled/b.slim-draft.md";
  manager.indexSession(meetingA);
  manager.indexSession(meetingB);
  manager.claimCapture(meetingB);

  manager.handleVaultDelete({ path: meetingB.draftRel });

  assert.equal(manager.activeCaptureId, null);
  assert.equal(manager.sessionForPath(meetingB.draftRel), null);
  assert.equal(manager.sessionForPath(meetingA.draftRel), meetingA);
  assert.equal(meetingA.state, "processing");
});

test("resume must acquire the same exclusive capture slot", async () => {
  const v = view();
  v.state = "finished";
  v.filedNotePath = "Capture/project/a.md";
  v.result = { title: "A", summary: "", notes_md: "", card: {} };
  v.noteParts = { summary: "", notes: "", transcript: "earlier words" };
  let claimed = null;
  v.manager = { claimCapture(session) { claimed = session; }, releaseCapture() {} };
  v.start = async () => {};

  await v.startAppendRecording();

  assert.equal(claimed, v);
});

test("resuming a paused draft also reacquires the capture slot", async () => {
  const v = view();
  v.state = "paused";
  let claimed = null;
  v.manager = { claimCapture(session) { claimed = session; }, releaseCapture() {} };
  v.start = async () => {};

  await v.resumeRecording();

  assert.equal(claimed, v);
});

test("the real recorder replaces a stale untouched setup draft with one named for today", async () => {
  const { v, files } = fileBackedView();
  const views = [v];
  const manager = new MeetingSessionManager({}, () => {
    if (views.length === 1) return views[0];
    const next = new RecorderView({ app: v.app }, v.plugin);
    next.app = v.app;
    next.type = v.type;
    next.render = () => {};
    next.attachEditorObject = () => {};
    return next;
  });
  const stale = manager.createSession();
  views.push(null);
  stale.recordingId = "20260915T213136-d16525f8";
  stale.render = () => {};
  stale.attachEditorObject = () => {};
  await stale.createDraft();
  manager.claimCapture(stale);
  const oldPath = stale.draftRel;

  const fresh = await manager.activate();

  assert.notEqual(fresh, stale);
  assert.equal(manager.sessions.has(stale.sessionId), false);
  assert.equal(files.has(oldPath), false);
  assert.match(fresh.draftRel, new RegExp(`^Capture/_unfiled/\\d{4}-\\d{2}-\\d{2}--recording--${today()}T`));
  assert.equal(files.has(fresh.draftRel), true);
  assert.equal(manager.activeCaptureId, fresh.sessionId);
});
