/*
 * Headless tests for the recorder plugin. Run: `node --test obsidian/`
 *
 * WHY THIS EXISTS. Nothing tested this file, and it showed: on 2026-08-22 the first recording
 * made the way they actually works — Obsidian launched from the Dock — failed outright, and the
 * defects behind it had sat through two weeks of review. Every one was found by RUNNING the
 * thing. `main.js` has no build step and no test framework, so this is plain `node --test`
 * with `Module._load` patched to serve stubs for `obsidian` and `@electron/remote`.
 *
 * Scope is the recorder's behavioral contract: acquisition, draft lifecycle, native-view
 * handoff, recovery, and image paste. Pixel rendering remains Obsidian's responsibility.
 */
import assert from "node:assert/strict";
import Module from "node:module";
import test from "node:test";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { readFileSync } from "node:fs";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// --- stubs ---------------------------------------------------------------------------------

class FakeTrack {
  constructor(kind) { this.kind = kind; this.stopped = false; }
  stop() { this.stopped = true; }
}

class FakeStream {
  constructor(kinds) { this.tracks = kinds.map((k) => new FakeTrack(k)); }
  getTracks() { return this.tracks; }
  getAudioTracks() { return this.tracks.filter((t) => t.kind === "audio"); }
  getVideoTracks() { return this.tracks.filter((t) => t.kind === "video"); }
}

let notices = [];
let noticeInstances = [];
function fakeContentEl() {
  const el = {
    classes: new Set(), children: [], isConnected: true, style: {}, events: {}, text: "",
    addClass(name) { this.classes.add(name); },
    removeClass(name) { this.classes.delete(name); },
    toggleClass(name, on) { if (on) this.classes.add(name); else this.classes.delete(name); },
    createDiv(options = {}) { const child = fakeContentEl(); child.options = options; if (options.cls) options.cls.split(" ").forEach((c) => child.classes.add(c)); this.children.push(child); return child; },
    createEl(tag, options = {}) { const child = fakeContentEl(); child.tag = tag; child.options = options; if (options.cls) options.cls.split(" ").forEach((c) => child.classes.add(c)); this.children.push(child); return child; },
    createSpan(options = {}) { const child = fakeContentEl(); child.tag = "span"; child.options = options; if (options.cls) options.cls.split(" ").forEach((c) => child.classes.add(c)); this.children.push(child); return child; },
    prepend(child) { this.children = [child, ...this.children.filter((x) => x !== child)]; },
    remove() { this.isConnected = false; },
    empty() { this.children = []; },
    setAttr() {}, setText(value) { this.text = value; },
    addEventListener(event, callback) { this.events[event] = callback; },
    querySelector() { return null; },
  };
  return el;
}

function findClass(root, name) {
  if (root.classes && root.classes.has(name)) return root;
  for (const child of root.children || []) {
    const found = findClass(child, name);
    if (found) return found;
  }
  return null;
}
const obsidianStub = {
  Plugin: class {},
  ItemView: class {
    constructor(leaf) { this.leaf = leaf; this.app = leaf && leaf.app; }
    async onOpen() {}
    async onClose() {}
  },
  MarkdownView: class {
    constructor(leaf) {
      this.leaf = leaf;
      this.app = leaf && leaf.app;
      this.file = null;
      this.contentEl = fakeContentEl();
      this.loadedStates = [];
      this.saved = 0;
      this.editor = {
        inserted: [],
        cursor: null,
        focused: 0,
        replaceSelection: (text) => this.editor.inserted.push(text),
        lastLine: () => Math.max(0, String(this.file && this.file.text || "").split("\n").length - 1),
        setCursor: (cursor) => { this.editor.cursor = cursor; },
        focus: () => { this.editor.focused += 1; },
      };
    }
    async onOpen() {}
    async onClose() {}
    async save() { this.saved += 1; }
    async setState(state) {
      this.loadedStates.push(state);
      this.file = this.app && this.app.vault.getAbstractFileByPath(state.file);
    }
  },
  Modal: class { constructor(app) { this.app = app; } },
  MarkdownRenderer: { render: async (_app, markdown, el) => el.setText(markdown) },
  PluginSettingTab: class {},
  Setting: class {},
  Notice: class {
    constructor(msg) { notices.push(msg); this.noticeEl = fakeContentEl(); noticeInstances.push(this); }
    hide() { this.hidden = true; }
  },
  // Indirection because main.js destructures `requestUrl` at load — tests need to swap it after.
  requestUrl: (...args) => requestUrlImpl(...args),
};
let requestUrlImpl = async () => ({ status: 200, json: {} });

// The SHARED session the handler is installed on. Every test asserts it is left at null.
const installed = { handler: undefined };
const remoteStub = {
  session: {
    defaultSession: { setDisplayMediaRequestHandler(h) { installed.handler = h; } },
  },
};

const origLoad = Module._load;
Module._load = function (request) {
  if (request === "obsidian") return obsidianStub;
  if (request === "@electron/remote") return remoteStub;
  return origLoad.apply(this, arguments);
};

const PluginClass = createRequire(import.meta.url)(path.join(HERE, "main.js"));
const { RecorderView, splitMeetingNote, embedLegacyMeetingBlock,
        markdownBlockShortcut, pcm16k, renderPassiveMeetingBlock,
        MeetingSessionManager, serverArgv, serverEnv } = PluginClass;

test("plugin-started servers are tied to the current Obsidian process", () => {
  // The plugin never asks for --reload: a restart cuts requests in flight, so a save during
  // a recording loses that recording's filing. `slim chat --reload` is a terminal thing.
  assert.deepEqual(serverArgv({ repoPath: "/repo", port: 7546 }, 4321),
    ["--directory", "/repo", "run", "slim", "chat", "--port", "7546",
     "--parent-pid", "4321"]);
  assert.ok(!serverArgv({ repoPath: "/repo", port: 7546, devReload: true }, 4321)
    .includes("--reload"));
});

test("a plugin-started server is told which vault this plugin is in", () => {
  // The plugin sends vault-RELATIVE paths and the server now DISCOVERS its own vault. With
  // two vaults registered, a server left to discover would resolve those paths against the
  // wrong one — the audio it looks for is in the vault the plugin wrote to, not that one.
  const env = serverEnv("/Users/x/Vaults/Work", { PATH: "/usr/bin", SLIM_VAULT: "/stale" });
  assert.equal(env.SLIM_VAULT, "/Users/x/Vaults/Work");
  assert.equal(env.PATH, "/usr/bin");
});

// --- harness -------------------------------------------------------------------------------

function view({ recordVoice = true, display, user } = {}) {
  notices = [];
  noticeInstances = [];
  installed.handler = undefined;
  // ⚠ Node 23 ships a read-only `navigator` global, so plain assignment throws.
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    writable: true,
    value: {
      mediaDevices: {
        getDisplayMedia: display || (async () => { throw new Error("no display media"); }),
        getUserMedia: user || (async () => new FakeStream(["audio"])),
      },
    },
  });
  globalThis.window = {
    AudioContext: class {
      createMediaStreamDestination() { return { stream: new FakeStream(["audio"]) }; }
      createMediaStreamSource() { return { connect() {} }; }
    },
  };
  const plugin = { settings: { recordVoice, lastType: "lecture" }, saveSettings: async () => {} };
  return new RecorderView({}, plugin);
}

function fileBackedView({ type = "lecture", existing = [] } = {}) {
  const files = new Map(existing.map((p) => [p, { path: p, bytes: null, text: "" }]));
  const created = [];
  const binaries = [];
  const vault = {
    adapter: {
      exists: async (p) => files.has(p),
      list: async (dir) => ({
        files: [...files.keys()].filter((p) => p.startsWith(`${dir}/`) && !files.get(p).folder),
        folders: [],
      }),
      stat: async (p) => (files.has(p) ? { size: (files.get(p).bytes || "x").length } : null),
    },
    createFolder: async (p) => { files.set(p, { path: p, folder: true }); },
    create: async (p, text) => {
      const f = { path: p, text };
      files.set(p, f); created.push([p, text]); return f;
    },
    createBinary: async (p, bytes) => {
      const f = { path: p, bytes: Buffer.from(bytes) };
      files.set(p, f); binaries.push([p, f.bytes]); return f;
    },
    read: async (f) => f.text,
    delete: async (f) => { files.delete(f.path); },
    getAbstractFileByPath: (p) => files.get(p) || null,
  };
  const app = { vault, workspace: null };
  const editorLeaf = {
    app,
    view: null,
    opened: [],
    async openFile(file) {
      if (!(this.view instanceof obsidianStub.MarkdownView)) {
        this.view = new obsidianStub.MarkdownView(this);
        this.view.contentEl = fakeContentEl();
      }
      this.view.file = file;
      this.opened.push(file.path);
    },
  };
  app.workspace = { trigger() {}, getLeaf: () => editorLeaf };
  const plugin = {
    settings: { recordVoice: true, lastType: type, port: 7546 },
    saveSettings: async () => {},
  };
  const v = new RecorderView({ app }, plugin);
  v.app = app;
  v.type = type;
  return { v, files, created, binaries, editorLeaf };
}

const okDisplay = async () => new FakeStream(["video", "audio"]);

// --- system audio ---------------------------------------------------------------------------

test("system audio is captured with no device selection at all", async () => {
  const v = view({ display: okDisplay });
  const stream = await v.openSystemAudio();
  assert.equal(stream.getAudioTracks().length, 1);
});

test("the video track is stopped — we only ever wanted audio", async () => {
  // ⚠ `video` is a REQUIRED field satisfied with Obsidian's own frame. Leaving the track
  // running would keep a capture alive for a feature that never wanted one.
  const v = view({ display: okDisplay });
  const stream = await v.openSystemAudio();
  assert.equal(stream.getVideoTracks()[0].stopped, true);
  assert.equal(stream.getAudioTracks()[0].stopped, false);
});

test("the handler is removed from the shared session afterwards", async () => {
  const v = view({ display: okDisplay });
  await v.openSystemAudio();
  assert.equal(installed.handler, null);
});

test("the handler is removed even when the request throws", async () => {
  // ⚠ THE ONE THAT MATTERS: `session.defaultSession` is shared with the whole app. A handler
  // left installed silently hijacks any other getDisplayMedia call in Obsidian.
  const v = view({ display: async () => { throw new Error("denied"); } });
  await assert.rejects(() => v.openSystemAudio());
  assert.equal(installed.handler, null);
});

test("a stream with no audio track is an error, not a silent success", async () => {
  const v = view({ display: async () => new FakeStream(["video"]) });
  await assert.rejects(() => v.openSystemAudio(), /no audio track/);
});

test("the handler asks for loopback audio and the requesting frame as video", async () => {
  let got = null;
  const v = view({
    display: async () => {
      installed.handler({ frame: "FRAME", audioRequested: true }, (streams) => { got = streams; });
      return new FakeStream(["video", "audio"]);
    },
  });
  await v.openSystemAudio();
  assert.deepEqual(got, { video: "FRAME", audio: "loopback" });
});

// --- the mix ---------------------------------------------------------------------------------

test("'Record my voice' off means the microphone is never opened", async () => {
  let asked = false;
  const v = view({
    recordVoice: false,
    display: okDisplay,
    user: async () => { asked = true; return new FakeStream(["audio"]); },
  });
  await v.buildStream();
  assert.equal(asked, false);
});

test("losing system audio still records the voice, and SAYS SO", async () => {
  // Degrading silently is how a two-hour loss goes unnoticed.
  const v = view({ display: async () => { throw new Error("denied"); } });
  const stream = await v.buildStream();
  assert.ok(stream);
  assert.equal(notices.length, 1);
  assert.match(notices[0], /could not capture system audio/i);
});

test("no voice and no system audio refuses to record rather than capturing silence", async () => {
  const v = view({
    recordVoice: false,
    display: async () => { throw new Error("denied"); },
  });
  await assert.rejects(() => v.buildStream(), /no audio source/);
});


// --- the stale-server guard ------------------------------------------------------------------
// Reuse is deliberate; a server OLDER THAN THE CODE being silently preferred is not. A server
// from the previous afternoon survived a Cmd+Q, was reused, and failed two recordings against
// a bug already fixed on disk (2026-08-23).

function plugin() {
  notices = [];
  const p = Object.create(PluginClass.prototype);
  p.settings = { port: 7546 };
  return p;
}

test("a stale server produces a warning naming the file and how to restart", async () => {
  requestUrlImpl = async () => ({
    status: 200,
    json: { stale: true, started_at: "2026-08-22T18:05:38Z",
            newest_source: "2026-08-23T14:00:00Z", newest_file: "transcribe.py" },
  });
  await plugin().warnIfStale();
  assert.equal(notices.length, 1);
  assert.match(notices[0], /transcribe\.py/);
  assert.match(notices[0], /lsof -tiTCP:7546/);
});

test("a fresh server says nothing at all", async () => {
  requestUrlImpl = async () => ({ status: 200, json: { stale: false } });
  await plugin().warnIfStale();
  assert.equal(notices.length, 0);
});

test("a server too old to have /api/health is itself the proof", async () => {
  // ⚠ 404 is not an error to swallow: only a server predating the endpoint can answer it.
  requestUrlImpl = async () => ({ status: 404, json: null });
  await plugin().warnIfStale();
  assert.equal(notices.length, 1);
  assert.match(notices[0], /OLD code/);
});

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


// --- a failed transcript is refused, and retrying is one button ------------------------------
// The server used to file an empty note and return 200, so there was nothing to retry — it had
// already reported success (2026-08-23).

test("retry posts the SAME staged audio, not a new recording", async () => {
  const posted = [];
  // start() always creates a draft before recording begins, so a real submitRecording call
  // always carries `draft`, never `notes_md` (obsidian/main.js:1336-1341, D3).
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


// --- a real Markdown draft is the notes field ------------------------------------------------

test("the recorder is a controller that mounts its object into a native MarkdownView", () => {
  const { v } = fileBackedView();
  assert.equal(v instanceof obsidianStub.ItemView, false);
  assert.equal(v instanceof obsidianStub.MarkdownView, false);
});

test("the meeting parser retains the complete existing frontmatter byte-for-byte", () => {
  // `index_terms` is RETIRED (deleted 2026-08-26) and stays in this fixture on purpose: the
  // notes already on disk carry it, and the parser must return their frontmatter untouched.
  const frontmatter = [
    "---", "title: \"Design sync\"", "date: 2026-08-24", "type: meeting-note",
    "tags: [meeting-note]", "topics: [ux, recorder]", "index_terms: [tabbed object]",
    "origin: recorded", "authority: evidence", "audio: Attachments/Recorder/x.webm",
    "recorded_at: 2026-08-24T14:00:00Z", "---",
  ].join("\n");
  const note = `${frontmatter}\n\n<!-- slim:summary model=x -->\n## Summary\nAI result.\n<!-- /slim:summary -->\n\n## Notes\n- exact note\n\n## Transcript\nExact transcript.`;
  const parts = splitMeetingNote(note);
  assert.equal(parts.frontmatter, frontmatter);
  assert.equal(parts.summary, "AI result.");
  assert.equal(parts.notes, "- exact note");
  assert.equal(parts.transcript, "Exact transcript.");
});

test("the meeting block is inline and leaves ordinary Markdown after it alone", () => {
  const legacy = [
    "---", "title: Test", "origin: recorded", "---", "",
    "<!-- slim:summary prompt=x -->", "## Summary", "Summary.",
    "<!-- /slim:summary -->", "", "## Transcript", "Words.",
  ].join("\n");
  const embedded = embedLegacyMeetingBlock(legacy) + "## Follow-up\n\nEditable after the block.\n";
  const parts = splitMeetingNote(embedded);
  assert.match(embedded, /^`{4,}slim-meeting$/m);
  assert.equal(parts.frontmatter, "---\ntitle: Test\norigin: recorded\n---");
  assert.equal(parts.transcript, "Words.");
  assert.equal(parts.tail, "\n## Follow-up\n\nEditable after the block.\n");
});

test("a second recorded note remains readable while another meeting is under review", () => {
  const block = fakeContentEl();
  const source = [
    "<!-- slim:summary model=x -->", "## Summary", "Saved summary.",
    "<!-- /slim:summary -->", "", "## Notes", "- saved note", "",
    "## Transcript", "Saved transcript.",
  ].join("\n");

  renderPassiveMeetingBlock({}, {}, block, source, "Notes/other.md");

  assert.ok(findClass(block, "slim-passive-meeting"));
  assert.equal(findClass(block, "slim-rendered-markdown").text, "Saved summary.");

  const tabs = findClass(block, "slim-object-tabs");
  const notesTab = tabs.children.find((child) => child.options.text === "My notes");
  notesTab.events.click();
  assert.equal(findClass(block, "slim-rendered-markdown").text, "- saved note");
});

test("Notion-style Markdown shortcuts map to live block formatting", () => {
  assert.deepEqual(markdownBlockShortcut("### "), { tag: "h3", kind: "block" });
  assert.deepEqual(markdownBlockShortcut("- "), { tag: "ul", kind: "list" });
  assert.deepEqual(markdownBlockShortcut("1. "), { tag: "ol", kind: "list" });
  assert.equal(markdownBlockShortcut("ordinary text"), null);
});

test("live captions downsample browser audio to local 16 kHz PCM", () => {
  const input = new Float32Array(48000).fill(0.5);
  const pcm = pcm16k(input, 48000);
  assert.equal(pcm.length, 32000);
  assert.ok(Math.abs(pcm.readInt16LE(0) - 16384) < 2);
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

test("the inline object has no raw-Markdown escape hatch", () => {
  const source = readFileSync(path.join(HERE, "main.js"), "utf8");
  assert.doesNotMatch(source, /Open raw Markdown/);
});

test("plugin reload adopts every already-visible meeting note", () => {
  const source = readFileSync(path.join(HERE, "main.js"), "utf8");
  assert.match(source,
               /onLayoutReady\(\(\) => \{[\s\S]*getLeavesOfType\("markdown"\)[\s\S]*meetings\.adoptFile\(file, leaf\)/);
  assert.match(source, /meetingBlockFor\(path\)[\s\S]*return el \|\| null/);
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
  const css = readFileSync(path.join(HERE, "styles.css"), "utf8");
  assert.match(css, /\.slim-recorder-draft-frozen \.cm-line:not\(:has\(\.slim-meeting-block\)\)[^{]*\{[^}]*pointer-events:\s*none/s);
  assert.match(css, /\.slim-recorder-draft-frozen \.slim-meeting-block[^{]*\{[^}]*pointer-events:\s*auto/s);
});

test("review panes own their scroll instead of painting over filing controls", () => {
  const css = readFileSync(path.join(HERE, "styles.css"), "utf8");
  assert.match(css, /\.slim-recorder\s*>\s*\*\s*\{[^}]*flex:\s*0 0 auto/s);
  assert.match(css, /\.slim-summary-editor,[^{]*\.slim-notes-editor\s*\{[^}]*width:\s*100%[^}]*overflow:\s*auto[^}]*resize:\s*none/s);
});

test("recording progress tolerates the initial race and streams the live state", async () => {
  const plugin = Object.create(PluginClass.prototype);
  plugin.settings = { port: 7546 };
  const states = [];
  let calls = 0;
  requestUrlImpl = async () => {
    calls += 1;
    if (calls === 1) return { status: 404, json: { error: "not started" } };
    return { status: 200, json: { stage: "summarizing", thinking: "actual token", done: true } };
  };
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

/* --- 2026-08-26: their notes are theirs, and a summary can be retried ----------------------- */

const inlineMarkdownForTest = (node) => PluginClass.inlineMarkdown(node);

function findButton(root, text) {
  if (root.tag === "button" && root.options && root.options.text === text) return root;
  for (const child of root.children || []) {
    const found = findButton(child, text);
    if (found) return found;
  }
  return null;
}

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

test("resuming carries the earlier transcript and the elapsed time forward", () => {
  // Seen live: resume showed an empty transcript pane and a timer back at 00:00:00, so the
  // first forty minutes looked lost even though the server had them cached.
  const v = view();
  v.liveTranscript = "the first part";
  v.elapsed = 41;
  v.carryTranscriptAndClock();
  assert.equal(v.liveCarry, "the first part");
  assert.equal(v.elapsedCarry, 41);

  v.liveTranscript = "the second part";
  assert.equal(v.shownTranscript(), "the first part\n\nthe second part");
  v.elapsed = v.elapsedCarry + 5;
  assert.equal(v.elapsed, 46, "the clock counts the whole recording, not this segment");
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

test("one helper decides which recording a staged file belongs to", () => {
  const source = readFileSync(path.join(HERE, "main.js"), "utf8");
  const spellings = source.match(/-\\d\{3\}\$/g) || [];
  assert.ok(spellings.length <= 1,
            `the segment-suffix rule is written ${spellings.length} times in main.js`);
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

test("editing rendered notes preserves embeds, math and wikilinks", () => {
  // ⚠ P1 DATA LOSS. `markdownFromEditable` serializes the RENDERED DOM on every keystroke, and
  // `inlineMarkdown` has no case for an image, a formula or a wikilink — so editing the notes
  // field in review silently deleted the screenshots they pasted while recording.
  const el = {
    nodeType: 1, tagName: "P",
    childNodes: [
      { nodeType: 3, nodeValue: "before " },
      { nodeType: 1, tagName: "IMG", childNodes: [],
        getAttribute: (k) => (k === "data-slim-src"
          ? "![[Attachments/Recorder/x/paste-001.png]]" : null) },
      { nodeType: 3, nodeValue: " and " },
      { nodeType: 1, tagName: "SPAN", childNodes: [{ nodeType: 3, nodeValue: "P(a|b)" }],
        getAttribute: (k) => (k === "data-slim-src" ? "$P(a|b)$" : null) },
      { nodeType: 3, nodeValue: " after" },
    ],
    getAttribute: () => null,
  };
  const out = inlineMarkdownForTest(el);
  assert.match(out, /!\[\[Attachments\/Recorder\/x\/paste-001\.png\]\]/);
  assert.match(out, /\$P\(a\|b\)\$/);
  assert.match(out, /^before .* and .* after$/);
});

// A DOM the way Obsidian's renderer builds it: element children, newline text nodes between
// them, `parentElement` links, and `querySelectorAll` in document order — what the notes-pane
// stamping and serializer actually run against. `fakeContentEl` above has none of that.
function domText(value) { return { nodeType: 3, nodeValue: value }; }
function domEl(tag, { cls = "", attrs = {} } = {}, kids = []) {
  const el = {
    nodeType: 1, tagName: tag.toUpperCase(), classList: new Set(cls.split(" ").filter(Boolean)),
    attrs: { ...attrs }, childNodes: kids, parentElement: null, events: {},
    get children() { return this.childNodes.filter((k) => k.nodeType === 1); },
    getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; },
    setAttribute(name, value) { this.attrs[name] = value; },
    setAttr(name, value) { this.attrs[name] = value; },
    addEventListener(event, callback) { this.events[event] = callback; },
    createDiv(options = {}) { const child = domEl("div", { cls: options.cls || "", attrs: options.attr || {} }); this.appendChild(child); return child; },
    createEl(tag, options = {}) { const child = domEl(tag, { cls: options.cls || "" }); this.appendChild(child); return child; },
    appendChild(child) { child.parentElement = this; this.childNodes.push(child); return child; },
    querySelectorAll(selector) {
      const wanted = selector.split(",").map((s) => s.trim());
      const matches = (node) => wanted.some((w) => {
        const [tagPart, ...classes] = w.split(".");
        return (!tagPart || node.tagName === tagPart.toUpperCase()) && classes.every((c) => node.classList.has(c));
      });
      const out = [];
      const walk = (node) => { for (const child of node.children) { if (matches(child)) out.push(child); walk(child); } };
      walk(this);
      return out;
    },
  };
  for (const kid of kids) if (kid.nodeType === 1) kid.parentElement = el;
  return el;
}
const domEmbed = (cls = "internal-embed image-embed") =>
  domEl("p", {}, [domEl("span", { cls }, [domEl("img")])]);

test("stamping lands on the embed wrapper, not on the image inside it", () => {
  // ⚠ P1 DATA LOSS (2026-09-05). Obsidian renders one `![[x.png]]` as TWO nodes the stamping
  // selector matches — `<span class="internal-embed"><img></span>` — and inline math as
  // `<span class="math"><mjx-container>`. Positional stamping counted both, so from the second
  // atom on every source landed on the wrong node and the second half landed nowhere. Each
  // blur-save of the notes pane then halved their screenshots: 7 → 4 → 2 in one lecture.
  const md = "![[a.png|600]]\n\n![[b.png|500]]\n\n$x^2$\n\n![[c.png]]";
  const root = domEl("div", {}, [
    domEmbed(), domText("\n"), domEmbed(), domText("\n"),
    domEl("p", {}, [domEl("span", { cls: "math math-inline" }, [domEl("mjx-container")])]),
    domText("\n"), domEmbed(),
  ]);
  PluginClass.stampSourceAtoms(root, md);
  assert.equal(PluginClass.markdownFromEditable(root), md);
});

test("nested list items keep their level when the notes pane is serialized", () => {
  // Obsidian renders `- a\n\t- b` as <ul><li>a<ul><li>b</li></ul></li></ul>, with newline text
  // nodes between the elements. Flattening the inner list into its parent item's text turned
  // their sub-bullets into a bare paragraph under the list (2026-09-05).
  const root = domEl("div", {}, [
    domEl("ul", {}, [domText("\n"),
      domEl("li", {}, [domText("a"), domText("\n"),
        domEl("ul", {}, [domText("\n"), domEl("li", {}, [domText("b")]), domText("\n")]),
        domText("\n")]),
      domText("\n"),
      domEl("li", {}, [domText("c")]), domText("\n")]),
  ]);
  assert.equal(PluginClass.markdownFromEditable(root), "- a\n\t- b\n- c");
});

test("atoms are stamped again once the async render has landed", async () => {
  // The renderer is a promise. At the synchronous pass and at setTimeout(0) the editor can
  // still be empty, and an unstamped embed is deleted by the first keystroke. Stamp when the
  // render actually resolves, not only when a timer guesses that it has.
  const v = view();
  const md = "![[a.png]]";
  const original = obsidianStub.MarkdownRenderer.render;
  obsidianStub.MarkdownRenderer.render = async (_app, _markdown, el) => {
    await new Promise((resolve) => setTimeout(resolve, 5));
    el.appendChild(domEl("p", {}, [domEl("span", { cls: "internal-embed" }, [domEl("img")])]));
  };
  try {
    const editor = v.renderLiveMarkdownEditor(domEl("div"), md, () => {});
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(PluginClass.markdownFromEditable(editor), md);
  } finally {
    obsidianStub.MarkdownRenderer.render = original;
  }
});

test("rendered nodes are stamped with the source they came from", () => {
  const source = readFileSync(path.join(HERE, "main.js"), "utf8");
  // The stamping pass and the serializer have to agree on one attribute name.
  assert.match(source, /data-slim-src/);
  assert.match(source, /function stampSourceAtoms/);
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

test("the live transcript keeps flushing after a pause and resume", async () => {
  const v = view();
  v.recordingId = "rec-live";
  const posts = [];
  v.plugin.postLiveTranscript = async (body) => {
    posts.push(body.action);
    return { transcript: `words ${posts.filter((a) => a === "chunk").length}` };
  };
  let onaudio = null;
  const node = () => ({ connect() {}, disconnect() {} });
  v.audioCtx = {
    sampleRate: 16000,
    destination: node(),
    createMediaStreamSource: () => node(),
    createGain: () => ({ ...node(), gain: { value: 1 } }),
    createScriptProcessor: () => {
      const p = node();
      Object.defineProperty(p, "onaudioprocess", {
        set(fn) { onaudio = fn; }, get() { return onaudio; },
      });
      return p;
    },
  };
  v.startLiveTranscript({});
  await v.liveTranscriptQueue;

  const frame = { inputBuffer: { getChannelData: () => new Float32Array(4096).fill(0.2) } };
  const speak = async () => {
    for (let i = 0; i < 10; i += 1) onaudio(frame);
    await v.liveTranscriptQueue;
  };

  await speak();
  const before = posts.filter((a) => a === "chunk").length;
  assert.ok(before > 0, "audio reaches the transcriber before the pause");

  v.recorder = { state: "recording", pause() { this.state = "paused"; },
                 resume() { this.state = "recording"; } };
  v.togglePause();                                  // Pause
  await speak();
  assert.equal(posts.filter((a) => a === "chunk").length, before, "paused sends nothing");

  v.togglePause();                                  // Resume
  await speak();
  assert.ok(posts.filter((a) => a === "chunk").length > before,
            "⚠ after resume the live transcript must start growing again");
});

test("a lost live session is restarted instead of switching captions off", async () => {
  // The chunk handler treated EVERY error as fatal and set liveTranscriptActive = false, so one
  // failed POST — a restarted server, an evicted session — killed captions for the rest of the
  // recording, with grey status text as the only sign.
  const v = view();
  v.recordingId = "rec-1";
  let failed = false;
  const actions = [];
  v.plugin.postLiveTranscript = async (body) => {
    actions.push(body.action);
    if (body.action === "chunk" && !failed) {
      failed = true;
      throw new Error("/api/record/live returned 400: no such live transcript");
    }
    return { transcript: "recovered words" };
  };
  v.liveTranscriptActive = true;
  v.livePcm = [Buffer.alloc(64000)];
  v.livePcmBytes = 64000;

  await v.flushLiveTranscript();
  await v.liveTranscriptQueue;

  assert.equal(v.liveTranscriptActive, true, "captions survive one lost session");
  assert.ok(actions.includes("start"), `it restarts the session: ${actions.join(",")}`);
});

test("the caption pane says which of the three states it is in", () => {
  const v = view();
  v.recorder = { state: "paused" };
  v.liveTranscriptActive = true;
  assert.match(v.liveStatusText(), /paused/i);

  v.recorder = { state: "recording" };
  v.liveTranscriptStatus = "Live transcript · draft";
  assert.match(v.liveStatusText(), /draft/i);

  v.liveTranscriptActive = false;
  assert.match(v.liveStatusText(), /off|unavailable|stopped/i);
});

test("live captions coalesce instead of queueing behind a slow round trip", async () => {
  // ⚠ Captions flushed every 2s onto an UNBOUNDED promise chain. One round trip slower than
  // 2s and every later flush waits behind it, so the lag compounds for the whole recording —
  // "it doesn't keep up", exactly as they described.
  const v = view();
  v.liveTranscriptActive = true;
  v.recordingId = "r";
  let inFlight = 0;
  let maxInFlight = 0;
  const sizes = [];
  let release;
  const gate = new Promise((r) => { release = r; });
  v.plugin.postLiveTranscript = async (body) => {
    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    sizes.push(Buffer.from(body.pcm16, "base64").length);
    await gate;
    inFlight -= 1;
    return { transcript: "words" };
  };

  v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
  v.flushLiveTranscript();
  for (let i = 0; i < 4; i += 1) {              // four more flushes while the first is stuck
    v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
    v.flushLiveTranscript();
  }
  release();
  await new Promise((r) => setTimeout(r, 0));
  await v.liveTranscriptQueue;

  assert.equal(maxInFlight, 1, "never more than one request in flight");
  assert.ok(sizes.length <= 2, `the backlog coalesced into ${sizes.length} sends`);
});

test("a decoder that revises its own prefix never duplicates the caption", () => {
  // ⚠ THE BUG THE OWNER REPORTED, 2026-08-27: captions "repeat excessively and spam words".
  //
  // The rule used to be "growth replaces, anything else appends", written against a
  // rolling-window ASR. Parakeet does not roll — it REWRITES what it already emitted as more
  // audio arrives. Measured over 99 s of speech: 19 of 49 chunks revised their prefix and 0
  // shrank, so the append branch fired constantly and appended the WHOLE cumulative string,
  // compounding each time. The pane showed 37,785 chars of a 1,477-char transcript (25.6x).
  //
  // The stream is cumulative — a server contract, not an assumption — so: replace, always.
  const v = view();
  v.applyLiveTranscript("The live caption lane feeds six.");
  v.applyLiveTranscript("The live caption lane feeds sixteen kilohertz audio into a stream.");
  // A real revision: "sixteen kilohertz" is rewritten as "16 kHz". This is the case that used
  // to append everything again.
  v.applyLiveTranscript("The live caption lane feeds 16 kHz audio into a streaming parakeet.");
  assert.equal(v.liveTranscript,
               "The live caption lane feeds 16 kHz audio into a streaming parakeet.");
  assert.equal(v.shownTranscript(), v.liveTranscript, "nothing is carried mid-segment");

  // Punctuation-only revision of the tail — the most common shape by far.
  v.applyLiveTranscript("The live caption lane feeds 16 kHz audio into a streaming parakeet model.");
  assert.ok(!v.liveTranscript.includes("parakeet. The"), "no duplicated prefix");
});

test("a live session that restarts keeps the words already on screen", () => {
  // The append branch was really protecting THIS case: a restarted session decodes from zero,
  // so a bare replace would erase the first half of a lecture. It belongs here, not in the
  // per-chunk rule — carry, then let the new session grow from empty.
  const v = view();
  v.applyLiveTranscript("everything they said before the restart");
  v.carryTranscript();
  assert.equal(v.liveTranscript, "");
  assert.equal(v.shownTranscript(), "everything they said before the restart");

  v.applyLiveTranscript("what the new session hears");
  assert.equal(v.shownTranscript(),
               "everything they said before the restart\n\nwhat the new session hears");
});

test("carrying mid-segment does not touch the clock", () => {
  // A restart happens INSIDE a segment, with the clock still running; only a segment boundary
  // may bank the elapsed time. One helper doing both would silently reset their timer.
  const v = view();
  v.elapsed = 42;
  v.elapsedCarry = 0;
  v.applyLiveTranscript("words");
  v.carryTranscript();
  assert.equal(v.elapsedCarry, 0, "a restart must not bank the clock");
  v.applyLiveTranscript("more words");
  v.carryTranscriptAndClock();
  assert.equal(v.elapsedCarry, 42, "a segment boundary still banks it");
});


// --- several durable meeting sessions, one capture ------------------------------------------

function sessionManagerHarness() {
  let next = 0;
  const made = [];
  const plugin = {
    app: { workspace: {} },
    settings: { lastType: "meeting-note", recordVoice: false },
  };
  const manager = new MeetingSessionManager(plugin, (leaf) => {
    const id = `session-${++next}`;
    const session = {
      sessionId: id,
      recordingId: id,
      state: "idle",
      editorLeaf: leaf,
      draftRel: "",
      filedNotePath: "",
      opened: 0,
      renders: 0,
      async open() { this.opened += 1; },
      render() { this.renders += 1; },
      handleVaultDelete() {},
      expectsDeletion() { return this.state === "recording" || this.state === "processing"; },
      onClose() {},
    };
    made.push(session);
    return session;
  });
  plugin.meetings = manager;
  return { manager, made };
}

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
  notices = [];
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

test("the plugin owns a meeting-session manager, not one global recorder", () => {
  const source = readFileSync(path.join(HERE, "main.js"), "utf8");
  assert.match(source, /this\.meetings\s*=\s*new MeetingSessionManager\(this\)/);
  assert.doesNotMatch(source, /this\.recorder\s*=\s*new RecorderView\(null, this\)/);
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

// --- 2026-08-28: three things they noticed in a week of real use ---------------------------------

test("a caption round trip that returns with less than 2 s buffered waits for the tap", async () => {
  // ⚠ THE COIL WHINE. The `finally` re-flushed WHATEVER had arrived during the request, so after
  // the first 2 s chunk the loop ran at one request per round trip: 2.7 requests/s measured on a
  // real recording, each re-encoding 20 s of context — GPU 60 % busy on silence, pulsing at
  // 3 Hz. Per call cost is flat (~170 ms), so the count of calls is the whole cost. At the tap's
  // own 2 s floor: 0.5/s and 9 %.
  const v = view();
  v.liveTranscriptActive = true;
  v.recordingId = "r";
  const sizes = [];
  v.plugin.postLiveTranscript = async (body) => {
    sizes.push(Buffer.from(body.pcm16, "base64").length);
    // One tap callback lands mid-flight. Once: with the old `finally` this fake would feed
    // the loop forever, which is exactly what the real tap did.
    if (sizes.length === 1) { v.livePcm.push(Buffer.alloc(2730)); v.livePcmBytes += 2730; }
    return { transcript: "" };
  };

  v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
  await v.flushLiveTranscript();
  await v.liveTranscriptQueue;

  assert.deepEqual(sizes, [64000], "85 ms that arrived mid-flight waits for the 2 s floor");
  assert.equal(v.livePcmBytes, 2730, "and stays buffered for the tap to send");
});

test("a backlog that reached the floor while a request was out still goes at once", async () => {
  // The coalescing that fixed "captions fall behind" (2026-08-26) is kept: a slow round trip
  // sends fewer, larger chunks — it must not wait for yet another 2 s on top of the lag.
  const v = view();
  v.liveTranscriptActive = true;
  v.recordingId = "r";
  const sizes = [];
  v.plugin.postLiveTranscript = async (body) => {
    sizes.push(Buffer.from(body.pcm16, "base64").length);
    if (sizes.length === 1) { v.livePcm.push(Buffer.alloc(70000)); v.livePcmBytes += 70000; }
    return { transcript: "" };
  };

  v.livePcm = [Buffer.alloc(64000)]; v.livePcmBytes = 64000;
  await v.flushLiveTranscript();
  await v.liveTranscriptQueue;

  assert.deepEqual(sizes, [64000, 70000]);
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
