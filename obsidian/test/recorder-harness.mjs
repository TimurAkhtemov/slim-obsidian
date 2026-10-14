/*
 * The recorder's test harness: a RecorderView over fake media, a fake vault and a fake editor.
 * Import it before requiring anything from src/; it installs the stubs (fakes.mjs) and draws
 * Markdown into the recorder's fake DOM as `.text`.
 *
 * WHY THIS EXISTS. Nothing tested the recorder, and it showed: on 2026-08-22 the first recording
 * made the way they actually works — Obsidian launched from the Dock — failed outright, and the
 * defects behind it had sat through two weeks of review. Every one was found by RUNNING the
 * thing, so this is plain `node --test` against src/ with `Module._load` patched to serve stubs
 * for `obsidian` and `@electron/remote`.
 *
 * Scope is the recorder's behavioral contract: acquisition, draft lifecycle, native-view
 * handoff, recovery, and image paste. Pixel rendering remains Obsidian's responsibility.
 */
import { createRequire } from "node:module";
import { fakeContentEl, installed, obsidianStub, resetNoticeInstances, resetNotices,
         setMarkdownRender } from "./fakes.mjs";

const require = createRequire(import.meta.url);
const PluginClass = require("../src/plugin.js");
const { RecorderView } = require("../src/recorder-view.js");
const { MeetingSessionManager } = require("../src/sessions.js");

setMarkdownRender(async (_app, markdown, el) => el.setText(markdown));

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
let wiring = null;

// --- harness -------------------------------------------------------------------------------

function view({ recordVoice = true, display, user } = {}) {
  resetNotices();
  resetNoticeInstances();
  wiring = null;
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
      createChannelMerger(inputs) {
        wiring = { inputs, routed: [], out: null };
        return { connect(node) { wiring.out = node; }, wiring };
      }
      createMediaStreamSource(stream) {
        // Tolerant of other targets (a caption tap, an analyser): only the merger has `wiring`.
        return { connect(node, output, input) { if (node && node.wiring) node.wiring.routed.push([stream, input]); } };
      }
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
      remove: async (p) => { files.delete(p); },
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


// --- the stale-server guard ------------------------------------------------------------------
// Reuse is deliberate; a server OLDER THAN THE CODE being silently preferred is not. A server
// from the previous afternoon survived a Cmd+Q, was reused, and failed two recordings against
// a bug already fixed on disk (2026-08-23).

function plugin() {
  resetNotices();
  const p = Object.create(PluginClass.prototype);
  p.settings = { port: 7546 };
  return p;
}

// --- the digital-silence warning -------------------------------------------------------------
// 2026-09-24: 157 s of zeros on both channels, and the empty-transcript refusal at the end was
// the first anyone knew. A meeting you think is being recorded is the worst failure there is.

const ZERO = { mic: 0, system: 0 };
const SPEECH = 0.05;

function findButton(root, text) {
  if (root.tag === "button" && root.options && root.options.text === text) return root;
  for (const child of root.children || []) {
    const found = findButton(child, text);
    if (found) return found;
  }
  return null;
}


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

export { FakeTrack, FakeStream, wiring, view, fileBackedView, plugin, ZERO, SPEECH, findButton, sessionManagerHarness };
