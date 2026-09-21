/*
 * SLIM Recorder — record, take notes, file the result.
 *
 * Deliberately a single CommonJS file with NO build step. Obsidian loads `main.js` directly, so
 * there is no esbuild, no npm install, and no bundler config standing between an edit and a
 * reload. That matters more than type checking here: this plugin's first real test is a live
 * meeting, and a toolchain that fails to build is a failure mode with no upside.
 *
 * THE DIVISION OF LABOUR IS THE ARCHITECTURE (CLAUDE.md, "Capture"):
 * this file owns the UI and writes files through the Vault API. Every judgement — transcribe,
 * summarize, suggest a path — belongs to the Python server on 127.0.0.1.
 * A judgement that migrates into this file is a bug, because it becomes untestable by the
 * suite that protects the rest of SLIM.
 *
 * ⚠ AUDIO GOES INTO THE VAULT, NOT OVER HTTP. The server caps request bodies far below what a
 * recording weighs, so the file is written to disk here and only its PATH is posted.
 *
 * ⚠ AUDIO IS FLUSHED TO DISK AS IT ARRIVES, through Node's `fs` rather than the Vault API,
 * because Obsidian can only create or overwrite a binary file and never append. `isDesktopOnly`
 * is what licenses that. The alternative — holding every chunk in browser memory until Stop —
 * makes a crash during a two-hour meeting cost the entire meeting, and long meetings are the
 * use case.
 *
 * ⚠ THIS PLUGIN OWNS `slim chat`'s LIFECYCLE. Nothing else starts it, so recording with the
 * server down used to fail at the POST. It spawns on load if the port is free and kills the
 * process GROUP on unload — see `stopServer`.
 */
"use strict";

const { Plugin, ItemView, MarkdownView, MarkdownRenderer, MarkdownRenderChild, Modal,
        PluginSettingTab, Setting, Notice, requestUrl, setIcon } = require("obsidian");
const { spawn } = require("child_process");
const { randomUUID } = require("crypto");
const http = require("http");
const net = require("net");
const os = require("os");
const nodePath = require("path");

const STAGING_DIR = "Attachments/_incoming";
const MIC_CHANNEL = 0;      // left  — the server labels it Me
const SYSTEM_CHANNEL = 1;   // right — the server labels it Them
const RECORDER_ATTACHMENTS = "Attachments/Recorder";
const DRAFT_SUFFIX = ".slim-draft.md";
const EMPTY_MEETING_BLOCK = "````slim-meeting\n\n````\n\n";

// Matches slim/voicetags.py TYPE_FRONTMATTER. A type they PICKS is testimony: the pipeline never
// overrides a declared type, so choosing here removes the whole axis from guesswork.
const TYPES = [
  ["meeting-note", "Meeting"],
  ["lecture", "Lecture"],
  ["learning", "Learning"],
  ["idea", "Idea"],
  ["journal", "Journal"],
  ["note", "Note"],
];

// ⚠ Defaults are DERIVED from the home directory, never a literal path — this file is in a
// git repo that may one day be public, and a hard-coded /Users/<name> is both a leak and a
// machine-specific bug.
const DEFAULTS = {
  port: 7546,
  repoPath: `${os.homedir()}/Programming/Personal/SLIM`,
  uvPath: `${os.homedir()}/.local/bin/uv`,
  recordVoice: true,
  lastType: "meeting-note",
};

function pad(n) { return String(n).padStart(2, "0"); }

function serverArgv(settings, parentPid = process.pid) {
  return ["--directory", settings.repoPath, "run", "slim", "chat",
          "--port", String(settings.port), "--parent-pid", String(parentPid)];
}

// ⚠ The server DISCOVERS its own vault (slim/vaultpath.py). This plugin sends vault-RELATIVE
// paths, so a server that discovered a different vault than this one would look for the audio
// in a vault nobody wrote to, and file the note there. The plugin knows which vault it is in;
// it says so, and the override wins over discovery.
function serverEnv(vaultBase, env = process.env) {
  return Object.assign({}, env, { SLIM_VAULT: vaultBase });
}

function hhmmss(totalSeconds) {
  const s = Math.floor(totalSeconds);
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}

// A TCP connect is the only honest test of "is
// something serving this port" — a pid file lies after a crash.
function portOpen(port) {
  return new Promise((resolve) => {
    const sock = net.connect({ port, host: "127.0.0.1" }, () => {
      sock.end();
      resolve(true);
    });
    sock.on("error", () => resolve(false));
    sock.setTimeout(1000, () => {
      sock.destroy();
      resolve(false);
    });
  });
}

async function waitForPort(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await portOpen(port)) return true;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

/* Which recording a staged file belongs to. `<id>-001.webm` -> `<id>`; a name with no segment
 * number is returned unchanged, because recordings made before segments existed still have to
 * resolve. ONE spelling: the rule lived in three places across two languages, they agreed only
 * by luck, and missing one of them made every recording fail on 2026-08-26. */
function recordingIdFromStaged(filePath) {
  return nodePath.posix.basename(filePath, ".webm").replace(/-\d{3}$/, "");
}

function localDay(d = new Date()) {
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
}

function recordingId() {
  const d = new Date();
  const date = localDay(d);
  const time = `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return `${date}T${time}-${randomUUID().slice(0, 8)}`;
}

/* Presentation-only parser for the finished meeting object.
 *
 * It NEVER writes the note. `slim/record.py` remains the sole owner of frontmatter and of the
 * Summary / Notes / Transcript boundaries. Keeping the complete frontmatter block here is a
 * regression seam: the tabbed object may hide it visually, but it must never normalize, reorder,
 * or replace a byte of it.
 */
function splitMeetingNote(text) {
  const source = String(text || "");
  let frontmatter = "";
  let body = source;
  if (source.startsWith("---\n")) {
    const end = source.indexOf("\n---", 4);
    if (end >= 0) {
      const after = end + 4;
      frontmatter = source.slice(0, after);
      body = source.slice(after).replace(/^\r?\n/, "");
    }
  }
  let managed = body;
  let tail = "";
  const blockOpen = body.match(/^(`{3,})slim-meeting[^\n]*\n/m);
  if (blockOpen) {
    const fence = blockOpen[1].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const afterOpen = body.slice(blockOpen.index + blockOpen[0].length);
    const close = afterOpen.match(new RegExp(`^${fence}[ \\t]*(?:\\r?\\n|$)`, "m"));
    if (close) {
      managed = afterOpen.slice(0, close.index);
      tail = afterOpen.slice(close.index + close[0].length);
    }
  }
  const summaryMatch = managed.match(/<!-- slim:summary[^\n]*-->\s*\n## Summary\s*\n([\s\S]*?)\n<!-- \/slim:summary -->/);
  const notesMatch = managed.match(/(?:^|\n)## Notes\s*\n([\s\S]*?)(?=\n## Transcript\s*(?:\n|$))/);
  const transcriptMatch = managed.match(/(?:^|\n)## Transcript\s*\n([\s\S]*)$/);
  return {
    frontmatter,
    summary: summaryMatch ? summaryMatch[1].trim() : "",
    notes: notesMatch ? notesMatch[1].replace(/^\r?\n/, "").replace(/\s+$/, "") : "",
    transcript: transcriptMatch ? transcriptMatch[1].trim() : "",
    tail,
  };
}

function frontmatterScalar(text, key) {
  const frontmatter = splitMeetingNote(text).frontmatter;
  if (!frontmatter) return "";
  const escaped = String(key).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = frontmatter.match(new RegExp(`^${escaped}:\\s*(.*?)\\s*$`, "m"));
  if (!match) return "";
  return match[1].trim().replace(/^(["'])(.*)\1$/, "$2");
}

function hasMeetingBlock(text) {
  return /^`{3,}slim-meeting[^\n]*$/m.test(String(text || ""));
}

function embedLegacyMeetingBlock(text) {
  const source = String(text || "");
  if (hasMeetingBlock(source)) return source;
  let split = 0;
  if (source.startsWith("---\n")) {
    const end = source.indexOf("\n---", 4);
    if (end >= 0) split = end + 4;
  }
  const before = source.slice(0, split);
  const body = source.slice(split).replace(/^\r?\n+/, "").replace(/\s+$/, "");
  const longest = Math.max(0, ...Array.from(body.matchAll(/`+/g), (m) => m[0].length));
  const fence = "`".repeat(Math.max(4, longest + 1));
  return `${before}${before ? "\n\n" : ""}${fence}slim-meeting\n${body}\n${fence}\n\n`;
}

function markdownBlockShortcut(text) {
  const heading = String(text || "").match(/^(#{1,6})\s$/);
  if (heading) return { tag: `h${heading[1].length}`, kind: "block" };
  if (/^[-*+]\s$/.test(text || "")) return { tag: "ul", kind: "list" };
  if (/^\d+\.\s$/.test(text || "")) return { tag: "ol", kind: "list" };
  if (/^>\s$/.test(text || "")) return { tag: "blockquote", kind: "block" };
  return null;
}

/* Markdown this editor RENDERS but cannot reconstruct from the DOM: an Obsidian embed, a
 * formula, a wikilink. Each rendered node keeps the source it came from on `data-slim-src`, and
 * the serializer hands that back verbatim.
 *
 * ⚠ WITHOUT THIS THE EDITOR EATS THEIR SCREENSHOTS. `markdownFromEditable` runs on every
 * keystroke over the RENDERED tree, and `inlineMarkdown` had no case for an <img> — so a note
 * containing `![[Attachments/Recorder/…png]]` lost the embed the moment they typed a character
 * next to it. Same for `$…$`, which came back as bare text with its delimiters gone.
 */
const SOURCE_ATOM = /(!?\[\[[^\]]+\]\]|!\[[^\]]*\]\([^)]*\)|\$\$[^$]+\$\$|\$[^$\n]+\$)/g;

function stampSourceAtoms(root, markdown) {
  if (!root || typeof root.querySelectorAll !== "function") return;
  const atoms = String(markdown || "").match(SOURCE_ATOM) || [];
  if (!atoms.length) return;
  const matched = Array.from(
    root.querySelectorAll("img, .internal-embed, a.internal-link, mjx-container, .math"));
  // ⚠ OUTERMOST ONLY. Obsidian renders an embed as `<span class="internal-embed"><img></span>`
  // and a formula as `<span class="math"><mjx-container>`: two matches for one atom. Counting
  // both put every source after the first on the wrong node and the second half on none, and
  // each blur-save of the notes pane then halved their screenshots — 7, 4, 2 (2026-09-05).
  const inside = new Set(matched);
  const nodes = matched.filter((node) => {
    for (let up = node.parentElement; up; up = up.parentElement) if (inside.has(up)) return false;
    return true;
  });
  for (let i = 0; i < nodes.length && i < atoms.length; i += 1) {
    if (typeof nodes[i].setAttribute === "function") {
      nodes[i].setAttribute("data-slim-src", atoms[i]);
    }
  }
}

function inlineMarkdown(node) {
  if (!node) return "";
  if (node.nodeType === 3) return node.nodeValue || "";
  // The source this node was rendered from, when it is something we cannot rebuild.
  const stamped = typeof node.getAttribute === "function"
    ? node.getAttribute("data-slim-src") : null;
  if (stamped) return stamped;
  const tag = String(node.tagName || "").toLowerCase();
  if (tag === "br") return "\n";
  const inner = Array.from(node.childNodes || [], inlineMarkdown).join("");
  if (tag === "strong" || tag === "b") return `**${inner}**`;
  if (tag === "em" || tag === "i") return `*${inner}*`;
  if (tag === "code") return `\`${inner}\``;
  if (tag === "a") return `[${inner}](${node.getAttribute("href") || ""})`;
  return inner;
}

function markdownFromEditable(root) {
  const list = (node, depth) => {
    const ordered = String(node.tagName || "").toLowerCase() === "ol";
    return Array.from(node.children || []).map((li, i) => {
      // ⚠ An inner list is a CHILD of its <li>. Serializing the item inline flattened it into
      // the item's text, and their sub-bullets came back as a bare paragraph (2026-09-05).
      const own = [], inner = [];
      for (const child of li.childNodes || []) {
        const tag = String(child.tagName || "").toLowerCase();
        if (tag === "ul" || tag === "ol") inner.push(list(child, depth + 1));
        else own.push(inlineMarkdown(child));
      }
      const marker = ordered ? `${i + 1}.` : "-";
      return [`${"\t".repeat(depth)}${marker} ${own.join("").trim()}`, ...inner].join("\n");
    }).join("\n");
  };
  const block = (node) => {
    const tag = String(node.tagName || "").toLowerCase();
    if (/^h[1-6]$/.test(tag)) return `${"#".repeat(Number(tag[1]))} ${inlineMarkdown(node).trim()}`;
    if (tag === "ul" || tag === "ol") return list(node, 0);
    if (tag === "blockquote") {
      return inlineMarkdown(node).split("\n").map((line) => `> ${line}`).join("\n").trim();
    }
    if (tag === "pre") return `\`\`\`\n${node.textContent || ""}\n\`\`\``;
    if (tag === "hr") return "---";
    return inlineMarkdown(node).trimEnd();
  };
  return Array.from((root && root.children) || []).map(block)
    .filter((part) => part.trim()).join("\n\n").trim();
}

/* ONE tabbed reader for a meeting note (2026-09-03). Two surfaces show the same three tabs —
 * the recorder's review and finished states, and every `slim-meeting` block rendered passively
 * in a note — and they had drifted into two copies of the same tab bar, body shell and section
 * labels. The recorder's extra affordances (an editable summary, a notes editor that saves on
 * blur) arrive as pane overrides rather than as a second copy of the reader.
 *
 * `editable` says which surface this is. False is the plain read-only reader; true means the
 * caller supplies `summaryPane`/`notesPane` and owns what goes inside them.
 */
function renderMeetingTabs(app, el, parts, sourcePath, {
  editable = false, component = null, activeTab = "summary", onTab = null,
  summaryPane = null, notesPane = null,
} = {}) {
  const renderMarkdown = (target, markdown) => {
    if (!markdown) {
      target.createEl("p", { cls: "slim-empty", text: "Nothing was captured here." });
      return;
    }
    if (MarkdownRenderer && typeof MarkdownRenderer.render === "function") {
      MarkdownRenderer.render(app, markdown, target, sourcePath || "", component).catch((e) => {
        console.error("[slim] render meeting", e);
        target.setText(markdown);
      });
    } else target.setText(markdown);
  };

  const tabs = el.createDiv({ cls: "slim-object-tabs", attr: { role: "tablist" } });
  [["summary", "AI summary"], ["notes", "My notes"], ["transcript", "Transcript"]]
    .forEach(([value, label]) => {
      const tab = tabs.createEl("button", {
        cls: `slim-object-tab${activeTab === value ? " is-active" : ""}`,
        text: label,
        attr: { role: "tab", "aria-selected": String(activeTab === value) },
      });
      tab.addEventListener("click", () => { if (onTab) onTab(value); });
    });

  const body = el.createDiv({ cls: `slim-object-body is-${activeTab}` });
  if (activeTab === "summary") {
    // ONE thing per tab (2026-08-26). Review used to show the generated summary beside their
    // notes as a reference column; two documents in the tab named for one of them read as
    // clutter, and "My notes" already exists one click away.
    body.createDiv({ cls: "slim-object-section-label", text: "Generated summary" });
    if (editable && summaryPane) summaryPane(body);
    else renderMarkdown(body.createDiv({ cls: "slim-rendered-markdown" }), parts.summary);
  } else if (activeTab === "notes") {
    body.createDiv({ cls: "slim-object-section-label", text: "Your notes" });
    if (editable && notesPane) notesPane(body);
    else renderMarkdown(body.createDiv({ cls: "slim-rendered-markdown" }), parts.notes);
  } else {
    body.createDiv({ cls: "slim-object-section-label", text: "Raw transcript — read only" });
    renderMarkdown(body.createDiv({ cls: "slim-rendered-markdown slim-transcript" }),
                   parts.transcript);
  }
  return body;
}

/* A review can be open for only one meeting at a time, but a vault can render many meeting
 * blocks at once. This renderer deliberately owns no RecorderView state: it keeps every other
 * recorded note visible while the active controller retains the pending approval unchanged.
 */
function renderPassiveMeetingBlock(app, plugin, el, source, sourcePath) {
  const body = String(source || "");
  const longestFence = Math.max(0, ...Array.from(body.matchAll(/`+/g), (m) => m[0].length));
  const fence = "`".repeat(Math.max(4, longestFence + 1));
  const parts = splitMeetingNote(`${fence}slim-meeting\n${body}\n${fence}`);
  let activeTab = "summary";

  const draw = () => {
    el.empty();
    el.addClass("slim-meeting-object");
    el.addClass("slim-passive-meeting");
    const head = el.createDiv({ cls: "slim-object-head" });
    const copy = head.createDiv({ cls: "slim-object-head-copy" });
    copy.createDiv({ cls: "slim-object-eyebrow", text: "SLIM meeting note" });
    copy.createEl("h3", { text: "Meeting note" });
    const badge = head.createDiv({ cls: "slim-object-status" });
    badge.createSpan({ cls: "slim-object-status-dot" });
    badge.createSpan({ text: "Saved" });

    renderMeetingTabs(app, el, parts, sourcePath, {
      editable: false, component: plugin, activeTab,
      onTab: (value) => { activeTab = value; draw(); },
    });
  };
  draw();
}

// One caption request per this much audio: 2.0 s of 16 kHz PCM16. The server re-encodes ~20 s
// of context on EVERY request whatever the chunk holds (parakeet-mlx's streaming design, ~170 ms
// per call on this machine), so the number of requests IS the cost — and a request that carries
// 85 ms of audio costs the same as one that carries two seconds.
const LIVE_FLUSH_BYTES = 64000;

function pcm16k(input, inputRate) {
  const source = input || new Float32Array();
  const ratio = Math.max(1, Number(inputRate || 16000) / 16000);
  const length = Math.floor(source.length / ratio);
  const out = Buffer.alloc(length * 2);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(source.length, Math.floor((i + 1) * ratio));
    let value = 0;
    for (let j = start; j < end; j++) value += source[j];
    value = Math.max(-1, Math.min(1, value / Math.max(1, end - start)));
    out.writeInt16LE(value < 0 ? value * 32768 : value * 32767, i * 2);
  }
  return out;
}

class RecorderView {
  constructor(leaf, plugin) {
    this.leaf = leaf || null;
    this.plugin = plugin;
    this.app = (leaf && leaf.app) || plugin.app;
    this.contentEl = null;
    this.editorView = null;
    this.activeTab = "notes";
    this.noteParts = { summary: "", notes: "", transcript: "" };
    this.summaryEditing = false;
    this.processingProgress = null;
    this.stopProgressWatch = null;
    this.liveTranscript = "";
    // What earlier SEGMENTS of this same recording said. Resume used to show an empty pane and
    // a clock back at zero, so the first forty minutes read as lost while the server had them.
    this.liveCarry = "";
    this.elapsedCarry = 0;
    this.liveTranscriptStatus = "Listening…";
    this.liveTranscriptQueue = Promise.resolve();
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveProcessor = null;
    this.liveTranscriptActive = false;
    this.liveTranscriptPaused = false;
    this.state = "idle";          // idle | recording | processing | paused | card | finished | error
    this.chunks = [];             // ONLY used when the disk lane is unavailable
    this.pending = [];            // written buffers that a failed append still owes the file
    this.writeQueue = Promise.resolve();
    this.diskWrite = false;
    this.audioRel = "";
    this.audioAbs = "";
    this.recorder = null;
    this.streams = [];            // every raw input stream, so all get stopped
    this.audioCtx = null;
    this.elapsed = 0;
    this.timer = null;
    this.recordingId = "";
    this.draftRel = "";
    this.editorFrozen = false;
    this.editorLeaf = null;
    this.title = "";
    this.type = plugin.settings.lastType || "meeting-note";
    this.recordVoice = plugin.settings.recordVoice !== false;
    this.result = null;           // the server's response
    this.edited = false;          // has they touched the card?
    this.filedNotePath = "";
    this.filingFrom = "";
    this.filing = false;
    this.cardTitle = null;
    this.reviewCloseNotified = false;
  }

  async onOpen() {
    this.attachEditorObject();
    this.render();
  }

  async onClose() {
    // The final note is already safe. Closing or reloading the plugin must not impersonate the
    // explicit approval action by applying half-reviewed edits.
    this.teardownStream();
    if (this.stopProgressWatch) this.stopProgressWatch();
    this.detachEditorObject();
  }

  ownsEditor(info) {
    return info === (this.editorLeaf && this.editorLeaf.view) && !this.editorFrozen &&
      !!this.draftRel && !!info.file && info.file.path === this.draftRel;
  }

  async open() {
    if (this.state === "idle" && !this.draftRel) {
      this.recordingId = recordingId();
      await this.createDraft();
    } else if (this.filedNotePath) {
      await this.loadEditorFile(this.filedNotePath);
    } else if (this.draftRel) {
      await this.loadEditorFile(this.draftRel);
    }
    this.attachEditorObject();
    this.render();
  }

  attachEditorObject() {
    const view = this.editorLeaf && this.editorLeaf.view;
    if (!(view instanceof MarkdownView) || !view.file) return false;
    const object = typeof this.plugin.meetingBlockFor === "function"
      ? this.plugin.meetingBlockFor(view.file.path) : null;
    if (!object) return false;
    if (this.editorView === view && this.contentEl && this.contentEl.isConnected !== false) {
      return true;
    }
    this.detachEditorObject();
    this.editorView = view;
    object.empty();
    object.addClass("slim-meeting-object");
    object.addClass("slim-recorder");
    this.contentEl = object;
    return true;
  }

  detachEditorObject() {
    if (this.editorView && this.editorView.contentEl) {
      this.editorView.contentEl.removeClass("slim-recorder-draft-frozen");
    }
    if (this.contentEl && typeof this.contentEl.empty === "function") this.contentEl.empty();
    this.contentEl = null;
    this.editorView = null;
  }

  async ensureVaultDirs(path) {
    const parts = path.split("/").filter(Boolean);
    let current = "";
    for (const part of parts) {
      current = current ? `${current}/${part}` : part;
      if (!(await this.app.vault.adapter.exists(current))) {
        try { await this.app.vault.createFolder(current); } catch (e) { /* raced */ }
      }
    }
  }

  async loadEditorFile(path) {
    const file = this.app.vault.getAbstractFileByPath(path);
    if (!file) throw new Error(`no note at ${path}`);
    if (!this.isOpenMarkdownLeaf(this.editorLeaf)) {
      this.detachEditorObject();
      const prior = this.plugin.editorLeaf;
      const priorOwner = this.manager && typeof this.manager.sessionForLeaf === "function"
        ? this.manager.sessionForLeaf(prior) : null;
      this.editorLeaf = this.isOpenMarkdownLeaf(prior) && (!priorOwner || priorOwner === this)
        ? prior : this.app.workspace.getLeaf("split");
    }
    await this.editorLeaf.openFile(file, { active: true, state: { mode: "source", source: false } });
    this.plugin.editorLeaf = this.editorLeaf;
    // Live Preview exposes the source of the block containing the cursor. A brand-new note
    // otherwise opens with its caret inside the empty slim-meeting fence and looks broken until
    // the user clicks away. Keep the object rendered while leaving the native editor ready for
    // ordinary Markdown immediately below it.
    if (path === this.draftRel && this.state === "idle") {
      const editor = this.editorLeaf.view && this.editorLeaf.view.editor;
      if (editor && typeof editor.lastLine === "function" &&
          typeof editor.setCursor === "function") {
        const moveBelowObject = () => {
          editor.setCursor({ line: editor.lastLine(), ch: 0 });
          if (typeof editor.focus === "function") editor.focus();
        };
        moveBelowObject();
        // `openFile()` resolves before Obsidian has finished restoring the leaf's cursor. Run
        // once more after that render cycle or its restoration puts the caret back inside the
        // fence and exposes the raw block on first open.
        if (typeof requestAnimationFrame === "function") {
          await new Promise((resolve) => requestAnimationFrame(() => {
            moveBelowObject();
            resolve();
          }));
        }
      }
    }
    this.attachEditorObject();
  }

  isOpenMarkdownLeaf(leaf) {
    if (!leaf || !(leaf.view instanceof MarkdownView)) return false;
    const workspace = this.app && this.app.workspace;
    if (!workspace || typeof workspace.getLeavesOfType !== "function") return true;
    return workspace.getLeavesOfType("markdown").includes(leaf);
  }

  async loadNoteParts() {
    if (!this.filedNotePath || !this.app || !this.app.vault) return this.noteParts;
    const file = this.app.vault.getAbstractFileByPath(this.filedNotePath);
    if (!file) return this.noteParts;
    const text = await this.app.vault.read(file);
    this.noteParts = splitMeetingNote(text);
    return this.noteParts;
  }

  /* A cancelled recording parks its segments, their caches and the draft on DISK, but the
   * session lives only in this view. After an Obsidian restart the draft rendered as a passive
   * block: no Resume, no Finish, and no way back to the audio except by hand.
   * Adopting it turns the draft back into the paused object it was. */
  async adoptPausedDraft(draftPath) {
    const name = nodePath.posix.basename(draftPath);
    if (!name.endsWith(DRAFT_SUFFIX) || !name.includes("--recording--")) return false;
    const id = name.slice(0, -DRAFT_SUFFIX.length).split("--recording--").pop();
    if (!id) return false;
    const adapter = this.app && this.app.vault && this.app.vault.adapter;
    if (!adapter || typeof adapter.list !== "function") return false;
    let listing;
    try { listing = await adapter.list(STAGING_DIR); } catch (e) { return false; }
    const segments = (listing.files || [])
      .filter((f) => f.toLowerCase().endsWith(".webm") && recordingIdFromStaged(f) === id)
      .sort();
    if (!segments.length) return false;      // a draft with no audio is just a draft

    this.resetSession();
    this.recordingId = id;
    this.draftRel = draftPath;
    this.segments = segments;
    this.audioRel = segments[segments.length - 1];
    this.type = draftPath.startsWith("Journal/") ? "journal" : this.type;
    this.pausedTranscript = null;
    this.state = "paused";
    return true;
  }

  async openExisting(file, leaf) {
    if (!file || !leaf || !(leaf.view instanceof MarkdownView)) return false;
    const cache = this.app.metadataCache && this.app.metadataCache.getFileCache(file);
    const fm = cache && cache.frontmatter;
    if (!fm || fm.origin !== "recorded") return false;
    let text = await this.app.vault.read(file);
    if (!hasMeetingBlock(text)) {
      text = embedLegacyMeetingBlock(text);
      await this.app.vault.modify(file, text);
    }
    const parts = splitMeetingNote(text);
    if (!parts.transcript) return false;
    this.resetSession();
    this.editorLeaf = leaf;
    this.filedNotePath = file.path;
    this.noteParts = parts;
    const parent = file.path.includes("/") ? file.path.slice(0, file.path.lastIndexOf("/")) : "";
    const topics = Array.isArray(fm.topics) ? fm.topics.slice() : [];
    const type = String(fm.type || "meeting-note");
    this.result = {
      note: file.path,
      title: String(fm.title || file.basename || "Meeting note"),
      summary: parts.summary,
      notes_md: parts.notes,
      card: { dest_dir: parent, type_tag: type, topics, folders: [] },
    };
    this.cardTitle = this.result.title;
    // Absence means complete for backwards compatibility: hundreds of historical recordings
    // predate durable review state and must not suddenly become an inbox. New recordings write
    // `pending`, which makes review survive a closed tab or Obsidian restart.
    this.state = frontmatterScalar(text, "review_status") === "pending" ? "card" : "finished";
    this.activeTab = "summary";
    this.attachEditorObject();
    this.render();
    return true;
  }

  /* Returns the render promise when there is one, so a caller can act when the DOM is
   * actually there; a synchronous fallback returns undefined. */
  renderMarkdown(el, markdown) {
    if (!markdown) {
      el.createEl("p", { cls: "slim-empty", text: "Nothing was captured here." });
      return undefined;
    }
    if (MarkdownRenderer && typeof MarkdownRenderer.render === "function") {
      return MarkdownRenderer.render(this.app, markdown, el,
                                     this.filedNotePath || this.draftRel || "",
                                     this.plugin).catch((e) => {
        console.error("[slim] render markdown", e);
        el.setText(markdown);
      });
    }
    el.setText(markdown);
    return undefined;
  }

  renderLiveMarkdownEditor(parent, markdown, onChange, cls = "") {
    const editor = parent.createDiv({
      cls: `slim-live-markdown-editor ${cls}`.trim(),
      attr: {
        contenteditable: "true", spellcheck: "true", role: "textbox",
        "aria-multiline": "true", "data-placeholder": "Write Markdown…",
      },
    });
    if (markdown) {
      const rendered = this.renderMarkdown(editor, markdown);
      // After the async render lands, mark every node whose source the serializer cannot
      // reconstruct — otherwise the first keystroke deletes it.
      const stamp = () => stampSourceAtoms(editor, markdown);
      stamp();
      if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
        window.setTimeout(stamp, 0);
      }
      // ⚠ The timer is a guess at when the render lands. Stamp when it actually resolves: an
      // embed rendered after the last pass has no source and is deleted by the first keystroke.
      if (rendered && typeof rendered.then === "function") rendered.then(stamp);
    }
    else {
      const p = editor.createEl("p");
      p.createEl("br");
    }
    const save = (event) => onChange(event && event.target && typeof event.target.value === "string"
      ? event.target.value : markdownFromEditable(editor));
    editor.addEventListener("input", save);
    editor.addEventListener("keyup", (event) => {
      if (event.key !== " ") return;
      const selection = typeof window !== "undefined" && window.getSelection
        ? window.getSelection() : null;
      let current = selection && selection.anchorNode;
      if (current && current.nodeType === 3) current = current.parentElement;
      while (current && current.parentElement !== editor) current = current.parentElement;
      if (!current) return;
      const shortcut = markdownBlockShortcut(current.textContent || "");
      if (!shortcut || typeof document === "undefined") return;
      let target;
      if (shortcut.kind === "list") {
        target = document.createElement(shortcut.tag);
        const item = document.createElement("li");
        item.appendChild(document.createElement("br"));
        target.appendChild(item);
        current.replaceWith(target);
        target = item;
      } else {
        target = document.createElement(shortcut.tag);
        target.appendChild(document.createElement("br"));
        current.replaceWith(target);
      }
      const range = document.createRange();
      range.selectNodeContents(target);
      range.collapse(true);
      selection.removeAllRanges();
      selection.addRange(range);
      save();
    });
    return editor;
  }

  async saveDraft() {
    const view = this.editorLeaf && this.editorLeaf.view;
    if (view instanceof MarkdownView && typeof view.save === "function") await view.save();
  }

  resetSession() {
    if (this.manager) this.manager.releaseCapture(this);
    this.teardownStream();
    this.detachEditorObject();
    this.state = "idle";
    this.activeTab = "notes";
    this.noteParts = { summary: "", notes: "", transcript: "" };
    this.summaryEditing = false;
    this.processingProgress = null;
    if (this.stopProgressWatch) this.stopProgressWatch();
    this.stopProgressWatch = null;
    this.liveTranscript = "";
    this.liveCarry = "";
    this.elapsedCarry = 0;
    this.liveTranscriptStatus = "Listening…";
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveTranscriptActive = false;
    this.liveTranscriptPaused = false;
    this.recordingId = "";
    this.draftRel = "";
    this.audioRel = "";
    this.segments = [];
    // ⚠ Everything a retry or a resume leaves behind. Left set, an in-flight retry on note A
    // resolved into note B's view — its summary written over B's — because `openExisting`
    // repoints the view but used to leave `retrying` true.
    this.appendTo = "";
    this.resumeReturnState = "";
    this.retrying = false;
    this.retryStatus = "";
    this.retryInstructions = "";
    this.pausedTranscript = null;
    this.jobId = "";
    this.attempt = 0;
    this.audioAbs = "";
    this.title = "";
    this.result = null;
    this.edited = false;
    this.filedNotePath = "";
    this.filingFrom = "";
    this.filing = false;
    this.cardTitle = null;
    this.reviewCloseNotified = false;
    this.errorText = "";
    this.freezeEditor(false);
  }

  async reconcileMissingSessionFiles() {
    const missingDraft = this.draftRel && !this.app.vault.getAbstractFileByPath(this.draftRel);
    const missingNote = this.filedNotePath &&
      !this.app.vault.getAbstractFileByPath(this.filedNotePath);
    if (!missingDraft && !missingNote) return false;
    const audio = this.audioRel;
    this.resetSession();
    if (audio) {
      new Notice(`SLIM: the deleted note was released. Its staged audio remains at ${audio}.`);
    }
    return true;
  }

  /* A deletion the server made on this session's behalf is not their instruction to let go:
   * filing removes the draft before its response arrives, and `apply` unlinks the old path
   * while it moves the note — Obsidian reports that unlink as a delete, usually mid-request.
   * Read as their deletion it reset the session under the Approve button: "the deleted note was
   * released", then "Cannot set properties of null (setting 'note')" (2026-09-04). */
  expectsDeletion(path) {
    return this.state === "processing" || this.state === "recording" || path === this.filingFrom;
  }

  handleVaultDelete(file) {
    if (!file || (file.path !== this.draftRel && file.path !== this.filedNotePath)) return;
    if (this.expectsDeletion(file.path)) return;
    this.reconcileMissingSessionFiles().catch((e) => console.error("[slim] delete reconcile", e));
  }

  async trashManagedPath(path) {
    if (!path) return;
    const file = this.app.vault.getAbstractFileByPath(path);
    if (!file) return;
    if (this.app.fileManager && typeof this.app.fileManager.trashFile === "function") {
      await this.app.fileManager.trashFile(file);
    } else if (typeof this.app.vault.trash === "function") {
      await this.app.vault.trash(file, true);
    } else {
      await this.app.vault.delete(file);
    }
  }

  async discardSession(restart) {
    const id = this.recordingId;
    const draft = this.draftRel;
    // Every segment of this recording, not just the one that was open — a resumed session
    // leaves several, and half a deleted recording is the worst of both outcomes.
    const audios = [...new Set([...(this.segments || []), this.audioRel].filter(Boolean))];
    const attachmentFolder = id ? `${RECORDER_ATTACHMENTS}/${id}` : "";
    const draftIsManaged = draft && draft.endsWith(DRAFT_SUFFIX) &&
      (draft.startsWith("Capture/_unfiled/") || draft.startsWith("Journal/"));
    const managed = (audio) => Boolean(audio) && Boolean(id) &&
      audio.startsWith(`${STAGING_DIR}/`) && recordingIdFromStaged(audio) === id;
    if (draftIsManaged) await this.trashManagedPath(draft);
    for (const audio of audios) {
      if (!managed(audio)) continue;
      await this.trashManagedPath(audio);
      // The server writes `<segment>.transcript.json` beside each segment; discarding the
      // recording discards its cached words too rather than leaving them in _incoming.
      await this.trashManagedPath(`${audio}.transcript.json`);
    }
    if (attachmentFolder) await this.trashManagedPath(attachmentFolder);
    this.resetSession();
    new Notice("SLIM: discarded the failed recording and its draft.");
    if (restart) {
      if (this.manager) this.manager.claimCapture(this);
      await this.open();
      if (this.manager) this.manager.indexSession(this);
    }
  }

  closeSession() {
    this.resetSession();
    if (this.manager) this.manager.removeSession(this);
    new Notice("SLIM: closed the recorder. Saved draft and audio were kept.");
  }

  async createDraft() {
    if (!this.recordingId) this.recordingId = recordingId();
    const digits = this.recordingId.slice(0, 8);
    const date = `${digits.slice(0, 4)}-${digits.slice(4, 6)}-${digits.slice(6, 8)}`;
    const folder = this.type === "journal" ? "Journal" : "Capture/_unfiled";
    await this.ensureVaultDirs(folder);
    this.draftRel = `${folder}/${date}--recording--${this.recordingId}${DRAFT_SUFFIX}`;
    await this.app.vault.create(this.draftRel, EMPTY_MEETING_BLOCK);
    if (this.manager) this.manager.indexSession(this);
    await this.loadEditorFile(this.draftRel);
    return this.draftRel;
  }

  async ensureDraftLocation() {
    if (!this.draftRel) return this.createDraft();
    const folder = this.type === "journal" ? "Journal" : "Capture/_unfiled";
    if (this.draftRel.startsWith(`${folder}/`)) return this.draftRel;
    const file = this.app.vault.getAbstractFileByPath(this.draftRel);
    if (!file) { this.draftRel = ""; return this.createDraft(); }
    await this.ensureVaultDirs(folder);
    const next = `${folder}/${nodePath.posix.basename(this.draftRel)}`;
    if (this.app.fileManager && typeof this.app.fileManager.renameFile === "function") {
      await this.app.fileManager.renameFile(file, next);
    } else if (typeof this.app.vault.rename === "function") {
      await this.app.vault.rename(file, next);
    } else {
      throw new Error("Obsidian cannot move the setup draft into its privacy-safe folder");
    }
    this.draftRel = next;
    if (this.manager) this.manager.indexSession(this);
    await this.loadEditorFile(next);
    return next;
  }

  async removeUntouchedDraft() {
    if (!this.draftRel) return false;
    const draft = this.app.vault.getAbstractFileByPath(this.draftRel);
    if (!draft) { this.draftRel = ""; return false; }
    const contents = await this.app.vault.read(draft);
    if (contents !== "" && contents !== EMPTY_MEETING_BLOCK) return false;
    await this.app.vault.delete(draft);
    this.draftRel = "";
    return true;
  }

  freezeEditor(frozen) {
    this.editorFrozen = !!frozen;
    const view = this.editorLeaf && this.editorLeaf.view;
    if (view && view.contentEl && typeof view.contentEl.toggleClass === "function") {
      view.contentEl.toggleClass("slim-recorder-draft-frozen", this.editorFrozen);
    }
    if (frozen && typeof document !== "undefined" && document.activeElement &&
        typeof document.activeElement.blur === "function") document.activeElement.blur();
  }

  async handleEditorPaste(event, editor) {
    const files = Array.from((event.clipboardData && event.clipboardData.files) || []);
    if (!files.length || files.some((f) => !String(f.type || "").startsWith("image/"))) {
      return false;
    }
    const extensions = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };
    if (files.some((f) => !extensions[f.type])) return false;

    event.preventDefault();
    const folder = `${RECORDER_ATTACHMENTS}/${this.recordingId}`;
    await this.ensureVaultDirs(folder);
    const embeds = [];
    for (const file of files) {
      let n = 1;
      let path;
      do {
        path = `${folder}/paste-${String(n).padStart(3, "0")}.${extensions[file.type]}`;
        n += 1;
      } while (await this.app.vault.adapter.exists(path));
      await this.app.vault.createBinary(path, await file.arrayBuffer());
      embeds.push(`![[${path}]]`);
    }
    editor.replaceSelection(embeds.join("\n"));
    return true;
  }

  // ---- recording -----------------------------------------------------------------------

  /* ⚠ The two inputs want OPPOSITE treatment, and getting this backwards is worse than
   * leaving it off.
   *
   * The microphone is in a room: chair movement, keyboard, fans. Chromium ships real DSP for
   * exactly that, and unasked-for it is not reliably on. So the mic gets noise suppression,
   * echo cancellation and auto gain.
   *
   * The loopback input is NOT a room — it is a clean digital copy of what the machine is
   * playing. Noise suppression there chews on the far end's voices, and echo cancellation
   * would try to remove the speaker audio, which is the entire thing we are recording. So it
   * is captured raw.
   */
  async openMic() {
    return navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  }

  /* What the machine is PLAYING, straight from macOS. No virtual device, no routing.
   *
   * Electron >= 39 defaults to Apple's Core Audio Tap API for desktop audio capture on
   * macOS >= 14.2, so `audio: "loopback"` returns a track labelled "System audio" carrying the
   * real mix. Measured 2026-08-22 against a YouTube lecture: peak 0.0 dBFS, RMS -18.7 dBFS —
   * within half a decibel of the same lecture captured through BlackHole.
   *
   * ⚠ `video` is REQUIRED and must be a WebFrameMain or a DesktopCapturerSource. We hand back
   * the requesting frame — Obsidian's own window — and stop the track immediately. That is not
   * a trick to save code: asking desktopCapturer for a SCREEN drags in screen-recording
   * permission for a feature that only ever wanted audio.
   *
   * ⚠ The handler lives on the SHARED default session, so it is installed for exactly one
   * request and always removed, including when the request throws. Leaving it installed would
   * silently hijack any other getDisplayMedia call in Obsidian.
   */
  async openSystemAudio() {
    const { session } = require("@electron/remote");
    session.defaultSession.setDisplayMediaRequestHandler((request, callback) => {
      callback({ video: request.frame, audio: "loopback" });
    });
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      stream.getVideoTracks().forEach((track) => track.stop());
      if (!stream.getAudioTracks().length) throw new Error("no audio track was returned");
      return stream;
    } finally {
      session.defaultSession.setDisplayMediaRequestHandler(null);
    }
  }

  /* Both sides of the conversation in ONE file, and kept APART inside it.
   *
   * A microphone records the room; on headphones that is only them, and the transcript ends up
   * a monologue with the other half missing. So the machine's own output is captured too —
   * directly, via `openSystemAudio`.
   *
   * ⚠ THIS USED TO COST THEM A DEVICE SETUP, and that was the whole complaint (2026-08-22):
   * `getUserMedia` sees only INPUT devices, so reaching system output meant installing
   * BlackHole to impersonate a microphone and building a Multi-Output Device — a NEW one per
   * pair of headphones — to fork the sound so they could still hear it. Notion is one click.
   * Nothing about that was necessary: the Electron Obsidian ships could already ask macOS.
   * The lesson is the pattern, not the fix — reach for what the HOST can do before reaching
   * for the universal workaround.
   *
   * The two are opened separately and summed here with Web Audio rather than depending on a
   * pre-built Aggregate Device having the right channel layout. It degrades honestly: system
   * audio can fail and you still get the mic, with a Notice saying so.
   *
   * ⚠ LEFT IS THE MICROPHONE, RIGHT IS THE MACHINE — `slim/speakers.py` reads them that way.
   * They used to be summed, and once summed nobody can say who spoke. Apart, "me or them" is
   * a fact of the capture, decided on the server with no model. A missing source leaves its
   * channel silent; it never moves to the other side. The caption tap and the level meter
   * each ask for ONE channel, so Web Audio downmixes for them — leave them alone.
   */
  async buildStream() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.audioCtx = new Ctx();
    const dest = this.audioCtx.createMediaStreamDestination();
    const sides = this.audioCtx.createChannelMerger(2);
    sides.connect(dest);
    this.streams = [];

    let sources = 0;

    if (this.recordVoice) {
      const mic = await this.openMic();
      this.streams.push(mic);
      this.audioCtx.createMediaStreamSource(mic).connect(sides, 0, MIC_CHANNEL);
      sources++;
    }

    try {
      const sys = await this.openSystemAudio();
      this.streams.push(sys);
      this.audioCtx.createMediaStreamSource(sys).connect(sides, 0, SYSTEM_CHANNEL);
      sources++;
    } catch (e) {
      // Losing the far side is bad; losing the recording is worse. Carry on with the mic and
      // SAY SO — ⚠ degrading silently is how a two-hour loss goes unnoticed. There is
      // deliberately no fall back to device pickers: that apparatus is what this removed.
      console.error("[slim] system audio", e);
      new Notice(this.recordVoice
        ? "SLIM: could not capture system audio — recording your voice only."
        : "SLIM: could not capture system audio, and 'Record my voice' is off.");
    }

    if (!sources) throw new Error("no audio source available — nothing would be recorded");
    return dest.stream;
  }

  /* One recording, several audio files. A second MediaRecorder session writes its own
   * container header, so appending a resumed take to the first file risks a recording nothing
   * can play; a numbered segment beside it cannot. The server joins their transcripts. */
  nextSegmentPath() {
    const n = String((this.segments || []).length + 1).padStart(3, "0");
    return `${STAGING_DIR}/${this.recordingId}-${n}.webm`;
  }

  async start() {
    // Idempotent for the holder; a setup draft whose lease a Resume took over is refused here.
    if (this.manager) this.manager.claimCapture(this);
    if (!this.recordingId) this.recordingId = recordingId();
    // ⚠ A FILED NOTE IS ITS OWN DRAFT. `ensureDraftLocation` creates one whenever `draftRel`
    // is empty, and it is empty for a filed note — so a resume used to mint an orphan in
    // Capture/_unfiled, switch the editor to it, and lose everything typed while resuming.
    if (!this.appendTo) await this.ensureDraftLocation();
    const mixed = await this.buildStream();
    if (!this.segments) this.segments = [];
    this.audioRel = this.nextSegmentPath();
    this.segments.push(this.audioRel);
    try {
      let mime = "audio/webm;codecs=opus";
      if (typeof MediaRecorder.isTypeSupported === "function" &&
          !MediaRecorder.isTypeSupported(mime)) {
        mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      }
      this.recorder = mime ? new MediaRecorder(mixed, { mimeType: mime })
                           : new MediaRecorder(mixed);
      this.chunks = [];
      this.pending = [];
      this.writeQueue = Promise.resolve();
      this.warnedWrite = false;
      await this.ensureStagingDirs();
      this.openDiskFile();
      this.recorder.ondataavailable = (e) => { if (e.data && e.data.size) this.onChunk(e.data); };
      // A timeslice means the buffer is flushed periodically rather than held whole until stop —
      // and every flushed chunk is appended to the file on disk immediately. A crash mid-meeting
      // then costs the last five seconds, not the meeting.
      this.recorder.start(5000);
    } catch (e) {
      try { await this.removeUntouchedDraft(); }
      catch (cleanupError) { console.error("[slim] draft cleanup", cleanupError); }
      this.teardownStream();
      throw e;
    }

    this.setupMeter(mixed);
    this.elapsed = this.elapsedCarry || 0;
    this.timer = window.setInterval(() => {
      if (this.recorder && this.recorder.state === "recording") {
        this.elapsed += 1;
        const el = this.contentEl.querySelector(".slim-timer");
        if (el) el.setText(hhmmss(this.elapsed));
      }
    }, 1000);
    if (typeof this.plugin.registerInterval === "function") this.plugin.registerInterval(this.timer);

    this.state = "recording";
    this.render();
    this.startLiveTranscript(mixed);
  }

  /* One recording's captions and clock, across every segment of it. */
  shownTranscript() {
    return [this.liveCarry, this.liveTranscript].filter(Boolean).join("\n\n");
  }

  /* Fold what is on screen into the carry. Used by a segment boundary AND by a mid-segment
   * session restart — the restart must NOT touch the clock, which is still running. */
  carryTranscript() {
    this.liveCarry = [this.liveCarry, this.liveTranscript].filter(Boolean).join("\n\n");
    this.liveTranscript = "";
  }

  carryTranscriptAndClock() {
    this.carryTranscript();
    this.elapsedCarry = this.elapsed || 0;
  }

  /* What the caption pane says. Three states, and they must be distinguishable: PAUSED (they did
   * that and nothing is wrong), running, and off. "Live transcript · draft" sitting frozen
   * because captions had quietly died is what sent them looking for a bug in the wrong place. */
  liveStatusText() {
    if (!this.liveTranscriptActive) return "Live transcript off — recording continues";
    if (this.recorder && this.recorder.state === "paused") return "Paused — not listening";
    return this.liveTranscriptStatus || "Live transcript · draft";
  }

  /* Fold one caption result into what is on screen.
   *
   * ⚠ THE STREAM IS CUMULATIVE, AND IT REVISES ITS OWN PREFIX. That second half is what this
   * used to get wrong. The rule was "growth replaces, anything else appends", written to guard
   * against a rolling-window ASR — but parakeet does not roll, it REWRITES: as more audio
   * arrives, `"...produces stable."` becomes `"...produces stable text we can compare..."` and
   * `"sixteen kilohertz"` becomes `"16 kHz"`. Every rewrite fails `startsWith`, so the append
   * branch appended the WHOLE cumulative text to what was already there, and compounded on the
   * next chunk. Measured 2026-08-27 over 99 s of speech: 19 of 49 chunks revised their prefix,
   * and the pane showed 37,785 chars of a 1,477-char transcript — 25.6x, which is what the owner
   * saw as captions spamming repeated words.
   *
   * So: REPLACE, always. Server contract, not an assumption — `/api/record/live` returns
   * `stream.result.text`, the full text for that session, and 0 of those 49 chunks shrank.
   * A session that RESTARTS does start empty, which is the case the old append branch was
   * really protecting; that is now handled where it belongs, by folding the words on screen
   * into `liveCarry` before asking for a new session, so nothing vanishes.
   */
  applyLiveTranscript(text) {
    const next = String(text || "").trim();
    if (!next) return;
    this.liveTranscript = next;
  }

  updateLiveTranscriptUI() {
    if (!this.contentEl) return;
    const status = this.contentEl.querySelector(".slim-live-transcript-status");
    if (status) status.setText(this.liveStatusText());
    const text = this.contentEl.querySelector(".slim-live-transcript-text");
    if (text) {
      text.setText(this.shownTranscript() || "Listening for speech…");
      if (typeof text.scrollTo === "function") text.scrollTo({ top: text.scrollHeight });
      else text.scrollTop = text.scrollHeight;
    }
  }

  startLiveTranscript(mixed) {
    this.liveTranscript = "";   // the carry holds the earlier segments; see shownTranscript()
    this.liveTranscriptStatus = "Starting local captions…";
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveTranscriptActive = true;
    this.liveTranscriptPaused = false;
    this.updateLiveTranscriptUI();
    this.liveTranscriptQueue = this.plugin.postLiveTranscript({
      action: "start", session_id: this.recordingId,
    }).then(() => {
      this.liveTranscriptStatus = "Live transcript · draft";
      this.updateLiveTranscriptUI();
    }).catch((e) => {
      console.error("[slim] live transcript start", e);
      this.liveTranscriptActive = false;
      this.liveTranscriptStatus = "Live transcript unavailable — recording continues";
      this.updateLiveTranscriptUI();
    });
    try {
      if (!this.audioCtx || typeof this.audioCtx.createScriptProcessor !== "function") return;
      const source = this.audioCtx.createMediaStreamSource(mixed);
      const processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
      const silent = this.audioCtx.createGain();
      silent.gain.value = 0;
      processor.onaudioprocess = (event) => {
        if (!this.liveTranscriptActive || this.liveTranscriptPaused) return;
        const channel = event.inputBuffer.getChannelData(0);
        const pcm = pcm16k(channel, this.audioCtx && this.audioCtx.sampleRate);
        if (!pcm.length) return;
        this.livePcm.push(pcm);
        this.livePcmBytes += pcm.length;
        if (this.livePcmBytes >= LIVE_FLUSH_BYTES) this.flushLiveTranscript();
      };
      source.connect(processor);
      processor.connect(silent);
      silent.connect(this.audioCtx.destination);
      this.liveProcessor = { source, processor, silent };
    } catch (e) {
      console.error("[slim] live transcript tap", e);
    }
  }

  /* One caption round trip at a time, and the audio waits in the buffer rather than in a
   * queue of requests.
   *
   * ⚠ THIS IS WHY CAPTIONS FELL BEHIND AND NEVER CAUGHT UP. A flush fired every 2 s of audio
   * and chained onto an unbounded promise queue; one round trip slower than 2 s and every
   * later flush waited behind it, so the lag compounded for the length of the recording.
   * Coalescing bounds the lag to a single round trip: a slow machine sends fewer, larger
   * chunks and the captions jump forward instead of drifting away.
   */
  flushLiveTranscript() {
    if (!this.liveTranscriptActive || !this.livePcmBytes) return this.liveTranscriptQueue;
    if (this.liveInFlight) return this.liveTranscriptQueue;   // audio keeps accumulating
    const pcm = Buffer.concat(this.livePcm, this.livePcmBytes);
    this.livePcm = [];
    this.livePcmBytes = 0;
    this.liveInFlight = true;
    this.liveTranscriptQueue = this.liveTranscriptQueue.then(async () => {
      if (!this.liveTranscriptActive) return;
      const result = await this.plugin.postLiveTranscript({
        action: "chunk", session_id: this.recordingId, pcm16: pcm.toString("base64"),
      });
      if (result && typeof result.transcript === "string") {
        this.applyLiveTranscript(result.transcript);
        this.liveTranscriptStatus = "Live transcript · draft";
        this.updateLiveTranscriptUI();
      }
    }).catch(async (e) => {
      // ⚠ NOT EVERY FAILURE IS FATAL. A lost session — a restarted server, an eviction, a
      // recording that never closed its predecessor — used to switch captions off for the rest
      // of the meeting behind one line of grey text. Ask for a new session and carry on; the
      // words already on screen are kept, and the authoritative transcript never depended on
      // this surface anyway.
      console.error("[slim] live transcript chunk", e);
      if (!this.liveTranscriptActive || this.liveRestarting) return;
      this.liveRestarting = true;
      // ⚠ CARRY BEFORE RESTARTING. A new session decodes from zero, and `applyLiveTranscript`
      // replaces — so without this the pane would drop everything said before the restart,
      // which is the "lost the first half of a lecture" failure, not a new one.
      this.carryTranscript();
      try {
        await this.plugin.postLiveTranscript({ action: "start", session_id: this.recordingId });
        this.liveTranscriptStatus = "Live transcript · draft";
      } catch (restartError) {
        console.error("[slim] live transcript restart", restartError);
        this.liveTranscriptActive = false;
      } finally {
        this.liveRestarting = false;
        this.updateLiveTranscriptUI();
      }
    });
    // ⚠ ALWAYS cleared, on every path. Left set by a failure, captions stop for the rest of
    // the recording and the pane simply freezes — no error, no status change.
    this.liveTranscriptQueue = this.liveTranscriptQueue.finally(() => {
      this.liveInFlight = false;
      // A backlog that reached the floor while the request was out goes now — that is the
      // coalescing, and it bounds the lag to one round trip. Anything smaller waits for the
      // tap, which flushes at the same floor.
      //
      // ⚠ THIS LINE WAS THE COIL WHINE (2026-08-28). It used to send WHATEVER had arrived, so
      // after the first 2 s chunk the loop ran at one request per round trip — 2.7 requests/s
      // measured on a real recording, each carrying ~340 ms of audio and each re-encoding 20 s
      // of context: GPU 60 % busy on silence, pulsing at 3 Hz. Faster machine, more requests.
      // At the floor: 0.5 requests/s and 9 %.
      if (this.liveTranscriptActive && this.livePcmBytes >= LIVE_FLUSH_BYTES) {
        this.flushLiveTranscript();
      }
    });
    return this.liveTranscriptQueue;
  }

  async stopLiveTranscript() {
    if (!this.liveTranscriptActive) return;
    await this.flushLiveTranscript();
    this.liveTranscriptActive = false;
    try {
      const result = await this.plugin.postLiveTranscript({
        action: "stop", session_id: this.recordingId,
      });
      if (result && result.transcript) this.liveTranscript = result.transcript;
    } catch (e) {
      console.error("[slim] live transcript stop", e);
    }
  }

  setupMeter(mixed) {
    try {
      const src = this.audioCtx.createMediaStreamSource(mixed);
      const analyser = this.audioCtx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        if (!this.audioCtx || this.state !== "recording") return;
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i] - 128));
        const bar = this.contentEl.querySelector(".slim-level-fill");
        if (bar) bar.style.width = `${Math.min(100, (peak / 128) * 160)}%`;
        window.requestAnimationFrame(tick);
      };
      window.requestAnimationFrame(tick);
    } catch (e) {
      console.error("[slim] meter", e);   // a missing level bar must never stop a recording
    }
  }

  togglePause() {
    if (!this.recorder) return;
    if (this.recorder.state === "recording") {
      this.recorder.pause();
      this.liveTranscriptPaused = true;
    } else if (this.recorder.state === "paused") {
      this.recorder.resume();
      this.liveTranscriptPaused = false;
    }
    this.render();
  }

  teardownStream() {
    if (this.timer) { window.clearInterval(this.timer); this.timer = null; }
    // EVERY input, not just the mic — a live loopback stream keeps the device busy and the
    // recording indicator on long after the pane is gone.
    this.streams.forEach((s) => s.getTracks().forEach((t) => t.stop()));
    this.streams = [];
    if (this.liveTranscriptActive) {
      this.liveTranscriptActive = false;
      this.plugin.postLiveTranscript({ action: "stop", session_id: this.recordingId })
        .catch((e) => console.debug("[slim] live transcript cleanup", e));
    }
    if (this.liveProcessor) {
      for (const node of Object.values(this.liveProcessor)) {
        try { if (node) node.disconnect(); } catch (e) { /* already disconnected */ }
      }
      this.liveProcessor = null;
    }
    if (this.audioCtx) { try { this.audioCtx.close(); } catch (e) { /* already closed */ } }
    this.audioCtx = null;
  }

  async stop() {
    if (!this.recorder) return;
    const blob = await new Promise((resolve) => {
      this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: "audio/webm" }));
      this.recorder.stop();
    });
    await this.stopLiveTranscript();
    this.teardownStream();
    if (this.manager) this.manager.releaseCapture(this);
    // This segment is over. Fold its captions and its seconds into the recording's running
    // totals so a resume continues the same transcript and the same clock.
    this.carryTranscriptAndClock();

    try {
      // The last chunk arrives just before `onstop`, so the queue is drained after it, not
      // before — otherwise the final five seconds are posted without having been written.
      let rel;
      if (this.diskWrite) {
        await this.writeQueue;
        this.flushPending();
        rel = this.audioRel;
        if (this.pending.length) {
          new Notice("SLIM: the tail of this recording could not be written — the file is " +
                     "playable but may be missing its last seconds.", 20000);
        }
      } else {
        rel = await this.writeAudio(blob);
      }

      // Persist both sources before changing the UI or posting either path. In particular, the
      // in-memory fallback has no audio file until `writeAudio` returns, so a draft-save error
      // must happen after that commit point rather than discard the only copy of the recording.
      await this.saveDraft();
      this.freezeEditor(true);
      this.state = "processing";
      this.processingProgress = {
        stage: "queued", label: "Starting the local pipeline",
        detail: "Your recording and notes are already safe", steps: [], metrics: {},
      };
      this.render();
      await this.submitRecording(this.segments.length ? this.segments : [rel]);
    } catch (e) {
      console.error("[slim] stop", e);
      this.errorText = String((e && e.message) || e);
      this.state = "error";
      this.freezeEditor(false);
    }
    this.render();
  }

  /* Post one staged recording and move to its card.
   *
   * Separated from stop() so RETRY runs exactly the same request. ⚠ The server now REFUSES a
   * recording it could not transcribe rather than filing an empty note and returning 200
   * (2026-08-23), which makes a retry meaningful: the audio is still in `_incoming`, and the
   * usual causes — Ollama not up, the model still loading — are fixed in the time it takes to
   * read the error. Before that change there was nothing to retry, because the server had
   * already reported success.
   */
  async submitRecording(rel) {
    const rels = Array.isArray(rel) ? rel : [rel];
    // Resuming a note that already exists is an APPEND, not a second recording: its audio is
    // archived, its staging cleared, and a new note would split one lecture across two files.
    if (this.appendTo) return this.submitAppend(rels);
    const body = { audio: rels.length > 1 ? rels : rels[0],
                   type: this.type, title: this.title.trim() };
    if (this.draftRel) {
      await this.saveDraft();
      body.draft = this.draftRel;
    } else {
      body.notes_md = "";
    }
    // ⚠ ONE JOB PER ATTEMPT. Reusing the recording id meant the progress poll could find the
    // PREVIOUS attempt's record — finished, and marked cancelled — and stop watching at once,
    // so a perfectly healthy run displayed "Cancelled" from beginning to end.
    const base = this.recordingId ||
      nodePath.posix.basename(rels[0], nodePath.posix.extname(rels[0]));
    this.attempt = (this.attempt || 0) + 1;
    const jobId = `${base}-${this.attempt}`;
    this.jobId = jobId;
    if (typeof this.plugin.watchRecording === "function") {
      body.job_id = jobId;
      this.stopProgressWatch = this.plugin.watchRecording(jobId, (progress) => {
        this.processingProgress = progress;
        if (this.state === "processing") this.render();
      });
    }
    try {
      this.result = await this.plugin.postRecording(body);
    } catch (e) {
      // ⚠ NEITHER OF THESE IS AN ERROR SCREEN. A cancel is their instruction; a silent take is
      // an outcome, not a failure. Both leave every file where it was, so both park in
      // `paused` — the ordinary way forward — instead of asking them to choose between
      // deleting and keeping their own audio over a recording that simply had nothing in it.
      if (e && (e.cancelled || e.silent)) {
        if (e.silent) new Notice(`SLIM: ${e.message}`);
        this.freezeEditor(false);
        this.state = "paused";
        this.pausedTranscript = null;      // re-read; a resume adds words
        this.errorText = "";
        this.processingProgress = null;
        this.render();
        return;
      }
      throw e;
    } finally {
      if (this.stopProgressWatch) this.stopProgressWatch();
      this.stopProgressWatch = null;
    }
    this.cardTitle = this.result.title || "";
    // ⚠ INDEX BEFORE OPENING. `openFile` fires `file-open`, and the handler adopts any recorded
    // note it cannot find in the index as a NEW session — a second card on the same block,
    // approved separately, and a "review still pending" notice on close for the one they never
    // saw (2026-09-05).
    const fromDraft = Boolean(this.draftRel);
    this.draftRel = "";
    this.filedNotePath = this.result.note;
    if (this.manager) this.manager.indexSession(this);
    if (fromDraft) await this.loadEditorFile(this.filedNotePath);
    await this.loadNoteParts();
    this.result.notes_md = this.noteParts.notes;
    this.freezeEditor(false);
    this.activeTab = "summary";
    this.state = "card";
  }

  /* Record more into a note that is already filed — from the review card, or from any
   * finished meeting note. The summary is deliberately NOT re-run (their call): it keeps
   * describing what it described until they presse Retry summary. */
  async startAppendRecording() {
    const back = this.state;
    this.appendTo = this.filedNotePath;
    // ⚠ COME BACK WHERE THEY WERE. Resuming from the review card used to file the note on Stop,
    // skipping the approval step they were standing in.
    this.resumeReturnState = this.state === "card" ? "card" : "finished";
    this.recordingId = recordingId();
    this.segments = [];
    // The pane shows the RECORDING's transcript, not this segment's: resuming a filed note
    // with an empty pane made the words already in the note look gone.
    this.liveCarry = (this.noteParts && this.noteParts.transcript) || "";
    this.elapsedCarry = 0;
    this.state = "recording";
    this.render();
    try {
      if (this.manager) this.manager.claimCapture(this);
      await this.start();
    } catch (e) {
      // ⚠ Permission denied or no device leaves a recording UI with no recorder behind it, and
      // an `appendTo` pointing at a note nothing is going to append to. Put them back.
      console.error("[slim] resume recording", e);
      new Notice(`SLIM: could not start recording — ${e.message || e}`);
      this.appendTo = "";
      this.resumeReturnState = "";
      this.segments = [];
      this.liveCarry = "";
      this.state = back;
      if (this.manager) this.manager.releaseCapture(this);
      this.render();
    }
  }

  async submitAppend(rels) {
    const note = this.appendTo;
    this.attempt = (this.attempt || 0) + 1;
    const jobId = `${this.recordingId}-append-${this.attempt}`;
    this.jobId = jobId;
    let stop = null;
    if (typeof this.plugin.watchRecording === "function") {
      stop = this.plugin.watchRecording(jobId, (progress) => {
        this.processingProgress = progress;
        if (this.state === "processing") this.render();
      });
    }
    try {
      await this.plugin.appendRecording({ note, audio: rels, job_id: jobId });
    } catch (e) {
      if (e && (e.cancelled || e.silent)) {
        // A silent resumed segment adds nothing and costs nothing: say so, and put them back
        // where they were rather than on a failure screen. ⚠ Its staged file has no words in it,
        // so it is dropped here — left behind it would sit in _incoming forever, unassociated
        // with any note, and the next Resume resets `segments` out from under it.
        if (e.silent) {
          new Notice(`SLIM: ${e.message}`);
          for (const seg of this.segments || []) {
            await this.trashManagedPath(seg).catch(() => {});
            await this.trashManagedPath(`${seg}.transcript.json`).catch(() => {});
          }
          this.segments = [];
          this.appendTo = "";
        }
        this.freezeEditor(false);
        this.state = e.silent ? (this.resumeReturnState || "finished") : "paused";
        this.pausedTranscript = null;
        this.errorText = "";
        this.render();
        return;
      }
      throw e;
    } finally {
      if (stop) stop();
    }
    // ⚠ The POST is the commit point: the words are in the note. Everything below is opening
    // it again, and a failure there is a display problem, not a reason to re-run an append
    // that already happened.
    this.appendTo = "";
    this.segments = [];
    this.filedNotePath = note;
    try {
      await this.loadEditorFile(note);
      await this.loadNoteParts();
    } catch (e) {
      console.error("[slim] reopen after append", e);
      new Notice(`SLIM: the recording was added to ${note}, but the note could not be opened — `
                 + `${e.message || e}`);
    }
    if (this.result) this.result.notes_md = this.noteParts.notes;
    this.freezeEditor(false);
    this.activeTab = "transcript";
    this.state = this.resumeReturnState || "finished";
    this.resumeReturnState = "";
    this.render();
  }

  async ensureStagingDirs() {
    await this.ensureVaultDirs(STAGING_DIR);
  }

  /* Open the file the chunks will be appended to, and prove it is writable BEFORE recording —
   * discovering a permissions problem five seconds into a meeting is discovering it too late.
   *
   * Node's `fs`, not the Vault API, because Obsidian offers only create and overwrite: there is
   * no appendBinary. `isDesktopOnly: true` in the manifest is precisely the licence for this.
   * The path is still a vault-relative path resolved against the vault's own base, and the
   * containment check keeps it that way.
   */
  openDiskFile() {
    try {
      const fs = require("fs");
      const adapter = this.app.vault.adapter;
      if (typeof adapter.getBasePath !== "function") throw new Error("no base path");
      const base = nodePath.resolve(adapter.getBasePath());
      const abs = nodePath.resolve(base, this.audioRel);
      if (abs !== base && !abs.startsWith(base + nodePath.sep)) {
        throw new Error(`${this.audioRel} resolves outside the vault`);
      }
      fs.mkdirSync(nodePath.dirname(abs), { recursive: true });
      fs.writeFileSync(abs, Buffer.alloc(0));   // create/truncate, so appends start from zero
      this.fs = fs;
      this.audioAbs = abs;
      this.diskWrite = true;
    } catch (e) {
      // ⚠ SAY SO. Falling back silently is exactly how a two-hour loss goes unnoticed: the
      // recording still works, so nothing looks wrong until the day something crashes.
      console.error("[slim] disk flush unavailable", e);
      this.diskWrite = false;
      new Notice(
        "SLIM: cannot write audio to disk while recording — holding it in memory until you " +
        "press Stop. A crash or a quit before then loses the whole recording. " +
        `(${(e && e.message) || e})`, 20000);
    }
  }

  onChunk(blob) {
    if (!this.diskWrite) { this.chunks.push(blob); return; }
    // Serialized: `Blob.arrayBuffer()` is async and the webm's clusters must land in the order
    // MediaRecorder produced them, or the container is garbage.
    // The `.catch` is not decoration: a rejected queue swallows every LATER chunk silently,
    // because `.then` on a rejected promise never runs.
    this.writeQueue = this.writeQueue
      .then(() => this.appendChunk(blob))
      .catch((e) => { console.error("[slim] write queue", e); });
  }

  async appendChunk(blob) {
    try {
      this.pending.push(Buffer.from(await blob.arrayBuffer()));
    } catch (e) {
      console.error("[slim] chunk read", e);
      return;
    }
    this.flushPending();
  }

  /* Append everything the file is still owed. `pending` normally holds exactly one buffer and
   * empties immediately; it only grows when a write fails, and the next chunk then retries the
   * lot in order. That keeps a transient failure (a full disk, an iCloud stall) from tearing a
   * hole in the middle of the container. */
  flushPending() {
    if (!this.pending.length) return;
    try {
      this.fs.appendFileSync(this.audioAbs, this.pending.length === 1
        ? this.pending[0] : Buffer.concat(this.pending));
      this.pending = [];
    } catch (e) {
      console.error("[slim] append", e);
      if (!this.warnedWrite) {
        this.warnedWrite = true;
        new Notice("SLIM: a chunk of audio did not reach disk. Still recording, and the next " +
                   "chunk retries it — but check free space.", 15000);
      }
    }
  }

  async writeAudio(blob) {
    // The in-memory fallback, used only when `openDiskFile` failed. Written by Obsidian itself;
    // the audio never travels over HTTP (the server caps bodies at 1 MB and a meeting is tens
    // of megabytes). From the moment this returns, the recording exists on disk.
    await this.ensureStagingDirs();
    await this.app.vault.createBinary(this.audioRel, await blob.arrayBuffer());
    return this.audioRel;
  }

  async applyCard() {
    // Retitling renames the file, so the server's response carries the new path — keep it, or
    // "Open note" and any later apply would point at a file that no longer exists.
    const c = this.result.card;
    this.filingFrom = this.filedNotePath;
    let moved;
    try {
      moved = await this.plugin.applyCard({
        note: this.filedNotePath,
        dest_dir: c.dest_dir,
        type_tag: c.type_tag,
        topics: c.topics || [],
        summary_md: this.result.summary || "",
        notes_md: this.result.notes_md == null ? this.noteParts.notes : this.result.notes_md,
        title: this.cardTitle || "",
      });
    } finally {
      this.filingFrom = "";
    }
    if (moved && moved.note) {
      this.filedNotePath = moved.note;
      this.result.note = moved.note;
      if (this.manager) this.manager.indexSession(this);
      // A move is a RENAME, and Obsidian renames in place: the leaf keeps its file object and
      // the object's path changes, so the note is already on screen. Reopening it reloaded an
      // editor for nothing — and it was the one step approve-with-edits took that
      // approve-without did not, when "Cannot set properties of null" surfaced after changing
      // the suggested location (2026-08-28).
      const view = this.editorLeaf && this.editorLeaf.view;
      const shown = view instanceof MarkdownView && view.file && view.file.path === moved.note;
      if (!shown) {
        // The server has filed the note by now; that is the fact. A pane that cannot be
        // refreshed is a console matter — reported as a failed filing, it re-enabled Approve,
        // left the session at `card`, and the next leaf change announced a pending review
        // for a note that was done.
        try { await this.loadEditorFile(this.filedNotePath); }
        catch (e) { console.error("[slim] refresh after filing", e); }
      }
    }
    this.result.title = this.cardTitle || this.result.title;
  }

  renderSavedLocation(root) {
    if (!this.filedNotePath) return;
    const row = root.createDiv({ cls: "slim-saved-location" });
    const copy = row.createDiv({ cls: "slim-saved-location-copy" });
    copy.createSpan({ cls: "slim-saved-location-label", text: "Saved in the vault" });
    copy.createSpan({ cls: "slim-saved-location-path", text: this.filedNotePath });
    const reveal = row.createEl("button", { text: "Reveal" });
    reveal.addEventListener("click", () => {
      if (typeof this.plugin.revealMeetingPath !== "function") return;
      this.plugin.revealMeetingPath(this.filedNotePath)
        .catch((e) => new Notice(`SLIM: could not reveal the note — ${e.message || e}`));
    });
  }

  // ---- rendering -----------------------------------------------------------------------

  render() {
    const root = this.contentEl;
    if (!root) return;
    root.empty();
    root.addClass("slim-recorder");
    if (typeof root.setAttr === "function") root.setAttr("data-state", this.state);
    if (this.state === "idle") this.renderIdle(root);
    else if (this.state === "recording") this.renderRecording(root);
    else if (this.state === "processing") this.renderProcessing(root);
    else if (this.state === "paused") this.renderPaused(root);
    else if (this.state === "card") this.renderCard(root);
    else if (this.state === "finished") this.renderFinished(root);
    else this.renderError(root);
  }

  /* Their notes, written back on their own. NOT `applyCard`: apply rewrites frontmatter, drops
   * `filed_by` and can MOVE the note. Typing is not a filing decision. */
  async saveNotes(baseline) {
    const value = (this.result && this.result.notes_md) || "";
    if (!this.filedNotePath || value === baseline.text) return;   // every blur is not a write
    try {
      await this.plugin.saveNotes({ note: this.filedNotePath, notes_md: value });
      baseline.text = value;
      this.noteParts.notes = value;
    } catch (e) {
      console.error("[slim] save notes", e);
      new Notice(`SLIM: could not save your notes — ${e.message || e}`);
    }
  }

  /* Retry the summary against the transcript the note already holds — Notion's button, and
   * the reason they asked for it: iterate on the summarizer without re-recording a lecture.
   * The instructions field is the SLIM-safe half of Notion's: nothing tells the model what
   * KIND of recording this is, they do, in their own words, for this one retry. */
  renderRetryRow(root) {
    const row = root.createDiv({ cls: "slim-retry" });
    const field = row.createDiv({ cls: "slim-field slim-retry-field" });
    field.createEl("label", { text: "Instructions (optional)" });
    const input = field.createEl("input", { type: "text", cls: "slim-retry-instructions" });
    input.value = this.retryInstructions || "";
    input.setAttr("placeholder", "organise by concept, keep the formulas…");
    input.addEventListener("input", (e) => { this.retryInstructions = e.target.value; });

    const button = row.createEl("button", { text: "Retry summary" });
    button.disabled = !!this.retrying;
    button.addEventListener("click", () => this.retrySummary());
    // Always present, even empty: the progress ticks update it in place rather than
    // re-rendering the card, so it has to be there before the first tick arrives.
    row.createSpan({ cls: "slim-retry-status", text: this.retryStatus || "" });
    return row;
  }

  updateRetryStatusUI() {
    if (!this.contentEl || typeof this.contentEl.querySelector !== "function") return;
    const el = this.contentEl.querySelector(".slim-retry-status");
    if (el && typeof el.setText === "function") el.setText(this.retryStatus || "");
  }

  async retrySummary() {
    if (this.retrying || !this.filedNotePath) return;
    this.retrying = true;
    this.retryStartedAt = Date.now();
    this.retryStatus = "Re-reading the transcript… 0s";
    this.render();
    const jobId = `retry-${this.recordingId || "note"}-${this.retryCount = (this.retryCount || 0) + 1}`;
    let stop = null;
    if (typeof this.plugin.watchRecording === "function") {
      stop = this.plugin.watchRecording(jobId, (progress) => {
        // The LABEL, not the detail: while the model streams, `detail` is the fixed string
        // "Live local-model activity", which says nothing. The seconds are what make a
        // minute-long wait read as progress rather than a stall.
        const elapsed = Math.round((Date.now() - this.retryStartedAt) / 1000);
        const label = progress.label || "Rewriting the summary";
        this.retryStatus = `${label}… ${elapsed}s`;
        // ⚠ IN PLACE, NEVER render(). This fires every 600 ms while the state is card or
        // finished — the states whose DOM holds the live notes editor, the summary editor and
        // the instructions field. A full rebuild every tick tore them out from under the caret
        // a few hundred times per retry.
        this.updateRetryStatusUI();
      });
    }
    try {
      const target = this.filedNotePath;
      const out = await this.plugin.retrySummary({
        note: target,
        instructions: (this.retryInstructions || "").trim(),
        job_id: jobId,
      });
      // ⚠ They may have clicked into another note during the minute this took. The note on disk
      // is correct either way; what must not happen is A's summary being painted into B's view.
      if (this.filedNotePath !== target) return;
      this.result.summary = out.summary;
      this.noteParts.summary = out.summary;
      this.activeTab = "summary";
    } catch (e) {
      // ⚠ The old summary stays. A retry that blanks a good summary because the model died is
      // the recorder reporting a loss as progress, which is its worst failure mode.
      console.error("[slim] retry summary", e);
      new Notice(`SLIM: the summary was not rewritten — ${e.message || e}`);
    } finally {
      if (stop) stop();
      this.retrying = false;
      this.retryStatus = "";
      this.render();
    }
  }

  async cancelProcessing() {
    const jobId = this.jobId || this.recordingId;
    if (!jobId || typeof this.plugin.cancelRecording !== "function") return;
    try {
      const out = await this.plugin.cancelRecording({ job_id: jobId });
      // ⚠ The click can beat the POST that registers the job, and the server then has nothing
      // to flag. Announcing success there is a lie that ends with a filed note.
      if (out && out.cancelling === false) {
        new Notice("SLIM: could not stop it — the job had not started yet. Try Cancel again in " +
                   "a moment.");
        return;
      }
      // Cancelling is cooperative: the summary stream stops mid-generation, but transcription
      // and the filing card are blocking calls that finish their current step first.
      new Notice("SLIM: stopping after the current step. Nothing is written or deleted.");
    } catch (e) {
      console.error("[slim] cancel", e);
      new Notice(`SLIM: could not cancel — ${e.message || e}`);
    }
  }

  async resumeRecording() {
    try {
      if (this.manager) this.manager.claimCapture(this);
      this.state = "recording";
      this.render();
      await this.start();
    } catch (e) {
      if (this.manager) this.manager.releaseCapture(this);
      console.error("[slim] resume", e);
      this.errorText = String((e && e.message) || e);
      this.state = "error";
      this.render();
    }
  }

  renderPaused(root) {
    const n = (this.segments || []).length;
    this.renderObjectHeader(root, "SLIM meeting note", this.title.trim() || "Recording paused",
                            "Paused");
    // Its own padded body: the header pads itself, so content hung directly off the root sat
    // flush against the left edge.
    const body = root.createDiv({ cls: "slim-object-body slim-paused" });
    body.createEl("p", { cls: "slim-hint",
      text: `${n} segment${n === 1 ? "" : "s"} saved, and your notes with them. Carry on ` +
            "recording into this same note, or finish it now." });

    // "1 segment saved" is a claim they cannot check. Show the words.
    const pane = body.createDiv({ cls: "slim-live-transcript" });
    const head = pane.createDiv({ cls: "slim-live-transcript-head" });
    head.createSpan({ cls: "slim-object-section-label", text: "Transcript so far" });
    const status = head.createSpan({ cls: "slim-live-transcript-status", text: "Reading…" });
    const text = pane.createEl("p", { cls: "slim-live-transcript-text",
                                      text: this.pausedTranscript || "Reading the transcript…" });
    if (this.pausedTranscript == null && typeof this.plugin.recordTranscript === "function") {
      this.plugin.recordTranscript({ audio: this.segments || [] }).then((out) => {
        this.pausedTranscript = out.text || "";
        status.setText(out.pending
          ? `${out.words} words · ${out.pending} segment${out.pending === 1 ? "" : "s"} not transcribed yet`
          : `${out.words} words`);
        text.setText(this.pausedTranscript || "Nothing was transcribed before you cancelled.");
      }).catch((e) => {
        console.error("[slim] paused transcript", e);
        status.setText("could not read it");
        text.setText("The audio is safe; the transcript is not available yet.");
      });
    } else {
      status.setText("");
      text.setText(this.pausedTranscript || "Nothing was transcribed before you cancelled.");
    }

    const row = body.createDiv({ cls: "slim-row" });
    const resume = row.createEl("button", { cls: "mod-cta", text: "Resume recording" });
    resume.addEventListener("click", () => this.resumeRecording());
    // Always offered: a cancel that left them with no way forward would be a worse trap than
    // the wait they cancelled.
    const finish = row.createEl("button", { text: "Finish" });
    finish.addEventListener("click", async () => {
      this.freezeEditor(true);
      this.state = "processing";
      this.processingProgress = {
        stage: "queued", label: "Starting the local pipeline",
        detail: "Your recording and notes are already safe", steps: [], metrics: {},
      };
      this.render();
      try {
        await this.submitRecording(this.segments);
      } catch (e) {
        console.error("[slim] finish", e);
        this.errorText = String((e && e.message) || e);
        this.state = "error";
        this.freezeEditor(false);
      }
      this.render();
    });
  }

  renderObjectHeader(root, eyebrow, title, status) {
    const head = root.createDiv({ cls: "slim-object-head" });
    const copy = head.createDiv({ cls: "slim-object-head-copy" });
    copy.createDiv({ cls: "slim-object-eyebrow", text: eyebrow });
    copy.createEl("h3", { text: title });
    const badge = head.createDiv({ cls: "slim-object-status" });
    badge.createSpan({ cls: "slim-object-status-dot" });
    badge.createSpan({ text: status });
    return head;
  }

  /* The three tabs, drawn by the shared `renderMeetingTabs` reader. What the recorder adds
   * over a passively rendered block is the editing: a summary they can click into, and a notes
   * editor. `editableSummary` gates only the DERIVED half — their notes are editable in every
   * state, because the meeting block hides them from Obsidian's own editor, so read-only here
   * meant not editable anywhere and reaching them meant reopening the review.
   */
  renderSource(root, editableSummary) {
    const parts = {
      summary: this.result ? (this.result.summary || "") : (this.noteParts.summary || ""),
      notes: this.result && this.result.notes_md != null
        ? this.result.notes_md : this.noteParts.notes,
      transcript: this.noteParts.transcript,
    };
    return renderMeetingTabs(
      this.app, root, parts, this.filedNotePath || this.draftRel || "", {
        editable: true,
        component: this.plugin,
        activeTab: this.activeTab,
        onTab: (value) => {
          this.summaryEditing = false;
          this.activeTab = value;
          this.render();
        },
        summaryPane: (body) => this.renderSummaryPane(body, parts.summary, editableSummary),
        notesPane: (body) => this.renderNotesPane(body, parts.notes),
      });
  }

  renderSummaryPane(generated, summary, editableSummary) {
    if (editableSummary && this.summaryEditing) {
      const edit = generated.createDiv({ cls: "slim-summary-editing" });
      const toolbar = edit.createDiv({ cls: "slim-summary-editing-head" });
      toolbar.createSpan({ text: "Live Markdown" });
      const done = toolbar.createEl("button", { text: "Done" });
      const live = this.renderLiveMarkdownEditor(edit, summary, (value) => {
        this.result.summary = value;
        this.edited = true;
      }, "slim-summary slim-summary-editor");
      const finish = () => {
        if (!this.summaryEditing) return;
        this.summaryEditing = false;
        this.render();
      };
      done.addEventListener("click", finish);
      live.addEventListener("keydown", (e) => {
        if (e.key === "Escape") finish();
      });
      // Defer until the click that moved focus has reached its destination; an immediate
      // rerender on mousedown would swallow the user's intended filing/tab click.
      live.addEventListener("blur", () => {
        if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
          window.setTimeout(finish, 0);
        } else finish();
      });
      if (typeof window !== "undefined" && typeof window.setTimeout === "function") {
        window.setTimeout(() => { if (typeof live.focus === "function") live.focus(); }, 0);
      }
      return;
    }
    const preview = generated.createDiv({
      cls: `slim-summary-preview${editableSummary ? " is-editable" : ""}`,
      attr: editableSummary ? { role: "button", tabindex: "0", "aria-label": "Edit AI summary" } : {},
    });
    this.renderMarkdown(preview.createDiv({ cls: "slim-rendered-markdown" }), summary);
    if (editableSummary) {
      preview.createSpan({ cls: "slim-summary-edit-cue", text: "Edit" });
      const begin = (e) => {
        if (e && e.target && typeof e.target.closest === "function" && e.target.closest("a")) return;
        this.summaryEditing = true;
        this.render();
      };
      preview.addEventListener("click", begin);
      preview.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); begin(e); }
      });
    }
  }

  // ⚠ SAVED ON BLUR, NOT ON EVERY KEYSTROKE, and that is a constraint rather than a
  // preference: their notes live INSIDE the meeting block, so writing them re-renders the very
  // object they are typing into. On blur the caret has already left, so nothing is lost.
  renderNotesPane(parent, notes) {
    // A BOX, not a value: `saveNotes` moves it on success. Captured by value, every later
    // click away re-posted the same notes — a full note rewrite and an index pass each time.
    const baseline = { text: notes || "" };
    const editor = this.renderLiveMarkdownEditor(parent, baseline.text, (value) => {
      this.result.notes_md = value;
      this.edited = true;
    }, "slim-notes-editor is-full");
    editor.setAttr("data-placeholder", "Add the notes you want to keep with this recording…");
    // In EVERY state that has a filed note behind it. Review was the gap: `Approve & finish`
    // was the only writer there, so typing in review and walking away lost the keystrokes —
    // and review is the state they are most likely to walk away from.
    if (this.filedNotePath) {
      editor.addEventListener("blur", () => this.saveNotes(baseline));
    }
    return editor;
  }

  renderProperties(root) {
    const c = this.result && this.result.card;
    if (!c) return;
    const props = root.createDiv({ cls: "slim-object-properties" });
    const property = (label, value) => {
      const p = props.createDiv({ cls: "slim-object-property" });
      p.createSpan({ cls: "slim-object-property-label", text: label });
      p.createSpan({ cls: "slim-object-property-value", text: value || "—" });
    };
    property("Location", c.dest_dir);
    property("Type", TYPES.find(([value]) => value === c.type_tag)?.[1] || c.type_tag);
    property("Topics", (c.topics || []).join(", "));
  }

  renderIdle(root) {
    this.renderObjectHeader(root, "SLIM meeting note", "Set up this recording", "Saved locally");
    const setup = root.createDiv({ cls: "slim-object-setup" });

    const t = setup.createDiv({ cls: "slim-field" });
    t.createEl("label", { text: "Title" });
    const titleInput = t.createEl("input", { type: "text", placeholder: "What is this?" });
    titleInput.value = this.title;
    titleInput.addEventListener("input", (e) => { this.title = e.target.value; });

    const ty = setup.createDiv({ cls: "slim-field" });
    ty.createEl("label", { text: "Type" });
    const sel = ty.createEl("select");
    TYPES.forEach(([value, label]) => {
      const o = sel.createEl("option", { text: label });
      o.value = value;
      if (value === this.type) o.selected = true;
    });
    sel.addEventListener("change", (e) => {
      this.type = e.target.value;
      this.plugin.settings.lastType = this.type;
      this.plugin.saveSettings();
    });
    ty.createEl("p", { cls: "slim-hint",
      text: "Chosen before you speak, so nothing has to guess it later." });

    const v = setup.createDiv({ cls: "slim-field slim-toggle" });
    const box = v.createEl("input", { type: "checkbox" });
    box.checked = this.recordVoice;
    v.createEl("label", { text: "Record my voice" });
    box.addEventListener("change", (e) => {
      this.recordVoice = e.target.checked;
      this.plugin.settings.recordVoice = this.recordVoice;
      this.plugin.saveSettings();
    });
    v.createEl("p", { cls: "slim-hint",
      text: "What the computer is playing is always captured — no device setup, whatever your "
          + "output is set to. Turn this off for a lecture you are only watching." });

    const btn = setup.createEl("button", { cls: "mod-cta slim-big", text: "Start recording" });
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try { await this.start(); }
      catch (e) {
        console.error("[slim] start", e);
        new Notice(`SLIM: could not start recording — ${e.message || e}`);
        btn.disabled = false;
      }
    });
  }

  renderRecording(root) {
    const paused = this.recorder && this.recorder.state === "paused";
    this.renderObjectHeader(root, "SLIM meeting note", this.title.trim() || "Untitled recording",
                            paused ? "Paused" : "Recording");
    const console = root.createDiv({ cls: "slim-recording-console" });
    const toolbar = console.createDiv({ cls: "slim-recording-toolbar" });
    const head = toolbar.createDiv({ cls: "slim-rec-head" });
    head.createSpan({ cls: `slim-dot ${paused ? "is-paused" : ""}` });
    head.createSpan({ cls: "slim-timer", text: hhmmss(this.elapsed) });
    head.createSpan({ cls: "slim-state", text: paused ? "Paused" : "Live" });
    const row = toolbar.createDiv({ cls: "slim-row slim-recording-actions" });
    const p = row.createEl("button", { text: paused ? "Resume" : "Pause" });
    p.addEventListener("click", () => this.togglePause());
    const s = row.createEl("button", { cls: "mod-cta", text: "Stop" });
    s.addEventListener("click", async () => {
      s.disabled = true; p.disabled = true;
      await this.stop();
    });
    const level = console.createDiv({ cls: "slim-level" });
    level.createDiv({ cls: "slim-level-fill" });
    const transcript = console.createDiv({ cls: "slim-live-transcript" });
    const transcriptHead = transcript.createDiv({ cls: "slim-live-transcript-head" });
    transcriptHead.createSpan({ cls: "slim-object-section-label", text: "Live transcript" });
    transcriptHead.createSpan({ cls: "slim-live-transcript-status",
      text: this.liveStatusText() });
    transcript.createEl("p", { cls: "slim-live-transcript-text",
      text: this.shownTranscript() || "Listening for speech…" });
    const bridge = console.createDiv({ cls: "slim-editor-bridge" });
    bridge.createDiv({ cls: "slim-object-section-label", text: "Your notes" });
    bridge.createEl("p", {
      text: "Keep writing in the native Markdown editor below. Every keystroke is already saved to this recording.",
    });
  }

  renderProcessing(root) {
    this.renderObjectHeader(root, "SLIM meeting note", this.title.trim() || "Untitled recording",
                            "Processing locally");
    const progress = this.processingProgress || {};
    const w = root.createDiv({ cls: "slim-processing" });
    const current = w.createDiv({ cls: "slim-processing-current" });
    current.createDiv({ cls: "slim-spinner" });
    const copy = current.createDiv();
    copy.createEl("h3", { text: progress.label || "Working on your note…" });
    copy.createEl("p", { cls: "slim-hint",
      text: progress.detail || "Your recording and notes are safe while SLIM finishes locally." });
    // Stop was hit by accident, 40 minutes into a lecture. Cancelling parks the session with
    // every segment and their notes intact instead of making them wait out the whole pipeline.
    const cancel = current.createEl("button", { cls: "slim-cancel", text: "Cancel" });
    cancel.addEventListener("click", () => this.cancelProcessing());

    const metrics = progress.metrics || {};
    const facts = [];
    if (metrics.transcript_words != null) facts.push(`${metrics.transcript_words} words`);
    if (metrics.transcription_seconds != null) facts.push(`transcribed in ${Number(metrics.transcription_seconds).toFixed(1)}s`);
    if (metrics.summary_output_tokens != null) facts.push(`${metrics.summary_output_tokens} summary tokens`);
    if (metrics.summary_decode_tok_s != null) facts.push(`${metrics.summary_decode_tok_s} tok/s`);

    const activity = w.createEl("details", { cls: "slim-thinking" });
    const activitySummary = activity.createEl("summary");
    activitySummary.createSpan({ cls: "slim-thinking-spark" });
    activitySummary.createSpan({ text: metrics.model_activity_events ? "Thinking locally" : "What SLIM is doing" });
    activitySummary.createSpan({ cls: "slim-thinking-stage", text: progress.label || "Starting" });
    activity.createEl("p", { cls: "slim-hint",
      text: metrics.model_activity_events
        ? "The local model is working. Its private reasoning is not shown."
        : "Stage and performance updates appear here as the local pipeline runs." });
    if (facts.length) activity.createDiv({ cls: "slim-thinking-meta", text: facts.join(" · ") });
  }

  renderError(root) {
    this.renderObjectHeader(root, "SLIM meeting note",
                            this.title.trim() || this.recordingId || "Untitled recording",
                            "Processing stopped · sources preserved");
    root.createEl("h3", { text: "That did not work" });
    root.createEl("p", { text: this.errorText || "Unknown error." });
    root.createEl("p", { cls: "slim-hint",
      text: "The audio is at " + ((this.segments || []).join(", ") || this.audioRel || STAGING_DIR) +
            (this.draftRel ? " and your notes are at " + this.draftRel : "") +
            ". Nothing is lost. " +
            "If the server is the problem, check the plugin's repo and uv settings — it is " +
            "started from here, on port " + this.plugin.settings.port + "." });
    const row = root.createDiv({ cls: "slim-row slim-error-actions" });
    const again = row.createEl("button", { cls: "mod-cta", text: "Try again" });
    again.addEventListener("click", async () => {
      // ⚠ EVERY segment, not the one that happens to be current. Posting `this.audioRel` would
      // re-file the last segment of a resumed recording and silently drop the earlier ones,
      // under a screen that says nothing is lost.
      const all = (this.segments || []).length ? this.segments
                : (this.audioRel ? [this.audioRel] : []);
      if (!all.length) return;
      try {
        await this.saveDraft();
        this.freezeEditor(true);
        this.state = "processing";
        this.processingProgress = {
          stage: "queued", label: "Retrying the local pipeline",
          detail: "Using the same saved audio and notes", steps: [], metrics: {},
        };
        this.render();
        await this.submitRecording(all);
      } catch (e) {
        console.error("[slim] retry", e);
        this.errorText = String((e && e.message) || e);
        this.state = "error";
        this.freezeEditor(false);
      }
      this.render();
    });
    const discard = row.createEl("button", { cls: "mod-warning", text: "Delete & start over" });
    discard.addEventListener("click", async () => {
      again.disabled = true; discard.disabled = true;
      try { await this.discardSession(true); }
      catch (e) {
        console.error("[slim] discard", e);
        new Notice(`SLIM: could not discard the recording — ${e.message || e}`);
        again.disabled = false; discard.disabled = false;
      }
    });
    const close = row.createEl("button", { text: "Keep files & close" });
    close.addEventListener("click", () => this.closeSession());
  }

  renderCard(root) {
    const c = this.result.card;
    this.renderObjectHeader(root, "AI review", this.result.title || "Review this meeting note",
                            "Ready for approval");
    this.renderSavedLocation(root);
    this.renderProperties(root);
    this.renderSource(root, true);
    this.renderRetryRow(root);
    const moreRow = root.createDiv({ cls: "slim-row slim-resume-row" });
    const moreBtn = moreRow.createEl("button", { text: "Resume recording" });
    moreBtn.addEventListener("click", () => this.startAppendRecording());
    const details = root.createEl("details", { cls: "slim-review-details" });
    details.createEl("summary", { text: "Review filing details" });
    root = details.createDiv({ cls: "slim-review-controls" });
    root.createEl("p", { cls: "slim-hint",
      text: "The note is already saved. Adjust any suggestion, then approve the final result." });

    // The title the server settled on — their if they typed one, otherwise written from the
    // transcript. Editable, because a generated title is a suggestion like everything else
    // here, and this is the one field that also names the FILE.
    if (this.cardTitle == null) this.cardTitle = this.result.title || "";
    const ti = root.createDiv({ cls: "slim-field" });
    ti.createEl("label", { text: "Title" });
    const tin = ti.createEl("input", { type: "text" });
    tin.value = this.cardTitle;
    tin.addEventListener("input", (e) => {
      this.cardTitle = e.target.value;
      this.edited = true;
    });

    // --- destination. EVERY segment is a control: confidence sets the default, never a lock.
    const dest = root.createDiv({ cls: "slim-field" });
    dest.createEl("label", { text: "Filing to" });
    const pathRow = dest.createDiv({ cls: "slim-path" });
    const renderPath = () => {
      pathRow.empty();
      const segs = c.dest_dir.split("/");
      segs.forEach((seg, i) => {
        if (i) pathRow.createSpan({ cls: "slim-sep", text: "/" });
        const cls = i >= (c.confident_depth || 0) ? "slim-seg slim-seg-open" : "slim-seg";
        pathRow.createSpan({ cls, text: seg });
      });
    };
    renderPath();

    const det = dest.createEl("details", { cls: "slim-tree-wrap" });
    det.createEl("summary", { text: "browse the vault" });
    const tree = det.createDiv({ cls: "slim-tree" });

    // The tree is the WHOLE vault now, so it has to collapse. A flat list was tolerable only
    // while a depth cap was hiding most of it — which was the bug, not the reason it was flat.
    // Everything starts collapsed except the ancestors of the suggested destination, so the
    // first thing they see is where SLIM wants to put it, in context.
    const roots = [];
    const byPath = new Map();
    (c.folders || []).slice().sort((a, b) => a.path.localeCompare(b.path)).forEach((f) => {
      const node = { path: f.path, count: f.count, name: f.path.split("/").pop(), children: [] };
      byPath.set(f.path, node);
      const parentPath = f.path.split("/").slice(0, -1).join("/");
      const parent = parentPath && byPath.get(parentPath);
      (parent ? parent.children : roots).push(node);
    });

    const expanded = new Set();
    (c.dest_dir || "").split("/").reduce((acc, seg) => {
      const p = acc ? `${acc}/${seg}` : seg;
      expanded.add(p);
      return p;
    }, "");

    const select = (path, isNew) => {
      c.dest_dir = path;
      this.edited = true;
      renderPath();
      drawTree();
      if (mi) mi.value = path;
      if (isNew) new Notice(`SLIM: will create ${path}`);
    };

    const drawNode = (node, depth, parentEl) => {
      const row = parentEl.createDiv({
        cls: "slim-tree-row" + (node.path === c.dest_dir ? " is-sel" : ""),
      });
      row.style.paddingLeft = `${6 + depth * 14}px`;
      const hasKids = node.children.length > 0;
      const tw = row.createSpan({ cls: "slim-tw" + (hasKids ? "" : " is-leaf") });
      tw.setText(hasKids ? (expanded.has(node.path) ? "▾" : "▸") : "·");
      if (hasKids) {
        tw.addEventListener("click", (e) => {
          e.stopPropagation();
          if (expanded.has(node.path)) expanded.delete(node.path);
          else expanded.add(node.path);
          drawTree();
        });
      }
      row.createSpan({ cls: "slim-tree-name", text: node.name });
      if (node.isNew) row.createSpan({ cls: "slim-tree-new", text: "new" });
      else row.createSpan({ cls: "slim-tree-count", text: String(node.count) });

      const plus = row.createSpan({ cls: "slim-tree-plus", text: "+" });
      plus.setAttr("aria-label", `New folder in ${node.path}`);
      plus.addEventListener("click", (e) => {
        e.stopPropagation();
        newFolderUnder(node);
      });

      row.addEventListener("click", () => select(node.path, !!node.isNew));

      if (hasKids && expanded.has(node.path)) {
        node.children.forEach((k) => drawNode(k, depth + 1, parentEl));
      }
    };

    const newFolderUnder = (node) => {
      const row = tree.createDiv({ cls: "slim-tree-row is-newrow" });
      row.createSpan({ cls: "slim-tw is-leaf", text: "+" });
      const input = row.createEl("input", { type: "text", cls: "slim-newfolder",
                                            placeholder: `new folder in ${node.path}` });
      input.focus();
      const commit = () => {
        const name = input.value.trim().replace(/^\/+|\/+$/g, "");
        if (!name) { drawTree(); return; }
        const path = `${node.path}/${name}`;
        // Not created on disk here: the apply endpoint makes parents when the note moves, so
        // abandoning the card leaves no empty folder behind.
        const fresh = { path, count: 0, name, children: [], isNew: true };
        node.children.push(fresh);
        byPath.set(path, fresh);
        expanded.add(node.path);
        select(path, true);
      };
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") drawTree();
      });
      input.addEventListener("blur", commit);
    };

    const drawTree = () => {
      tree.empty();
      roots.forEach((r) => drawNode(r, 0, tree));
    };
    drawTree();

    const manual = det.createDiv({ cls: "slim-field" });
    manual.createEl("label", { text: "or type a path" });
    const mi = manual.createEl("input", { type: "text" });
    mi.value = c.dest_dir;
    mi.addEventListener("change", (e) => {
      c.dest_dir = e.target.value.replace(/^\/+|\/+$/g, "");
      this.edited = true;
      renderPath();
      drawTree();
    });
    manual.createEl("p", { cls: "slim-hint",
      text: "A folder that does not exist yet is created when the note is filed." });

    // --- type
    const ty = root.createDiv({ cls: "slim-field" });
    ty.createEl("label", { text: "Type" });
    const tsel = ty.createEl("select");
    TYPES.forEach(([value, label]) => {
      const o = tsel.createEl("option", { text: label });
      o.value = value;
      if (value === c.type_tag) o.selected = true;
    });
    tsel.addEventListener("change", (e) => { c.type_tag = e.target.value; this.edited = true; });

    // --- topics. Creating is visibly heavier than reusing: an existing topic shows how often
    //     it has been used, a brand-new one says so. Informational, never blocking — and it
    //     applies to one THEY type too, since they are the likeliest source of a one-off.
    //
    // ⚠ THIS SAID "Tags" AND WROTE `topics:`, and `tags:` in the note is a DIFFERENT FIELD —
    // it carries the TYPE, for Obsidian's tag pane. So the card's "Tags" and the note's `tags:`
    // named two unrelated things and they had to hold both in their head (2026-08-23). The label
    // now matches the field it edits. The `#` went with it: `topics:` are not Obsidian tags,
    // and a `#` promises a clickable one.
    const tg = root.createDiv({ cls: "slim-field" });
    tg.createEl("label", { text: "Topics" });
    const chips = tg.createDiv({ cls: "slim-chips" });
    const counts = c.tag_counts || {};
    const drawChips = () => {
      chips.empty();
      (c.topics || []).forEach((t, i) => {
        const n = counts[String(t).toLowerCase()] || 0;
        const chip = chips.createSpan({ cls: "slim-chip" + (n ? "" : " is-new") });
        chip.createSpan({ text: String(t) });
        chip.createSpan({ cls: "slim-chip-count", text: n ? String(n) : "new" });
        const x = chip.createSpan({ cls: "slim-chip-x", text: "×" });
        x.addEventListener("click", () => {
          c.topics.splice(i, 1); this.edited = true; drawChips();
        });
      });
      const add = chips.createEl("input", { type: "text", cls: "slim-chip-add",
                                            placeholder: "+ add tag" });
      add.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        const v = add.value.trim().replace(/^#/, "");
        if (!v) return;
        c.topics = c.topics || [];
        c.topics.push(v);
        this.edited = true;
        drawChips();
      });
    };
    drawChips();

    const row = root.createDiv({ cls: "slim-row" });
    const file = row.createEl("button", { cls: "mod-cta", text: "Approve & finish" });
    file.addEventListener("click", async () => {
      file.disabled = true;
      // ⚠ `filing` covers the whole request. A move drops the old file from its leaf, and
      // that layout change ran the pending-review notice while Approve was still out — state
      // `card`, pane empty, the path from before the move (2026-09-09).
      this.filing = true;
      try {
        if (this.edited) await this.applyCard();
        else await this.plugin.completeReview({ note: this.filedNotePath });
        await this.loadNoteParts();
        new Notice("SLIM: meeting note finished.");
        this.state = "finished";
        this.activeTab = "summary";
        this.edited = false;
        this.freezeEditor(false);
        this.render();
      } catch (e) {
        console.error("[slim] file", e);
        new Notice(`SLIM: ${e.message || e}`);
        file.disabled = false;
      } finally {
        this.filing = false;
      }
    });
  }

  renderFinished(root) {
    const title = (this.result && this.result.title) || this.cardTitle || "Meeting note";
    this.renderObjectHeader(root, "SLIM meeting note", title, "Finished");
    this.renderSavedLocation(root);
    this.renderProperties(root);
    this.renderSource(root, false);
    this.renderRetryRow(root);
    const row = root.createDiv({ cls: "slim-row slim-finished-actions" });
    const more = row.createEl("button", { text: "Resume recording" });
    more.addEventListener("click", () => this.startAppendRecording());
    if (this.result && this.result.card) {
      const edit = row.createEl("button", { text: "Edit review details" });
      edit.addEventListener("click", async () => {
        try {
          await this.saveDraft();
          await this.loadNoteParts();
          this.result.summary = this.noteParts.summary;
          this.result.notes_md = this.noteParts.notes;
          this.summaryEditing = false;
          this.state = "card";
          this.render();
        } catch (e) {
          console.error("[slim] reopen review", e);
          new Notice(`SLIM: could not reopen the review — ${e.message || e}`);
        }
      });
    }
    root.createEl("p", { cls: "slim-native-editor-hint",
      text: "This block is part of the note. Keep writing normally before or after it." });
  }
}

class SlimSettingTab extends PluginSettingTab {
  constructor(app, plugin) { super(app, plugin); this.plugin = plugin; }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    new Setting(containerEl)
      .setName("Server port")
      .setDesc("Where `slim chat` listens on 127.0.0.1. A server already on this port is " +
               "reused, never duplicated.")
      .addText((t) => t
        .setValue(String(this.plugin.settings.port))
        .onChange(async (v) => {
          const n = parseInt(v, 10);
          if (!isNaN(n)) { this.plugin.settings.port = n; await this.plugin.saveSettings(); }
        }));

    new Setting(containerEl)
      .setName("SLIM repo")
      .setDesc("The checkout `uv run slim chat` is started from.")
      .addText((t) => t
        .setValue(this.plugin.settings.repoPath)
        .onChange(async (v) => {
          this.plugin.settings.repoPath = v.trim();
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("uv binary")
      .setDesc("Full path — Obsidian launches with a bare PATH, so a bare `uv` will not resolve.")
      .addText((t) => t
        .setValue(this.plugin.settings.uvPath)
        .onChange(async (v) => {
          this.plugin.settings.uvPath = v.trim();
          await this.plugin.saveSettings();
        }));

    containerEl.createEl("p", { cls: "slim-hint",
      text: "Changes take effect the next time the server is started — disable and re-enable " +
            "the plugin to restart it now." });
  }
}

// ===========================================================================================
// Open-note copilot (sidebar). ⚠ Lives in THIS file on purpose: Obsidian evals main.js with a
// shim `require` that serves `obsidian` and CodeMirror and hands everything else to the
// renderer's global require — bound to Obsidian's bundle, not the plugin folder — so a
// relative require of `./copilot` throws before the plugin class exists and takes the
// recorder down with it. `node --test` resolves relative paths and cannot see that. One
// file, no build step.
// ===========================================================================================

const VIEW_TYPE_COPILOT = "slim-copilot";

class SSEParser {
  constructor() { this.buffer = ""; }
  push(chunk) {
    this.buffer += chunk;
    const events = [];
    while (true) {
      const boundary = this.buffer.indexOf("\n\n");
      if (boundary < 0) break;
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      try { events.push({ event, data: JSON.parse(data.join("\n")) }); }
      catch (_error) { events.push({ event: "error", data: { error: "Malformed server event" } }); }
    }
    return events;
  }
}

function groupThreads(threads, activeSourceId) {
  const grouped = new Map();
  for (const thread of threads || []) {
    if (!thread.source_id || !thread.source_path) continue;
    if (!grouped.has(thread.source_id)) grouped.set(thread.source_id, {
      sourceId: thread.source_id, path: thread.source_path, threads: [], expanded: false,
    });
    const group = grouped.get(thread.source_id);
    group.threads.push(thread);
    if (thread.source_path) group.path = thread.source_path;
  }
  const groups = [...grouped.values()];
  for (const group of groups) {
    group.threads.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
    group.updatedAt = group.threads[0]?.updated_at || "";
    group.expanded = group.sourceId === activeSourceId;
  }
  groups.sort((a, b) => {
    if (a.sourceId === activeSourceId) return -1;
    if (b.sourceId === activeSourceId) return 1;
    return b.updatedAt.localeCompare(a.updatedAt);
  });
  return groups;
}

// --- what the sidebar shows instead of paths ------------------------------------------------

function pathStem(path) {
  const name = String(path || "").split("/").pop() || "";
  return name.replace(/\.md$/i, "");
}

// Frontmatter title, then the file name, then the path's last segment. The server already
// sends titles with the source, the citations and the nearby notes; only a thread group
// carries a bare path, frozen when the thread was created, so it may be stale.
function noteTitle(app, path) {
  const file = app?.vault?.getAbstractFileByPath?.(path);
  if (file && file.extension) {
    const title = app.metadataCache?.getFileCache?.(file)?.frontmatter?.title;
    return (typeof title === "string" && title.trim()) || file.basename || pathStem(path);
  }
  return pathStem(path);
}

// One row per NOTE, in order of first appearance: [1]–[7] were one note, seven fragments.
function groupCitations(citations) {
  const groups = new Map();
  for (const citation of citations || []) {
    const key = citation.source_id || citation.path;
    if (!groups.has(key)) {
      groups.set(key, { source_id: citation.source_id, path: citation.path,
                        title: citation.title || pathStem(citation.path), ns: [] });
    }
    groups.get(key).ns.push(citation.n);
  }
  return [...groups.values()];
}

// qwen writes \( \) and \[ \] as often as $ $; Obsidian's renderer only knows the dollar
// forms, and Markdown eats the escaped brackets, so an equation rendered as nothing. Display
// only: the stored answer is what the model wrote. Code spans and fences are left alone.
// Only content that reads as math converts: a TeX command, an operator, or a lone symbol.
// `\[1, 3\]`, `arr\[0\]`, `\[here\](url)` and an empty `\( \)` are Markdown-escaped
// brackets and stay as they are.
function looksLikeMath(inner) {
  const text = inner.trim();
  return text.length > 0 && (/\\[A-Za-z]|[=^_+{}<>]|\d\s*[*/-]\s*\d/.test(text) || /^[A-Za-z]$/.test(text));
}

function normalizeMath(markdown) {
  const parts = String(markdown || "").split(/(```[\s\S]*?```|`[^`\n]*`)/);
  return parts.map((part, index) => index % 2 ? part : part
    .replace(/\\\[([\s\S]*?)\\\](?!\()/g, (match, inner) => looksLikeMath(inner) ? `$$${inner}$$` : match)
    .replace(/\\\((.*?)\\\)(?!\()/g, (match, inner) => looksLikeMath(inner) ? `$${inner}$` : match)).join("");
}

// Live Markdown cadence: a detached render swapped in at most this often. Per-token rendering
// is async and tore scroll and focus (2026-08-26); 350 ms reads as live and costs nothing.
const LIVE_RENDER_MS = 350;

/* ⚠ A thread saved before 2026-09-03 can say "auto" on disk, and the server refuses that word
 * now (threads.save_copilot). Coerce rather than refuse, in the ONE place every save passes
 * through: an old conversation must still open, still rename and still take a new turn. Quick
 * is what Auto resolved to nearly every time, so it is the honest fallback.
 */
function quickOrDeep(value) {
  return value === "deep" ? "deep" : "quick";
}

function metaText(turn) {
  const mode = turn?.mode === "deep" ? "Deep" : "Quick";
  const ms = Number(turn?.timing?.total_ms);
  return Number.isFinite(ms) && ms > 0 ? `${mode} · ${Math.max(1, Math.round(ms / 1000))} s` : mode;
}

const SUGGESTED_PROMPTS = [
  "Explain this note like I'm new to it",
  "Quiz me on this note",
  "What is unclear or missing here?",
];

function streamRequest(port, path, body, handlers, signal) {
  return new Promise((resolve, reject) => {
    const encoded = Buffer.from(JSON.stringify(body));
    const request = http.request({
      hostname: "127.0.0.1", port, path, method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": encoded.length },
    }, (response) => {
      if (response.statusCode !== 200) {
        const parts = [];
        response.on("data", (chunk) => parts.push(chunk));
        response.on("end", () => reject(new Error(
          `${path} returned ${response.statusCode}: ${Buffer.concat(parts).toString("utf8")}`)));
        return;
      }
      const parser = new SSEParser();
      response.setEncoding("utf8");
      response.on("data", (chunk) => {
        try {
          for (const item of parser.push(chunk)) {
            const handler = handlers[item.event];
            if (handler) handler(item.data);
          }
        } catch (error) {
          response.destroy();
          request.destroy();
          reject(error);
        }
      });
      response.on("end", resolve);
      response.on("error", reject);
    });
    request.on("error", reject);
    if (signal) signal.addEventListener("abort", () => {
      const error = new Error("Cancelled");
      error.name = "AbortError";
      request.destroy(error);
    }, { once: true });
    request.end(encoded);
  });
}

function clear(element) {
  if (element.empty) element.empty();
  else while (element.firstChild) element.removeChild(element.firstChild);
}

function create(parent, tag, cls, text) {
  const options = {};
  if (cls) options.cls = cls;
  if (text !== undefined) options.text = text;
  if (parent.createEl) return parent.createEl(tag, options);
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function iconButton(parent, icon, label, cls = "clickable-icon") {
  const button = create(parent, "button", cls);
  button.setAttribute("aria-label", label);
  button.setAttribute("title", label);
  if (typeof setIcon === "function") setIcon(button, icon);
  else button.textContent = label;
  return button;
}

// One line until they type more; capped so a long paste does not push the chat off-screen.
function autoGrow(input, max = 180) {
  if (!input || !input.style) return;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight || 0, max)}px`;
}

function titleFromQuestion(question) {
  const title = question.trim().replace(/\s+/g, " ");
  return title.length > 60 ? `${title.slice(0, 57)}...` : title || "Conversation";
}

function threadId() {
  return `chat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function atBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= 40;
}

function cancelledError() {
  const error = new Error("Cancelled");
  error.name = "AbortError";
  return error;
}

// Mirrors chat.py MAX_HISTORY_TURNS: what the server keeps of a question's history.
const MAX_HISTORY_TURNS = 12;
const MAX_COPILOT_IMAGES = 3;
const MAX_COPILOT_IMAGE_BYTES = 2_000_000;
const MAX_COPILOT_IMAGE_INPUT_BYTES = 20_000_000;
const MAX_COPILOT_IMAGE_EDGE = 1600;
const COPILOT_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);

function imageBytes(data) {
  return Buffer.byteLength(String(data || ""), "base64");
}

function imageName(name, mime) {
  const safe = String(name || "image").replace(/\\/g, "/").split("/").pop().slice(0, 120);
  const ext = mime === "image/png" ? "png" : mime === "image/webp" ? "webp" : "jpg";
  return `${safe.replace(/\.[^.]+$/, "") || "image"}.${ext}`;
}

function loadBrowserImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("This image could not be decoded."));
    image.src = source;
  });
}

async function resizeCopilotImage(file) {
  const mime = String(file?.type || "").toLowerCase();
  if (!COPILOT_IMAGE_TYPES.has(mime)) throw new Error("Use a PNG, JPEG, or WebP image.");
  if (!Number.isFinite(file.size) || file.size <= 0) throw new Error("This image is empty.");
  if (file.size > MAX_COPILOT_IMAGE_INPUT_BYTES) throw new Error("Images must be 20 MB or smaller before resizing.");
  const buffer = Buffer.from(await file.arrayBuffer());
  const original = `data:${mime};base64,${buffer.toString("base64")}`;
  const image = await loadBrowserImage(original);
  if (Math.max(image.naturalWidth, image.naturalHeight) <= MAX_COPILOT_IMAGE_EDGE
      && buffer.length <= MAX_COPILOT_IMAGE_BYTES) {
    return { name: imageName(file.name, mime), mime, data: original.split(",")[1],
             preview: original, bytes: buffer.length };
  }
  let scale = Math.min(1, MAX_COPILOT_IMAGE_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d");
    context.fillStyle = "#fff";              // transparent textbook captures stay legible
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const quality = Math.max(0.65, 0.9 - attempt * 0.07);
    const resized = canvas.toDataURL("image/jpeg", quality);
    const data = resized.split(",")[1];
    const bytes = imageBytes(data);
    if (bytes <= MAX_COPILOT_IMAGE_BYTES) {
      return { name: imageName(file.name, "image/jpeg"), mime: "image/jpeg", data,
               preview: resized, bytes };
    }
    scale *= 0.8;
  }
  throw new Error("The image is still over 2 MB after resizing. Crop it and try again.");
}

// window.prompt is not an option: Electron defines it as a function that THROWS
// ("prompt() is not supported."), so a `!window.prompt` guard passes and the click rejects
// silently. `onSubmit` receives the trimmed title, or null when the modal is dismissed.
class RenameModal extends Modal {
  constructor(app, current, onSubmit) {
    super(app);
    this.current = current;
    this.onSubmit = onSubmit;
    this.done = false;
  }

  finish(value) {
    if (this.done) return;
    this.done = true;
    this.onSubmit(value);
  }

  onOpen() {
    const root = this.contentEl;
    create(root, "h3", "", "Rename chat");
    const input = create(root, "input", "slim-rename-input");
    input.type = "text";
    input.value = this.current;
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); submit(); }
    });
    const row = create(root, "div", "slim-row");
    const cancel = create(row, "button", "", "Cancel");
    cancel.onclick = () => this.close();
    const ok = create(row, "button", "mod-cta", "Rename");
    const submit = () => {
      const title = String(input.value || "").trim();
      if (title) this.finish(title);
      this.close();
    };
    ok.onclick = submit;
    if (typeof input.focus === "function") { input.focus(); input.select?.(); }
  }

  onClose() {
    clear(this.contentEl);
    this.finish(null);
  }
}

class CopilotView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.app = plugin.app;
    this.context = null;
    this.threads = [];
    this.activeThread = null;
    this.turns = [];
    this.mode = "quick";
    this.stage = "";
    this.draft = "";
    this.error = "";
    this.loading = false;
    this.sending = false;
    this.streamingText = "";
    this.abort = null;
    this.requestVersion = 0;
    this.pendingThreadId = null;
    this.deleteArmed = null;
    this.retryQuestion = "";
    this.pendingImages = [];
    this.retryImages = [];
    this.loadingPath = null;
    this.streamingThread = null;
    this.streamingTurns = null;
    this.el = null;                 // live handles from the last full render()
    this.composerFocused = false;
    this.groupOpen = new Map();     // sourceId -> open, where they toggled a group by hand
    this.showThreads = false;       // the history button: the chat list in the conversation's slot
    this.nearbyOpen = false;        // the context chip under the title
    this.liveRenderTimer = null;    // the throttled Markdown render of the streaming answer
    this.liveRenderVersion = 0;
    this.liveRenderedAt = 0;
  }

  imageURL(threadId, imageId) {
    const port = this.plugin.settings?.port || 7546;
    return `http://127.0.0.1:${port}/api/copilot/image?thread=${encodeURIComponent(threadId)}`
      + `&id=${encodeURIComponent(imageId)}`;
  }

  getViewType() { return VIEW_TYPE_COPILOT; }
  getDisplayText() { return "SLIM Copilot"; }
  getIcon() { return "message-square-text"; }

  async onOpen() {
    const file = this.app.workspace.getActiveFile ? this.app.workspace.getActiveFile() : null;
    await this.setActiveFile(file);
  }

  async onClose() { this.cancel(); }

  async setActiveFile(file) {
    const path = file && String(file.extension || "").toLowerCase() === "md" ? file.path : null;
    // Same note, loaded or loading: nothing to do. Focus moving between the editor and this
    // sidebar fires active-leaf-change with the SAME file; re-syncing here cancelled the
    // answer, erased the question, and — with onOpen, the event and activateCopilot all
    // racing — ran ingest_note twice into an IntegrityError. Edits since the open are picked
    // up by send() (refreshNote), not by focus changes.
    if (path && (this.context?.source.path === path || this.loadingPath === path)) return;
    const version = ++this.requestVersion;
    this.cancel();
    this.error = "";
    this.context = null;
    this.activeThread = null;
    this.turns = [];
    this.retryQuestion = "";
    this.pendingImages = [];
    this.retryImages = [];
    if (!file || String(file.extension || "").toLowerCase() !== "md") {
      this.error = "Open a saved Markdown note to use SLIM Copilot.";
      this.render();
      return;
    }
    this.loading = true;
    this.loadingPath = path;
    this.render();
    try {
      if (typeof this.plugin.saveActiveNote === "function") {
        await this.plugin.saveActiveNote(file);
      }
      if (version !== this.requestVersion) return;
      const [context, listing] = await Promise.all([
        this.plugin.postJSON("/api/copilot/context", { path: file.path }),
        this.plugin.getJSON("/api/copilot/threads"),
      ]);
      if (version !== this.requestVersion) return;
      this.context = context;
      this.threads = listing.threads || [];
      const wanted = (this.pendingThreadId
        ? this.threads.find((item) => item.id === this.pendingThreadId
          && item.source_id === context.source.id)
        : null) || this.threads.find((item) => item.source_id === context.source.id);
      if (wanted) {
        this.pendingThreadId = null;
        await this.openThread(wanted.id, version, true);
      }
    } catch (error) {
      if (version === this.requestVersion) this.error = error.message || String(error);
    } finally {
      if (version === this.requestVersion) {
        this.loading = false;
        this.loadingPath = null;
        this.render();
      }
    }
  }

  // The note as it is NOW: save the editor, re-sync the row (ingest_note is hash-keyed, so an
  // unchanged note costs one stat), skip the nearby-notes search. Called by send(), so every
  // question is answered from what they have typed, not from the version indexed at open.
  async refreshNote() {
    if (!this.context) return;
    const path = this.context.source.path;
    const file = this.app.workspace.getActiveFile ? this.app.workspace.getActiveFile() : null;
    if (file && file.path === path && typeof this.plugin.saveActiveNote === "function") {
      await this.plugin.saveActiveNote(file);
    }
    const fresh = await this.plugin.postJSON("/api/copilot/context", { path, related: false });
    this.context = { ...this.context, source: fresh.source };
  }

  newThread() {
    if (!this.context) return;
    this.activeThread = {
      id: threadId(), title: "New chat", source_id: this.context.source.id,
      source_path: this.context.source.path, reasoning_mode: this.mode,
    };
    this.turns = [];
    this.stage = "";
    this.error = "";
    this.retryQuestion = "";
    this.pendingImages = [];
    this.retryImages = [];
    this.showThreads = false;
    this.render();
  }

  async openThread(id, version = this.requestVersion, afterNavigation = false) {
    const data = await this.plugin.getJSON(`/api/copilot/thread?id=${encodeURIComponent(id)}`);
    if (version !== this.requestVersion) return;
    if (!afterNavigation && this.context && data.source_id !== this.context.source.id) {
      this.pendingThreadId = id;
      this.activeThread = null;
      this.turns = [];
      this.openNote(data.source_path);
      return;
    }
    this.activeThread = data;
    this.turns = data.turns || [];
    this.pendingImages = [];
    this.retryImages = [];
    // A legacy "auto" on disk must not reach the mode selector; `saveRecord` normalizes the
    // record itself, so nothing has to correct `data` here.
    this.mode = quickOrDeep(data.reasoning_mode);
    this.showThreads = false;
    this.render();
  }

  async saveRecord(thread, turns) {
    const body = {
      id: thread.id, title: thread.title, turns,
      source_id: thread.source_id, source_path: thread.source_path,
      // ⚠ Normalized HERE, not at the point a thread is opened: `renameThread` fetches a
      // thread it never opened and hands it straight to this method, so a pre-2026-09-03
      // conversation would post "auto" and be refused with nothing catching it.
      reasoning_mode: quickOrDeep(thread.reasoning_mode || this.mode),
    };
    const saved = await this.plugin.postJSON("/api/copilot/thread/save", body);
    Object.assign(thread, saved);
    const index = this.threads.findIndex((item) => item.id === thread.id);
    if (index >= 0) this.threads[index] = { ...thread };
    else this.threads.unshift({ ...thread });
    return saved;
  }

  async addImages(files) {
    if (this.sending) return;
    const images = [...(files || [])].filter((file) => String(file?.type || "").startsWith("image/"));
    if (!images.length) return;
    if (this.pendingImages.length + images.length > MAX_COPILOT_IMAGES) {
      this.error = `Attach at most ${MAX_COPILOT_IMAGES} images to one message.`;
      this.render();
      return;
    }
    try {
      for (const file of images) this.pendingImages.push(await resizeCopilotImage(file));
      this.error = "";
    } catch (error) {
      this.error = error.message || String(error);
    }
    this.render();
  }

  async send(text = this.draft) {
    const outgoingImages = this.pendingImages;
    const question = String(text || "").trim() || (outgoingImages.length ? "Explain this image." : "");
    if (!question || !this.context || this.sending) return;
    if (!this.activeThread) this.newThread();
    // Snapshot the thread that ASKED. openThread/newThread/deleteThread stay clickable while
    // an answer streams, and the answer belongs to this thread whatever is active when it
    // lands — otherwise B was saved with A's answer and A's question was never saved.
    const thread = this.activeThread;
    const turns = this.turns;
    if (!turns.length) thread.title = titleFromQuestion(question);
    const asked = { role: "you", text: question,
      ...(outgoingImages.length ? { attachments: outgoingImages } : {}) };
    turns.push(asked);
    this.draft = "";
    this.pendingImages = [];
    this.showThreads = false;
    this.sending = true;
    this.stage = "Starting";
    this.error = "";
    let streamed = "";
    this.streamingText = "";
    this.streamingThread = thread;
    this.streamingTurns = turns;
    let completed = null;
    const abort = new AbortController();
    this.abort = abort;
    this.render();
    try {
      await this.refreshNote();
      if (abort.signal.aborted) throw cancelledError();
      await this.plugin.streamCopilot({
        thread_id: thread.id, source_id: this.context.source.id, question,
        // Role, text and small image refs only, and only what the server keeps
        // (MAX_HISTORY_TURNS). Slim turns also carry evidence and stats; re-uploading those
        // on every question grew with the square of the thread. Image bytes stay server-side.
        history: turns.slice(0, -1).slice(-MAX_HISTORY_TURNS)
          .map(({ role, text, attachments }) => ({ role, text,
            ...(attachments?.length ? { attachments: attachments.map(
              ({ id, name, mime, bytes }) => ({ id, name, mime, bytes })) } : {}) })),
        // A regenerated question re-sends its stored refs; the server has the bytes.
        images: outgoingImages.map(({ id, name, mime, data, bytes }) =>
          data ? { name, mime, data } : { id, name, mime, bytes }),
        reasoning_mode: this.mode,
      }, {
        stage: (data) => { this.stage = data.stage || "Working"; this.renderStage(); },
        delta: (data) => {
          streamed += data.text || "";
          this.streamingText = streamed;
          this.renderDelta();
        },
        turn: (data) => { completed = data; },
        error: (data) => { throw new Error(data.error || "Copilot failed"); },
      }, abort.signal);
      if (!completed) throw new Error("The server ended without a completed turn.");
      asked.attachments = completed.attachments || [];
      delete completed.attachments;          // refs belong to the user turn, not the answer
      turns.push({ role: "slim", text: completed.answer || streamed, turn: completed });
      this.retryQuestion = "";
      this.retryImages = [];
      thread.reasoning_mode = this.mode;
      await this.saveRecord(thread, turns);
    } catch (error) {
      if (error.name === "AbortError") {
        // Stopped before an answer: the question is theirs — hand it back rather than leave a
        // dangling "you" turn to ride along as history and be persisted with the next answer.
        if (turns.at(-1) === asked) {
          turns.pop(); this.draft = question; this.pendingImages = outgoingImages;
        }
      } else if (turns.at(-1) === asked) {
        // The model never answered: show the error under the question and offer a retry.
        this.error = error.message || String(error);
        this.retryQuestion = question;
        this.retryImages = outgoingImages;
      } else {
        // Answered but not saved (e.g. the thread is over its cap): the answer stays visible
        // with the reason. No retry — it would append a second you/slim pair every click.
        this.error = error.message || String(error);
      }
    } finally {
      this.abort = null;
      this.sending = false;
      this.stopLiveRender();
      this.streamingText = "";
      this.streamingThread = null;
      this.streamingTurns = null;
      this.stage = "";
      this.render();
    }
  }

  cancel() {
    if (this.abort) this.abort.abort();
    this.abort = null;
    this.sending = false;
    this.stopLiveRender();
    this.streamingText = "";
    this.stage = "";
  }

  // Ask the last question again: the pair comes off the thread and the question goes back
  // through send() with the image refs it had, so the answer lands where the first one did.
  async regenerate() {
    if (this.sending || !this.activeThread) return;
    const answer = this.turns.at(-1);
    const asked = this.turns.at(-2);
    if (answer?.role !== "slim" || asked?.role !== "you") return;
    this.turns.splice(-2, 2);
    this.pendingImages = (asked.attachments || []).map((ref) => ({ ...ref }));
    await this.send(asked.text);
  }

  async retry() {
    const question = this.retryQuestion;
    if (!question || this.sending) return;
    const last = this.turns.at(-1);
    if (last?.role === "you" && last.text === question) this.turns.pop();
    this.retryQuestion = "";
    this.pendingImages = this.retryImages;
    this.retryImages = [];
    await this.send(question);
  }

  async deleteThread(id) {
    if (this.deleteArmed !== id) {
      this.deleteArmed = id;
      this.render();
      return;
    }
    // A late answer would save into — and so resurrect — the thread being deleted.
    if (this.sending && this.streamingThread?.id === id) this.cancel();
    await this.plugin.postJSON("/api/copilot/thread/delete", { id });
    this.threads = this.threads.filter((item) => item.id !== id);
    if (this.activeThread?.id === id) this.newThread();
    this.deleteArmed = null;
    this.render();
  }

  async renameThread(id) {
    let thread = this.activeThread?.id === id ? this.activeThread : null;
    let turns = this.activeThread?.id === id ? this.turns : null;
    if (!thread) {
      thread = await this.plugin.getJSON(`/api/copilot/thread?id=${encodeURIComponent(id)}`);
      turns = thread.turns || [];
    }
    const title = await this.promptTitle(thread.title);
    if (!title) return;
    thread.title = title.slice(0, 120);
    if (this.activeThread?.id === id) this.activeThread.title = thread.title;
    await this.saveRecord(thread, turns);
    this.render();
  }

  // Guarded like the recorder's "Open note" button: openLinkText CREATES a missing target,
  // and citation / thread paths are only as fresh as the last ingest — a note dragged since
  // then would become an empty phantom, which the next active-leaf-change indexes and embeds.
  promptTitle(current) {
    return new Promise((resolve) => new RenameModal(this.app, current, resolve).open());
  }

  async openNote(path) {
    const file = this.app.vault && typeof this.app.vault.getAbstractFileByPath === "function"
      ? this.app.vault.getAbstractFileByPath(path) : null;
    if (!file) {
      this.error = `Note not found: ${path} — it may have moved since the last index.`;
      this.render();
      return;
    }
    await this.app.workspace.getLeaf(false).openFile(file);
  }

  // Per-token updates touch two live nodes. A full render() per delta tore the whole view
  // down ~40 times a second: the transcript's scroll position reset to the top on every
  // token, the composer lost focus per keystroke, and a hand-expanded group snapped shut.
  renderDelta() {
    if (!this.el) { this.render(); return; }
    if (!this.el.liveText) return;          // the chat list is showing; the next render catches up
    const transcript = this.el.transcript;
    const follow = atBottom(transcript);      // decided BEFORE the text grows the pane
    this.el.liveText.textContent = this.streamingText;
    if (follow) transcript.scrollTop = transcript.scrollHeight;
    this.scheduleLiveRender();
  }

  // Markdown while it streams: plain text shows until the first rendered swap, then a detached
  // render replaces the visible node at most every LIVE_RENDER_MS. The version guard drops a
  // slow render that resolves after a newer one; the turn event's full render is the final word.
  scheduleLiveRender() {
    if (this.liveRenderTimer || !this.el?.liveText) return;
    const wait = Math.max(0, LIVE_RENDER_MS - (Date.now() - this.liveRenderedAt));
    this.liveRenderTimer = setTimeout(() => { this.liveRenderTimer = null; this.renderLive(); }, wait);
  }

  stopLiveRender() {
    if (this.liveRenderTimer) clearTimeout(this.liveRenderTimer);
    this.liveRenderTimer = null;
    this.liveRenderVersion += 1;            // an in-flight render must not land on the next view
    for (const host of this.liveHosts || []) this.removeChild?.(host);
    this.liveHosts = new Set();
  }

  // Each live pass renders under its own child component, dropped with its buffer: a Deep
  // answer is ~170 passes, and post-processor children registered on the shared mdHost would
  // otherwise pile up until the turn's full render.
  liveHost(buffer) {
    if (!MarkdownRenderChild || typeof this.addChild !== "function") return undefined;
    const host = this.addChild(new MarkdownRenderChild(buffer));
    (this.liveHosts ||= new Set()).add(host);
    return host;
  }

  dropLive(buffer, host) {
    buffer?.remove?.();
    if (host) { this.removeChild?.(host); this.liveHosts?.delete(host); }
  }

  renderLive() {
    const el = this.el;
    if (!el?.live || !this.sending) return;
    const version = ++this.liveRenderVersion;
    this.liveRenderedAt = Date.now();
    const buffer = create(el.live, "div", "slim-message-text slim-rendered-markdown");
    buffer.hidden = true;
    const host = this.liveHost(buffer);
    return this.renderMarkdown(buffer, this.streamingText, host).then(() => {
      if (version !== this.liveRenderVersion || this.el !== el || !this.sending) { this.dropLive(buffer, host); return; }
      const follow = atBottom(el.transcript);
      this.dropLive(el.liveRendered, el.liveHost);
      el.liveRendered = buffer;
      el.liveHost = host;
      buffer.hidden = false;
      el.liveText.hidden = true;
      if (follow) el.transcript.scrollTop = el.transcript.scrollHeight;
    }).catch((e) => console.error("[slim] live render", e));
  }

  renderStage() {
    if (!this.el) { this.render(); return; }
    if (!this.el.stage) return;
    this.el.stage.textContent = this.stage || "Working";
  }

  renderMarkdown(el, markdown, host = this.mdHost || this) {
    const text = normalizeMath(markdown);
    if (MarkdownRenderer && typeof MarkdownRenderer.render === "function") {
      return MarkdownRenderer.render(this.app, text, el, this.context?.source?.path || "",
                                     host).catch((e) => {
        console.error("[slim] render answer", e);
        el.setText?.(text);
      });
    }
    el.textContent = text;
    return Promise.resolve();
  }

  render() {
    const root = this.contentEl;
    if (!root || !root.createEl) return;
    const focused = this.composerFocused;
    clear(root);
    this.el = {};
    // Rendered Markdown attaches child components (embeds, callouts) to the component it is
    // given. One host per render, replaced by the next, so they do not pile up on the view.
    if (this.mdHost) { this.removeChild(this.mdHost); this.mdHost = null; }
    if (MarkdownRenderChild && typeof this.addChild === "function") {
      this.mdHost = this.addChild(new MarkdownRenderChild(root));
    }
    root.addClass?.("slim-copilot");
    const header = create(root, "header", "slim-copilot-head");
    const headRow = create(header, "div", "slim-copilot-head-row");
    create(headRow, "div", "slim-copilot-eyebrow", "SLIM");
    const tools = create(headRow, "div", "slim-copilot-tools");
    const history = iconButton(tools, "history", this.showThreads ? "Back to the chat" : "Chats",
      this.showThreads ? "clickable-icon slim-history is-active" : "clickable-icon slim-history");
    history.onclick = () => { this.showThreads = !this.showThreads; this.render(); };
    const add = iconButton(tools, "plus", "New chat");
    add.onclick = () => this.newThread();

    if (this.loading) { create(root, "p", "slim-copilot-status", "Indexing this note..."); return; }
    if (this.error && !this.context) { create(root, "p", "slim-copilot-error", this.error); return; }
    if (!this.context) { create(root, "p", "slim-copilot-status", "Open a Markdown note."); return; }

    const sourcePath = this.context.source.path;
    const sourceTitle = this.context.source.title || pathStem(sourcePath);
    const related = this.context.related || [];
    if (related.length) {
      // The title is the chip; Nearby folds under it. Closed by default: it is context for
      // them to reach for, not the reason the pane is open.
      const context = create(header, "details", "slim-copilot-context");
      context.open = this.nearbyOpen;
      const summary = create(context, "summary", "slim-copilot-context-summary");
      summary.onclick = () => { this.nearbyOpen = !context.open; };   // pre-toggle state, as with groups
      const note = create(summary, "span", "slim-copilot-note", sourceTitle);
      note.setAttribute("title", sourcePath);
      create(summary, "small", "slim-copilot-nearby-count", `${related.length} nearby`);
      for (const item of related) {
        const button = create(context, "button", "slim-related-note");
        button.setAttribute("title", item.path);
        create(button, "span", "", item.title || pathStem(item.path));
        create(button, "small", "", item.scope || "");
        button.onclick = () => this.openNote(item.path);
      }
    } else {
      const note = create(header, "div", "slim-copilot-note", sourceTitle);
      note.setAttribute("title", sourcePath);
    }

    if (this.showThreads) this.renderThreadList(root, sourceTitle);
    else this.renderTranscript(root);

    const composer = create(root, "footer", "slim-copilot-composer");
    if (this.pendingImages.length) {
      const previews = create(composer, "div", "slim-copilot-image-previews");
      this.pendingImages.forEach((attachment, index) => {
        const item = create(previews, "div", "slim-copilot-image-preview");
        const image = create(item, "img", "");
        image.src = attachment.preview
          || (attachment.id && this.activeThread ? this.imageURL(this.activeThread.id, attachment.id) : "");
        image.alt = attachment.name;
        const remove = iconButton(item, "x", `Remove ${attachment.name}`);
        remove.onclick = () => { this.pendingImages.splice(index, 1); this.render(); };
      });
    }
    const modes = create(composer, "select", "slim-mode-select");
    for (const value of ["quick", "deep"]) {
      const option = create(modes, "option", "", value[0].toUpperCase() + value.slice(1));
      option.value = value;
      option.selected = value === this.mode;
    }
    modes.onchange = () => { this.mode = modes.value; };
    const input = create(composer, "textarea", "slim-copilot-input");
    input.placeholder = "Ask SLIM";
    input.value = this.draft;
    input.rows = 1;
    autoGrow(input);
    input.oninput = () => { this.draft = input.value; autoGrow(input); };
    input.onfocus = () => { this.composerFocused = true; };
    input.onblur = () => { this.composerFocused = false; };
    this.el.input = input;
    input.onkeydown = (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); this.send(); }
      if ((event.key === "Backspace" || event.key === "Delete") &&
          !event.metaKey && !event.altKey && !event.ctrlKey) {
        // Obsidian's sidebar keymap can consume deletion before Electron applies the
        // textarea's native edit. Do the ordinary character/selection deletion here; keep
        // modified word/line deletion native so macOS retains its platform conventions.
        let start = Number.isInteger(input.selectionStart) ? input.selectionStart : input.value.length;
        let end = Number.isInteger(input.selectionEnd) ? input.selectionEnd : start;
        if (start === end && event.key === "Backspace" && start > 0) start -= 1;
        if (start === end && event.key === "Delete" && end < input.value.length) end += 1;
        if (start !== end) {
          event.preventDefault();
          input.value = `${input.value.slice(0, start)}${input.value.slice(end)}`;
          input.selectionStart = input.selectionEnd = start;
          this.draft = input.value;
        }
      }
    };
    input.onpaste = (event) => {
      const files = [...(event.clipboardData?.files || [])];
      if (files.some((file) => String(file.type || "").startsWith("image/"))) {
        event.preventDefault();
        this.addImages(files);
        return;
      }
      // Electron does not consistently perform the textarea's native insertion after a
      // plugin paste listener has inspected clipboardData. Insert plain text ourselves while
      // preserving the browser selection: paste must replace highlighted text, not append to
      // the draft or trigger a full render that moves the caret.
      const text = event.clipboardData?.getData?.("text/plain");
      if (typeof text !== "string") return;
      event.preventDefault();
      const start = Number.isInteger(input.selectionStart) ? input.selectionStart : input.value.length;
      const end = Number.isInteger(input.selectionEnd) ? input.selectionEnd : start;
      input.value = `${input.value.slice(0, start)}${text}${input.value.slice(end)}`;
      input.selectionStart = input.selectionEnd = start + text.length;
      this.draft = input.value;
      autoGrow(input);
    };
    const picker = create(composer, "input", "slim-copilot-image-picker");
    picker.type = "file";
    picker.accept = "image/png,image/jpeg,image/webp";
    picker.multiple = true;
    picker.onchange = () => { this.addImages(picker.files); picker.value = ""; };
    const attach = iconButton(composer, "image-plus", "Attach images", "clickable-icon slim-attach");
    attach.onclick = () => picker.click();
    composer.ondragover = (event) => {
      if ([...(event.dataTransfer?.items || [])].some((item) => item.kind === "file")) event.preventDefault();
    };
    composer.ondrop = (event) => {
      const files = [...(event.dataTransfer?.files || [])];
      if (files.some((file) => String(file.type || "").startsWith("image/"))) {
        event.preventDefault();
        this.addImages(files);
      }
    };
    const send = iconButton(
      composer, this.sending ? "square" : "send", this.sending ? "Stop response" : "Send",
      "mod-cta slim-send");
    send.onclick = () => this.sending ? this.cancel() : this.send();
    if (focused && typeof input.focus === "function") input.focus();
  }

  renderThreadList(root, sourceTitle) {
    const list = create(root, "nav", "slim-copilot-threads");
    if (!this.threads.length) {
      create(list, "p", "slim-copilot-status", "No chats yet.");
      return;
    }
    for (const group of groupThreads(this.threads, this.context.source.id)) {
      const details = create(list, "details", "slim-note-group");
      details.open = this.groupOpen.has(group.sourceId)
        ? this.groupOpen.get(group.sourceId) : group.expanded;
      const label = group.sourceId === this.context.source.id
        ? sourceTitle : noteTitle(this.app, group.path);
      const summary = create(details, "summary", "", label);
      summary.setAttribute("title", group.path);
      // At click time `open` is still the pre-toggle state; the browser flips it afterwards.
      summary.onclick = () => { this.groupOpen.set(group.sourceId, !details.open); };
      for (const thread of group.threads) {
        const row = create(details, "div", `slim-chat-row${thread.id === this.activeThread?.id ? " is-active" : ""}`);
        const open = create(row, "button", "slim-chat-open", thread.title || "Conversation");
        open.onclick = () => this.openThread(thread.id);
        const rename = iconButton(row, "pencil", `Rename ${thread.title}`);
        rename.onclick = () => this.renameThread(thread.id);
        const armed = this.deleteArmed === thread.id;
        const remove = iconButton(
          row, armed ? "check" : "trash-2",
          armed ? `Confirm delete ${thread.title}` : `Delete ${thread.title}`,
          armed ? "clickable-icon is-armed" : "clickable-icon");
        remove.onclick = () => this.deleteThread(thread.id);
      }
    }
  }

  renderTranscript(root) {
    const transcript = create(root, "section", "slim-copilot-transcript");
    this.el.transcript = transcript;
    const pending = [];
    for (const turn of this.turns) {
      const message = create(transcript, "article", `slim-message is-${turn.role}`);
      if (turn.role === "slim") {
        const body = create(message, "div", "slim-message-text slim-rendered-markdown");
        pending.push(this.renderMarkdown(body, turn.text)
          .catch((e) => console.error("[slim] render answer", e)));
      } else {
        create(message, "div", "slim-message-text", turn.text);
      }
      if (turn.attachments?.length) this.renderAttachments(message, turn.attachments);
      // Sources are a footnote row, one chip per note, only when the answer drew on notes.
      const sources = groupCitations(turn.turn?.citations);
      if (sources.length) {
        const list = create(message, "div", "slim-sources");
        create(list, "span", "slim-copilot-label", "From your notes");
        for (const source of sources) {
          const chip = create(list, "button", "slim-source", source.title);
          chip.setAttribute("title", source.path);
          chip.onclick = () => this.openNote(source.path);
        }
      }
      if (turn.turn?.verification?.truncated) {
        create(message, "small", "slim-verification", "Cut off at the length limit — ask for the rest.");
      }
      if (turn.role === "slim" && turn.turn) this.renderActions(message, turn);
    }
    const streamingHere = this.sending && this.turns === this.streamingTurns;
    if (streamingHere) {
      const live = create(transcript, "article", "slim-message is-slim is-streaming");
      this.el.live = live;
      this.el.liveText = create(live, "div", "slim-message-text", this.streamingText);
      this.el.stage = create(transcript, "div", "slim-copilot-stage", this.stage || "Working");
    }
    if (this.error) {
      const error = create(transcript, "div", "slim-copilot-error", this.error);
      if (this.retryQuestion) {
        const retry = iconButton(error, "rotate-ccw", "Retry response");
        retry.onclick = () => this.retry();
      }
    }
    if (!this.turns.length && !streamingHere && !this.error) {
      const suggestions = create(transcript, "div", "slim-suggestions");
      for (const prompt of SUGGESTED_PROMPTS) {
        const button = create(suggestions, "button", "slim-suggest", prompt);
        button.onclick = () => this.send(prompt);
      }
    }
    // A full render is a new transcript element at scrollTop 0: the newest turn — the
    // reason for the render — would sit below the fold.
    transcript.scrollTop = transcript.scrollHeight;
    if (pending.length) {
      // The Markdown lands after that scroll. Re-apply it once — unless they scrolled meanwhile.
      const anchor = transcript.scrollTop;
      Promise.all(pending).then(() => {
        if (this.el.transcript === transcript && transcript.scrollTop === anchor) {
          transcript.scrollTop = transcript.scrollHeight;
        }
      }).catch((e) => console.error("[slim] render turns", e));
    }
  }

  // A sent image stays a thumbnail: the server serves the stored bytes back. A pending image
  // has its own preview; a ref that fails to load falls back to the name chip.
  renderAttachments(message, attachments) {
    const gallery = create(message, "div", "slim-message-images");
    const threadId = this.activeThread?.id;
    for (const attachment of attachments) {
      const name = attachment.name || "Attached image";
      const src = attachment.preview || (attachment.id && threadId ? this.imageURL(threadId, attachment.id) : "");
      if (!src) { create(gallery, "span", "slim-image-chip", name); continue; }
      const image = create(gallery, "img", "slim-message-image");
      image.src = src;
      image.alt = name;
      image.onerror = () => { image.remove?.(); create(gallery, "span", "slim-image-chip", name); };
    }
  }

  // Under an answer: copy, regenerate (last answer only, while idle), and how it was made.
  renderActions(message, turn) {
    const actions = create(message, "div", "slim-message-actions");
    const copy = iconButton(actions, "copy", "Copy answer");
    copy.onclick = () => {
      if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) navigator.clipboard.writeText(turn.text);
    };
    if (turn === this.turns.at(-1) && !this.sending) {
      const again = iconButton(actions, "refresh-cw", "Regenerate");
      again.onclick = () => this.regenerate();
    }
    create(actions, "small", "slim-message-meta", metaText(turn.turn));
  }
}

/* One durable meeting object per recording/note, with one reserved capture slot.
 *
 * RecorderView deliberately remains the session controller: it already owns the draft, staged
 * segments, processing promise, review edits and recovery behavior. This manager owns only the
 * facts that are global across those controllers — lookup, path changes, and the exclusive
 * microphone/system-audio lease. Processing and review sessions never occupy that lease.
 */
class MeetingSessionManager {
  constructor(plugin, sessionFactory = null) {
    this.plugin = plugin;
    this.sessionFactory = sessionFactory || ((leaf) => new RecorderView(leaf, plugin));
    this.sessions = new Map();
    this.paths = new Map();
    this.activeCaptureId = null;
  }

  createSession(leaf = null) {
    const session = this.sessionFactory(leaf);
    session.sessionId = session.sessionId || recordingId();
    if (!session.recordingId) session.recordingId = session.sessionId;
    session.manager = this;
    this.sessions.set(session.sessionId, session);
    this.indexSession(session);
    return session;
  }

  indexSession(session) {
    if (!session) return null;
    for (const [path, owner] of this.paths) {
      if (owner === session && path !== session.draftRel && path !== session.filedNotePath) {
        this.paths.delete(path);
      }
    }
    for (const path of [session.draftRel, session.filedNotePath]) {
      if (path) this.paths.set(path, session);
    }
    return session;
  }

  sessionForPath(path) { return (path && this.paths.get(path)) || null; }

  sessionForLeaf(leaf) {
    if (!leaf) return null;
    for (const session of this.sessions.values()) {
      if (session.editorLeaf === leaf) return session;
    }
    return null;
  }

  claimCapture(session) {
    if (!session || !session.sessionId) throw new Error("meeting session has no identity");
    // Only a session that is recording holds the microphone. The ribbon's setup draft takes
    // the lease so a second click returns to it, but it captures nothing — a Resume elsewhere
    // takes over rather than reporting a recording they could not see (2026-09-05).
    const holder = this.activeCapture();
    if (holder && holder !== session && holder.state === "recording") {
      throw new Error("another meeting capture is already active");
    }
    this.activeCaptureId = session.sessionId;
    return session;
  }

  /* A setup draft that was never started keeps the lease for as long as Obsidian stays open, so
   * a click days later reopened it under the name of the day it was made (2026-09-17). An
   * untouched one from an earlier day is deleted and replaced; one with their input is kept. */
  async releaseStaleSetupDraft(session) {
    if (session.state !== "idle" || (session.segments && session.segments.length)) return false;
    if (String(session.recordingId || "").slice(0, 8) === localDay()) return false;
    if (typeof session.removeUntouchedDraft !== "function") return false;
    if (!(await session.removeUntouchedDraft())) return false;
    this.removeSession(session);
    return true;
  }

  releaseCapture(session) {
    if (session && this.activeCaptureId === session.sessionId) this.activeCaptureId = null;
  }

  activeCapture() {
    const session = this.activeCaptureId && this.sessions.get(this.activeCaptureId);
    if (!session && this.activeCaptureId) this.activeCaptureId = null;
    return session || null;
  }

  async activate(leaf = null) {
    let session = this.activeCapture();
    if (session && await this.releaseStaleSetupDraft(session)) session = null;
    if (!session) {
      // A processing session may navigate its leaf to the filed note at any moment. Starting B
      // in that same leaf lets A's completion replace B's live recorder. Give the new capture a
      // fresh pane whenever the current one already belongs to another durable meeting object.
      const availableLeaf = this.sessionForLeaf(leaf) ? null : leaf;
      session = this.createSession(availableLeaf);
      this.claimCapture(session);
    }
    await session.open();
    this.indexSession(session);
    return session;
  }

  async openPath(path, leaf = null) {
    const session = this.sessionForPath(path);
    if (!session) return null;
    if (leaf) session.editorLeaf = leaf;
    await session.open();
    this.indexSession(session);
    return session;
  }

  async adoptFile(file, leaf = null) {
    if (!file || !file.path) return null;
    const existing = this.sessionForPath(file.path);
    if (existing) {
      if (leaf) existing.editorLeaf = leaf;
      return existing;
    }
    const session = this.createSession(leaf);
    let adopted = false;
    if (file.path.endsWith(DRAFT_SUFFIX) && typeof session.adoptPausedDraft === "function") {
      adopted = await session.adoptPausedDraft(file.path);
      if (adopted) session.editorLeaf = leaf;
    } else if (typeof session.openExisting === "function") {
      adopted = await session.openExisting(file, leaf);
    }
    if (!adopted) {
      this.removeSession(session);
      return null;
    }
    this.indexSession(session);
    if (typeof session.attachEditorObject === "function") session.attachEditorObject();
    if (typeof session.render === "function") session.render();
    return session;
  }

  removeSession(session) {
    if (!session) return;
    this.releaseCapture(session);
    for (const [path, owner] of this.paths) {
      if (owner === session) this.paths.delete(path);
    }
    this.sessions.delete(session.sessionId);
  }

  ownsEditor(info) {
    for (const session of this.sessions.values()) {
      if (typeof session.ownsEditor === "function" && session.ownsEditor(info)) return session;
    }
    return null;
  }

  handleVaultDelete(file) {
    const session = file && this.sessionForPath(file.path);
    if (!session) return;
    // The session knows which deletions are the server's own work (the draft during filing,
    // the old path during a move). Every other deletion is the user's instruction to release
    // this one object, including the setup draft that reserves the microphone action.
    if (session.expectsDeletion(file.path)) return;
    if (typeof session.handleVaultDelete === "function") session.handleVaultDelete(file);
    this.removeSession(session);
  }

  handleVaultRename(file, oldPath) {
    const session = this.sessionForPath(oldPath);
    if (!session || !file || !file.path) return;
    if (session.draftRel === oldPath) session.draftRel = file.path;
    if (session.filedNotePath === oldPath) {
      session.filedNotePath = file.path;
      if (session.result) session.result.note = file.path;
    }
    this.indexSession(session);
  }

  onClose() {
    for (const session of this.sessions.values()) {
      if (typeof session.onClose === "function") session.onClose();
    }
    this.sessions.clear();
    this.paths.clear();
    this.activeCaptureId = null;
  }
}

module.exports = class SlimRecorderPlugin extends Plugin {
  async onload() {
    this.settings = Object.assign({}, DEFAULTS, await this.loadData());
    this.server = null;
    this._ensuring = null;
    this.meetingBlocks = new Map();
    this.meetings = new MeetingSessionManager(this);
    this.registerMarkdownCodeBlockProcessor("slim-meeting", (_source, el, ctx) => {
      const path = ctx.sourcePath;
      el.addClass("slim-meeting-block");
      this.meetingBlocks.set(path, el);
      if (ctx && typeof ctx.addChild === "function" && MarkdownRenderChild) {
        const child = new MarkdownRenderChild(el);
        child.onunload = () => {
          if (this.meetingBlocks.get(path) === el) this.meetingBlocks.delete(path);
          const session = this.meetings.sessionForPath(path);
          if (session && session.contentEl === el) {
            session.contentEl = null;
            session.editorView = null;
          }
        };
        ctx.addChild(child);
      }
      const session = this.meetings.sessionForPath(path);
      if (session) {
        session.attachEditorObject();
        session.render();
      } else {
        // Adoption follows from the real Markdown leaf below. Until then the durable block is
        // still readable; no session is allowed to paint another meeting's state into it.
        renderPassiveMeetingBlock(this.app, this, el, _source, path);
      }
    });
    const activate = () => this.activate().catch((e) => {
      console.error("[slim] activate", e);
      new Notice(`SLIM: could not open the recorder — ${e.message || e}`);
    });
    this.addRibbonIcon("mic", "SLIM: record", activate);
    this.addCommand({ id: "start-recording", name: "Start recording",
                      callback: activate });
    this.registerView(VIEW_TYPE_COPILOT, (leaf) => new CopilotView(leaf, this));
    this.addRibbonIcon("message-square-text", "SLIM: open-note copilot",
                       () => this.activateCopilot());
    this.addCommand({ id: "open-note-copilot", name: "Open note copilot",
                      callback: () => this.activateCopilot() });
    if (this.app.workspace && typeof this.app.workspace.on === "function") {
      this.registerEvent(this.app.workspace.on("active-leaf-change", () => {
        const file = this.app.workspace.getActiveFile();
        for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE_COPILOT)) {
          if (leaf.view && typeof leaf.view.setActiveFile === "function") {
            leaf.view.setActiveFile(file);
          }
        }
        queueMicrotask(() => this.noticeClosedPendingReviews());
      }));
      this.registerEvent(this.app.workspace.on("layout-change", () => {
        this.noticeClosedPendingReviews();
      }));
    }
    this.addSettingTab(new SlimSettingTab(this.app, this));

    if (this.app.workspace && typeof this.app.workspace.on === "function" &&
        typeof this.registerEvent === "function") {
      this.registerEvent(this.app.workspace.on("editor-paste", (event, editor, info) => {
        const session = this.meetings.ownsEditor(info);
        if (!session) return;
        session.handleEditorPaste(event, editor)
          .catch((e) => { console.error("[slim] image paste", e); new Notice(`SLIM: ${e.message || e}`); });
      }));
      this.registerEvent(this.app.workspace.on("file-open", (file) => {
        const leaf = this.app.workspace.getMostRecentLeaf && this.app.workspace.getMostRecentLeaf();
        if (!file || !leaf || !(leaf.view instanceof MarkdownView)) return;
        for (const session of this.meetings.sessions.values()) {
          if (session.editorLeaf === leaf && file.path !== session.draftRel &&
              file.path !== session.filedNotePath) session.detachEditorObject();
        }
        const tracked = this.meetings.sessionForPath(file.path);
        if (tracked) {
          tracked.editorLeaf = leaf;
          tracked.attachEditorObject();
          tracked.render();
          return;
        }
        this.meetings.adoptFile(file, leaf).catch((e) => console.error("[slim] open note", e));
      }));
      if (this.app.vault && typeof this.app.vault.on === "function") {
        this.registerEvent(this.app.vault.on("delete", (file) => this.meetings.handleVaultDelete(file)));
        this.registerEvent(this.app.vault.on("rename", (file, oldPath) =>
          this.meetings.handleVaultRename(file, oldPath)));
      }
    }

    // Not awaited: the server takes seconds to bind and onload blocks Obsidian's startup.
    // `post()` waits on it before the first request, which is the moment it actually matters.
    this.ensureServer().catch((e) => console.error("[slim] ensureServer", e));
    if (this.app.workspace && typeof this.app.workspace.onLayoutReady === "function") {
      this.app.workspace.onLayoutReady(() => {
        this.warnAboutStrandedAudio();
        // `file-open` does not fire for leaves already visible when a plugin reloads. Adopt all
        // of them, not whichever one happened to be most recent.
        const leaves = this.app.workspace.getLeavesOfType("markdown") || [];
        for (const leaf of leaves) {
          const file = leaf && leaf.view instanceof MarkdownView && leaf.view.file;
          if (file) this.meetings.adoptFile(file, leaf)
            .catch((e) => console.error("[slim] open visible note", e));
        }
      });
    }
  }

  onunload() {
    if (this.meetings) this.meetings.onClose();
    this.stopServer();
  }

  async saveSettings() { await this.saveData(this.settings); }

  meetingBlockFor(path) {
    const el = this.meetingBlocks && this.meetingBlocks.get(path);
    // A code-block processor is invoked while Obsidian is still constructing the Live Preview
    // DOM, so its element can legitimately report `isConnected === false` at the exact moment
    // the already-open note is adopted after a plugin reload. The MarkdownRenderChild unload
    // hook removes genuinely stale entries; rejecting the construction-phase element here
    // loses the only mount opportunity and leaves an empty block until the user changes files.
    return el || null;
  }

  // ---- the server's lifecycle ------------------------------------------------------------

  /* Nothing else starts `slim chat`: there is no launchd agent, and the Electron shell that
   * used to do it was cut on 2026-08-27. Without this, recording with the server down fails
   * at the POST — the audio survives in the staging folder but recovery is by hand.
   *
   * Not a memory decision: measured 2026-08-06, the server is 31 MB and holds NO model. Ollama
   * loads weights on demand and evicts on its own keep_alive.
   */
  ensureServer() {
    // One spawn at a time. Both `post()` calls of a single recording can arrive here at once,
    // and two spawns would race for one port.
    if (!this._ensuring) {
      this._ensuring = this._ensureServer().finally(() => { this._ensuring = null; });
    }
    return this._ensuring;
  }

  async _ensureServer() {
    const port = this.settings.port;
    // A CLI-started server is REUSED, not duplicated — the second bind would fail anyway, and
    // killing their own `uv run slim chat` on unload would be worse than not managing it at all.
    // ⚠ waitForPort, not a single portOpen: `--reload` closes the port for a beat on every
    // restart, and a one-shot probe there reads "nothing serving" and spawns a SECOND server
    // that then fights the reloader's respawn for the bind. 3s covers a restart. It probes
    // once immediately, so a live server still returns at once; the 3s is paid only when
    // nothing is listening at all, and that path is the un-awaited one at onload().
    if (await waitForPort(port, 3000)) {
      this.warnIfStale();          // not awaited: a reused server must not delay a recording
      return "reused";
    }

    try {
      // ⚠ detached: its own process GROUP. `uv run` wraps
      // the real python server, so a plain child.kill() reaches only uv — measured there, the
      // server outlived the app. See stopServer().
      const argv = serverArgv(this.settings);
      const base = this.vaultBasePath();
      // ⚠ stdio is "ignore", so the server's own output goes nowhere from here. Run
      // `uv run slim chat` in a terminal when you want to watch it work — and `--reload`
      // there too, which is the only place it ever belonged (a restart cuts requests in
      // flight, so a save during a recording loses that recording's filing).
      this.server = spawn(this.settings.uvPath, argv,
                          { stdio: "ignore", detached: true,
                            env: base ? serverEnv(base) : process.env });
    } catch (e) {
      this.server = null;
      new Notice(`SLIM: could not launch ${this.settings.uvPath} — ${(e && e.message) || e}`,
                 20000);
      return "failed";
    }
    const child = this.server;
    child.on("error", (e) => {
      // A bad uv path fails here, asynchronously, not at spawn().
      if (this.server === child) this.server = null;
      console.error("[slim] server spawn", e);
      new Notice(`SLIM: the server did not start — check the repo and uv paths in settings. ` +
                 `(${(e && e.message) || e})`, 20000);
    });
    child.on("exit", () => { if (this.server === child) this.server = null; });

    if (await waitForPort(port, 30000)) return "started";
    new Notice(`SLIM: the server did not come up on 127.0.0.1:${port} within 30s. Check that ` +
               "Ollama is running and that `uv run slim chat` works from the repo.", 20000);
    return "timeout";
  }

  /* ⚠ A REUSED SERVER CAN BE OLDER THAN THE CODE, and nothing about it looks wrong.
   *
   * Reuse is deliberate — it is what stops this plugin killing a terminal `slim chat --reload`.
   * But `uv run` reads Python from disk at spawn time and never hot-reloads, and "the plugin
   * owns the lifecycle" turned out not to hold: Obsidian does not reliably call `onunload` when
   * the whole APP quits, so a server started the previous afternoon survived a Cmd+Q and a
   * relaunch, was reused, and failed two recordings against a bug already fixed on disk
   * (2026-08-23). In July the same trap cost a full day of debugging.
   *
   * It WARNS rather than restarting: the server on the port might be their, with a reloader
   * attached, and killing that out from under them is its own surprise. Once at load, not per
   * recording — a warning that appears when they presse Stop is a warning that arrives too late.
   */
  /* This vault's own base path, or null on a mobile adapter that has none. `openDiskFile()`
   * already resolves audio against it; the server has to agree on the same directory. */
  vaultBasePath() {
    try {
      const adapter = this.app.vault.adapter;
      if (typeof adapter.getBasePath !== "function") return null;
      return nodePath.resolve(adapter.getBasePath());
    } catch (e) {
      console.error("[slim] no vault base path", e);
      return null;
    }
  }

  async warnIfStale() {
    // ⚠ Deliberately NOT this.post(): that calls ensureServer() and would recurse.
    const res = await requestUrl({
      url: `http://127.0.0.1:${this.settings.port}/api/health`,
      method: "GET",
      throw: false,
    }).catch(() => null);
    // A server old enough to predate this endpoint answers 404 — which is itself proof that it
    // is stale, and worth saying so rather than swallowing.
    if (!res || res.status === 404) {
      new Notice("SLIM: the running server predates /api/health, so it is serving OLD code. " +
                 `Restart it:\n\nlsof -tiTCP:${this.settings.port} -sTCP:LISTEN | xargs kill`, 30000);
      return;
    }
    const health = res.json;
    if (!health) return;
    // A REUSED server resolved its own vault, possibly before this vault was opened or moved.
    // Wrong vault is worse than old code: the recording is filed somewhere nobody is looking.
    // Python's Path.resolve() follows symlinks; Node's nodePath.resolve() does not. Use
    // fs.realpathSync to match, or fall back to the lexical resolve if it fails.
    const base = this.vaultBasePath();
    const fs = require("fs");
    let resolvedBase = base;
    try { if (base) resolvedBase = fs.realpathSync(base); } catch {}
    const killCmd = `lsof -tiTCP:${this.settings.port} -sTCP:LISTEN | xargs kill`;
    if (resolvedBase && health.vault && health.vault !== resolvedBase) {
      new Notice(
        `SLIM: the server on port ${this.settings.port} is using the vault ${health.vault}, ` +
        `but this vault is ${base}. Recordings would be filed in the wrong one. Restart it:` +
        `\n\n${killCmd}`, 30000);
      return;
    }
    if (!health.stale) return;
    new Notice(
      `SLIM: the running server started ${health.started_at} but ${health.newest_file} changed ` +
      `${health.newest_source}. It is serving OLD code. Restart it:\n\n` +
      killCmd, 30000);
  }

  async openMeetingPath(path) {
    const file = this.app.vault.getAbstractFileByPath(path);
    if (!file) throw new Error(`no note at ${path}`);
    const workspace = this.app.workspace;
    const markdownLeaves = typeof workspace.getLeavesOfType === "function"
      ? workspace.getLeavesOfType("markdown") : [];
    let leaf = markdownLeaves.find((candidate) =>
      candidate.view && candidate.view.file && candidate.view.file.path === path);
    if (!leaf) {
      leaf = workspace.getLeaf(true);
      await leaf.openFile(file, { active: true, state: { mode: "source", source: false } });
    }
    let session = this.meetings.sessionForPath(path);
    if (session) {
      session.editorLeaf = leaf;
      await session.open();
    } else {
      session = await this.meetings.adoptFile(file, leaf);
    }
    if (typeof workspace.revealLeaf === "function") workspace.revealLeaf(leaf);
    return { leaf, session };
  }

  async revealMeetingPath(path) {
    await this.openMeetingPath(path);
    if (this.app.commands && typeof this.app.commands.executeCommandById === "function") {
      this.app.commands.executeCommandById("file-explorer:reveal-active-file");
    }
  }

  showPendingReviewNotice(session) {
    const path = session && session.filedNotePath;
    if (!path) return;
    const notice = new Notice(
      `SLIM: recording saved at ${path}. Review is still pending.`, 20000);
    const el = notice.noticeEl;
    if (!el || typeof el.empty !== "function") return;
    el.empty();
    el.createDiv({ text: "SLIM: recording saved. Review is still pending." });
    el.createDiv({ cls: "slim-notice-path", text: path });
    const actions = el.createDiv({ cls: "slim-notice-actions" });
    const reopen = actions.createEl("button", { cls: "mod-cta", text: "Reopen" });
    reopen.addEventListener("click", () => {
      this.openMeetingPath(path)
        .then(() => { if (typeof notice.hide === "function") notice.hide(); })
        .catch((e) => new Notice(`SLIM: could not reopen the note — ${e.message || e}`));
    });
    const reveal = actions.createEl("button", { text: "Reveal" });
    reveal.addEventListener("click", () => {
      this.revealMeetingPath(path)
        .then(() => { if (typeof notice.hide === "function") notice.hide(); })
        .catch((e) => new Notice(`SLIM: could not reveal the note — ${e.message || e}`));
    });
  }

  noticeClosedPendingReviews() {
    if (!this.meetings || !this.app.workspace ||
        typeof this.app.workspace.getLeavesOfType !== "function") return;
    const live = new Set(this.app.workspace.getLeavesOfType("markdown"));
    for (const session of this.meetings.sessions.values()) {
      const shown = session.editorLeaf && session.editorLeaf.view && session.editorLeaf.view.file;
      const stillShowingMeeting = shown &&
        (shown.path === session.filedNotePath || shown.path === session.draftRel);
      if (live.has(session.editorLeaf) && stillShowingMeeting) {
        session.reviewCloseNotified = false;
      } else if (session.state === "card" && !session.filing && session.editorLeaf &&
                 !session.reviewCloseNotified) {
        session.reviewCloseNotified = true;
        this.showPendingReviewNotice(session);
      }
    }
  }

  stopServer() {
    const child = this.server;
    this.server = null;
    if (!child || child.pid == null) return;
    try {
      process.kill(-child.pid, "SIGTERM");   // the whole group: uv AND the python server
    } catch (e) {
      try { child.kill(); } catch (e2) { /* already gone */ }
    }
  }

  /* Writing chunks to disk is only half a guarantee if the survivor is never mentioned again.
   * `record.write_recording` unlinks the staged file once the audio is archived, so anything
   * non-empty still sitting here is a recording that never finished being filed. */
  async warnAboutStrandedAudio() {
    try {
      const adapter = this.app.vault.adapter;
      if (!(await adapter.exists(STAGING_DIR))) return;
      const listing = await adapter.list(STAGING_DIR);
      const webm = (listing.files || []).filter((f) => f.toLowerCase().endsWith(".webm"));
      const stranded = [];
      for (const f of webm) {
        // A zero-byte file is a recording that failed before its first chunk — nothing to
        // recover, and warning about it would train them to ignore this notice.
        try {
          const st = await adapter.stat(f);
          if (!st || st.size > 0) stranded.push(f);
        } catch (e) { stranded.push(f); }
      }
      if (!stranded.length) return;

      const draftsById = new Map();
      for (const folder of ["Capture/_unfiled", "Journal"]) {
        if (!(await adapter.exists(folder))) continue;
        const drafts = await adapter.list(folder);
        for (const path of drafts.files || []) {
          const name = nodePath.posix.basename(path);
          if (!name.endsWith(DRAFT_SUFFIX) || !name.includes("--recording--")) continue;
          const id = name.slice(0, -DRAFT_SUFFIX.length).split("--recording--").pop();
          if (id) draftsById.set(id, path);
        }
      }

      // ⚠ A resumed recording stages `<id>-001.webm`, `-002.webm`. Without stripping that
      // suffix the segments look stranded and their draft looks orphaned — the notice whose
      // whole job is to say "nothing was lost" would say the opposite.
      const paired = [...new Set(stranded.map(recordingIdFromStaged))].filter((id) => draftsById.has(id));
      if (paired.length) {
        new Notice(
          `SLIM: ${paired.length} recoverable recording${paired.length > 1 ? "s" : ""} ` +
          `have both a saved Markdown draft and audio (${paired.join(", ")}). Open the draft ` +
          "to review it; SLIM will not delete either source automatically.", 30000);
      }
      const unpaired = stranded.filter((path) => !draftsById.has(recordingIdFromStaged(path)));
      if (!unpaired.length) return;
      new Notice(
        `SLIM: ${unpaired.length} recording${unpaired.length > 1 ? "s" : ""} in ${STAGING_DIR} ` +
        "never finished filing. The audio is intact and playable — file it by hand.", 30000);
    } catch (e) {
      console.error("[slim] stranded audio check", e);
    }
  }

  async activate() {
    const { workspace } = this.app;
    const recent = typeof workspace.getMostRecentLeaf === "function"
      ? workspace.getMostRecentLeaf() : null;
    const leaf = recent && recent.view instanceof MarkdownView ? recent : null;
    const session = await this.meetings.activate(leaf);
    if (session.editorLeaf && typeof workspace.revealLeaf === "function") {
      workspace.revealLeaf(session.editorLeaf);
    }
  }

  /* The recording POST stays synchronous so its durable-source guarantees remain simple.
   * This lightweight companion request exposes only progress for that POST. A first 404 is
   * normal: the poll can win the race with the request handler creating its job record. */
  watchRecording(jobId, onProgress) {
    let stopped = false;
    let timer = null;
    const tick = async () => {
      if (stopped) return;
      try {
        const res = await requestUrl({
          url: `http://127.0.0.1:${this.settings.port}/api/record/progress?job_id=${encodeURIComponent(jobId)}`,
          method: "GET",
          throw: false,
        });
        if (res.status === 200 && res.json) {
          onProgress(res.json);
          if (res.json.done) { stopped = true; return; }
        }
      } catch (e) {
        // Progress is supplementary. The POST below remains the authoritative success/error
        // path, so a transient poll failure must never turn a recoverable recording into an
        // apparent failure.
        console.debug("[slim] progress poll", e);
      }
      if (!stopped) timer = window.setTimeout(tick, 600);
    };
    tick();
    return () => {
      stopped = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }

  async activateCopilot() {
    const { workspace } = this.app;
    const existing = workspace.getLeavesOfType(VIEW_TYPE_COPILOT);
    const leaf = existing.length ? existing[0] : workspace.getRightLeaf(false);
    await leaf.setViewState({ type: VIEW_TYPE_COPILOT, active: true });
    workspace.revealLeaf(leaf);
    if (leaf.view && typeof leaf.view.setActiveFile === "function") {
      await leaf.view.setActiveFile(workspace.getActiveFile());
    }
  }

  async saveActiveNote(file) {
    const view = this.app.workspace.getActiveViewOfType
      ? this.app.workspace.getActiveViewOfType(MarkdownView) : null;
    if (view && view.file?.path === file?.path && typeof view.save === "function") {
      await view.save();
    }
  }

  // `requestUrl` rather than fetch: it bypasses CORS, which a plugin origin otherwise trips on.
  async post(path, body) {
    // Cheap when the server is up (one loopback connect) and the whole point when it is not:
    // a recording made minutes after launch, or after the server died, still lands.
    await this.ensureServer();
    const res = await requestUrl({
      url: `http://127.0.0.1:${this.settings.port}${path}`,
      method: "POST",
      contentType: "application/json",
      body: JSON.stringify(body),
      throw: false,
    });
    if (res.status !== 200) {
      let detail = res.text;
      let cancelled = false;
      let silent = false;
      try {
        detail = (res.json && res.json.error) || detail;
        cancelled = !!(res.json && res.json.cancelled);
        silent = !!(res.json && res.json.silent);
      } catch (e) { /* not json */ }
      // A cancel is THEIR instruction, not a failure, and the caller has to be able to tell the
      // two apart: one parks the session, the other shows an error screen.
      const err = new Error(cancelled || silent
        ? detail : `${path} returned ${res.status}: ${detail}`);
      err.status = res.status;
      err.cancelled = cancelled;
      err.silent = silent;
      throw err;
    }
    return res.json;
  }

  postJSON(path, body) { return this.post(path, body); }

  async getJSON(path) {
    await this.ensureServer();
    const res = await requestUrl({
      url: `http://127.0.0.1:${this.settings.port}${path}`,
      method: "GET", throw: false,
    });
    if (res.status !== 200) {
      const detail = (res.json && res.json.error) || res.text;
      throw new Error(`${path} returned ${res.status}: ${detail}`);
    }
    return res.json;
  }

  async streamCopilot(body, handlers, signal) {
    await this.ensureServer();
    return streamRequest(this.settings.port, "/api/copilot/stream", body, handlers, signal);
  }

  postRecording(body) { return this.post("/api/record", body); }
  postLiveTranscript(body) { return this.post("/api/record/live", body); }
  applyCard(body) { return this.post("/api/record/apply", body); }
  completeReview(body) { return this.post("/api/record/review", body); }
  saveNotes(body) { return this.post("/api/record/notes", body); }
  retrySummary(body) { return this.post("/api/record/summary", body); }
  cancelRecording(body) { return this.post("/api/record/cancel", body); }
  recordTranscript(body) { return this.post("/api/record/transcript", body); }
  appendRecording(body) { return this.post("/api/record/append", body); }
};

// A testing seam, and the only one. `obsidian/main.test.mjs` needs the view class; Obsidian
// itself only ever reads the default export, so an extra property costs nothing at runtime.
module.exports.RecorderView = RecorderView;
module.exports.MeetingSessionManager = MeetingSessionManager;
module.exports.splitMeetingNote = splitMeetingNote;
module.exports.embedLegacyMeetingBlock = embedLegacyMeetingBlock;
module.exports.markdownBlockShortcut = markdownBlockShortcut;
module.exports.pcm16k = pcm16k;
module.exports.renderPassiveMeetingBlock = renderPassiveMeetingBlock;
module.exports.CopilotView = CopilotView;
module.exports.RenameModal = RenameModal;
module.exports.SSEParser = SSEParser;
module.exports.groupThreads = groupThreads;
module.exports.groupCitations = groupCitations;
module.exports.normalizeMath = normalizeMath;
module.exports.SUGGESTED_PROMPTS = SUGGESTED_PROMPTS;
module.exports.metaText = metaText;
module.exports.noteTitle = noteTitle;
module.exports.streamRequest = streamRequest;
module.exports.inlineMarkdown = inlineMarkdown;
module.exports.stampSourceAtoms = stampSourceAtoms;
module.exports.markdownFromEditable = markdownFromEditable;
module.exports.serverArgv = serverArgv;
module.exports.serverEnv = serverEnv;
