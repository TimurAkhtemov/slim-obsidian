// The meeting note: its paths, its ids, its parts, and the tabbed reader that shows them.
"use strict";

const { MarkdownRenderer } = require("obsidian");
const { randomUUID } = require("crypto");
const nodePath = require("path");
const { pad } = require("./clock");

const STAGING_DIR = "Attachments/_incoming";
const RECORDER_ATTACHMENTS = "Attachments/Recorder";
const DRAFT_SUFFIX = ".slim-draft.md";
const EMPTY_MEETING_BLOCK = "````slim-meeting\n\n````\n\n";

// Matches slim/capture/voicetags.py TYPE_FRONTMATTER. A type they PICKS is testimony: the
// pipeline never overrides a declared type, so choosing here removes the whole axis from
// guesswork.
const TYPES = [
  ["meeting-note", "Meeting"],
  ["lecture", "Lecture"],
  ["idea", "Idea"],
  ["journal", "Journal"],
  ["note", "Note"],
];

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
 * It NEVER writes the note. `slim/recorder/note.py` remains the sole owner of frontmatter and of
 * the Summary / Notes / Transcript boundaries. Keeping the complete frontmatter block here is a
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

module.exports = { STAGING_DIR, RECORDER_ATTACHMENTS, DRAFT_SUFFIX, EMPTY_MEETING_BLOCK, TYPES,
        recordingIdFromStaged, localDay, recordingId, splitMeetingNote, frontmatterScalar, hasMeetingBlock,
        embedLegacyMeetingBlock, renderMeetingTabs, renderPassiveMeetingBlock };
