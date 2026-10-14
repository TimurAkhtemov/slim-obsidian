// The copilot's pure helpers: threads, titles, citations, skills, math, and what a run did.
"use strict";

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

/* ---- copilot skills and editing runs -------------------------------------------------------
 * The server never writes a note from the copilot. One chat answers or edits: a question gets an
 * answer; a request to change the note gets a plan, then changes that land in the note's editor
 * as they stream (`makeLiveEditor`). A new note from a skill is created here, through the Vault
 * API. The note as it was before a run is kept with the chat, outside the vault.
 */
function sha256Hex(text) {
  return require("crypto").createHash("sha256").update(String(text), "utf8").digest("hex");
}

// "/con" → "con" while the first word is still being typed; null once there is a space.
function slashQuery(draft) {
  const match = /^\/([a-z0-9-]*)$/.exec(String(draft || ""));
  return match ? match[1] : null;
}

function matchSkills(skills, query) {
  if (query === null || query === undefined) return [];
  return (skills || []).filter((skill) => skill.name.startsWith(query)).slice(0, 8);
}

// One line under a skill or edit answer: what code actually handed the model.
function inputsText(inputs) {
  if (!inputs) return "";
  const parts = [];
  const used = inputs.used?.length || 0;
  if (used) parts.push(`Read ${used} note${used === 1 ? "" : "s"}${inputs.selection ? " and your selection" : ""}`);
  else if (inputs.selection) parts.push("Read your selection");
  if (inputs.dropped?.length) parts.push(`${inputs.dropped.length} left out (too long together)`);
  if (inputs.cut?.length) parts.push(`${inputs.cut.length} cut to fit`);
  if (inputs.skipped?.length) parts.push(`${inputs.skipped.length} without that section`);
  for (const name of inputs.missing || []) parts.push(`[[${name}]] not found`);
  return parts.join(" · ");
}

// The open note's editor selection, from whichever pane shows it: the sidebar has focus when
// they press Send, so "the active editor" is not the question. Its text, and where it starts as
// the editor counts (UTF-16 units): the same words can be in the note twice.
function editorSelection(app, path) {
  const recent = app?.workspace?.getMostRecentLeaf?.();
  const leaves = app?.workspace?.getLeavesOfType?.("markdown") || [];
  for (const leaf of recent && leaves.includes(recent) ? [recent, ...leaves] : leaves) {
    const view = leaf.view;
    // Reading mode keeps CodeMirror's last selection, which the owner can no longer see.
    if (view?.file?.path !== path || view.getMode?.() === "preview") continue;
    if (typeof view.editor?.getSelection !== "function") continue;
    const text = view.editor.getSelection();
    if (!text) continue;
    const at = view.editor.getCursor?.("from");
    const from = at ? view.editor.posToOffset?.(at) : undefined;
    return { text: text.slice(0, MAX_SELECTION_CHARS), ...(Number.isInteger(from) ? { from } : {}) };
  }
  return { text: "" };
}

// Mirrors copilot.py MAX_SELECTION_CHARS.
const MAX_SELECTION_CHARS = 20_000;
// Mirrors edits.MAX_EDIT_CHARS: a longer note is still asked about, from the file, but not edited.
const MAX_EDIT_CHARS = 400_000;

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

// The line under a turn while a model call runs: what it is doing and, from its first second, for
// how long. A call writes for 10 to 30 s before anything else shows.
function stageLine(stage, since, now) {
  const seconds = since ? Math.floor((now - since) / 1000) : 0;
  return `${stage || "Working"}…${seconds >= 1 ? ` ${seconds} s` : ""}`;
}

const SUGGESTED_PROMPTS = [
  "Explain this note like I'm new to it",
  "Quiz me on this note",
  "What is unclear or missing here?",
];

function titleFromQuestion(question) {
  const title = question.trim().replace(/\s+/g, " ");
  return title.length > 60 ? `${title.slice(0, 57)}...` : title || "Conversation";
}

function threadId() {
  return `chat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function cancelledError() {
  const error = new Error("Cancelled");
  error.name = "AbortError";
  return error;
}

// The changes a run made that the owner decides on, and how many of those are undone. A change
// `part_of` another only rewrote SLIM's own text from earlier in the run: it is that one's.
function runCounts(run) {
  const landed = (run.changes || [])
    .filter((change) => !change.part_of && !["skipped", "failed", "creating"].includes(change.state));
  const undone = run.restored ? landed.length
    : landed.filter((change) => change.state === "undone" || change.state === "trashed").length;
  return { landed, undone };
}

// What the model sees of an earlier turn: an answer's prose; an edit turn as its plan and what
// became of its changes, never the text it wrote (that is in the note, which the model reads).
function historyText(turn) {
  const run = turn?.turn?.edit_run;
  if (turn?.role !== "slim" || !run) return turn?.text;
  const { landed, undone } = runCounts(run);
  const steps = run.plan?.steps || [];
  return `[Changed the note: ${run.plan?.intro || run.summary || "no plan"}`
    + `${steps.length ? `; steps: ${steps.join("; ")}` : ""}`
    + `; ${landed.length} change${landed.length === 1 ? "" : "s"}, ${undone} undone`
    + `${run.stopped && steps.length ? `; stopped after ${run.steps_done || 0} of ${steps.length} steps` : ""}]`;
}

const STOPPED_BECAUSE = {
  refusals: "its changes kept being refused",
  "call limit": "it reached its limit of calls",
  "token budget": "it reached its output budget",
  "empty plan": "the plan had no steps",
};

// "Done: …", or how far a run got, why it stopped, and what it left undone.
function editOutcome(run) {
  const steps = run.plan?.steps || [];
  const { landed } = runCounts(run);
  if (!run.stopped) {
    const said = run.summary && run.summary !== run.plan?.intro ? run.summary : "";
    if (!said) return landed.length ? "Done." : "Done. Nothing needed changing.";
    return /^done\b/i.test(said) ? said : `Done: ${said}`;
  }
  const why = run.stopped === "stopped" ? ""
    : STOPPED_BECAUSE[run.stopped] || String(run.stopped).replace(/^error: /, "");
  if (!landed.length) return `Stopped before changing anything${why ? `: ${why}` : ""}.`;
  if (!steps.length) return `Stopped${why ? `: ${why}` : ""}.`;
  const done = Math.min(run.steps_done || 0, steps.length);
  const states = run.step_states || [];
  const left = [
    ...steps.slice(0, done).filter((_step, i) => states[i] === "partial").map((step) => `the rest of ${step}`),
    ...steps.slice(done),
  ];
  const list = left.length < 2 ? left.join("") : `${left.slice(0, -1).join(", ")} and ${left.at(-1)}`;
  return `Stopped after ${done} of ${steps.length} step${steps.length === 1 ? "" : "s"}${why ? `: ${why}` : ""}.`
    + `${left.length ? ` Not done: ${list}.` : ""}`;
}

module.exports = { groupThreads, pathStem, noteTitle, groupCitations, sha256Hex, slashQuery, matchSkills,
        inputsText, editorSelection, MAX_SELECTION_CHARS, MAX_EDIT_CHARS, looksLikeMath, normalizeMath,
        LIVE_RENDER_MS, quickOrDeep, metaText, stageLine, SUGGESTED_PROMPTS, titleFromQuestion, threadId,
        cancelledError, runCounts, historyText, STOPPED_BECAUSE, editOutcome };
