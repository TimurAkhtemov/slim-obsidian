// The copilot's pure helpers: threads, math, citations, titles, slash commands, the selection.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import "../copilot-harness.mjs";

const require = createRequire(import.meta.url);
const { editorSelection, groupCitations, relativeDate, threadRows, inputsText, matchSkills, normalizeMath,
        slashQuery } = require("../../src/chat/text.js");

test("thread rows are one flat list, newest first", () => {
  const threads = [
    { id: "old", source_path: "Notes/one.md", updated_at: "2026-01-01" },
    { id: "new", source_path: "Notes/one.md", updated_at: "2026-02-01" },
    { id: "other", source_path: "Notes/two.md", updated_at: "2026-03-01" },
    { id: "none", source_path: null, updated_at: "2026-02-15" },
  ];
  assert.deepEqual(threadRows(threads).map((item) => item.id), ["other", "none", "new", "old"]);
  assert.deepEqual(threads.map((item) => item.id), ["old", "new", "other", "none"]);   // not sorted in place
});

// --- the redesign: rendered Markdown, titles not paths, one source per note ------------------

test("LaTeX delimiters qwen writes become the dollar forms Obsidian renders", () => {
  assert.equal(normalizeMath("Let \\(x\\) be \\[ y = 2 \\]."), "Let $x$ be $$ y = 2 $$.");
  assert.equal(normalizeMath("multi \\[\na\n\\] line"), "multi $$\na\n$$ line");
  assert.equal(normalizeMath("keep `\\[1, 3\\]` and ```\n\\(code\\)\n``` alone"),
               "keep `\\[1, 3\\]` and ```\n\\(code\\)\n``` alone");
  assert.equal(normalizeMath("already $x$ and $$y$$"), "already $x$ and $$y$$");
  assert.equal(normalizeMath(""), "");
  // Markdown-escaped brackets that are not math stay as written.
  const prose = "The list is \\[1, 3\\] and a link \\[here\\](x). Also arr\\[0\\] and A: \\( \\).";
  assert.equal(normalizeMath(prose), prose);
  assert.equal(normalizeMath("\\(\\theta\\) and \\(x\\) and \\(a+b\\)"), "$\\theta$ and $x$ and $a+b$");
});

test("citations group by note in order of first appearance", () => {
  const intro = { source_id: "a", path: "Notes/nlp/Introduction to NLP.md", title: "Introduction to NLP" };
  const groups = groupCitations([
    { n: 1, ...intro }, { n: 2, ...intro },
    { n: 3, source_id: "b", path: "Notes/nlp/Word embeddings.md", title: "" },
    { n: 4, ...intro },
  ]);
  assert.deepEqual(groups.map((group) => [group.title, group.ns]),
    [["Introduction to NLP", [1, 2, 4]], ["Word embeddings", [3]]]);
});

// ---- skills ----------------------------------------------------------------------------------

test("a slash command completes only while its first word is being typed", () => {
  const skills = [{ name: "actions" }, { name: "consolidate-my-notes" }, { name: "quiz" }];
  assert.equal(slashQuery("/con"), "con");
  assert.equal(slashQuery("/"), "");
  assert.equal(slashQuery("/con focus"), null);
  assert.equal(slashQuery("what /con"), null);
  assert.deepEqual(matchSkills(skills, "con").map((s) => s.name), ["consolidate-my-notes"]);
  assert.equal(matchSkills(skills, "").length, 3);
  assert.deepEqual(matchSkills(skills, null), []);
});

test("the inputs line says what code handed the model, including what it left out", () => {
  assert.equal(inputsText({ used: ["a", "b"], selection: false, dropped: ["c"], cut: [],
                            skipped: ["d"], missing: ["Nope"] }),
    "Read 2 notes · 1 left out (too long together) · 1 without that section · [[Nope]] not found");
  assert.equal(inputsText({ used: [], selection: true }), "Read your selection");
  assert.equal(inputsText(undefined), "");
});

test("the selection comes from the pane showing the open note, not the focused one, with where it starts", () => {
  // Where, as the editor counts: the same words can be in the note twice.
  const leaf = (path, text, from) => ({ view: { file: { path }, editor: {
    getSelection: () => text, getCursor: (which) => (which === "from" ? { line: 1, ch: 2 } : null),
    posToOffset: ({ line, ch }) => (line === 1 ? from + ch - 2 : null) } } });
  const app = { workspace: { getLeavesOfType: () => [leaf("Notes/other.md", "wrong", 3), leaf("Notes/a.md", "right", 7)] } };
  assert.deepEqual(editorSelection(app, "Notes/a.md"), { text: "right", from: 7 });
  assert.deepEqual(editorSelection(app, "Notes/none.md"), { text: "" });
  const bare = { workspace: { getLeavesOfType: () => [{ view: { file: { path: "Notes/a.md" },
    editor: { getSelection: () => "right" } } }] } };
  assert.deepEqual(editorSelection(bare, "Notes/a.md"), { text: "right" });   // an editor that cannot say where
});

test("@ starts a mention only at the start of a word, up to the caret", () => {
  const { mentionQuery, mentionMatches } = require("../../src/chat/text.js");
  assert.deepEqual(mentionQuery("ask @wee", 8), { query: "wee", from: 4 });
  assert.deepEqual(mentionQuery("@", 1), { query: "", from: 0 });
  assert.equal(mentionQuery("a@b", 3), null);
  assert.equal(mentionQuery("ask @wee now", 12), null);
  assert.deepEqual(mentionQuery("ask @wee now", 8), { query: "wee", from: 4 });
});

test("mentions match by substring, folders first, then shorter paths, at most 20", () => {
  const { mentionMatches } = require("../../src/chat/text.js");
  const hits = mentionMatches("WEEK", ["Notes/week-long.md", "week.md", "x/y/week2.md"], ["Notes/weekly", "Notes/other"]);
  assert.deepEqual(hits, [
    { kind: "folder", path: "Notes/weekly" }, { kind: "note", path: "week.md" },
    { kind: "note", path: "x/y/week2.md" }, { kind: "note", path: "Notes/week-long.md" }]);
  const many = Array.from({ length: 30 }, (_, i) => `n${i}.md`);
  assert.equal(mentionMatches("", many, []).length, 20);
});

test("a history date is Today, Yesterday, days ago within the week, then the date", () => {
  const now = new Date(2026, 10, 14, 9, 0).getTime();
  const at = (day, hour = 12) => new Date(2026, 10, day, hour, 0).toISOString();
  assert.equal(relativeDate(at(14, 1), now), "Today");
  assert.equal(relativeDate(at(13, 23), now), "Yesterday");
  assert.equal(relativeDate(at(11), now), "3 days ago");
  assert.equal(relativeDate(at(1), now), "Nov 1");
  assert.equal(relativeDate("not a date", now), "");
});
