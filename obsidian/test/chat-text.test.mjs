// The copilot's pure helpers: threads, math, citations, titles, slash commands, the selection.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import "./copilot-harness.mjs";

const require = createRequire(import.meta.url);
const { editorSelection, groupCitations, groupThreads, inputsText, matchSkills, normalizeMath,
        noteTitle, slashQuery } = require("../src/chat-text.js");

test("thread hierarchy puts the active note first and sorts chats by recency", () => {
  const groups = groupThreads([
    { id: "old", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-01-01" },
    { id: "new", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-02-01" },
    { id: "other", source_id: "s2", source_path: "Notes/two.md", updated_at: "2026-03-01" },
  ], "s1");
  assert.equal(groups[0].sourceId, "s1");
  assert.deepEqual(groups[0].threads.map((item) => item.id), ["new", "old"]);
  assert.equal(groups[0].expanded, true);
  assert.equal(groups[1].expanded, false);
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

test("a note title comes from the frontmatter, then the file name, then the path", () => {
  const file = { path: "Notes/nlp/Word embeddings.md", basename: "Word embeddings", extension: "md" };
  const folder = { path: "Notes/nlp", name: "nlp" };
  const app = {
    vault: { getAbstractFileByPath: (p) => (p === file.path ? file : p === folder.path ? folder : null) },
    metadataCache: { getFileCache: () => ({ frontmatter: { title: "Embeddings, week 3" } }) },
  };
  assert.equal(noteTitle(app, file.path), "Embeddings, week 3");
  app.metadataCache.getFileCache = () => null;
  assert.equal(noteTitle(app, file.path), "Word embeddings");
  assert.equal(noteTitle(app, folder.path), "nlp");                // a folder has no cache
  assert.equal(noteTitle({ workspace: {} }, "Notes/gone/Old note.md"), "Old note");
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
