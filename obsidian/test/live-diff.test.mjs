// Word and block diffs: what a change marks, and the note as it was before.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import "./copilot-harness.mjs";

const require = createRequire(import.meta.url);
const { beforeBlocks, changeMarks, excerpt, formatChange, markable, textDiff, wordOps,
        wordTokens } = require("../src/live-diff.js");

test("words keep math, bold, code and links whole, so a mark never splits their syntax", () => {
  assert.deepEqual(wordTokens("a $x + y$ **b c** `d e` [[f g]] h"),
    ["a ", "$x + y$ ", "**b c** ", "`d e` ", "[[f g]] ", "h"]);
  assert.deepEqual(wordOps("For a vector, the softmax is:", "For a vector:"), [
    ["=", "For ", "For "], ["=", "a ", "a "], ["-", "vector, "], ["-", "the "], ["-", "softmax "],
    ["-", "is:"], ["+", "vector:"]]);
});

test("a change gets word marks only when its lines keep their kind and half their words", () => {
  assert.equal(markable("## Properties\n", "## Key Properties\n"), true);
  assert.equal(markable("For a vector $z$ of $K$ real numbers, defined as:\n", "For a vector $z$ of $K$ real numbers:\n"), true);
  assert.equal(markable("1. **Non-negativity**: positive.\n", "*   **Output**: positive.\n"), false);   // marker changed
  assert.equal(markable("one two three four\n", "five six seven eight\n"), false);                      // rewritten
  assert.equal(markable("- Syntactic ambiguity: scope.\n", "- **Syntactic ambiguity**: scope.\n"), true); // every word kept
  assert.equal(markable("$$ a + b $$\n", "$$ a + c $$\n"), false);                                     // math stays whole
  assert.equal(markable("a\nb\n", "a\n"), false);
  assert.equal(markable("", "new\n"), false);
});

test("a change that keeps every word is named for the formatting it moves", () => {
  assert.equal(formatChange("## Why is NLP Hard?\n", "### Why is NLP Hard?\n"), "Heading level 2 → 3");
  assert.equal(formatChange("- Lexical ambiguity: many senses.\n- Syntactic ambiguity: scope.\n",
                            "- **Lexical ambiguity**: many senses.\n- **Syntactic ambiguity**: scope.\n"), "Bold added");
  assert.equal(formatChange("- Open vocabulary: new words.\n", "- **Open vocabulary**: new words.\n\n"), "Bold added");
  assert.equal(formatChange("**Note**: x\n", "Note: x\n"), "Bold removed");
  assert.equal(formatChange("a\nb\n", "a\n\nb\n"), "Spacing only");
  assert.equal(formatChange("- a\n", "  - a\n"), "Indentation changed");
  assert.equal(formatChange("- a\n", "1. a\n"), "Formatting only");
  // Words that change, math that changes, and additions and removals are not formatting.
  assert.equal(formatChange("## Properties\n", "## Key Properties\n"), "");
  assert.equal(formatChange("$w_1$\n", "$w1$\n"), "");
  assert.equal(formatChange("", "new\n"), "");
  assert.equal(formatChange("old\n", ""), "");
});

// ---- the live editor's pure parts (the editor itself runs in the real-CodeMirror harness) ----

test("a change's marks: added words, removed words where they were, formatting, and folds", () => {
  const [words] = changeMarks("replace", [{ old: "The quick brown fox jumps.\n\n", now: "The quick red fox jumps.\n\n" }]);
  const now = "The quick red fox jumps.\n\n";
  assert.deepEqual(words.rail, [0, now.length]);
  assert.deepEqual(words.ins.map(([f, t]) => now.slice(f, t)), ["red"]);
  assert.deepEqual(words.dels, [[now.indexOf("red"), "brown"]]);         // struck before what replaced it
  assert.equal(words.fold, null);
  const [format] = changeMarks("replace", [{ old: "## Why\n", now: "### Why\n" }]);
  assert.deepEqual([format.formats, format.ins, format.dels], [[[0, 7]], [], []]);
  const [rewrite] = changeMarks("replace", [{ old: "one two three four\n", now: "five six seven eight\n" }]);
  assert.deepEqual(rewrite.fold, { text: "one two three four\n", moved: false });  // old text folds above the new
  const [gone, landed] = changeMarks("move", [{ old: "## A\n\nx\n\n", now: "" }, { old: "", now: "## A\n\nx\n\n" }]);
  assert.deepEqual([gone.rail, gone.fold.moved, landed.fold, landed.rail], [null, true, null, [0, 9]]);
});

test("a word fixed with edit_text and a heading renamed are marked word by word, not folded", () => {
  const [word] = changeMarks("edit_text", [{ old: "Fine.\n", now: "Good.\n" }]);
  assert.deepEqual([word.fold, word.dels, word.ins], [null, [[0, "Fine."]], [[0, 5]]]);
  const [renamed] = changeMarks("replace", [{ old: "## Middle\n", now: "## Centre\n" }]);
  assert.deepEqual([renamed.fold, renamed.dels, renamed.ins], [null, [[3, "Middle"]], [[3, 9]]]);
  // A paragraph rewritten whole still folds above the new one, and math stays whole.
  assert.equal(changeMarks("replace", [{ old: "Fine.\n", now: "Good.\n" }])[0].fold.text, "Fine.\n");
  assert.equal(changeMarks("edit_text", [{ old: "$$ a + b $$\n", now: "$$ a + c $$\n" }])[0].fold.text, "$$ a + b $$\n");
  // A rename that also changes the level keeps its words and marks only that.
  assert.equal(changeMarks("replace", [{ old: "## Middle\n", now: "### Centre\n" }])[0].fold.text, "## Middle\n");
});

test("the change between two texts is one replace from the common prefix to the common suffix", () => {
  assert.deepEqual(textDiff("abc\ndef\n", "abc\nXYZ\ndef\n"), [{ from: 4, to: 4, insert: "XYZ\n" }]);
  assert.deepEqual(textDiff("same", "same"), []);
  assert.deepEqual(textDiff("🎙️ a", "🎙️ b"), [{ from: 4, to: 5, insert: "b" }]);   // UTF-16, as the editor counts
});

test("a folded removal shows its first line without markers", () => {
  assert.equal(excerpt("\n## **Key** Ideas\n\nbody"), "Key Ideas");
  assert.equal(excerpt("- `item` one\n"), "item one");
});

test("See before marks what is gone, what changed, what moved and where something was added", () => {
  const before = "# T\n\nThe quick brown fox.\n\n## Gone\n\nM block.\n\nKeep one.\n\nKeep two.\n";
  const now = "# T\n\nThe quick red fox.\n\nA brand new paragraph.\n\nKeep one.\n\nKeep two.\n\nM block.\n";
  const blocks = beforeBlocks(before, now);
  assert.deepEqual(blocks.map((b) => b.kind), ["same", "changed", "removed", "moved", "added", "same", "same"]);
  assert.equal(blocks[1].markdown, "The quick ~~brown~~ fox.");
  assert.equal(blocks[4].markdown, "A brand new paragraph.");
  // A fence with blank lines inside is one block.
  assert.deepEqual(beforeBlocks("```\na\n\nb\n```\n", "```\na\n\nb\n```\n").map((b) => b.kind), ["same"]);
});
