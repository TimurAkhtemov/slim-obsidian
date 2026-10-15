// The notes editor's Markdown: shortcuts, and embeds, math and links surviving an edit.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { obsidianStub, srcText } from "../fakes.mjs";
import { view } from "../recorder-harness.mjs";

const require = createRequire(import.meta.url);
const { inlineMarkdown, markdownBlockShortcut, markdownFromEditable,
        stampSourceAtoms } = require("../../src/recorder/editable-markdown.js");

test("Notion-style Markdown shortcuts map to live block formatting", () => {
  assert.deepEqual(markdownBlockShortcut("### "), { tag: "h3", kind: "block" });
  assert.deepEqual(markdownBlockShortcut("- "), { tag: "ul", kind: "list" });
  assert.deepEqual(markdownBlockShortcut("1. "), { tag: "ol", kind: "list" });
  assert.equal(markdownBlockShortcut("ordinary text"), null);
});

/* --- 2026-08-26: their notes are theirs, and a summary can be retried ----------------------- */

const inlineMarkdownForTest = (node) => inlineMarkdown(node);

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
  stampSourceAtoms(root, md);
  assert.equal(markdownFromEditable(root), md);
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
  assert.equal(markdownFromEditable(root), "- a\n\t- b\n- c");
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
    assert.equal(markdownFromEditable(editor), md);
  } finally {
    obsidianStub.MarkdownRenderer.render = original;
  }
});

test("rendered nodes are stamped with the source they came from", () => {
  const source = srcText("recorder/editable-markdown.js");
  // The stamping pass and the serializer have to agree on one attribute name.
  assert.match(source, /data-slim-src/);
  assert.match(source, /function stampSourceAtoms/);
});
