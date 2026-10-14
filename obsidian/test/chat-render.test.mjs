// The copilot sidebar as drawn: streaming, stages, Markdown, sources, wikilinks.
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { find } from "./fakes.mjs";
import { PLAN, change, editView, fakeLive, makeView, renderedView,
         text } from "./copilot-harness.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { SUGGESTED_PROMPTS, sha256Hex, stageLine } = require("../src/chat-text.js");

const transcriptOf = (root) => find(root, (node) => node.cls === "slim-copilot-transcript");

test("a delta updates the streaming answer in place instead of rebuilding the view", () => {
  const { view, root, rebuilds } = renderedView();
  view.sending = true;
  view.streamingTurns = view.turns;
  view.streamingText = "Gro";
  view.render();
  const before = rebuilds();
  view.streamingText = "Grounded";
  view.renderDelta();
  assert.equal(rebuilds(), before);                       // ~40 tokens/s must not mean 40 teardowns/s
  assert.equal(text(root).textContent, "Grounded");
  view.stage = "Writing";
  view.stageSince = Date.now();
  view.renderStage();
  assert.equal(rebuilds(), before);
  assert.equal(find(root, (node) => node.cls === "slim-copilot-stage").textContent, "Writing…");
});

test("the stage line names the stage and counts the seconds of the call that runs", () => {
  assert.equal(stageLine("Writing", 1000, 1900), "Writing…");
  assert.equal(stageLine("Writing", 1000, 13_400), "Writing… 12 s");
  assert.equal(stageLine("", 1000, 5000), "Working… 4 s");
  assert.equal(stageLine("Writing", 0, 5000), "Writing…");
});

test("a new stage or a new step starts the count again; the same stage sent again does not", async () => {
  const realNow = Date.now;
  let now = 1_000_000;
  Date.now = () => now;
  try {
    const { live, editor } = fakeLive();
    const seen = [];
    const view = editView({ live, editor, events: async (_body, handlers) => {
      const line = () => stageLine(view.stage, view.stageSince, Date.now());
      now += 2000;
      seen.push(line());                                   // the gate: still Starting
      handlers.stage({ stage: "Thinking deeply" });
      now += 5000;
      handlers.stage({ stage: "Thinking deeply" });        // a heartbeat: the same call
      now += 4000;
      seen.push(line());
      handlers.edit({ before_ref: "a".repeat(32), hash_before: sha256Hex("# A\n\nold\n") });
      handlers.plan(PLAN);
      handlers.step({ index: 1, state: "doing" });
      handlers.stage({ stage: "Writing" });
      now += 3000;
      seen.push(line());
      handlers.change(change("c1", 1));
      handlers.step({ index: 1, state: "done" });
      handlers.step({ index: 2, state: "doing" });
      handlers.stage({ stage: "Writing" });
      now += 7000;
      seen.push(line());
      handlers.turn({ answer: PLAN.intro, mode: "deep", citations: [], edit_run: {
        plan: PLAN, steps_done: 2, changes: [change("c1", 1)], stopped: null } });
    } });
    view.mode = "deep";
    await view.send("Make the revisions");
    assert.deepEqual(seen, ["Starting… 2 s", "Thinking deeply… 9 s", "Writing… 3 s", "Writing… 7 s"]);
    assert.equal(view.stageClock, null);                    // the count stops with the turn
  } finally {
    Date.now = realNow;
  }
});

test("the stage line stays under the steps while the run's calls go on", () => {
  const { view, root } = renderedView();
  view.sending = true;
  view.streamingTurns = view.turns;
  view.stage = "Writing";
  view.stageSince = Date.now();
  view.editRun = { plan: PLAN, steps: ["doing"], changes: [], started: Date.now() };
  view.render();
  view.renderEdit();
  const transcript = transcriptOf(root);
  const stage = find(root, (node) => node.cls === "slim-copilot-stage");
  assert.equal(stage?.textContent, "Writing…");
  assert.equal(transcript.children.at(-1), stage);           // after the run's steps
});

test("the transcript follows the streaming answer when they were reading the bottom", () => {
  const { view, root } = renderedView();
  view.sending = true;
  view.streamingTurns = view.turns;
  view.streamingText = "Gro";
  view.render();
  const transcript = transcriptOf(root);
  transcript.scrollHeight = 500;                          // the answer grew past the pane
  transcript.scrollTop = 400;                             // and they are reading its end
  view.streamingText = "Grounded, at length";
  view.renderDelta();
  assert.equal(transcript.scrollTop, 500);
  transcript.scrollTop = 100;                             // they scrolled up to read something
  transcript.scrollHeight = 900;
  view.streamingText = "Grounded, at even more length";
  view.renderDelta();
  assert.equal(transcript.scrollTop, 100);                // and is not yanked back down
});

test("the composer keeps focus across a full render", () => {
  const { view, root } = renderedView();
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  input.onfocus();
  view.render();
  assert.equal(find(root, (node) => node.tag === "textarea").focused, true);
});

test("a note group they expanded by hand stays expanded across renders", () => {
  const { view, root } = renderedView();
  view.showThreads = true;
  view.threads = [
    { id: "a", title: "A", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-02-01" },
    { id: "b", title: "B", source_id: "s2", source_path: "Notes/two.md", updated_at: "2026-01-01" },
  ];
  view.render();
  const other = () => find(root, (node) => node.tag === "details"
    && node.children.some((child) => child.tag === "summary" && child.attrs.title === "Notes/two.md"));
  assert.equal(other().children.find((child) => child.tag === "summary").textContent, "two");
  assert.equal(other().open, false);
  other().children.find((child) => child.tag === "summary").onclick();
  view.render();
  assert.equal(other().open, true);
});

test("a turn shows when its answer was cut off; a bracket in prose is plain text", () => {
  // Sources come from the server's follow-up call, never from markers, so [9] in prose is
  // text and nothing flags it. Truncation is still their to know about.
  const { view, root } = renderedView();
  view.turns = [{ role: "you", text: "Q" }, {
    role: "slim", text: "Half [1] and [9]",
    turn: { citations: [], verification: { truncated: true } },
  }];
  view.render();
  const notes = [];
  find(root, (node) => { if (node.cls === "slim-verification") notes.push(node.textContent); return false; });
  assert.equal(notes.length, 1);
  assert.match(notes[0], /cut off/i);
});

test("the header shows the note title and keeps the path as a tooltip", () => {
  const { view, root } = renderedView();
  view.context.source = { id: "s1", path: "Notes/nlp/Introduction to NLP.md", title: "Introduction to NLP" };
  view.render();
  const note = find(root, (node) => node.cls === "slim-copilot-note");
  assert.equal(note.textContent, "Introduction to NLP");
  assert.equal(note.attrs.title, "Notes/nlp/Introduction to NLP.md");
  assert.equal(find(root, (node) => node.cls === "slim-copilot-path"), null);
  assert.equal(find(root, (node) => node.cls === "slim-copilot-eyebrow").textContent, "SLIM");
});

test("nearby notes fold under the title, closed until they open them, and stay open across renders", () => {
  const { view, root } = renderedView();
  view.context.related = [
    { source_id: "s2", path: "Notes/nlp/Word embeddings.md", title: "Word embeddings", scope: "course" },
    { source_id: "s3", path: "Notes/nlp/Tokens.md", title: "Tokens", scope: "folder" },
  ];
  view.render();
  const chip = () => find(root, (node) => node.cls === "slim-copilot-context");
  assert.equal(chip().open, false);
  const summary = chip().children.find((child) => child.tag === "summary");
  assert.deepEqual(summary.children.map((child) => child.textContent), ["One", "2 nearby"]);
  assert.equal(summary.children[0].attrs.title, "Notes/one.md");
  summary.onclick();
  view.render();
  assert.equal(chip().open, true);
  view.context.related = [];
  view.render();
  assert.equal(chip(), null);                                       // no chip without nearby notes
  assert.equal(find(root, (node) => node.cls === "slim-copilot-note").textContent, "One");
});

test("the history button swaps the chat list into the conversation's slot; opening a chat swaps back", async () => {
  const { view, root } = renderedView({
    getJSON: async () => ({ id: "a", source_id: "s1", source_path: "Notes/one.md", turns: [], reasoning_mode: "quick" }),
  });
  view.threads = [{ id: "a", title: "A", source_id: "s1", source_path: "Notes/one.md", updated_at: "2026-02-01" }];
  view.render();
  assert.notEqual(find(root, (node) => node.cls === "slim-copilot-transcript"), null);
  assert.equal(find(root, (node) => node.cls === "slim-copilot-threads"), null);
  find(root, (node) => node.attrs["aria-label"] === "Chats").onclick();
  assert.equal(find(root, (node) => node.cls === "slim-copilot-transcript"), null);
  const list = find(root, (node) => node.cls === "slim-copilot-threads");
  assert.notEqual(list, null);
  // A delta while the list shows must not rebuild the view per token.
  view.sending = true; view.streamingTurns = view.turns; view.streamingText = "x";
  const built = view.render.bind(view); let rebuilt = 0; view.render = () => { rebuilt += 1; built(); };
  view.renderDelta(); view.renderStage();
  assert.equal(rebuilt, 0);
  view.sending = false; view.render = built;
  find(root, (node) => node.cls === "slim-chat-open").onclick();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(view.showThreads, false);
  assert.notEqual(find(root, (node) => node.cls === "slim-copilot-transcript"), null);
});

test("an empty thread offers suggested prompts, and one click sends it", () => {
  const { view, root } = renderedView();
  const sent = [];
  view.send = async (text) => { sent.push(text); };
  view.render();
  const chips = [];
  find(root, (node) => { if (node.cls === "slim-suggest") chips.push(node); return false; });
  assert.deepEqual(chips.map((chip) => chip.textContent), SUGGESTED_PROMPTS);
  chips[1].onclick();
  assert.deepEqual(sent, ["Quiz me on this note"]);
  view.turns = [{ role: "you", text: "Q" }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-suggest"), null);
});

test("their turn is a bubble on the right and the composer starts as one line", () => {
  const styles = fs.readFileSync(path.join(HERE, "..", "styles.css"), "utf8");
  assert.match(styles, /\.slim-message\.is-you\s*\{[^}]*align-self:\s*flex-end/s);
  assert.match(styles, /\.slim-message\.is-you\s*\{[^}]*border-radius/s);
  assert.match(styles, /\.slim-copilot-input\s*\{[^}]*min-height:\s*36px/s);
  const { view, root } = renderedView();
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  assert.equal(input.placeholder, "Ask SLIM, or / for skills");
});

test("nearby notes and chat groups read as titles, with the path on hover", () => {
  const { view, root } = renderedView();
  view.context.related = [
    { source_id: "s2", path: "Notes/nlp/Word embeddings.md", title: "Word embeddings", scope: "course" },
  ];
  view.threads = [
    { id: "b", title: "B", source_id: "s2", source_path: "Notes/nlp/Word embeddings.md", updated_at: "2026-01-01" },
  ];
  view.render();
  const nearby = find(root, (node) => node.cls === "slim-related-note");
  assert.equal(nearby.attrs.title, "Notes/nlp/Word embeddings.md");
  assert.deepEqual(nearby.children.map((child) => child.textContent), ["Word embeddings", "course"]);
  view.showThreads = true;
  view.render();
  const summary = find(root, (node) => node.tag === "summary" && node.attrs.title === "Notes/nlp/Word embeddings.md");
  assert.equal(summary.textContent, "Word embeddings");
});

test("an answer is rendered as Markdown while a question stays plain text", () => {
  const { view, root } = renderedView();
  view.turns = [{ role: "you", text: "**Q**" }, { role: "slim", text: "**A**", turn: { citations: [] } }];
  view.render();
  const question = find(root, (node) => node.cls.includes("slim-message-text") && node.parent.cls.includes("is-you"));
  assert.equal(question.textContent, "**Q**");
  assert.deepEqual(question.children, []);
  const answer = find(root, (node) => node.cls.includes("slim-message-text") && node.parent.cls.includes("is-slim"));
  assert.equal(answer.textContent, "");                             // nothing set as raw text
  assert.deepEqual(answer.children.map((child) => [child.tag, child.textContent]), [["p", "**A**"]]);
});

const citedTurn = () => {
  const intro = { source_id: "a", path: "Notes/nlp/Introduction to NLP.md", title: "Introduction to NLP" };
  const embeddings = { source_id: "b", path: "Notes/nlp/Word embeddings.md", title: "Word embeddings" };
  return { role: "slim", text: "A [1] B [6] C [7]", turn: { citations: [
    ...[1, 2, 3, 4, 5].map((n) => ({ n, ...intro })), { n: 6, ...embeddings }, { n: 7, ...intro },
  ] } };
};

test("sources are one chip per note under the answer, and open that note", () => {
  const { view, root } = renderedView();
  const opened = [];
  view.openNote = async (path) => { opened.push(path); };
  view.turns = [citedTurn()];
  view.render();
  const rows = [];
  find(root, (node) => { if (node.cls === "slim-source") rows.push(node); return false; });
  assert.equal(rows.length, 2);                                     // seven fragments, two notes
  assert.deepEqual(rows.map((row) => row.textContent), ["Introduction to NLP", "Word embeddings"]);
  assert.equal(rows[0].attrs.title, "Notes/nlp/Introduction to NLP.md");
  assert.equal(find(root, (node) => node.cls === "slim-sources").children[0].textContent, "From your notes");
  rows[0].onclick();
  assert.deepEqual(opened, ["Notes/nlp/Introduction to NLP.md"]);
  view.turns = [{ role: "slim", text: "General knowledge.", turn: { citations: [] } }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-sources"), null);   // nothing when no note was used
});

test("the rendered-answer chain settles without a throw on a DOM that has no text nodes", async () => {
  // Marker linking and the re-scroll run after the Markdown promise. Under this fake DOM
  // there are no childNodes and no document; a walk without guards rejects unhandled here
  // and, in Obsidian, would leave the answer rendered but the markers dead.
  const rejections = [];
  const onRejection = (error) => rejections.push(error);
  process.on("unhandledRejection", onRejection);
  try {
    const { view } = renderedView();
    view.turns = [citedTurn()];
    view.render();
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.off("unhandledRejection", onRejection);
  }
  assert.deepEqual(rejections, []);
});

test("completed chat text can be selected and copied in Obsidian's ItemView", () => {
  // Obsidian gives custom ItemViews `user-select: none`. Textareas remain editable because
  // Chromium treats form controls specially, but ordinary question and answer text inherits
  // the prohibition unless the plugin explicitly restores selection on the transcript.
  const styles = fs.readFileSync(path.join(HERE, "..", "styles.css"), "utf8");
  assert.match(styles, /\.slim-copilot-transcript\s*\{[^}]*user-select:\s*text\s*;/s);
});

// A [[wikilink]] in an answer is rendered by MarkdownRenderer as <a class="internal-link">, and
// Obsidian wires clicks on those only inside its own note views — in this sidebar it was dead.
function linkView(existing) {
  const opened = [];
  const view = makeView({
    app: {
      metadataCache: {
        getFirstLinkpathDest(linkpath, source) {
          return existing.includes(linkpath) ? { path: `Notes/${linkpath}.md`, source } : null;
        },
      },
      workspace: { openLinkText(...args) { opened.push(args); } },
    },
  });
  view.context = { source: { path: "Notes/open.md" } };
  view.render = () => {};
  return { view, opened };
}

function linkClick(href, mods = {}) {
  const link = { getAttribute: (name) => (name === "data-href" ? href : null) };
  const event = {
    ...mods,
    target: { closest: (sel) => (sel === "a.internal-link" ? link : null) },
    prevented: false,
    preventDefault() { this.prevented = true; },
  };
  return event;
}

test("a wikilink in an answer opens the note it names", () => {
  const { view, opened } = linkView(["Bayes rule"]);
  const event = linkClick("Bayes rule#Evidence");
  view.onLinkClick(event);
  assert.equal(event.prevented, true);
  assert.deepEqual(opened, [["Bayes rule#Evidence", "Notes/open.md", false]]);
});

test("cmd-click opens the linked note in a new tab", () => {
  const { view, opened } = linkView(["Bayes rule"]);
  view.onLinkClick(linkClick("Bayes rule", { metaKey: true }));
  assert.deepEqual(opened, [["Bayes rule", "Notes/open.md", true]]);
});

test("a wikilink to a note that does not exist never creates one", () => {
  // openLinkText CREATES a missing target — an empty phantom the next ingest indexes.
  const { view, opened } = linkView([]);
  const event = linkClick("Invented title");
  view.onLinkClick(event);
  assert.equal(event.prevented, true);
  assert.deepEqual(opened, []);
  assert.match(view.error, /Note not found: Invented title/);
});

test("a click that is not on a wikilink is left alone", () => {
  const { view, opened } = linkView(["Bayes rule"]);
  const event = { target: { closest: () => null }, preventDefault() { this.prevented = true; } };
  view.onLinkClick(event);
  assert.equal(event.prevented, undefined);
  assert.deepEqual(opened, []);
});
