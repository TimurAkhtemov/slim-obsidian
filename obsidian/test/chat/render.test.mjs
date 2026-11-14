// The copilot sidebar as drawn: streaming, stages, Markdown, sources, wikilinks.
import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { fakeEl, find } from "../fakes.mjs";
import { PLAN, change, editView, fakeLive, makeView, renderedView,
         text } from "../copilot-harness.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { sha256Hex, stageLine, toolSummary } = require("../../src/chat/text.js");

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
      seen.push(line());                                   // before the first stage: still Starting
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
  const meta = find(root, (node) => node.cls === "slim-edit-meta");
  assert.equal(meta?.children[0].textContent, "Medium · 0 s");   // the effort it runs at
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

const composerOf = (root) => find(root, (node) => node.cls === "slim-copilot-composer");

test("the header is the chat's title; the open note is a chip in the composer with its path as a tooltip", () => {
  const { view, root } = renderedView();
  view.openNote = { path: "Notes/nlp/Introduction to NLP.md", id: "s1" };
  view.context.source = { id: "s1", path: "Notes/nlp/Introduction to NLP.md", title: "Introduction to NLP" };
  view.render();
  const title = find(root, (node) => node.cls === "slim-copilot-title");
  assert.equal(title.textContent, "New chat");                      // no chat yet
  const note = find(composerOf(root), (node) => node.cls === "slim-copilot-note");
  assert.equal(note.textContent, "Introduction to NLP");
  assert.equal(note.attrs.title, "Notes/nlp/Introduction to NLP.md");
  view.activeThread = { id: "chat-1", title: "What is a token?" };
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-copilot-title").textContent, "What is a token?");
  view.openNote = null;
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-copilot-note"), null);   // no note open: no chip
});

test("nearby notes open from the note chip's count, closed until they open them, and stay open across renders", () => {
  const { view, root } = renderedView();
  view.context.related = [
    { source_id: "s2", path: "Notes/nlp/Word embeddings.md", title: "Word embeddings", scope: "course" },
    { source_id: "s3", path: "Notes/nlp/Tokens.md", title: "Tokens", scope: "folder" },
  ];
  view.render();
  const list = () => find(root, (node) => node.cls === "slim-copilot-nearby");
  const count = () => find(root, (node) => node.cls?.includes("slim-copilot-nearby-count"));
  assert.equal(list().hidden, true);
  assert.equal(count().textContent, "2 nearby");
  const chip = find(root, (node) => node.cls?.includes("slim-copilot-note-chip"));
  assert.deepEqual(chip.children.map((child) => child.textContent), ["", "One", "\u2715"]);   // icon, title, ✕
  count().onclick();
  assert.equal(list().hidden, false);
  assert.equal(count().attrs["aria-expanded"], "true");
  view.render();
  assert.equal(list().hidden, false);
  view.onRootClick({ target: { closest: () => null } });           // a click elsewhere closes it
  assert.equal(list().hidden, true);
  view.context.related = [];
  view.render();
  assert.equal(list(), null);                                       // nothing without nearby notes
  assert.equal(count(), null);
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

test("a new chat shows SLIM, one line and three starters; a starter fills the composer and sends nothing", () => {
  const { view, root } = renderedView();
  const sent = [];
  view.send = async (text) => { sent.push(text); };
  view.render();
  const empty = find(root, (node) => node.cls === "slim-empty");
  assert.equal(find(empty, (node) => node.cls === "slim-empty-name").textContent, "SLIM");
  assert.equal(find(empty, (node) => node.cls === "slim-empty-line").textContent,
               "Ask about your notes, or have SLIM edit the open one.");
  const chips = () => { const out = []; find(root, (node) => { if (node.cls === "slim-suggest") out.push(node); return false; }); return out; };
  assert.deepEqual(chips().map((chip) => chip.textContent),
                   ["Summarize this note", "What's in my notes about \u2026", "Tidy this note"]);
  chips()[1].onclick();
  assert.deepEqual(sent, []);
  const input = find(root, (node) => node.tag === "textarea");
  assert.equal(input.value, "What's in my notes about ");
  assert.equal(input.focused, true);
  assert.equal(input.selectionStart, "What's in my notes about ".length);   // the caret after "about "
  view.draft = "";
  view.openNote = null;                                             // nothing to tidy
  view.render();
  assert.deepEqual(chips().map((chip) => chip.textContent),
                   ["Summarize this note", "What's in my notes about \u2026", "Find a note about \u2026"]);
  chips()[2].onclick();
  assert.equal(view.draft, "Find a note about ");
  view.turns = [{ role: "you", text: "Q" }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-suggest"), null);
  assert.equal(find(root, (node) => node.cls === "slim-empty"), null);
});

test("their turn is a bubble on the right and the composer starts as one line", () => {
  const styles = fs.readFileSync(path.join(HERE, "..", "..", "styles.css"), "utf8");
  assert.match(styles, /\.slim-message\.is-you\s*\{[^}]*align-self:\s*flex-end/s);
  assert.match(styles, /\.slim-message\.is-you\s*\{[^}]*border-radius/s);
  assert.match(styles, /\.slim-copilot-input\s*\{[^}]*min-height:\s*36px/s);
  const { view, root } = renderedView();
  view.render();
  const input = find(root, (node) => node.tag === "textarea");
  assert.equal(input.placeholder, "Ask anything \u00B7 @ to add context");
});

test("nearby notes and history rows read as titles, with the path on hover", () => {
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
  const faint = find(root, (node) => node.cls === "slim-chat-note");
  assert.equal(faint.attrs.title, "Notes/nlp/Word embeddings.md");
  assert.equal(faint.textContent.trim(), "Word embeddings");
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

test("sources fold into one row under the answer that opens to a title per note, each opening it", () => {
  const { view, root } = renderedView();
  const opened = [];
  view.openFile = async (path) => { opened.push(path); };
  view.turns = [citedTurn()];
  view.render();
  const toggle = () => find(root, (node) => node.cls === "slim-sources-toggle");
  const list = () => find(root, (node) => node.cls === "slim-sources-list");
  assert.equal(find(toggle(), (node) => node.cls === "slim-sources-count").textContent, "2 sources");   // seven fragments, two notes
  assert.equal(find(toggle(), (node) => node.cls === "slim-sources-stack").children.length, 2);        // a file icon per note
  assert.equal(list().hidden, true);                                // closed until clicked
  toggle().onclick();
  assert.equal(list().hidden, false);
  assert.equal(toggle().attrs["aria-expanded"], "true");
  const rows = list().children;
  assert.deepEqual(rows.map((row) => row.children.map((part) => part.textContent)),
                   [["Introduction to NLP", "Notes/nlp"], ["Word embeddings", "Notes/nlp"]]);
  assert.equal(rows[0].attrs.title, "Notes/nlp/Introduction to NLP.md");
  rows[0].onclick();
  assert.deepEqual(opened, ["Notes/nlp/Introduction to NLP.md"]);
  view.render();
  assert.equal(list().hidden, false);                               // stays open across renders
  view.turns = [{ role: "slim", text: "General knowledge.", turn: { citations: [] } }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-sources"), null);   // nothing when no note was used
  view.turns = [{ role: "slim", text: "One note.", turn: { citations: [{ n: 1, path: "Top.md", title: "Top" }] } }];
  view.render();
  assert.equal(find(root, (node) => node.cls === "slim-sources-count").textContent, "1 source");
  assert.deepEqual(list().children[0].children.map((part) => part.textContent), ["Top"]);   // at the root: no folder
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
  const styles = fs.readFileSync(path.join(HERE, "..", "..", "styles.css"), "utf8");
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
  view.openNote = { path: "Notes/open.md", id: null };
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

// ---- every lookup a line in the chat, as Pi shows each tool call --------------------------------

const toolLines = (root) => {
  const rows = [];
  find(root, (node) => { if (node.cls?.split(" ").includes("slim-tool")) rows.push(node); return false; });
  return rows.map((row) => [row.cls.includes("is-refused"), row.children.at(-1).textContent]);
};

test("each lookup shows as a muted line above the streaming answer as it arrives", async () => {
  const { view, root } = renderedView({
    streamCopilot: async (_body, handlers) => {
      handlers.tool({ name: "search", label: 'Searched the vault for "comp"', refused: false });
      assert.deepEqual(toolLines(root), [[false, 'Searched the vault for "comp"']]);
      handlers.delta({ text: "Your offer" });
      handlers.tool({ name: "read_note", label: "Read nope — refused", refused: true });
      assert.deepEqual(toolLines(root), [[false, 'Searched the vault for "comp"'], [true, "Read nope — refused"]]);
      const live = find(root, (node) => node.cls?.includes("is-streaming"));
      assert.ok(live.children[0].cls.includes("slim-tools"));          // above the answer
      handlers.turn({ answer: "Your offer is fixed.", mode: "quick", citations: [],
                      tools: [{ name: "search", label: 'Searched the vault for "comp"', refused: false },
                              { name: "read_note", label: "Read nope — refused", refused: true }] });
    },
  });
  view.render();
  await view.send("what is the total comp?");
  assert.deepEqual(view.liveTools, []);
  assert.deepEqual(toolLines(root), [[false, 'Searched the vault for "comp"'], [true, "Read nope — refused"]]);
});

test("a saved turn shows its lookups again on reload, folded into one row; a turn with none shows no row", () => {
  const { view, root } = renderedView();
  view.turns = [{ role: "slim", text: "Fixed.", turn: { citations: [], tools: [
    { name: "read_note", label: "Read Screen notes", refused: false }] } }];
  view.render();
  const message = find(root, (node) => node.cls === "slim-message is-slim");
  assert.equal(message.children[0].cls, "slim-tool-group");          // above the answer
  assert.equal(find(message, (node) => node.cls === "slim-tool-summary-text").textContent, "Read 1");
  assert.deepEqual(toolLines(root), [[false, "Read Screen notes"]]);
  view.turns = [{ role: "slim", text: "Hi.", turn: { citations: [] } }];
  view.render();
  assert.equal(find(root, (node) => node.cls?.includes("slim-tools")), null);
  assert.equal(find(root, (node) => node.cls === "slim-tool-group"), null);
});

test("after the answer the lookups fold into one summary row that opens on click, live and on reload", async () => {
  const tools = [{ name: "search", label: 'Searched the vault for "bees"', refused: false },
                 { name: "search", label: 'Searched the vault for "hives"', refused: false },
                 { name: "read_note", label: "Read Bees", refused: false },
                 { name: "list_folder", label: "Listed Notes", refused: false },
                 { name: "read_note", label: "Read nope — refused", refused: true }];
  const { view, root } = renderedView({
    streamCopilot: async (_body, handlers) => {
      for (const tool of tools) handlers.tool(tool);
      assert.equal(find(root, (node) => node.cls === "slim-tool-group"), null);   // live: a row each, no fold
      assert.equal(toolLines(root).length, 5);
      handlers.turn({ answer: "Bees.", citations: [], tools });
    },
  });
  view.render();
  await view.send("bees?");
  const summary = () => find(root, (node) => node.cls === "slim-tool-summary");
  const rows = () => find(root, (node) => node.cls === "slim-tool-group").children[1];
  assert.equal(find(summary(), (node) => node.cls === "slim-tool-summary-text").textContent,
               "Searched 2 \u00B7 Read 1 \u00B7 Listed 1 \u00B7 1 refused");
  assert.equal(rows().hidden, true);
  assert.equal(summary().attrs["aria-expanded"], "false");
  summary().onclick();
  assert.equal(rows().hidden, false);
  assert.equal(summary().attrs["aria-expanded"], "true");
  assert.deepEqual(toolLines(root).at(-1), [true, "Read nope — refused"]);       // refused: shown faint
  view.render();
  assert.equal(rows().hidden, false);                               // stays open across renders
  summary().onclick();
  assert.equal(rows().hidden, true);
  // Reloaded from the saved chat: the same fold.
  const saved = structuredClone(view.turns);
  view.turns = saved;
  view.render();
  assert.equal(rows().hidden, true);
  assert.equal(find(summary(), (node) => node.cls === "slim-tool-summary-text").textContent,
               "Searched 2 \u00B7 Read 1 \u00B7 Listed 1 \u00B7 1 refused");
  assert.equal(toolSummary([{ name: "read_note", refused: true }]), "1 refused");
});

test("lookups before an edit stay above the run's steps", async () => {
  const { live, editor } = fakeLive();
  const view = editView({ live, editor, events: async (_body, handlers) => {
    handlers.tool({ name: "read_note", label: "Read Bee facts", refused: false });
    handlers.edit({ before_ref: "a".repeat(32), hash_before: sha256Hex("# A\n\nold\n") });
    handlers.plan(PLAN);
    const root = view.contentEl;
    view.render();
    assert.deepEqual(toolLines(root), [[false, "Read Bee facts"]]);
    handlers.turn({ answer: PLAN.intro, mode: "quick", citations: [], edit_run: {},
                    tools: [{ name: "read_note", label: "Read Bee facts", refused: false }] });
  } });
  view.contentEl = fakeEl("div");
  delete view.render;                                             // editView stubs it; draw for real
  await view.send("add the bee facts");
  assert.equal(view.turns.at(-1).turn.tools[0].label, "Read Bee facts");
});

// ---- the conversation's details: the note a question was on, motion, the way back down -----------

test("the note a question was asked on shows faint above its bubble, only when it changed", () => {
  const { view, root } = renderedView();
  const answer = { role: "slim", text: "A", turn: { citations: [] } };
  view.turns = [{ role: "you", text: "a", note: "Notes/one.md" }, answer,
                { role: "you", text: "b", note: "Notes/one.md" }, { ...answer },
                { role: "you", text: "c", note: "Notes/two.md" }, { ...answer },
                { role: "you", text: "d" }, { ...answer }];
  view.render();
  const transcript = transcriptOf(root);
  const contexts = transcript.children.filter((child) => child.cls === "slim-message-context");
  assert.deepEqual(contexts.map((chip) => [chip.children.at(-1).textContent, chip.attrs.title]),
                   [["one", "Notes/one.md"], ["two", "Notes/two.md"]]);
  const next = (chip) => transcript.children[transcript.children.indexOf(chip) + 1];
  assert.deepEqual(contexts.map((chip) => [next(chip).cls, next(chip).children.at(-1).textContent]),
                   [["slim-message is-you", "a"], ["slim-message is-you", "c"]]);
});

test("a message fades in the first time it is drawn; a redraw, or the answer taking the stream's place, does not", async () => {
  const fresh = (root) => {
    const out = [];
    find(root, (node) => { if (node.cls?.startsWith("slim-message ") && "data-new" in node.attrs) out.push(node.cls); return false; });
    return out;
  };
  let during = null;
  const { view, root } = renderedView({
    streamCopilot: async (_body, handlers) => {
      during = fresh(root);
      view.render();
      assert.deepEqual(fresh(root), []);                            // drawn already: no second fade
      handlers.turn({ answer: "Hi.", citations: [] });
    },
  });
  view.render();
  await view.send("hello");
  assert.deepEqual(during, ["slim-message is-you", "slim-message is-slim is-streaming"]);
  assert.deepEqual(fresh(root), []);                                // the answer replaces the stream in place
  view.turns = [{ role: "you", text: "q" }];
  view.render();
  assert.deepEqual(fresh(root), ["slim-message is-you"]);
  view.render();
  assert.deepEqual(fresh(root), []);
});

test("scrolled up while an answer streams, a ↓ button appears and brings them back down", () => {
  const { view, root } = renderedView();
  view.sending = true;
  view.streamingTurns = view.turns;
  view.streamingText = "Gro";
  view.render();
  const transcript = transcriptOf(root);
  const jump = find(root, (node) => node.cls?.includes("slim-jump"));
  assert.equal(jump.hidden, true);
  transcript.scrollHeight = 900;
  transcript.scrollTop = 100;                                       // reading further up
  view.streamingText = "Grounded";
  view.renderDelta();
  assert.equal(transcript.scrollTop, 100);
  assert.equal(jump.hidden, false);
  jump.onclick();
  assert.equal(transcript.scrollTop, 900);
  assert.equal(jump.hidden, true);
  transcript.scrollTop = 0;
  transcript.onscroll();
  assert.equal(jump.hidden, false);
});
