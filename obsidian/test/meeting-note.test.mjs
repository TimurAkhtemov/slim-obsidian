// The meeting note's parts, its inline block, and the staged-file id rule.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { fakeContentEl, findClass, srcText } from "./fakes.mjs";
import "./recorder-harness.mjs";

const require = createRequire(import.meta.url);
const { embedLegacyMeetingBlock, renderPassiveMeetingBlock,
        splitMeetingNote } = require("../src/recorder/meeting-note.js");

test("the meeting parser retains the complete existing frontmatter byte-for-byte", () => {
  // `index_terms` is RETIRED (deleted 2026-08-26) and stays in this fixture on purpose: the
  // notes already on disk carry it, and the parser must return their frontmatter untouched.
  const frontmatter = [
    "---", "title: \"Design sync\"", "date: 2026-08-24", "type: meeting-note",
    "tags: [meeting-note]", "topics: [ux, recorder]", "index_terms: [tabbed object]",
    "origin: recorded", "authority: evidence", "audio: Attachments/Recorder/x.webm",
    "recorded_at: 2026-08-24T14:00:00Z", "---",
  ].join("\n");
  const note = `${frontmatter}\n\n<!-- slim:summary model=x -->\n## Summary\nAI result.\n<!-- /slim:summary -->\n\n## Notes\n- exact note\n\n## Transcript\nExact transcript.`;
  const parts = splitMeetingNote(note);
  assert.equal(parts.frontmatter, frontmatter);
  assert.equal(parts.summary, "AI result.");
  assert.equal(parts.notes, "- exact note");
  assert.equal(parts.transcript, "Exact transcript.");
});

test("the meeting block is inline and leaves ordinary Markdown after it alone", () => {
  const legacy = [
    "---", "title: Test", "origin: recorded", "---", "",
    "<!-- slim:summary prompt=x -->", "## Summary", "Summary.",
    "<!-- /slim:summary -->", "", "## Transcript", "Words.",
  ].join("\n");
  const embedded = embedLegacyMeetingBlock(legacy) + "## Follow-up\n\nEditable after the block.\n";
  const parts = splitMeetingNote(embedded);
  assert.match(embedded, /^`{4,}slim-meeting$/m);
  assert.equal(parts.frontmatter, "---\ntitle: Test\norigin: recorded\n---");
  assert.equal(parts.transcript, "Words.");
  assert.equal(parts.tail, "\n## Follow-up\n\nEditable after the block.\n");
});

test("a second recorded note remains readable while another meeting is under review", () => {
  const block = fakeContentEl();
  const source = [
    "<!-- slim:summary model=x -->", "## Summary", "Saved summary.",
    "<!-- /slim:summary -->", "", "## Notes", "- saved note", "",
    "## Transcript", "Saved transcript.",
  ].join("\n");

  renderPassiveMeetingBlock({}, {}, block, source, "Notes/other.md");

  assert.ok(findClass(block, "slim-passive-meeting"));
  assert.equal(findClass(block, "slim-rendered-markdown").text, "Saved summary.");

  const tabs = findClass(block, "slim-object-tabs");
  const notesTab = tabs.children.find((child) => child.options.text === "My notes");
  notesTab.events.click();
  assert.equal(findClass(block, "slim-rendered-markdown").text, "- saved note");
});

test("one helper decides which recording a staged file belongs to", () => {
  const source = srcText();
  const spellings = source.match(/-\\d\{3\}\$/g) || [];
  assert.ok(spellings.length <= 1,
            `the segment-suffix rule is written ${spellings.length} times in src/`);
});
