// The card's masking line, and a reopened note's masking read back from its frontmatter.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { piiHint, piiFromFrontmatter } = require("../../src/recorder/pii.js");

test("the line reads like a sentence, SSN never pluralised", () => {
  assert.equal(piiHint({ masked: { email: 2, ssn: 3, card: 1 }, partial: [] }),
               "Masked 3 SSN, 1 card, 2 emails.");
  assert.equal(piiHint({ masked: {}, partial: [] }), "");
  assert.equal(piiHint(undefined), "");
});

test("every class reads right at one and at two", () => {
  const want = { ssn: ["1 SSN", "2 SSN"], card: ["1 card", "2 cards"],
                 account: ["1 account", "2 accounts"], password: ["1 password", "2 passwords"],
                 email: ["1 email", "2 emails"], address: ["1 address", "2 addresses"],
                 phone: ["1 phone", "2 phones"] };
  for (const [cls, [one, two]] of Object.entries(want)) {
    assert.equal(piiHint({ masked: { [cls]: 1 }, partial: [] }), `Masked ${one}.`);
    assert.equal(piiHint({ masked: { [cls]: 2 }, partial: [] }), `Masked ${two}.`);
  }
});

test("each unfinished reason says what is left, and only that", () => {
  const password = "The password check didn't run. Run `slim mask` on this note.";
  const audio = "Some secrets could not be silenced in the audio.";
  assert.equal(piiHint({ masked: {}, partial: ["password"] }), password);
  assert.equal(piiHint({ masked: {}, partial: ["audio"] }), audio);
  assert.equal(piiHint({ masked: { ssn: 1 }, partial: ["audio", "password"] }),
               `Masked 1 SSN. ${password} ${audio}`);
  assert.equal(piiHint({ masked: {}, partial: ["bogus"] }), "");
});

test("frontmatter counts and partial come back as the server sent them", () => {
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["email:2", "ssn:1"], pii_by: "pii-1" }),
                   { masked: { email: 2, ssn: 1 }, partial: [] });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["ssn:1"], pii_partial: ["audio", "password"] }),
                   { masked: { ssn: 1 }, partial: ["audio", "password"] });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: [], pii_partial: ["password", "bogus", 3] }),
                   { masked: {}, partial: ["password"] });
});

test("a legacy pii_partial: true was the password pass", () => {
  assert.deepEqual(piiFromFrontmatter({ pii_masked: [], pii_partial: true }),
                   { masked: {}, partial: ["password"] });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["ssn:1"], pii_partial: "true" }),
                   { masked: { ssn: 1 }, partial: ["password"] });
});

test("a note with no masking keys gets no line", () => {
  assert.equal(piiFromFrontmatter({ origin: "recorded" }), null);
  assert.equal(piiFromFrontmatter(null), null);
  assert.equal(piiFromFrontmatter(undefined), null);
});

test("malformed items are skipped, never thrown on", () => {
  assert.deepEqual(
    piiFromFrontmatter({ pii_masked: ["email", "email:x", "ssn:1", 7, null, { a: 1 }, "__proto__:3"] }),
    { masked: { ssn: 1 }, partial: [] });
  assert.equal(piiFromFrontmatter({ pii_masked: "email:2" }), null, "not an array: nothing to read");
  assert.deepEqual(piiFromFrontmatter({ pii_masked: "email:2", pii_partial: true }),
                   { masked: {}, partial: ["password"] });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["email:2"], pii_partial: "false" }),
                   { masked: { email: 2 }, partial: [] });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["email:2"], pii_partial: { audio: 1 } }),
                   { masked: { email: 2 }, partial: [] });
  assert.equal(piiFromFrontmatter({ pii_partial: ["bogus"] }), null);
});
