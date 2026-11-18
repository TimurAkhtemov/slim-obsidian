// The card's masking line, and a reopened note's masking read back from its frontmatter.
import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { piiHint, piiFromFrontmatter } = require("../../src/recorder/pii.js");

test("the line reads like a sentence, SSN never pluralised", () => {
  assert.equal(piiHint({ masked: { email: 2, ssn: 3, card: 1 }, partial: false }),
               "Masked 3 SSN, 1 card, 2 emails.");
  assert.equal(piiHint({ masked: {}, partial: false }), "");
  assert.equal(piiHint(undefined), "");
  assert.match(piiHint({ masked: {}, partial: true }), /^The password check didn't run\./);
});

test("frontmatter counts and partial come back as the server sent them", () => {
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["email:2", "ssn:1"], pii_by: "pii-1" }),
                   { masked: { email: 2, ssn: 1 }, partial: false });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: [], pii_partial: true }),
                   { masked: {}, partial: true });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["ssn:1"], pii_partial: "true" }),
                   { masked: { ssn: 1 }, partial: true });
});

test("a note with no masking keys gets no line", () => {
  assert.equal(piiFromFrontmatter({ origin: "recorded" }), null);
  assert.equal(piiFromFrontmatter(null), null);
  assert.equal(piiFromFrontmatter(undefined), null);
});

test("malformed items are skipped, never thrown on", () => {
  assert.deepEqual(
    piiFromFrontmatter({ pii_masked: ["email", "email:x", "ssn:1", 7, null, { a: 1 }, "__proto__:3"] }),
    { masked: { ssn: 1 }, partial: false });
  assert.equal(piiFromFrontmatter({ pii_masked: "email:2" }), null, "not an array: nothing to read");
  assert.deepEqual(piiFromFrontmatter({ pii_masked: "email:2", pii_partial: true }),
                   { masked: {}, partial: true });
  assert.deepEqual(piiFromFrontmatter({ pii_masked: ["email:2"], pii_partial: "false" }),
                   { masked: { email: 2 }, partial: false });
});
