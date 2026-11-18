// What the server masked in a recording, as the card says it. Counts only: the values never
// leave the server.
"use strict";

// In the order a reader weighs them.
const PII_LABELS = [["ssn", "SSN"], ["card", "card"], ["account", "account"],
                    ["password", "password"], ["email", "email"], ["address", "address"],
                    ["phone", "phone"]];
const KNOWN = new Set(PII_LABELS.map(([cls]) => cls));

// Why masking is unfinished (`pii_partial`), and what each leaves them to do. `password` is
// fixable from the text; `audio` is not, so it never tells them to rerun.
const PARTIAL_LINES = [
  ["password", "The password check didn't run. Run `slim mask` on this note."],
  ["audio", "Some secrets could not be silenced in the audio."],
];
const REASONS = new Set(PARTIAL_LINES.map(([reason]) => reason));

/* `pii_partial` as a list of known reasons: a list is filtered, a legacy `true` was the
 * password pass, anything else is none. */
function partialReasons(value) {
  if (Array.isArray(value)) return [...new Set(value.filter((r) => REASONS.has(r)))].sort();
  if (value === true || String(value).trim() === "true") return ["password"];
  return [];
}

/* The card's line: "Masked 1 SSN, 2 emails." and, per unfinished reason, what is left. "" when
 * there is nothing to say. */
function piiHint(pii) {
  if (!pii) return "";
  const counts = pii.masked || {};
  const parts = PII_LABELS.filter(([cls]) => counts[cls] > 0).map(([cls, label]) =>
    `${counts[cls]} ${label}${counts[cls] === 1 || cls === "ssn" ? "" : "s"}`);
  const lines = [];
  if (parts.length) lines.push(`Masked ${parts.join(", ")}.`);
  const unfinished = partialReasons(pii.partial);
  for (const [reason, line] of PARTIAL_LINES) if (unfinished.includes(reason)) lines.push(line);
  return lines.join(" ");
}

/* A reopened note's `pii`, from the frontmatter Obsidian parsed: `pii_masked: [email:2]` and
 * `pii_partial: [audio, password]`. null when the note carries neither, so no line shows. A
 * malformed item is skipped, never thrown on: the frontmatter is theirs to edit. */
function piiFromFrontmatter(fm) {
  if (!fm || typeof fm !== "object") return null;
  const list = fm.pii_masked;
  const partial = partialReasons(fm.pii_partial);
  if (!Array.isArray(list) && !partial.length) return null;
  const masked = {};
  for (const item of Array.isArray(list) ? list : []) {
    const m = /^\s*([a-z]+)\s*:\s*(\d+)\s*$/.exec(String(item));
    if (!m || !KNOWN.has(m[1])) continue;
    masked[m[1]] = (masked[m[1]] || 0) + Number(m[2]);
  }
  return { masked, partial };
}

module.exports = { piiHint, piiFromFrontmatter };
