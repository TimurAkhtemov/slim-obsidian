// Word and block diffs for SLIM's changes in the note: what to mark, and the note as it was.
"use strict";

/* ---- live editing: SLIM's changes land in the note as they arrive ----------------------------
 * The server runs the editing loop on its own copy of the note and streams each change as
 * splices on that copy. The plugin applies them through the editor, mapped through whatever the
 * owner typed meanwhile, and draws what each change did. Nothing here talks to the server.
 */

// Words with the whitespace after them. Math, bold, code and links stay whole, so a mark never
// splits their syntax.
function wordTokens(text) {
  const s = String(text || "");
  const out = [];
  let i = 0;
  while (i < s.length) {
    let j = i;
    while (j < s.length && !/\s/.test(s[j])) {
      let close = 0;
      if (s[j] === "$") close = s.indexOf("$", j + 1) + 1;
      else if (s.startsWith("**", j)) close = s.indexOf("**", j + 2) + 2;
      else if (s[j] === "`") close = s.indexOf("`", j + 1) + 1;
      else if (s.startsWith("[[", j)) close = s.indexOf("]]", j + 2) + 2;
      j = close > j + 1 ? close : j + 1;
    }
    while (j < s.length && /\s/.test(s[j])) j += 1;
    out.push(s.slice(i, j));
    i = j;
  }
  return out;
}

// The longest common run of words; a replaced run reads removals first.
function wordOps(before, after) {
  const a = wordTokens(before), b = wordTokens(after);
  const n = a.length, m = b.length;
  const same = (i, j) => a[i].trimEnd() === b[j].trimEnd();
  const t = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      t[i][j] = same(i, j) ? t[i + 1][j + 1] + 1 : Math.max(t[i + 1][j], t[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && same(i, j)) { ops.push(["=", a[i], b[j]]); i += 1; j += 1; }
    else if (j >= m || (i < n && t[i + 1][j] >= t[i][j + 1])) { ops.push(["-", a[i]]); i += 1; }
    else { ops.push(["+", b[j]]); j += 1; }
  }
  return ops;
}

const MARKER = /^\s*(?:#{1,6}\s+|(?:[-*+]|\d+[.)])\s+|>\s?)?/;
const UNMARKABLE = /^\s*(?:\$\$|```|~~~|\|)/;
const MAX_WORD_CELLS = 250_000;

// A change that keeps every word and moves only markdown (a heading level, bold, a list marker,
// blank lines), named for what it does; "" for a change to the words. The note draws it quietly,
// since striking the words and drawing them again reads as a rewrite.
function formatChange(before, after) {
  if (!before || !after) return "";
  const words = (text) => String(text).split(/\s+/).filter(Boolean).join(" ");
  const plain = (text) => words(String(text)
    .replace(/^[ \t]*(?:#{1,6}[ \t]+|>[ \t]?|(?:[-*+]|\d+[.)])[ \t]+)/gm, "")
    .replace(/\*\*|__|~~|==|`/g, ""));
  if (plain(before) !== plain(after)) return "";
  const lines = (text) => String(text).split("\n").filter((line) => line.trim());
  if (words(before) === words(after)) {
    const indents = (text) => lines(text).map((line) => /^\s*/.exec(line)[0].length).join();
    return indents(before) === indents(after) ? "Spacing only" : "Indentation changed";
  }
  const what = [];
  const level = (line) => (/^(#{1,6})\s/.exec(line) || ["", ""])[1].length;
  const b = lines(after);
  lines(before).forEach((line, k) => {
    const from = level(line), to = level(b[k] || "");
    if (from && to && from !== to) what.push(`Heading level ${from} → ${to}`);
  });
  const bold = (text) => (String(text).match(/\*\*|__/g) || []).length;
  if (bold(after) > bold(before)) what.push("Bold added");
  if (bold(after) < bold(before)) what.push("Bold removed");
  return [...new Set(what)].join(" · ") || "Formatting only";
}

// Line for line the same kind (a heading stays a heading of its level, an item an item), none
// of them math, code or a table: what word marks can be drawn on. `half` also asks that each
// line keeps half its words.
function markable(before, after, half = true) {
  const lines = (text) => String(text || "").split("\n").filter((line) => line.trim());
  const a = lines(before), b = lines(after);
  if (!a.length || a.length !== b.length) return false;
  const kept = Boolean(formatChange(before, after));
  return a.every((line, k) => {
    const other = b[k];
    if (UNMARKABLE.test(line) || UNMARKABLE.test(other)) return false;
    const ma = MARKER.exec(line)[0], mb = MARKER.exec(other)[0];
    if (ma.trim() !== mb.trim()) return false;
    const x = line.slice(ma.length), y = other.slice(mb.length);
    const xs = wordTokens(x).length, ys = wordTokens(y).length;
    if (!(xs + ys) || xs * ys > MAX_WORD_CELLS) return false;
    return !half || kept || (2 * wordOps(x, y).filter((op) => op[0] === "=").length) / (xs + ys) >= 0.5;
  });
}

// What one change did, as offsets into each splice's new text: the rail over the new text, the
// words it added, the words it removed (drawn struck where they were), a formatting-only change's
// lines, and removed text folded into one struck line. Computed once per change.
function changeMarks(op, pieces) {
  return pieces.map(({ old, now }) => {
    const out = { rail: now ? [0, now.length] : null, ins: [], dels: [], formats: [], fold: null };
    if (!old) return out;
    if (!now) { out.fold = { text: old, moved: op === "move" }; return out; }
    const lines = [];
    let pos = 0;
    for (const line of now.split("\n")) {
      if (line.trim()) lines.push([pos, line]);
      pos += line.length + 1;
    }
    // A change that keeps most of its words is marked word by word, and so is a wording fix
    // (edit_text) or a heading renamed, however few words stay; one mostly rewritten shows the
    // old text folded above the new, not a scatter of marks.
    const renamed = lines.length === 1 && /^\s*#{1,6}\s/.test(now);
    if (formatChange(old, now)) {
      for (const [start, line] of lines) out.formats.push([start, start + line.length]);
    } else if (markable(old, now, op !== "edit_text" && !renamed)) {
      const olds = old.split("\n").filter((line) => line.trim());
      lines.forEach(([start, line], k) => {
        const other = olds[k];
        const ma = MARKER.exec(other)[0], mb = MARKER.exec(line)[0];
        let at = start + mb.length, gone = "", goneAt = at, from = -1, to = -1;
        const flushGone = () => { if (gone.trim()) out.dels.push([goneAt, gone.trimEnd()]); gone = ""; };
        const flushAdded = () => { if (from >= 0) out.ins.push([from, to]); from = -1; };
        for (const [kind, word, kept] of wordOps(other.slice(ma.length), line.slice(mb.length))) {
          if (kind === "-") { if (!gone) goneAt = at; gone += word; continue; }
          flushGone();
          if (kind === "+") {
            if (from < 0) from = at;
            to = at + word.trimEnd().length;
            at += word.length;
            continue;
          }
          flushAdded();
          at += kept.length;
        }
        flushGone();
        flushAdded();
      });
    } else {
      out.fold = { text: old, moved: false };
    }
    return out;
  });
}

// One replace from the common prefix to the common suffix: the change between two texts.
function textDiff(before, after) {
  const a = String(before), b = String(after);
  const max = Math.min(a.length, b.length);
  let start = 0;
  while (start < max && a.charCodeAt(start) === b.charCodeAt(start)) start += 1;
  let end = 0;
  while (end < max - start && a.charCodeAt(a.length - 1 - end) === b.charCodeAt(b.length - 1 - end)) end += 1;
  if (start === a.length && start === b.length) return [];
  return [{ from: start, to: a.length - end, insert: b.slice(start, b.length - end) }];
}

// The first line of some markdown, without its markers: what a folded removal shows.
function excerpt(markdown) {
  const line = String(markdown || "").split("\n").find((text) => text.trim()) || "";
  return line.replace(/^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s?)/, "").replace(/\*\*|__|`/g, "").trim();
}

// The note's blocks: runs of lines between blank lines, with a fenced block kept whole.
function noteBlocks(text) {
  const out = [];
  let lines = [], fence = null;
  for (const line of String(text || "").split("\n")) {
    const trimmed = line.trim();
    if (fence) {
      lines.push(line);
      if (trimmed.startsWith(fence) || (fence === "$$" && trimmed.endsWith("$$"))) fence = null;
      continue;
    }
    if (!trimmed) {
      if (lines.length) out.push(lines.join("\n"));
      lines = [];
      continue;
    }
    lines.push(line);
    const open = /^(`{3,}|~{3,}|\$\$)/.exec(trimmed);
    if (open && !(open[1] === "$$" && trimmed.length > 2 && trimmed.endsWith("$$"))) fence = open[1];
  }
  if (lines.length) out.push(lines.join("\n"));
  return out;
}

// A block as it was, with the words that are gone now struck (Markdown's ~~).
function struckWords(before, after) {
  const others = String(after).split("\n").filter((line) => line.trim());
  let k = 0;
  return String(before).split("\n").map((line) => {
    if (!line.trim()) return line;
    const other = others[k++] || "";
    const ma = MARKER.exec(line)[0], mb = MARKER.exec(other)[0];
    let out = ma, gone = "";
    const flush = () => {
      if (!gone) return;
      const word = gone.trimEnd();
      out += `~~${word}~~${gone.slice(word.length)}`;
      gone = "";
    };
    for (const [kind, word] of wordOps(line.slice(ma.length), other.slice(mb.length))) {
      if (kind === "-") gone += word;
      else if (kind === "=") { flush(); out += word; }
    }
    flush();
    return out;
  }).join("\n");
}

// The note as it was, block by block, against the note now: what is gone, what changed, what
// moved, and where something was added. See before draws it.
function beforeBlocks(before, now) {
  const a = noteBlocks(before), b = noteBlocks(now);
  const key = (text) => text.trim();
  const n = a.length, m = b.length;
  const t = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      t[i][j] = key(a[i]) === key(b[j]) ? t[i + 1][j + 1] + 1 : Math.max(t[i + 1][j], t[i][j + 1]);
    }
  }
  const runs = [];
  let gone = [], added = [];
  const flush = () => { if (gone.length || added.length) runs.push({ gone, added }); gone = []; added = []; };
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && key(a[i]) === key(b[j])) { flush(); runs.push({ same: a[i] }); i += 1; j += 1; }
    else if (j >= m || (i < n && t[i + 1][j] >= t[i][j + 1])) gone.push(a[i++]);
    else added.push(b[j++]);
  }
  flush();
  // A block gone from one place and added at another moved: as many times as both happened.
  const count = (field) => {
    const out = new Map();
    for (const run of runs) for (const text of run[field] || []) out.set(key(text), (out.get(key(text)) || 0) + 1);
    return out;
  };
  const goneCount = count("gone"), addedCount = count("added");
  const movesOut = new Map(), movesIn = new Map();
  for (const [k, n] of goneCount) {
    const moves = Math.min(n, addedCount.get(k) || 0);
    if (moves) { movesOut.set(k, moves); movesIn.set(k, moves); }
  }
  const take = (moves, text) => {
    const left = moves.get(key(text)) || 0;
    if (left) moves.set(key(text), left - 1);
    return left > 0;
  };
  const out = [];
  for (const run of runs) {
    if (run.same !== undefined) { out.push({ kind: "same", markdown: run.same }); continue; }
    const fresh = run.added.filter((text) => !take(movesIn, text));
    let k = 0;
    for (const text of run.gone) {
      if (take(movesOut, text)) { out.push({ kind: "moved", markdown: text }); continue; }
      const other = fresh[k];
      if (other !== undefined && markable(text, other)) {
        out.push({ kind: "changed", markdown: struckWords(text, other) });
        k += 1;
      } else out.push({ kind: "removed", markdown: text });
    }
    const left = fresh.slice(k);
    if (left.length) out.push({ kind: "added", markdown: excerpt(left[0]), count: left.length });
  }
  return out;
}

module.exports = { wordTokens, wordOps, MARKER, UNMARKABLE, MAX_WORD_CELLS, formatChange, markable,
        changeMarks, textDiff, excerpt, noteBlocks, struckWords, beforeBlocks };
