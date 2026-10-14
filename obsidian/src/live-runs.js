// The arithmetic of an editing run in the note: where its changes are, what each one needs, and the
// note with a different set of them kept. Pure: CodeMirror's ChangeSet is passed in.
"use strict";

// The owner's edits against a range of SLIM's text: inside it, not merely beside it.
const inside = (changes, from, to) => {
  let hit = false;
  changes.iterChangedRanges((a, b) => { if (!hit) hit = a === b ? from < a && a < to : a < to && b > from; });
  return hit;
};
// Undo or redo would replace text the owner has edited.
const clashes = (owner, delta) => {
  let hit = false;
  delta.iterChangedRanges((from, to) => { if (!hit && from < to) hit = inside(owner, from, to); });
  return hit;
};

// Where each splice's new text starts once the change is in, given agent → doc.
const starts = (splices, toDoc) => {
  const out = [];
  let shift = 0;
  for (const { from, to, insert } of splices) {
    const a = toDoc.mapPos(from, 1), b = from === to ? a : Math.max(a, toDoc.mapPos(to, -1));
    out.push(a + shift);
    shift += insert.length - (b - a);
  }
  return out;
};
const place = (rel, at) => {
  const out = { rails: [], ins: [], dels: [], formats: [], folds: [] };
  rel.forEach((piece, i) => {
    const s = at[i];
    if (piece.rail) out.rails.push([s + piece.rail[0], s + piece.rail[1]]);
    for (const [f, t] of piece.ins) out.ins.push([s + f, s + t]);
    for (const [p, text] of piece.dels) out.dels.push([s + p, text]);
    for (const [f, t] of piece.formats) out.formats.push([s + f, s + t]);
    if (piece.fold) out.folds.push([s, piece.fold.text, piece.fold.moved]);
  });
  return out;
};
// A range keeps text typed at its edges outside it; a point stays before text put there.
const mapMarks = (at, changes) => {
  const range = ([f, t]) => { const a = changes.mapPos(f, 1); return [a, Math.max(a, changes.mapPos(t, -1))]; };
  const point = ([p, ...rest]) => [changes.mapPos(p, -1), ...rest];
  return { rails: at.rails.map(range), ins: at.ins.map(range), dels: at.dels.map(point),
           formats: at.formats.map(range), folds: at.folds.map(point) };
};

// The run's changes from base with `kept` in the note: agent → doc, and each change's marks.
function replay(ChangeSet, run, kept) {
  let agent = run.base, toDoc = ChangeSet.empty(run.base.length);
  const marks = new Map();
  for (const step of run.steps) {
    if (kept.has(step.id)) {
      const onDoc = step.changes.map(toDoc);
      for (const [id, m] of marks) marks.set(id, mapMarks(m, onDoc));
      marks.set(step.id, place(step.rel, starts(step.splices, toDoc)));
      toDoc = toDoc.map(step.changes, true);
    } else {
      toDoc = step.changes.invert(agent).compose(toDoc);
    }
    agent = step.changes.apply(agent);
  }
  return { toDoc, marks };
}

// From one doc to another, both images of the agent's copy: back to that copy and forward
// again. A change neither doc keeps comes back as a replace of its text with itself; equal
// whole lines are trimmed from each end, so that text is never touched.
function between(ChangeSet, run, toDoc) {
  const specs = [];
  const lines = (text) => text.match(/[^\n]*\n|[^\n]+$/g) || [];
  run.toDoc.invert(run.agent).compose(toDoc).iterChanges((fromA, toA, _fromB, _toB, inserted) => {
    const a = lines(run.doc.sliceString(fromA, toA)), b = lines(inserted.toString());
    let i = 0, j = 0;
    while (i < a.length && i < b.length && a[i] === b[i]) i += 1;
    while (j < a.length - i && j < b.length - i && a[a.length - 1 - j] === b[b.length - 1 - j]) j += 1;
    if (i === a.length && i === b.length) return;
    const size = (list) => list.reduce((n, line) => n + line.length, 0);
    specs.push({ from: fromA + size(a.slice(0, i)), to: toA - size(a.slice(a.length - j)),
                 insert: b.slice(i, b.length - j).join("") });
  });
  return ChangeSet.of(specs, run.doc.length);
}

// Everything that needs these changes, directly or through another.
const dependents = (run, ids) => {
  const out = new Set(ids);
  for (let grew = true; grew;) {
    grew = false;
    for (const step of run.steps) {
      if (!out.has(step.id) && step.needs.some((id) => out.has(id))) { out.add(step.id); grew = true; }
    }
  }
  return out;
};
// A kept set the note can hold: every change in it has what it needs, and nothing that needs
// a skipped change is ever in it.
const holdable = (run, ids) => {
  const byId = new Map(run.steps.map((step) => [step.id, step]));
  const blocked = new Set(run.steps.filter((step) => step.skipped).map((step) => step.id));
  for (let grew = true; grew;) {
    grew = false;
    for (const step of run.steps) {
      if (!blocked.has(step.id) && step.needs.some((id) => blocked.has(id))) { blocked.add(step.id); grew = true; }
    }
  }
  const out = new Set();
  const add = (id) => {
    if (out.has(id) || blocked.has(id) || !byId.has(id)) return;
    out.add(id);
    for (const need of byId.get(id).needs) add(need);
  };
  for (const id of ids) add(id);
  return out;
};
const same = (a, b) => a.size === b.size && [...a].every((id) => b.has(id));

// A run moved to another kept set; null when the owner's edits are in the way.
function moved(ChangeSet, run, kept, push) {
  const w = replay(ChangeSet, run, kept);
  const delta = between(ChangeSet, run, w.toDoc);
  if (clashes(run.owner, delta)) return null;
  const redo = (push ? [...run.redo, push] : run.redo).filter((group) => group.some((id) => !kept.has(id)));
  return {
    changes: delta.map(run.owner),
    run: { ...run, kept, doc: delta.apply(run.doc), toDoc: w.toDoc, owner: run.owner.map(delta, true),
           marks: w.marks, redo, latest: null, version: run.version + 1 },
  };
}

// Cmd-Z and Cmd-Shift-Z are the editor's own history: when one takes a change out or puts it
// back, the run follows. Recognized by its result, the note as that move would leave it.
function fromHistory(ChangeSet, run, tr) {
  const tries = [...run.redo].reverse().map((group) => ({ kept: holdable(run, [...run.kept, ...group]) }));
  for (const step of [...run.steps].reverse()) {
    if (step.skipped) continue;
    if (run.kept.has(step.id)) {
      const gone = dependents(run, [step.id]);
      tries.push({ kept: new Set([...run.kept].filter((id) => !gone.has(id))),
                   push: [...gone].filter((id) => run.kept.has(id)) });
    } else {
      tries.push({ kept: holdable(run, [...run.kept, step.id]) });
    }
  }
  for (const { kept, push } of tries) {
    if (same(kept, run.kept)) continue;
    const next = moved(ChangeSet, run, kept, push);
    if (next && next.changes.apply(tr.startState.doc).eq(tr.newDoc)) return next.run;
  }
  return null;
}

// ↺'s words: the change, and how many other changes made inside it go with it. Its own
// rewrites (`part_of` it) are the change itself and are not counted.
const tipFor = (run, step) => {
  const parts = new Set(run.steps.filter((other) => other.partOf).map((other) => other.id));
  const inside = [...dependents(run, [step.id])]
    .filter((id) => id !== step.id && run.kept.has(id) && !parts.has(id)).length;
  return `${step.label || step.id}${inside ? ` · also undoes ${inside} inside it` : ""}`;
};

module.exports = { inside, clashes, starts, place, mapMarks, replay, between, dependents, holdable, same, moved,
                   fromHistory, tipFor };
