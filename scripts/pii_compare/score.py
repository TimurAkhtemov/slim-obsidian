"""Scoring: a secret counts as masked only when every digit of it is under a mask (a partial mask is
a miss); an ordinary value is damaged when any digit of it is under a secret-class mask, the masks
that also mute audio and cannot be undone."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from slim.capture.pii_detect import Hit

from .arms import SCORED

DATA = Path(__file__).parent / "data"
# An ordinary sentence's values: digit groups joined by one separator or ", " (as the detector
# reads a run), with ":" for times.
_VALUE = re.compile(r"\d+(?:(?:, |[ ./:-])\d+)*")


def _digits_at(text: str, a: int, b: int) -> list[int]:
    return [k for k in range(a, b) if text[k].isdigit()]


def _under(k: int, hits: list[Hit], classes=None) -> bool:
    return any(h.start <= k < h.end and (classes is None or h.cls in classes) for h in hits)


def masked_fully(text: str, sub: str, hits: list[Hit]) -> bool:
    a = text.index(sub)
    return all(_under(k, hits) for k in _digits_at(text, a, a + len(sub)))


def damaged(text: str, a: int, b: int, hits: list[Hit]) -> bool:
    return any(_under(k, hits, SCORED) for k in _digits_at(text, a, b))


def score_labelled(cases: list[dict], results: dict[str, list[Hit]]) -> dict:
    """`cases` carry `secrets` [(substring, class)] and `ordinary` [substring]; ambiguous cases are
    counted apart and never scored."""
    by_class = {c: Counter() for c in SCORED}
    ordinary = Counter()
    leaks, damage = [], []
    ambiguous = 0
    for c in cases:
        if c.get("ambiguous"):
            ambiguous += 1
            continue
        hits, text = results[c["id"]], c["text"]
        for sub, cls in c["secrets"]:
            ok = masked_fully(text, sub, hits)
            by_class[cls]["n"] += 1
            by_class[cls]["masked"] += ok
            if not ok:
                leaks.append({"id": c["id"], "class": cls, "value": sub})
        for sub in c["ordinary"]:
            a = text.index(sub)
            hurt = damaged(text, a, a + len(sub), hits)
            ordinary["n"] += 1
            ordinary["damaged"] += hurt
            if hurt:
                damage.append({"id": c["id"], "value": sub})
    total = Counter()
    for v in by_class.values():
        total.update(v)
    return {"secrets": {"masked": total["masked"], "n": total["n"]},
            "by_class": {c: dict(v) for c, v in by_class.items()},
            "ordinary": dict(ordinary), "ambiguous": ambiguous, "leaks": leaks, "damage": damage}


def score_ordinary(cases: list[dict], results: dict[str, list[Hit]]) -> dict:
    """Every number in an ordinary sentence is ordinary."""
    values = Counter()
    sentences = Counter()
    damage = []
    for c in cases:
        text, hits = c["text"], results[c["id"]]
        hurt_any = False
        for m in _VALUE.finditer(text):
            hurt = damaged(text, *m.span(), hits)
            values["n"] += 1
            values["damaged"] += hurt
            hurt_any |= hurt
        sentences["n"] += 1
        sentences["damaged"] += hurt_any
        if hurt_any:
            damage.append({"id": c["id"], "family": c["family"], "text": text,
                           "masks": [[h.start, h.end, h.cls] for h in hits if h.cls in SCORED]})
    return {"values": dict(values), "sentences": dict(sentences), "damage": damage}


def load(name: str) -> list[dict]:
    return json.loads((DATA / f"{name}.json").read_text())


# --- calls: the shipped scorer, text only ---

def load_calls():
    from scripts.pii_eval.gen import Distractor, Item, Script

    raw = json.loads((DATA / "calls" / "scripts.json").read_text())
    scripts = [Script(s["sid"], [tuple(x) for x in s["lines"]], [Item(**i) for i in s["items"]],
                      [Distractor(**x) for x in s["distractors"]], s.get("transcript", ""))
               for s in raw]
    return [(s, (DATA / "calls" / f"{s.sid}.raw.md").read_text()) for s in scripts]


def score_calls(calls, masked: dict[str, str], classes) -> dict:
    """`scripts/pii_eval/score.score_call` on each masked transcript, audio checks off: a partial
    mask is a miss, an over-wide mask a false positive. Reported for the scored classes."""
    from scripts.pii_eval import score as ev

    results = []
    for script, raw in calls:
        r = ev.score_call(script, raw, masked[script.sid], masked[script.sid], classes, spans=[],
                          reasons=set(), manifest={}, measure=lambda *a: -100.0, silenced=Path("."),
                          model=True)
        results.append(r)
    agg = ev.aggregate_calls(results, classes)
    by_class = {c: {"n": agg["by_class"][c]["n"], "masked": agg["by_class"][c]["masked"]} for c in SCORED}
    garbled = {c: {"n": agg["garbled"][c]["n"], "masked": agg["garbled"][c]["masked"]} for c in SCORED}
    fps = [f for f in agg["false_positives"] if f["class"] in SCORED]
    n = sum(v["n"] for v in by_class.values())
    m = sum(v["masked"] for v in by_class.values())
    distractors = sum(len(s.distractors) for s, _ in calls)
    return {"secrets": {"masked": m, "n": n}, "by_class": by_class, "garbled": garbled,
            "false_positives": [{k: f[k] for k in ("script", "class", "text", "kind", "distractor")} for f in fps],
            "distractors": {"n": distractors,
                            "damaged": sum(1 for f in fps for _ in (f.get("distractors") or []))},
            "misses": [{k: x[k] for k in ("script", "class", "kind", "cue", "heard", "result")}
                       for x in agg["misses"] if x["class"] in SCORED]}
