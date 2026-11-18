"""uv run python -m scripts.pii_compare <arm> <set> --out DIR [--split dev|test] [--no-cache]

arm: A rules, B shapes, C shapes+model, E rules+model, D<threshold> shapes+tagger (D0.5 reads the
tagger's entities from --entities). set: hard, realistic, ordinary, calls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import arms, score


def _texts(name: str, split: str | None):
    if name == "calls":
        calls = score.load_calls()
        return [{"id": s.sid, "text": raw} for s, raw in calls], calls
    cases = score.load(name)
    if split:
        cases = [c for c in cases if c["split"] == split]
    return cases, None


def _tagger_classify(entities: dict, threshold: float):
    from .tagger_map import kind_of

    def classify(text, base, seg):
        ents = entities.get(text, [])
        best, kind = 0.0, "none"
        for a, b, label, s in ents:
            if s >= threshold and a < seg.end and seg.start < b and s > best:
                best, kind = s, kind_of(label)
        return kind
    return classify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm")
    ap.add_argument("set", choices=("hard", "realistic", "ordinary", "calls"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=("dev", "test"))
    ap.add_argument("--only", help="comma list of case ids")
    ap.add_argument("--cache", help="model answer cache (JSON)")
    ap.add_argument("--no-cache", action="store_true", help="ask the model every time")
    ap.add_argument("--entities", help="tagger output (JSON) for a D arm")
    ap.add_argument("--tag", default="", help="suffix for the result file")
    a = ap.parse_args(argv)

    cases, calls = _texts(a.set, a.split)
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    classifier = None
    if a.arm in ("C", "E"):
        from .model import Classifier
        classifier = Classifier(Path(a.cache), use_cache=not a.no_cache)
        classify = classifier.classify
    elif a.arm.startswith("D"):
        entities = json.loads(Path(a.entities).read_text())["entities"]
        classify = _tagger_classify(entities, float(a.arm[1:]))

    hits_by, per_case = {}, []
    t0 = time.monotonic()
    for c in cases:
        text = c["text"]
        c0 = time.monotonic()
        n0 = classifier.calls if classifier else 0
        if a.arm == "A":
            hits, n = arms.rules(text), 0
        elif a.arm == "B":
            hits, n = arms.shapes(text), 0
        elif a.arm == "C" or a.arm.startswith("D"):
            hits, n = arms.layered(text, arms.shapes(text), classify)
        elif a.arm == "E":
            hits, n = arms.layered(text, arms.rules(text), classify)
        else:
            raise SystemExit(f"unknown arm {a.arm}")
        hits_by[c["id"]] = hits
        per_case.append({"id": c["id"], "candidates": n,
                         "model_calls": (classifier.calls - n0) if classifier else 0,
                         "seconds": round(time.monotonic() - c0, 3),
                         "hits": [[h.start, h.end, h.cls] for h in hits]})
        if classifier and len(per_case) % 25 == 0:
            print(f"  {len(per_case)}/{len(cases)} calls={classifier.calls} cached={classifier.cached} "
                  f"{time.monotonic() - t0:.0f}s", file=sys.stderr, flush=True)
    if classifier:
        classifier.save()

    if a.set == "calls":
        masked = {c["id"]: arms.apply(c["text"], hits_by[c["id"]]) for c in cases}
        keep = {c["id"] for c in cases}
        scored = score.score_calls([x for x in calls if x[0].sid in keep], masked, arms.CLASSES)
    elif a.set == "ordinary":
        scored = score.score_ordinary(cases, hits_by)
    else:
        scored = score.score_labelled(cases, hits_by)
    report = {"arm": a.arm, "set": a.set, "split": a.split, "cases": len(cases),
              "seconds": round(time.monotonic() - t0, 1), "score": scored, "per_case": per_case}
    if classifier:
        report["model"] = {"calls": classifier.calls, "cached": classifier.cached,
                           "failures": classifier.failures, "errors": classifier.errors,
                           "call_seconds": classifier.seconds}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.arm}-{a.set}{'-' + a.split if a.split else ''}{a.tag}.json"
    (out / name).write_text(json.dumps(report, indent=1))
    s = scored.get("secrets")
    line = f"{a.arm} {a.set} {a.split or ''}: "
    if s:
        line += f"secrets {s['masked']}/{s['n']} " + " ".join(
            f"{k}:{v.get('masked', 0)}/{v.get('n', 0)}" for k, v in scored["by_class"].items())
    if "ordinary" in scored:
        line += f" ordinary damaged {scored['ordinary'].get('damaged', 0)}/{scored['ordinary'].get('n', 0)}"
    if "values" in scored:
        line += (f"values damaged {scored['values'].get('damaged', 0)}/{scored['values']['n']}, sentences "
                 f"{scored['sentences'].get('damaged', 0)}/{scored['sentences']['n']}")
    if "distractors" in scored:
        line += (f" false positives {len(scored['false_positives'])}, distractors damaged "
                 f"{scored['distractors']['damaged']}/{scored['distractors']['n']}")
    if classifier:
        line += f" | model calls {classifier.calls} (cached {classifier.cached}, failed {classifier.failures})"
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
