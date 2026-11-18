"""Arm D's tagger, run in its own virtualenv (gliner + torch, CPU), never beside a model job:

    <venv>/bin/python scripts/pii_compare/tagger.py <model id or path> <out.json>

Imports nothing from `slim`. Reads every set's texts from `data/`, tags each turn with the
previous turn as context, and writes {text: [[start, end, label, score], ...]} at the lowest
threshold measured (0.3), plus seconds per text and the peak resident memory."""

import json
import re
import resource
import sys
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"
TURN = re.compile(r"\*\*[^*\n]+:\*\*")
LABELS = ["ssn", "credit card", "account number", "bank account", "routing number", "phone number",
          "location zip", "dob", "date", "money", "credit card expiration", "cvv", "password",
          "email address", "location address"]
THRESHOLD = 0.3


def texts() -> dict[str, list[str]]:
    out = {name: [c["text"] for c in json.loads((DATA / f"{name}.json").read_text())]
           for name in ("hard", "realistic", "ordinary")}
    out["calls"] = [p.read_text() for p in sorted((DATA / "calls").glob("s*.raw.md"))]
    return out


def turns(text: str) -> list[tuple[int, int]]:
    starts = [m.start() for m in TURN.finditer(text)]
    if not starts or starts[0] > 0:
        starts = [0] + starts
    return list(zip(starts, starts[1:] + [len(text)]))


def tag(model, text: str) -> list[list]:
    out = []
    spans = turns(text)
    for k, (a, b) in enumerate(spans):
        ctx = text[spans[k - 1][0]:spans[k - 1][1]] if k else ""
        joined = ctx + text[a:b]
        for e in model.predict_entities(joined, LABELS, threshold=THRESHOLD):
            if e["start"] >= len(ctx):
                out.append([a + e["start"] - len(ctx), a + e["end"] - len(ctx), e["label"], round(e["score"], 4)])
    return out


def main() -> None:
    from gliner import GLiNER

    t0 = time.monotonic()
    model = GLiNER.from_pretrained(sys.argv[1])
    model.eval()
    load = time.monotonic() - t0
    result, seconds = {}, {}
    for name, items in texts().items():
        secs = []
        for text in items:
            t = time.monotonic()
            result[text] = tag(model, text)
            secs.append(time.monotonic() - t)
        seconds[name] = {"n": len(secs), "mean": sum(secs) / len(secs), "max": max(secs)}
        print(name, seconds[name], flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20     # bytes on macOS -> MiB
    Path(sys.argv[2]).write_text(json.dumps({"entities": result, "seconds": seconds,
                                             "load_seconds": load, "peak_mib": peak,
                                             "labels": LABELS, "threshold": THRESHOLD}))
    print(f"load {load:.1f}s, peak {peak:.0f} MiB")


if __name__ == "__main__":
    main()
