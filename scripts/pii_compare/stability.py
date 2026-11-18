"""Compare two uncached runs of the same arm and set: every case whose masks changed.

    uv run python -m scripts.pii_compare.stability RUN1.json RUN2.json [CACHED.json]"""

import json
import statistics
import sys


def _hits(path):
    r = json.loads(open(path).read())
    return r, {p["id"]: [tuple(h) for h in p["hits"]] for p in r["per_case"]}


def main(paths):
    runs = [_hits(p) for p in paths]
    base_r, base = runs[0]
    for (r, hits), p in zip(runs[1:], paths[1:]):
        changed = [k for k in base if base[k] != hits.get(k)]
        print(f"{paths[0]} vs {p}: {len(changed)} of {len(base)} cases changed")
        for k in changed:
            print(f"  {k}: {base[k]} -> {hits[k]}")
    for r, _ in runs[:2]:
        pc = r["per_case"]
        calls = [x["model_calls"] for x in pc]
        secs = [x["seconds"] for x in pc]
        cs = r.get("model", {}).get("call_seconds") or [0]
        print(f"{r['arm']} {r['set']}: calls/recording mean {statistics.mean(calls):.1f} max {max(calls)}; "
              f"seconds/recording mean {statistics.mean(secs):.2f} max {max(secs):.2f}; "
              f"per call median {statistics.median(cs):.2f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
