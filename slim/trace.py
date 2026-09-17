"""Always-on traces. One JSONL record per operation;
`slim trace` replays the latest. No chain-of-thought, just the machine facts."""

import json
from datetime import datetime, timezone

from .config import TRACE_DIR


def record(kind: str, payload: dict) -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{datetime.now(timezone.utc):%Y-%m}.jsonl"
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, **payload}
    with path.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def last(n: int = 1) -> list[dict]:
    files = sorted(TRACE_DIR.glob("*.jsonl"))
    if not files:
        return []
    lines = files[-1].read_text().splitlines()
    return [json.loads(l) for l in lines[-n:]]
