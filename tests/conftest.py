"""Shared test isolation.

The trace layer is an ALWAYS-ON durable writer (`trace.record` appends to `TRACE_DIR`,
resolved from `config.DATA_DIR` at import). Nothing in a test should reach the real
`~/Library/Application Support/slim/traces` — but until this fixture existed, every test that
exercised a traced path (ask, summarize, …) silently appended to the production
trace. A `remember` unit run left 65 `fake-model` records in the live trace before it was
caught in review. This autouse fixture points the trace directory at a per-test tmp path so a
test can never again write a durable operational record into production.
"""
import json
import urllib.request

import pytest

from slim import embed, enrich, db, threads, trace


def ollama_available() -> bool:
    """Can a local Ollama actually embed? Vector-arm tests skip when it cannot.

    A reachable daemon is not enough: `embed` asks for one specific model, so a clone with
    Ollama running for some other reason but `embeddinggemma` never pulled used to FAIL the
    two vector tests rather than skip them. The suite has to be green on a machine that has
    none of the weights.
    """
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            tags = json.load(r)
    except Exception:  # noqa: BLE001
        return False
    return any(str(m.get("name", "")).startswith(embed.MODEL)
               for m in tags.get("models") or [])


@pytest.fixture(autouse=True)
def _isolate_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path / "traces")


@pytest.fixture(autouse=True)
def _isolate_default_db(tmp_path, monkeypatch):
    """A bare `db.connect()` in a test must never open the production brain.

    Part 3's initial argparse regression accidentally accepted the still-live
    `ingest --remember` flag and invoked its command body; trace/profile were isolated, but the
    default DB was not, so the test queued eight disposable candidates in the live brain. Exact
    rows were removed immediately. Keep explicit test DB paths unchanged while routing every
    omitted path to this test's temporary directory.
    """
    real_connect = db.connect

    def isolated_connect(db_path=None):
        return real_connect(db_path if db_path is not None else tmp_path / "default-slim.db")

    monkeypatch.setattr(db, "connect", isolated_connect)


@pytest.fixture(autouse=True)
def _isolate_threads(tmp_path, monkeypatch):
    """Same rationale as the trace isolation: once the chat surface persists threads
    (chat-fluidity F5), the store is an always-on durable writer — no test may read or
    write the production conversation history."""
    monkeypatch.setattr(threads, "THREADS_DIR", tmp_path / "threads")


@pytest.fixture(autouse=True)
def _disable_enrichment(monkeypatch):
    """Enrichment is ON in production (the owner enabled it 2026-07-20) and must be OFF in
    tests. It calls a live model, so leaving it on would put a real Ollama round-trip on
    the critical path of every inbox test — slow, non-deterministic, and silently broken on
    a machine where Ollama is not running. Tests that exercise enrichment set it True and
    fake `llm.chat_json` themselves."""
    monkeypatch.setattr(enrich, "ENABLED", False)
