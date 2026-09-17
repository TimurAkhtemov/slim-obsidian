"""S2-E: one live Ollama call at a time.

There is one resident model and the chat surface is a ThreadingHTTPServer, so concurrent
calls were always possible — nothing serialized them before the tool-loop. The cost is
not correctness at the HTTP layer (Ollama would answer both) but the per-runner KV cache:
interleaved calls evict each other's prefix, and the tool-loop's growing conversation is
only affordable because that prefix survives between calls.

These tests drive `llm` through a fake transport rather than a real model — the property
under test is the lock, not the decode.
"""

import json
import threading
import time

import pytest

from slim import llm


class _Overlap:
    """Records whether two calls were ever inside the client at the same time."""

    def __init__(self):
        self.inside = 0
        self.max_inside = 0
        self.lock = threading.Lock()

    def enter(self):
        with self.lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)

    def exit(self):
        with self.lock:
            self.inside -= 1


def test_non_streaming_calls_do_not_overlap(monkeypatch):
    overlap = _Overlap()
    payload = json.dumps({"message": {"content": "{}"}, "done": True}).encode()

    def fake_urlopen(req, timeout=None):
        overlap.enter()
        try:
            time.sleep(0.05)
            return _Payload(payload)
        finally:
            overlap.exit()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    threads = [threading.Thread(target=lambda: llm.chat([{"role": "user", "content": "hi"}]))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlap.max_inside == 1, "two live calls were inside the client at once"


class _Payload:
    """urlopen result for the non-streaming path: json.load() reads it directly."""

    def __init__(self, data):
        self._data = data

    def read(self, *a):
        out, self._data = self._data, b""
        return out


def test_a_stream_holds_the_lock_for_its_whole_decode(monkeypatch):
    """The load-bearing half: a stream that released at first byte would let a decision
    call interleave with its own answer, which is exactly the tool-loop's shape."""
    overlap = _Overlap()
    lines = [json.dumps({"message": {"content": "tok"}}).encode() for _ in range(4)]
    lines.append(json.dumps({"message": {"content": "!"}, "done": True,
                             "eval_count": 5}).encode())

    def fake_urlopen(req, timeout=None):
        overlap.enter()
        return _StreamBody(lines, overlap)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    results = []

    def run():
        text, _ = llm.chat_stream([{"role": "user", "content": "hi"}])
        results.append(text)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlap.max_inside == 1, "a second call ran while a stream was still decoding"
    assert results == ["tok" * 4 + "!"] * 3


class _StreamBody:
    """Line-iterable urlopen result that reports when the stream is truly done."""

    def __init__(self, lines, overlap):
        self._lines = lines
        self._overlap = overlap

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._overlap.exit()
        return False

    def __iter__(self):
        for line in self._lines:
            time.sleep(0.01)
            yield line

    def read(self):
        return b""


def test_a_failing_call_releases_the_lock(monkeypatch):
    """A refusal must not wedge every later call — the lock is released on the error
    path too, or one unreachable-Ollama moment would deadlock the whole surface."""
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", boom)

    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "hi"}])

    assert not llm._CALL_LOCK.locked(), "the lock survived a failed call"

    with pytest.raises(llm.LLMError):
        llm.chat_stream([{"role": "user", "content": "hi"}])

    assert not llm._CALL_LOCK.locked(), "the lock survived a failed stream"


def test_every_call_path_shares_the_residents_one_window():
    """One model, one size (2026-09-01: gemma dropped, qwen everywhere). `RESIDENT_CTX` is
    cheap ONLY because qwen35moe is MoE (~16 MB of KV per 1k tokens; 25 GB at 128k, all GPU).
    A dense model at this window is ~11 GB of KV — the configuration that let macOS kill two
    labeling passes (2026-08-01) and a recording pass (2026-08-22) with zero output. If a
    second model ever returns, its ctx must NOT be tidied up to match the resident's."""
    assert llm.RECORD_CTX == llm.RESIDENT_CTX, "the recording path IS the resident"



