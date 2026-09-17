"""chat_stream's callbacks: content goes to on_delta, reasoning only pings on_thinking."""

import json

from slim import llm


class _Body:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self._lines


def test_thinking_chunks_ping_on_thinking_but_never_reach_on_delta(monkeypatch):
    # A client that hung up can only be noticed by WRITING to it. During a long think no
    # content arrives, so without this hook the socket, the Ollama request and _CALL_LOCK
    # all stay held until the reasoning ends (~50 s measured).
    lines = [json.dumps({"message": {"thinking": "let me see"}}).encode(),
             json.dumps({"message": {"thinking": "hmm"}}).encode(),
             json.dumps({"message": {"content": "Answer"}}).encode(),
             json.dumps({"message": {"content": ""}, "done": True, "eval_count": 3}).encode()]
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout=None: _Body(lines))
    deltas, pings = [], []
    content, stats = llm.chat_stream(
        [{"role": "user", "content": "q"}], think=True,
        on_delta=deltas.append, on_thinking=pings.append)
    assert content == "Answer"
    assert deltas == ["Answer"]
    assert pings == ["let me see", "hmm"]      # the recorder shows it as activity; the copilot ignores it
    assert stats["thinking_chars"] == len("let me seehmm")
