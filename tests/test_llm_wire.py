"""What `llm` puts on the wire. qwen3.6 thinks unless told not to, and the thinking eats
`num_predict`: a body without `think` returns content="" and the recorder dies at its first
call. Every other test fakes the model's answer; these decode the request itself."""

import io
import json

import pytest

from slim import llm


def _capture(monkeypatch, reply: bytes):
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return io.BytesIO(reply)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return sent


@pytest.mark.parametrize("think", [False, True])
def test_chat_sends_think_explicitly(monkeypatch, think):
    sent = _capture(monkeypatch, json.dumps({"message": {"content": "ok"}}).encode())
    llm.chat([{"role": "user", "content": "q"}], think=think)
    assert sent[0]["think"] is think


def test_chat_sends_think_false_when_the_caller_says_nothing(monkeypatch):
    sent = _capture(monkeypatch, json.dumps({"message": {"content": "ok"}}).encode())
    llm.chat([{"role": "user", "content": "q"}])
    assert sent[0]["think"] is False


@pytest.mark.parametrize("think", [False, True])
def test_chat_stream_sends_think_explicitly(monkeypatch, think):
    reply = json.dumps({"message": {"content": "ok"}, "done": True}).encode() + b"\n"
    sent = _capture(monkeypatch, reply)
    llm.chat_stream([{"role": "user", "content": "q"}], think=think)
    assert sent[0]["think"] is think


def test_chat_stream_sends_think_false_when_the_caller_says_nothing(monkeypatch):
    reply = json.dumps({"message": {"content": "ok"}, "done": True}).encode() + b"\n"
    sent = _capture(monkeypatch, reply)
    llm.chat_stream([{"role": "user", "content": "q"}])
    assert sent[0]["think"] is False
