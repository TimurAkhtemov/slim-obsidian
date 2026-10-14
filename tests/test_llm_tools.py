"""Tool calls on the wire without streaming: the intent gate's request and the reply's parse. The
transport is faked: the properties under test are the request body and the parse, not the decode."""

import io
import json
import urllib.error

import pytest

from slim import llm

REPLY = {"message": {"content": "", "tool_calls": [
            {"function": {"name": "replace", "arguments": {"target": "b3", "markdown": "## A"}}}]},
         "prompt_eval_count": 40, "eval_count": 12, "total_duration": 2_000_000_000,
         "prompt_eval_duration": 500_000_000, "eval_duration": 1_000_000_000, "done_reason": "stop"}


def _capture(monkeypatch, reply: dict = REPLY):
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_chat_tools_without_tools_sends_the_format_and_every_bound(monkeypatch):
    # The intent gate. A call WITH tools streams raw instead (tests/test_llm_wire.py).
    sent = _capture(monkeypatch)
    schema = {"type": "object"}
    llm.chat_tools([{"role": "user", "content": "q"}], think=True, temperature=0.0,
                   num_predict=2048, fmt=schema)
    (body,) = sent
    assert "tools" not in body and body["format"] == schema
    assert body["think"] is True and body["stream"] is False and body["model"] == llm.MODEL
    assert body["options"] == {"num_ctx": llm.RESIDENT_CTX, "temperature": 0.0, "num_predict": 2048}


def test_chat_tools_leaves_out_what_was_not_given(monkeypatch):
    sent = _capture(monkeypatch)
    llm.chat_tools([{"role": "user", "content": "q"}], think=False, temperature=0.7, num_predict=64)
    assert "tools" not in sent[0] and "format" not in sent[0] and sent[0]["think"] is False


def test_chat_tools_parses_tool_calls_and_keeps_them_raw(monkeypatch):
    _capture(monkeypatch)
    msg, stats = llm.chat_tools([], [], think=False, temperature=0.0, num_predict=10)
    assert msg["content"] == ""
    assert msg["tool_calls"] == [{"name": "replace", "arguments": {"target": "b3", "markdown": "## A"}}]
    assert msg["raw_tool_calls"] == REPLY["message"]["tool_calls"]
    _content, chat_stats = llm.chat([], num_predict=10)
    assert set(stats) == set(chat_stats)               # traces stay uniform
    assert stats["output_tokens"] == 12 and stats["duration_s"] == 2.0 and stats["num_ctx"] == llm.RESIDENT_CTX


def test_chat_tools_turns_an_http_error_into_an_llm_error(monkeypatch):
    def refuse(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {},
                                     io.BytesIO(b"XML syntax error on line 10"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", refuse)
    with pytest.raises(llm.LLMError, match="HTTP 500 — XML syntax error on line 10"):
        llm.chat_tools([], think=False, temperature=0.0, num_predict=10)
    assert not llm._CALL_LOCK.locked()
