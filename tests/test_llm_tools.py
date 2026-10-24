"""Tool calls on the wire without streaming: the intent gate's request and the reply's parse. The
transport is faked: the properties under test are the request body and the parse, not the decode."""

import io
import json
import urllib.error

import pytest

from slim import llm

REPLY = {"choices": [{"message": {"content": "", "tool_calls": [
            {"type": "function", "function": {"name": "replace",
                                              "arguments": json.dumps({"target": "b3", "markdown": "## A"})}}]},
                      "finish_reason": "tool_calls"}],
         "timings": {"prompt_n": 40, "predicted_n": 12, "prompt_ms": 500.0, "predicted_ms": 1500.0}}


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
    assert "tools" not in body
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["stream"] is False and body["model"] == llm.MODEL
    assert body["max_tokens"] == 2048 and body["temperature"] == 0.0


def test_chat_tools_leaves_out_what_was_not_given(monkeypatch):
    sent = _capture(monkeypatch)
    llm.chat_tools([{"role": "user", "content": "q"}], think=False, temperature=0.7, num_predict=64)
    assert "tools" not in sent[0] and "response_format" not in sent[0]
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_tools_parses_tool_calls_and_keeps_them_raw(monkeypatch):
    _capture(monkeypatch)
    msg, stats = llm.chat_tools([], [], think=False, temperature=0.0, num_predict=10)
    assert msg["content"] == ""
    assert msg["tool_calls"] == [{"name": "replace", "arguments": {"target": "b3", "markdown": "## A"}}]
    assert msg["raw_tool_calls"] == [{"function": {"name": "replace", "arguments": {"target": "b3", "markdown": "## A"}}}]
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
