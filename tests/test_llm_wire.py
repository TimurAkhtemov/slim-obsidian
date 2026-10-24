"""What `llm` puts on the wire. qwen3.6 thinks unless told not to, and the thinking eats the
output cap: a body without `enable_thinking` returns content="" and the recorder dies at its first
call. Every other test fakes the model's answer; these decode the request itself."""

import io
import json

import pytest

from slim import llm
from tests.test_llm_messages import EDIT, PNG, TOOLS
from tests.test_llm_reader import INSERT

CHAT_REPLY = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def _event(data: dict) -> bytes:
    return b"data: " + json.dumps(data).encode() + b"\n\n"


def _capture(monkeypatch, reply: bytes):
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return io.BytesIO(reply)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return sent


@pytest.mark.parametrize("think", [False, True])
def test_chat_sends_think_explicitly(monkeypatch, think):
    sent = _capture(monkeypatch, json.dumps(CHAT_REPLY).encode())
    llm.chat([{"role": "user", "content": "q"}], think=think)
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": think}


def test_chat_sends_think_false_when_the_caller_says_nothing(monkeypatch):
    sent = _capture(monkeypatch, json.dumps(CHAT_REPLY).encode())
    llm.chat([{"role": "user", "content": "q"}])
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_turns_the_servers_thinking_budget_off_and_bounds_the_output(monkeypatch):
    # The server's budget is the coding agent's; SLIM's bound is the output cap.
    sent = _capture(monkeypatch, json.dumps(CHAT_REPLY).encode())
    llm.chat([{"role": "user", "content": "q"}], num_predict=300, temperature=0.7, think=True)
    assert sent[0]["thinking_budget_tokens"] == llm.NO_THINKING_BUDGET == -1
    assert sent[0]["max_tokens"] == 300 and sent[0]["temperature"] == 0.7 and sent[0]["stream"] is False


def test_a_schema_constrains_the_reply(monkeypatch):
    sent = _capture(monkeypatch, json.dumps(CHAT_REPLY).encode())
    llm.chat([{"role": "user", "content": "q"}], schema={"type": "object"})
    assert sent[0]["response_format"] == {"type": "json_schema",
                                          "json_schema": {"name": "reply", "schema": {"type": "object"}}}


@pytest.mark.parametrize("think", [False, True])
def test_chat_stream_sends_think_explicitly(monkeypatch, think):
    reply = _event({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}) + b"data: [DONE]\n\n"
    sent = _capture(monkeypatch, reply)
    llm.chat_stream([{"role": "user", "content": "q"}], think=think)
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": think} and sent[0]["stream"] is True


def test_chat_stream_sends_think_false_when_the_caller_says_nothing(monkeypatch):
    reply = _event({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
    sent = _capture(monkeypatch, reply)
    llm.chat_stream([{"role": "user", "content": "q"}])
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


# --- a call with tools: the server's prompt, raw and streamed ---------------------------------------

PROMPT = "<|im_start|>system\n# Tools ... <|im_start|>assistant\n<think>\n\n</think>\n\n"


def _stream(monkeypatch, events, prompt=PROMPT):
    """The template call answers with `prompt`; the completion streams `events`."""
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, json.loads(req.data)))
        if req.full_url == llm.TEMPLATE_URL:
            return io.BytesIO(json.dumps({"prompt": prompt}).encode())
        return io.BytesIO(b"".join(event if isinstance(event, bytes) else _event(event) for event in events))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return sent


def _chunks(text, size=4, **final):
    out = [{"content": text[i:i + size], "stop": False} for i in range(0, len(text), size)]
    return out + [{"content": "", "stop": True, "stop_type": "eos",
                   "timings": {"prompt_n": 30, "cache_n": 10, "predicted_n": 12, "prompt_ms": 500.0,
                               "predicted_ms": 1500.0}, **final}]


def test_a_call_with_tools_is_the_servers_prompt_raw_and_streamed(monkeypatch):
    sent = _stream(monkeypatch, _chunks(INSERT))
    parts = []
    msg, stats = llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=2048,
                                on_part=lambda *part: parts.append(part))
    (template_url, template), (url, body) = sent
    assert template_url == llm.TEMPLATE_URL
    assert template == {"messages": llm._wire(EDIT), "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}}
    assert url == llm.COMPLETION_URL and body["stream"] is True and body["cache_prompt"] is True
    assert body["prompt"] == {"prompt_string": PROMPT, "multimodal_data": [PNG, PNG, PNG]}
    assert body["n_predict"] == 2048 and body["temperature"] == 0.0 and "json_schema" not in body
    assert msg["content"] == "I'll add it." and msg["tool_calls"][0]["arguments"]["markdown"] == "# T2\n\nA line."
    assert msg["raw_tool_calls"] == [{"function": msg["tool_calls"][0]}]
    assert stats["output_tokens"] == 12 and stats["prompt_tokens"] == 40 and stats["duration_s"] == 2.0
    assert stats["num_ctx"] == llm.RESIDENT_CTX and stats["done_reason"] == "stop"
    assert any(p[0] == "text" for p in parts)
    assert not llm._CALL_LOCK.locked()


def test_a_call_without_images_sends_the_prompt_as_text(monkeypatch):
    sent = _stream(monkeypatch, _chunks(INSERT))
    llm.chat_tools(EDIT[:3], TOOLS, think=True, temperature=0.0, num_predict=64)
    (_url, template), (_url2, body) = sent
    assert template["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["prompt"] == PROMPT


def test_a_call_that_hits_its_cap_says_so(monkeypatch):
    _stream(monkeypatch, _chunks(INSERT, stop_type="limit"))
    _msg, stats = llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64)
    assert stats["done_reason"] == "length" and stats["output_truncated"] is True


def test_stop_abandons_a_streamed_call_and_frees_the_lock(monkeypatch):
    _stream(monkeypatch, _chunks(INSERT, size=1))
    monkeypatch.setattr(llm, "STOP_CHECK_S", 0.0)
    asked = []
    with pytest.raises(llm.Stopped):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64,
                       should_stop=lambda: asked.append(1) or len(asked) > 3)
    assert len(asked) == 4 and not llm._CALL_LOCK.locked()


@pytest.mark.parametrize("failure", [{"error": {"code": 500, "message": "runner crashed"}},
                                     b"error: {\"message\": \"runner crashed\"}\n\n"])
def test_an_error_in_the_stream_is_an_llm_error(monkeypatch, failure):
    _stream(monkeypatch, [{"content": "<tool_call>", "stop": False}, failure])
    with pytest.raises(llm.LLMError, match="runner crashed"):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64)
    assert not llm._CALL_LOCK.locked()


def test_no_test_reaches_a_live_model_through_the_raw_call():
    assert llm.TEMPLATE_URL.startswith("http://127.0.0.1:9/") and llm.COMPLETION_URL.startswith("http://127.0.0.1:9/")
    with pytest.raises(llm.LLMError):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=8)


def test_a_stream_that_ends_before_its_stop_is_an_error(monkeypatch):
    _stream(monkeypatch, _chunks(INSERT)[:-1])
    with pytest.raises(llm.LLMError, match="ended before"):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64)
