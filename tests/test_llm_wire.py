"""What `llm` puts on the wire. qwen3.6 thinks unless told not to, and the thinking eats
`num_predict`: a body without `think` returns content="" and the recorder dies at its first
call. Every other test fakes the model's answer; these decode the request itself."""

import io
import json

import pytest

from slim import llm
from tests.test_llm_reader import INSERT
from tests.test_llm_render import EDIT, EDIT_PROMPT, PNG, TOOLS


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


# --- a call with tools: the rendered prompt, raw and streamed -------------------------------------

def _stream(monkeypatch, lines):
    sent = []

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, json.loads(req.data)))
        return io.BytesIO(b"".join(json.dumps(line).encode() + b"\n" for line in lines))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return sent


def _chunks(text, size=4, **final):
    out = [{"response": text[i:i + size], "done": False} for i in range(0, len(text), size)]
    return out + [{"response": "", "done": True, "prompt_eval_count": 40, "eval_count": 12, "done_reason": "stop",
                   "total_duration": 2_000_000_000, "prompt_eval_duration": 500_000_000,
                   "eval_duration": 1_000_000_000, **final}]


def test_a_call_with_tools_is_the_rendered_prompt_raw_and_streamed(monkeypatch):
    sent = _stream(monkeypatch, _chunks(INSERT))
    parts = []
    msg, stats = llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=2048,
                                on_part=lambda *part: parts.append(part))
    ((url, body),) = sent
    assert url == llm.OLLAMA_GENERATE and body["raw"] is True and body["stream"] is True
    assert body["prompt"] == EDIT_PROMPT and body["images"] == [PNG, PNG, PNG] and body["think"] is False
    assert body["model"] == llm.MODEL and body["keep_alive"] == llm.KEEP_ALIVE
    assert body["options"] == {"num_ctx": llm.RESIDENT_CTX, "temperature": 0.0, "num_predict": 2048}
    assert msg["content"] == "I'll add it." and msg["tool_calls"][0]["arguments"]["markdown"] == "# T2\n\nA line."
    assert msg["raw_tool_calls"] == [{"function": msg["tool_calls"][0]}]
    assert stats["output_tokens"] == 12 and stats["num_ctx"] == llm.RESIDENT_CTX and stats["done_reason"] == "stop"
    assert any(p[0] == "text" for p in parts)
    assert not llm._CALL_LOCK.locked()


def test_stop_abandons_a_streamed_call_and_frees_the_lock(monkeypatch):
    _stream(monkeypatch, _chunks(INSERT, size=1))
    monkeypatch.setattr(llm, "STOP_CHECK_S", 0.0)
    asked = []
    with pytest.raises(llm.Stopped):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64,
                       should_stop=lambda: asked.append(1) or len(asked) > 3)
    assert len(asked) == 4 and not llm._CALL_LOCK.locked()


def test_an_error_in_the_stream_is_an_llm_error(monkeypatch):
    _stream(monkeypatch, [{"response": "<tool_call>", "done": False}, {"error": "runner crashed"}])
    with pytest.raises(llm.LLMError, match="runner crashed"):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64)
    assert not llm._CALL_LOCK.locked()


def test_no_test_reaches_a_live_model_through_the_raw_call():
    assert llm.OLLAMA_GENERATE.startswith("http://127.0.0.1:9/")
    with pytest.raises(llm.LLMError):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=8)


def test_a_stream_that_ends_before_done_is_an_error(monkeypatch):
    _stream(monkeypatch, _chunks(INSERT)[:-1])
    with pytest.raises(llm.LLMError, match="ended before"):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64)
