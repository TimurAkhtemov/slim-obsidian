"""The prompt a call with tools streams: rendered here as Ollama 0.35's qwen3.5 renderer renders it, byte
for byte, and read back as its parser reads a reply. The expected prompts are Ollama's own
(`_debug_render_only`, 2026-10-13), for synthetic conversations; the transport is faked."""

import base64
import io
import json

import pytest

from slim import llm, llm_render

PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode()
TOOLS = [
    {"type": "function", "function": {"name": "insert", "description": "Insert after or before <b> & a block.",
     "parameters": {"type": "object", "properties": {"after": {"type": "string", "description": "a block (b12)"},
                                                     "before": {"type": "string", "description": "a block (b12)"},
                                                     "markdown": {"type": "string"}}, "required": ["markdown"]}}},
    {"type": "function", "function": {"name": "plan", "description": "Plan: one line per step.",
     "parameters": {"type": "object", "properties": {
         "intro": {"type": "string", "description": "for the owner"},
         "steps": {"type": "array", "items": {"type": "string"}, "description": "1 to 8 steps"}},
         "required": ["intro", "steps"]}}},
]
EDIT = [
    {"role": "system", "content": "You edit notes.\n\n=== Notes/n.md ===\n[b1] # T\n"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "(These are the images embedded in the open note itself.)", "images": [PNG, PNG]},
    {"role": "user", "content": "Fix it", "images": [PNG]},
    {"role": "assistant", "content": "Sure.", "tool_calls": [
        {"function": {"name": "plan", "arguments": {"intro": "A & B <x>", "steps": ["one", "<details> two", "\\frac{a}{b}"]}}},
        {"function": {"name": "insert", "arguments": {"markdown": "# T2\n\n$x$ & y", "after": "b1"}}}]},
    {"role": "tool", "content": "ok: plan accepted"},
    {"role": "tool", "content": "refused: no\n\nStep 1 of 2: do it"},
]
THINK = [
    {"role": "system", "content": "You edit notes."},
    {"role": "user", "content": "Do it"},
    {"role": "assistant", "content": "", "thinking": "  plan it  ", "tool_calls": [
        {"function": {"name": "insert", "arguments": {"after": "b1", "markdown": "x"}}}]},
    {"role": "tool", "content": "ok: inserted"},
]

# Ollama's own rendering of each case.
EDIT_PROMPT = '<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "insert", "description": "Insert after or before \\u003cb\\u003e \\u0026 a block.", "parameters": {"type": "object", "required": ["markdown"], "properties": {"after": {"type": "string", "description": "a block (b12)"}, "before": {"type": "string", "description": "a block (b12)"}, "markdown": {"type": "string"}}}}}\n{"type": "function", "function": {"name": "plan", "description": "Plan: one line\\u2028per step.", "parameters": {"type": "object", "required": ["intro", "steps"], "properties": {"intro": {"type": "string", "description": "for the owner"}, "steps": {"type": "array", "items": {"type": "string"}, "description": "1 to 8 steps"}}}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou edit notes.\n\n=== Notes/n.md ===\n[b1] # T<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>user\n[img-0][img-1] (These are the images embedded in the open note itself.)<|im_end|>\n<|im_start|>user\n[img-2] Fix it<|im_end|>\n<|im_start|>assistant\nSure.\n\n<tool_call>\n<function=plan>\n<parameter=intro>\nA & B <x>\n</parameter>\n<parameter=steps>\n["one","\\u003cdetails\\u003e two","\\\\frac{a}{b}"]\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=insert>\n<parameter=markdown>\n# T2\n\n$x$ & y\n</parameter>\n<parameter=after>\nb1\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nok: plan accepted\n</tool_response>\n<tool_response>\nrefused: no\n\nStep 1 of 2: do it\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
THINK_PROMPT = '<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"type": "function", "function": {"name": "insert", "description": "Insert after or before \\u003cb\\u003e \\u0026 a block.", "parameters": {"type": "object", "required": ["markdown"], "properties": {"after": {"type": "string", "description": "a block (b12)"}, "before": {"type": "string", "description": "a block (b12)"}, "markdown": {"type": "string"}}}}}\n{"type": "function", "function": {"name": "plan", "description": "Plan: one line\\u2028per step.", "parameters": {"type": "object", "required": ["intro", "steps"], "properties": {"intro": {"type": "string", "description": "for the owner"}, "steps": {"type": "array", "items": {"type": "string"}, "description": "1 to 8 steps"}}}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou edit notes.<|im_end|>\n<|im_start|>user\nDo it<|im_end|>\n<|im_start|>assistant\n<think>\nplan it\n</think>\n\n<tool_call>\n<function=insert>\n<parameter=after>\nb1\n</parameter>\n<parameter=markdown>\nx\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nok: inserted\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n'
GATE_PROMPT = '<|im_start|>system\nYou edit notes.\n\n=== Notes/n.md ===\n[b1] # T<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello'


@pytest.mark.parametrize("messages, tools, think, prompt", [(EDIT, TOOLS, False, EDIT_PROMPT),
                                                           (THINK, TOOLS, True, THINK_PROMPT),
                                                           (EDIT[:3], None, False, GATE_PROMPT)])
def test_the_prompt_is_the_one_ollama_renders(messages, tools, think, prompt):
    assert llm_render.render(messages, tools, think) == prompt


# --- a reply as it decodes ----------------------------------------------------------------------

INSERT = ("I'll add it.\n\n<tool_call>\n<function=insert>\n<parameter=after>\nb1\n</parameter>\n"
          "<parameter=markdown>\n# T2\n\nA line.\n</parameter>\n</function>\n</tool_call>")


def read(text, *, think=False, size=1, tools=TOOLS):
    parts = []
    reader = llm_render._Reader(think, tools, lambda *part: parts.append(part))
    for i in range(0, len(text), size):
        reader.feed(text[i:i + size])
    return reader.close(), parts


@pytest.mark.parametrize("size", [1, 2, 5, 11, len(INSERT)])
def test_a_tool_call_is_read_the_same_however_the_reply_is_cut(size):
    (content, thinking, calls, raw), parts = read(INSERT, size=size)
    assert content == "I'll add it." and thinking == 0
    assert calls == [{"name": "insert", "arguments": {"after": "b1", "markdown": "# T2\n\nA line."}}]
    assert raw == [{"function": {"name": "insert", "arguments": {"after": "b1", "markdown": "# T2\n\nA line."}}}]
    assert [p for p in parts if p[0] != "text"] == [
        ("call", 0, "insert", None), ("parameter", 0, "after", None), ("value", 0, "after", "b1"),
        ("parameter", 0, "markdown", None), ("value", 0, "markdown", "# T2\n\nA line."), ("end", 0, None, None)]
    assert "".join(p[3] for p in parts if p[:3] == ("text", 0, "markdown")) == "# T2\n\nA line."


def test_a_parameters_text_arrives_as_it_decodes():
    reader, seen = llm_render._Reader(False, TOOLS, lambda *part: seen.append(part)), []
    reader.feed("<tool_call>\n<function=insert>\n<parameter=markdown>\nFirst words")
    assert [p[3] for p in seen if p[0] == "text"] == ["First words"]
    reader.feed(" and more\n</para")                        # what may be the closing tag waits
    assert "".join(p[3] for p in seen if p[0] == "text") == "First words and more"
    reader.feed("graph></p>\n</parameter>\n</function>\n</tool_call>")
    assert "".join(p[3] for p in seen if p[0] == "text") == "First words and more\n</paragraph></p>"


def test_values_are_typed_by_the_tools_schema_as_ollama_types_them():
    text = ("<tool_call>\n<function=plan>\n<parameter=intro>\nnull\n</parameter>\n<parameter=steps>\n"
            '["a", "<b>"]\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=plan>\n'
            "<parameter=steps>\n[not json\n</parameter>\n<parameter=other>\n[1]\n</parameter>\n</function>\n</tool_call>")
    (_content, _thinking, calls, _raw), _ = read(text, size=7)
    assert calls == [{"name": "plan", "arguments": {"intro": "null", "steps": ["a", "<b>"]}},
                     {"name": "plan", "arguments": {"steps": "[not json", "other": "[1]"}}]
    (_content, _thinking, calls, _raw), _ = read("<tool_call>\n<function=plan>\n<parameter=steps>\nNull\n</parameter>\n"
                                                 "<parameter=other>\nnull\n</parameter>\n</function>\n</tool_call>")
    assert calls == [{"name": "plan", "arguments": {"steps": None, "other": None}}]


def test_a_text_parameter_written_null_is_that_text_and_renders_as_ollamas_null():
    # Ollama types any "null" as None, so "change undefined to null" deleted the word (review,
    # 2026-10-13). A string parameter's text is its value; written back, it is the same bytes.
    text = ("<tool_call>\n<function=insert>\n<parameter=after>\nb1\n</parameter>\n<parameter=markdown>\nnull\n"
            "</parameter>\n</function>\n</tool_call>")
    (_content, _thinking, calls, raw), parts = read(text, size=3)
    assert calls == [{"name": "insert", "arguments": {"after": "b1", "markdown": "null"}}]
    assert ("value", 0, "markdown", "null") in parts
    ollamas = [{"function": {"name": "insert", "arguments": {"after": "b1", "markdown": None}}}]
    history = lambda calls: [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": calls},
                             {"role": "tool", "content": "ok: inserted"}]
    assert llm_render.render(history(raw), TOOLS, False) == llm_render.render(history(ollamas), TOOLS, False)


def test_thinking_ends_at_its_close_or_at_a_tool_call():
    for text in ["<think>\nPondering.\n</think>\n\nOK." + INSERT, "Pondering.\n</think>\n\nOK." + INSERT,
                 "Pondering." + INSERT[INSERT.index("<tool_call>"):]]:
        (content, thinking, calls, _raw), _ = read(text, think=True, size=3)
        assert thinking > 0 and calls[0]["arguments"]["after"] == "b1"
        assert content in ("OK.I'll add it.", "")


def test_text_around_calls_is_joined_as_ollama_joins_it_and_an_unclosed_call_is_text():
    call = "<tool_call>\n<function=insert>\n<parameter=markdown>\nx\n</parameter>\n</function>\n</tool_call>"
    (content, _t, calls, _raw), parts = read("Intro\n<tool_call>\n<function=insert>\n<parameter=markdown>\npartial")
    assert calls == [] and content == "Intro<tool_call>\n<function=insert>\n<parameter=markdown>\npartial"
    assert ("end", 0, None, None) not in parts


# --- the call on the wire -----------------------------------------------------------------------

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


# --- review (2026-10-13): what Go rejects is never a call; what Go normalizes, so is it here ---

def test_a_parameter_never_closed_is_no_call_however_its_call_ends():
    text = ("<tool_call>\n<function=replace>\n<parameter=target>\nb2\n</parameter>\n<parameter=markdown>\n"
            "Partial\n</function></tool_call>")
    for size in (1, 7, len(text)):
        (content, _t, calls, _raw), parts = read(text, size=size)
        assert calls == [] and content.startswith("<tool_call>") and ("end", 0, None, None) not in parts


def test_a_call_whose_function_is_never_closed_is_no_call():
    (content, _t, calls, _raw), _ = read("<tool_call>\n<function=insert>\n<parameter=after>\nb1\n</parameter>\n</tool_call>")
    assert calls == [] and content.startswith("<tool_call>")


def test_a_call_that_turns_into_text_is_dropped_and_the_next_call_takes_its_index():
    dead = "<tool_call>\n<function=replace>\n<parameter=target>\nb2\n</parameter>\n</tool_call>\n"
    for size in (1, 7, len(dead + INSERT)):
        (_content, _t, calls, _raw), parts = read(dead + INSERT, size=size)
        assert [p[:2] for p in parts if p[0] in ("call", "drop", "end")] == [("call", 0), ("drop", 0), ("call", 0),
                                                                             ("end", 0)]
        assert calls == [{"name": "insert", "arguments": {"after": "b1", "markdown": "# T2\n\nA line."}}]
    (_content, _t, calls, _raw), parts = read("<tool_call>\n<function=insert>\n<parameter=markdown>\npartial")
    assert calls == [] and parts[-1] == ("drop", 0, None, None)      # cut off: dropped as the reply closes


def test_line_endings_are_normalized_before_a_values_framing_newlines_go():
    text = "<tool_call>\r\n<function=insert>\r\n<parameter=markdown>\r\nx\r\ny\r\n</parameter>\r\n</function>\r\n</tool_call>"
    for size in (1, 2, len(text)):
        (_content, _t, calls, _raw), parts = read(text, size=size)
        assert calls[0]["arguments"]["markdown"] == "x\ny"
        assert "".join(p[3] for p in parts if p[:3] == ("text", 0, "markdown")) == "x\ny"


def test_what_follows_a_call_in_the_same_piece_is_trimmed_as_ollama_trims_it():
    call = "<tool_call>\n<function=insert>\n<parameter=markdown>\nx\n</parameter>\n</function>\n</tool_call>"
    assert read(f"A\n{call}\nB\n{call}", size=1000)[0][0] == "AB"
    assert read(f"A\n{call}\nB\n{call}", size=1)[0][0] == "A\nB"     # the token after </tool_call> is its own piece


def test_a_stream_that_ends_before_done_is_an_error(monkeypatch):
    _stream(monkeypatch, _chunks(INSERT)[:-1])
    with pytest.raises(llm.LLMError, match="ended before"):
        llm.chat_tools(EDIT, TOOLS, think=False, temperature=0.0, num_predict=64)


def test_numbers_in_history_arguments_are_written_as_go_writes_them():
    message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "plan", "arguments": {
        "steps": [1.0, 1e-7, 0.00001, 2.5, 1e21, 7, True, None, {"b": 1, "a": "<"}]}}}]}
    prompt = llm_render.render([{"role": "user", "content": "q"}, message, {"role": "tool", "content": "ok"}], TOOLS, False)
    assert '["1","1e-7","1e-05"' not in prompt
    assert '[1,1e-7,0.00001,2.5,1e+21,7,true,null,{"a":"\\u003c","b":1}]' in prompt
