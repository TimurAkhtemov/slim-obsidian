"""SLIM's messages as the server reads them (`llm._wire`), and its tool calls read back
(`llm._calls`). The server renders the prompt itself, with the model's own chat template; what is
pinned here is what it is given. The reply's parse is `test_llm_reader.py`; the calls on the wire,
`test_llm_wire.py`."""

import base64
import json

from slim import llm

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


def _image(data):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def test_a_messages_images_are_content_parts_before_its_text():
    wired = llm._wire(EDIT)
    assert wired[1] == {"role": "user", "content": "hi"}
    assert wired[3]["content"] == [_image(PNG), _image(PNG),
                                   {"type": "text", "text": "(These are the images embedded in the open note itself.)"}]
    assert wired[4]["content"] == [_image(PNG), {"type": "text", "text": "Fix it"}]


def test_tool_calls_go_out_as_json_text_and_come_back_as_the_same_objects():
    (wired,) = [m for m in llm._wire(EDIT) if m.get("tool_calls")]
    sent = [call["function"]["arguments"] for call in wired["tool_calls"]]
    assert all(isinstance(text, str) for text in sent)
    assert [json.loads(text) for text in sent] == [call["function"]["arguments"] for call in EDIT[5]["tool_calls"]]
    assert "<details> two" in sent[0]                  # written as it is, not escaped for HTML
    assert llm._calls(wired["tool_calls"]) == EDIT[5]["tool_calls"]


def test_thinking_is_reasoning_content_and_tool_results_stay_tool_messages():
    wired = llm._wire(THINK)
    assert wired[2]["reasoning_content"] == "  plan it  " and wired[2]["content"] == ""
    assert wired[3] == {"role": "tool", "content": "ok: inserted"}


def test_an_image_goes_with_its_own_type():
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0 a jpeg").decode()
    assert llm._image(jpeg)["image_url"]["url"] == "data:image/jpeg;base64," + jpeg
    assert llm._image(PNG) == _image(PNG)
