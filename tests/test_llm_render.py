"""The prompt a call with tools streams: rendered here as Ollama 0.35's qwen3.5 renderer renders it, byte
for byte, and read back as its parser reads a reply. The expected prompts are Ollama's own
(`_debug_render_only`, 2026-10-13), for synthetic conversations. The reply's parse is
`test_llm_reader.py`; the streamed call, `test_llm_wire.py`."""

import base64

import pytest

from slim import llm_render

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


# --- review (2026-10-13): what Go rejects is never a call; what Go normalizes, so is it here ---

def test_numbers_in_history_arguments_are_written_as_go_writes_them():
    message = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "plan", "arguments": {
        "steps": [1.0, 1e-7, 0.00001, 2.5, 1e21, 7, True, None, {"b": 1, "a": "<"}]}}}]}
    prompt = llm_render.render([{"role": "user", "content": "q"}, message, {"role": "tool", "content": "ok"}], TOOLS, False)
    assert '["1","1e-7","1e-05"' not in prompt
    assert '[1,1e-7,0.00001,2.5,1e+21,7,true,null,{"a":"\\u003c","b":1}]' in prompt
