"""A reply read back as Ollama 0.35's qwen3.5 parser reads it, piece by piece as it decodes."""

import pytest

from slim import llm_render
from tests.test_llm_render import TOOLS


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
