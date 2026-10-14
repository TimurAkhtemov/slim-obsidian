"""The prompt of a call with tools, rendered as Ollama renders it, and its reply read as it
decodes. No network and no lock: `llm` sends the prompt and feeds the reply through `_Reader`.
"""

import json
import math
import re
from decimal import Decimal

# --- a call with tools: the prompt rendered here, the reply read as it decodes ------------------
#
# /api/chat runs the model's own parser (Ollama 0.35's `qwen3.5`) on every reply, tools or not, and
# it holds a tool call back until `</tool_call>`: an insert's 700 characters arrived in one piece
# at the end (measured 2026-10-13, /api/chat with no tools=: 0 content chunks, the call whole at
# 3.66 s). A RAW /api/generate runs no parser, so the call streams; its prompt is rendered here, as
# that renderer renders it (model/renderers/qwen35.go with RenderImgTags on, json.go, image_tags.go,
# qwen3coder.go's formatToolCallArgument). Byte for byte: equal to Ollama's own `_debug_render_only`
# for every call of a real run, and a raw call after a chat call re-read 2 of its 8,006 prompt
# tokens. tests/test_llm_render.py pins it against Ollama's rendering.

_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_TAIL = """
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""
# What Go's strings.TrimSpace trims (unicode.IsSpace); Python's strip() also takes U+001C-U+001F.
_SPACE = ("\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
          "\u2028\u2029\u202f\u205f\u3000")


def _trim(text: str) -> str:
    return text.strip(_SPACE)


def _go_json(value, sort: bool = True) -> str:
    """Go's json.Marshal: compact, `<>&` and U+2028/9 escaped, a map's keys sorted, and every
    number a float64, as Ollama holds any number it read from JSON."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _go_number(value)
    if isinstance(value, list):
        return "[" + ",".join(_go_json(v, sort) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items()) if sort else value.items()
        return "{" + ",".join(f"{_go_json(str(k))}:{_go_json(v, sort)}" for k, v in items) + "}"
    out = json.dumps(str(value), ensure_ascii=False)
    return (out.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
               .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _go_number(value) -> str:
    """A float64 as encoding/json writes it: its shortest digits, positional between 1e-6 and
    1e21, otherwise with an exponent (1e-7, 1.5e+21)."""
    number = float(value)
    if number == 0:
        return "-0" if math.copysign(1, number) < 0 else "0"
    digits = Decimal(repr(number)).normalize()
    return format(digits, "e" if not 1e-6 <= abs(number) < 1e21 else "f")


def _spaced(compact: str) -> str:
    """marshalWithSpaces: a space after each ':' and ',' outside strings."""
    out, quoted, escaped = [], False, False
    for c in compact:
        out.append(c + " " if not quoted and c in ":," else c)
        if escaped:
            escaped = False
        elif quoted and c == "\\":
            escaped = True
        elif c == '"':
            quoted = not quoted
    return "".join(out)


def _property(prop: dict) -> dict:
    """api.ToolProperty's fields in Go's order, empty ones left out; `items` is a map, so sorted."""
    out = {}
    if prop.get("anyOf"):
        out["anyOf"] = [_property(p) for p in prop["anyOf"]]
    kind = prop.get("type")
    if kind:
        out["type"] = kind[0] if isinstance(kind, list) and len(kind) == 1 else kind
    if prop.get("items") is not None:
        out["items"] = json.loads(json.dumps(prop["items"], sort_keys=True))
    for key in ("description", "enum"):
        if prop.get(key):
            out[key] = prop[key]
    if prop.get("properties"):
        out["properties"] = {k: _property(v) for k, v in prop["properties"].items()}
    if prop.get("required"):
        out["required"] = prop["required"]
    return out


def _tool_json(tool: dict) -> str:
    """A tool as the prompt lists it: type, function {name, description, parameters {type,
    required, properties}}, the properties in the order given."""
    fn = tool["function"]
    params = fn.get("parameters") or {}
    shown = {"type": params.get("type", "object")}
    if params.get("required"):
        shown["required"] = params["required"]
    shown["properties"] = {k: _property(v) for k, v in (params.get("properties") or {}).items()}
    function = {"name": fn["name"], **({"description": fn["description"]} if fn.get("description") else {}),
                "parameters": shown}
    return _spaced(_go_json({"type": tool.get("type", "function"), "function": function}, sort=False))


def _argument(value) -> str:
    """An argument of a call in the history: text as it is, a list or a map as compact JSON."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return _go_json(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _content(message: dict, offset: int) -> tuple[str, int]:
    """A message's text with an [img-N] for each of its images before it, numbered across the
    prompt; the raw call sends the images in that order."""
    text, count = message.get("content") or "", len(message.get("images") or ())
    if not count:
        return text, offset
    if "[img-" in text:
        return text, offset + count
    prefix = ""
    for n in range(offset, offset + count):
        if "[img]" in text:
            text = text.replace("[img]", f"[img-{n}]", 1)
        else:
            prefix += f"[img-{n}]"
    if prefix and text and text[0] not in _SPACE:
        prefix += " "
    return prefix + text, offset + count


def _reasoning(content: str, thinking: str, shown: bool) -> tuple[str, str]:
    if shown and thinking:
        return _trim(thinking), content
    reasoning, at = "", content.find("</think>")
    if at != -1:
        before = content[:at]
        opened = before.rfind("<think>")
        reasoning = before[opened + len("<think>"):] if opened != -1 else before
        content = content[at + len("</think>"):].lstrip("\n")
    return _trim(reasoning), content


def render(messages: list[dict], tools: list[dict] | None, think: bool) -> str:
    """The prompt Ollama builds from these messages and tools, ending in the assistant's open turn:
    `<think>\\n` when it thinks, an empty think block when it does not."""
    out = []
    if tools:
        out.append(_IM_START + "system\n" + _TOOLS_HEAD + "".join("\n" + _tool_json(t) for t in tools) + _TOOLS_TAIL)
        if messages and messages[0]["role"] == "system":
            system = _trim(_content(messages[0], 0)[0])
            if system:
                out.append("\n\n" + system)
        out.append(_IM_END + "\n")
    elif messages and messages[0]["role"] == "system":
        out.append(_IM_START + "system\n" + _trim(_content(messages[0], 0)[0]) + _IM_END + "\n")
    query = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user" and not (
        (text := _trim(_content(messages[i], 0)[0])).startswith("<tool_response>") and text.endswith("</tool_response>"))),
        len(messages) - 1)
    offset = 0
    for i, message in enumerate(messages):
        content, offset = _content(message, offset)
        content = _trim(content)
        role, last = message["role"], i == len(messages) - 1
        if role == "user" or (role == "system" and i):
            out.append(_IM_START + role + "\n" + content + _IM_END + "\n")
        elif role == "assistant":
            shown = think and i > query
            reasoning, content = _reasoning(content, message.get("thinking") or "", shown)
            out.append(_IM_START + "assistant\n" + (f"<think>\n{reasoning}\n</think>\n\n" if shown else "") + content)
            for j, call in enumerate(message.get("tool_calls") or ()):
                out.append(("\n\n" if _trim(content) else "") if j == 0 else "\n")
                fn = call["function"]
                out.append(f"<tool_call>\n<function={fn['name']}>\n" + "".join(
                    f"<parameter={k}>\n{_argument(v)}\n</parameter>\n" for k, v in (fn.get("arguments") or {}).items())
                    + "</function>\n</tool_call>")
            if not last:
                out.append(_IM_END + "\n")
        elif role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                out.append(_IM_START + "user")
            out.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if last or messages[i + 1]["role"] != "tool":
                out.append(_IM_END + "\n")
        if last and role != "assistant":
            out.append(_IM_START + "assistant\n" + ("<think>\n" if think else "<think>\n\n</think>\n\n"))
    return "".join(out)


_FUNCTION = re.compile(r"<function=([^>\n]+)>")
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>")
_CALL_OPEN, _CALL_CLOSE, _VALUE_CLOSE, _FUNCTION_CLOSE = "<tool_call>", "</tool_call>", "</parameter>", "</function>"


def _lf(text: str) -> str:
    """Line endings as XML reads them (Ollama parses a call as XML): CR LF and a lone CR are LF.
    A CR at the end waits: its LF may come next."""
    held = text.endswith("\r")
    text = text[:-1] if held else text
    return text.replace("\r\n", "\n").replace("\r", "\n") + ("\r" if held else "")


def _held(text: str, *tags: str) -> int:
    """How much of the end of `text` may be the start of one of `tags`: it waits for more."""
    return max((n for tag in tags for n in range(min(len(text), len(tag) - 1), 0, -1)
                if tag.startswith(text[-n:])), default=0)


class _Reader:
    """A reply as it decodes, read as Ollama's qwen3.5 parser reads one (model/parsers/qwen35.go and
    qwen3coder.go): thinking up to `</think>`, text, and `<tool_call>` blocks, each parameter's value
    typed by the tool's schema. Unlike that parser it hands a value on as it arrives:
    `on_part(kind, index, name, value)` hears ("call", i, tool, None), ("parameter", i, name, None),
    ("text", i, name, chunk), ("value", i, name, value) and ("end", i, None, None) — or, for a call
    that turns into text, ("drop", i, None, None), and the next call is i again."""

    def __init__(self, think: bool, tools: list[dict] | None, on_part=None):
        self.types = {t["function"]["name"]: {k: (v or {}).get("type") for k, v in
                                              ((t["function"].get("parameters") or {}).get("properties") or {}).items()}
                      for t in tools or ()}
        self.on_part = on_part or (lambda *_part: None)
        self.state, self.lead = ("think", True) if think else ("text", False)
        self.buf, self.text, self.thinking, self.calls = "", [], 0, []
        self.skip = False                     # the whitespace after </think>, however it arrives
        self.call = None                      # {"name", "arguments", "written", "closed"}: the call being read
        self.param = None                     # [name, text so far, its leading newline to come]

    def feed(self, chunk: str) -> None:
        self.buf += chunk
        while {"think": self._think, "text": self._text, "call": self._in_call, "value": self._value}[self.state]():
            pass

    def _eat(self, n: int) -> str:
        eaten, self.buf = self.buf[:n], self.buf[n:]
        if self.call is not None:
            self.call["written"] += eaten
        return eaten

    def _think(self) -> bool:
        if self.lead:                          # one <think> may open the reply
            head = self.buf.lstrip(_SPACE)
            if head.startswith("<think>"):
                self.buf, self.lead = head[len("<think>"):].lstrip(_SPACE), False
                return bool(self.buf)
            if "<think>".startswith(head):
                return False
            self.lead = False
        for tag, kept in (("</think>", False), (_CALL_OPEN, True)):   # a call ends thinking too
            at = self.buf.find(tag)
            if at >= 0:
                self.thinking += at
                self.buf = self.buf[at:] if kept else self.buf[at + len(tag):]
                self.skip, self.state = not kept, "text"
                return True
        n = len(self.buf) - _held(self.buf, "</think>", _CALL_OPEN)
        self.thinking, self.buf = self.thinking + n, self.buf[n:]
        return False

    def _text(self) -> bool:
        if self.skip:
            self.buf = self.buf.lstrip(_SPACE)
            self.skip = not self.buf
        at = self.buf.find(_CALL_OPEN)
        if at < 0:
            return False
        self.text.append(self.buf[:at].rstrip(_SPACE))
        self.buf = self.buf[at + len(_CALL_OPEN):]
        self.call, self.state = {"name": None, "arguments": {}, "written": "", "closed": False}, "call"
        return True

    def _in_call(self) -> bool:
        self.buf = _lf(self.buf)
        index, end = len(self.calls), self.buf.find(_CALL_CLOSE)
        if self.call["name"] is None:
            found = _FUNCTION.search(self.buf)
            if found and (end < 0 or found.start() < end):
                self._eat(found.end())
                self.call["name"] = found.group(1).strip()
                self.on_part("call", index, self.call["name"], None)
                return True
            if end >= 0:                       # a call with no function: text, as the model wrote it
                self.text.append(_CALL_OPEN + self.call["written"] + self.buf[:end + len(_CALL_CLOSE)])
                self.buf, self.call, self.state = self.buf[end + len(_CALL_CLOSE):], None, "text"
                return True
            return False
        found, closing = _PARAMETER.search(self.buf), self.buf.find(_FUNCTION_CLOSE)
        first = min((at for at in (found.start() if found else -1, closing, end) if at >= 0), default=-1)
        if first < 0:
            return False
        if found and first == found.start():
            self._eat(found.end())
            self.param, self.state = [found.group(1).strip(), "", True], "value"
            self.on_part("parameter", index, self.param[0], None)
            return True
        if first == closing:
            self._eat(closing + len(_FUNCTION_CLOSE))
            self.call["closed"] = True
            return True
        if not self.call["closed"]:            # XML Ollama would refuse: text, never a call
            self.text.append(_CALL_OPEN + self.call["written"] + self.buf[:end + len(_CALL_CLOSE)])
            self.buf, self.call, self.state = self.buf[end + len(_CALL_CLOSE):], None, "text"
            self.on_part("drop", index, None, None)
            return True
        self._eat(end + len(_CALL_CLOSE))
        self.calls.append({"name": self.call["name"], "arguments": self.call["arguments"]})
        # What follows in the same piece is trimmed, as Ollama trims it; the token after a call
        # is a piece of its own, so in a stream it never is.
        self.call, self.state, self.buf = None, "text", self.buf.lstrip(_SPACE)
        self.on_part("end", index, None, None)
        return True

    def _value(self) -> bool:
        self.buf = _lf(self.buf)
        index, (name, said, lead) = len(self.calls), self.param
        if lead:                               # one leading newline is not the value's
            if not self.buf or self.buf == "\r":
                return False
            if self.buf[0] == "\n":
                self._eat(1)
            self.param[2] = False
        end = self.buf.find(_VALUE_CLOSE)      # only its own close ends a value, as in XML
        if end < 0:
            n = len(self.buf) - max(_held(self.buf, "\n" + _VALUE_CLOSE, _VALUE_CLOSE), self.buf.endswith("\r"))
            if n > 0:
                chunk = self._eat(n)
                self.param[1] += chunk
                self.on_part("text", index, name, chunk)
            return False
        value = said + self.buf[:end]
        value = value[:-1] if value.endswith("\n") else value     # nor one trailing newline
        if len(value) > len(said):
            self.on_part("text", index, name, value[len(said):])
        self._eat(end + len(_VALUE_CLOSE))
        typed = self._typed(self.call["name"], name, value)
        self.call["arguments"][name] = typed
        self.param, self.state = None, "call"
        self.on_part("value", index, name, typed)
        return True

    def _typed(self, tool: str, name: str, value: str):
        """As Ollama types it: "null" is None, and an array or object parameter is JSON if it parses.
        But a string parameter's text is its value, "null" too: Ollama's None made "change undefined
        to null" delete the word. Written back into the prompt, both are `null`."""
        kind = self.types.get(tool, {}).get(name)
        kinds = kind if isinstance(kind, list) else [kind]
        if value.lower() == "null" and "string" not in kinds:
            return None
        for want, shape in (("array", list), ("object", dict)):
            if want in kinds:
                try:
                    parsed = json.loads(value)
                except ValueError:
                    continue
                if isinstance(parsed, shape):
                    return parsed
        return value

    def close(self) -> tuple[str, int, list[dict], list[dict]]:
        """(content, thinking characters, tool calls, the calls as the conversation carries them).
        A call never closed is text, as Ollama leaves it: the caller reads it as malformed."""
        if self.state == "think":
            self.thinking += len(self.buf)
        elif self.state == "text":
            self.text.append(self.buf)
        else:
            self.text.append(_CALL_OPEN + self.call["written"] + self.buf)
            if self.call["name"] is not None:
                self.on_part("drop", len(self.calls), None, None)
        self.buf = ""
        return ("".join(self.text), self.thinking, self.calls,
                [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in self.calls])

