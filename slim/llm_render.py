"""A call with tools: its reply read as it decodes. No network and no lock: `llm` streams the
reply of a raw completion through `_Reader`.

The prompt is the server's own rendering (`llm._tools_raw`). Until 2026-10-23 it was rendered
here, byte for byte as Ollama 0.35's `qwen3.5` renderer built it, because Ollama's /api/chat held a
tool call back until it was whole. llama-server renders with the model's own chat template; the
reply's format is the model's, Qwen's tool-call XML, and is read here as before.
"""

import json
import re

# --- the reply, read as it decodes ---------------------------------------------------------------

# What Go's strings.TrimSpace trims (unicode.IsSpace); Python's strip() also takes U+001C-U+001F.
_SPACE = ("\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
          "\u2028\u2029\u202f\u205f\u3000")


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
    """A reply as it decodes, read as Ollama's qwen3.5 parser read one (model/parsers/qwen35.go and
    qwen3coder.go), which SLIM's editing run was built against: thinking up to `</think>`, text, and
    `<tool_call>` blocks, each parameter's value typed by the tool's schema. Unlike that parser it
    hands a value on as it arrives:
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
        to null" delete the word. Written back into the prompt, the text renders as itself."""
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

