"""Ollama chat client. The single place a local model gets called.

Two rules are enforced here rather than trusted to callers. `num_ctx` is ALWAYS
set explicitly, because Ollama's default is 4096 tokens and it truncates silently.
Structured calls pass a JSON schema to Ollama's `format`, so the decoder is
constrained to valid JSON — but a constrained decode still produces schema-valid
nonsense, so the caller validates meaning, not syntax.
"""

import json
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, nullcontext

OLLAMA_CHAT = "http://localhost:11434/api/chat"
OLLAMA_GENERATE = "http://localhost:11434/api/generate"
# A streamed call asks its caller's should_stop() this often: a write to a closed client is how
# Stop is noticed, and a stopped call frees the model at once rather than at its last token.
STOP_CHECK_S = 1.0

# ONE live call at a time. The chat surface is a ThreadingHTTPServer, and the KV cache is
# per-runner: concurrent calls evict each other's prefix, which is the whole reason a
# growing conversation is affordable. Serializing costs a queued caller some latency; not
# serializing costs every caller its cached prefill. Held across the ENTIRE stream in
# chat_stream, and across a whole editing run through `hold()`: a request slipping in between
# two steps would evict the prefix every later step reads. Nothing inside a call may re-enter
# this module (on_delta must not call the model) — the non-reentrant Lock is deliberate, so
# that mistake fails loudly.
_CALL_LOCK = threading.Lock()
_HELD = threading.local()


@contextmanager
def hold():
    """Hold the call lock across several calls. Calls this thread makes inside it go straight
    through; every other thread's call waits until it exits."""
    if getattr(_HELD, "on", False):
        yield
        return
    with _CALL_LOCK:
        _HELD.on = True
        try:
            yield
        finally:
            _HELD.on = False


def _lock():
    return nullcontext() if getattr(_HELD, "on", False) else _CALL_LOCK

# Resident model. MoE only — dense 31–32B models are disqualified for interactive use on this
# machine (~95 tok/s prefill measured). Switched to qwen3.6:35b on 2026-07-14 — THE OWNER'S CALL,
# and the summarization evidence argued against it:
#
#   summarization, on 98 real meetings (experiment M4-summarization, in the vault's slim-docs)
#                            over-extraction    fabrication    coverage
#     qwen3:30b (previous)      3% (1/31)          0.086         0.579
#     qwen3.6:35b (now)        13% (4/31)          0.093         0.606
#
#   i.e. +2.7pp coverage for 4x the rate of inventing commitments nobody made. On the
#   summarization task this is the worse model.
#
# Answering rewards saying only what the evidence supports; summarizing rewards saying
# nothing when nobody committed. The grounding-best model is the summarization-worst, and
# "qwen for everything" accepts that trade (2026-09-01).
#
# ⚠ qwen3.6 THINKS BY DEFAULT, and Ollama bills thinking tokens against num_predict while
#   returning them in a separate field. Every call path therefore sets `think` EXPLICITLY —
#   never by inheriting the default. Left unset, the model spends its whole output budget
#   reasoning and returns content="", and the call dies at the first step.
#   `tests/test_llm_wire.py` pins it.
MODEL = "qwen3.6:35b"

# ⚠ ONE context size for every call, and it is not a style choice. OLLAMA SPAWNS A SEPARATE
# RUNNER PER (model, num_ctx) PAIR. Two sizes in one flow UNLOAD 19 GB of weights and reload
# them at the other size, mid-query — `ollama ps` sitting in "Stopping..." while the GPU
# thrashes, turning a ~20s answer into a hang. A SMALLER bespoke window is the same cliff from
# the other side: a 32k call beside the loaded 128k resident is a SECOND ~23 GB runner
# (measured 2026-08-27). One size, everywhere.
#
# 128k since 2026-08-26, and the number is a MEASUREMENT, not a preference. `ollama ps`
# on this machine, same model, one size at a time:
#
#     32768 -> 23 GB     65536 -> 24 GB     131072 -> 25 GB      (all 100% GPU)
#
# ~16 MB of KV per 1k tokens, because qwen35moe is MoE with a 2048 embedding length. The
# weights dominate and the window is nearly free — none of which transfers to a dense model.
#
# It buys headroom for images and fewer summaries lost to ctx_saturated. It costs the guard:
# at 32k an overlong prompt fails visibly, and at 128k the same prompt succeeds with weak
# recall of its own middle. A visible failure traded for an invisible one, knowingly.
RESIDENT_CTX = 131072

# THE RECORDING PATH. All three calls a recording makes — `summarize`, `enrich`'s title and
# `suggest.build_card` — run here, and ⚠ THEY MUST MOVE TOGETHER: see the runner-per-(model,
# num_ctx) note on RESIDENT_CTX, which a recording would pay while the owner waits for their note.
#
# ⚠ AN ALIAS, NOT A NUMBER. Two constants that must be equal will drift; one that is defined
# as the other cannot.
RECORD_CTX = RESIDENT_CTX
# The recorder's three calls run on the RESIDENT (2026-08-25, their call).
RECORD_MODEL = MODEL

# Keep the resident model in VRAM between calls (the default 5m is fine, but being
# explicit is the point — an evicted model costs a 19 GB reload on the next query).
KEEP_ALIVE = "30m"

# Temperature is split by JOB. Structured extraction keeps the 0.0 default, where
# determinism IS correctness. PROSE THE OWNER READS is a different job, and greedy decoding is
# why it reads canned: measured by observation 2026-07-20, the same note produced
# near-identical sentences across separate conversations.
# 0.7 is the conventional conversational value, not tuned here; the copilot's reply and
# `reflect` sample.
REPLY_TEMPERATURE = 0.7


class LLMError(RuntimeError):
    """Ollama unreachable, timed out, or returned unusable output."""


class Stopped(Exception):
    """The caller's should_stop() said stop while a streamed call ran: the call was abandoned."""


def chat(messages: list[dict], schema: dict | None = None, model: str | None = None,
         num_ctx: int = RESIDENT_CTX, temperature: float = 0.0,
         num_predict: int = 1024, timeout: int = 600,
         think: bool = False) -> tuple[str, dict]:
    """One bounded chat turn. Returns (content, stats). No streaming; tools are chat_tools'.

    `num_predict` is a HARD output cap and a correctness control, not a performance
    knob. Greedy decoding into a JSON grammar with an unbounded string field is a
    repetition trap: the model loops one phrase until the context runs out, an
    eight-minute "hang" that is a model talking to itself. Cap the output, and bound
    free-text fields with maxLength so the grammar itself forbids the loop.

    `think` defaults to False, explicitly — the ⚠ note on `MODEL` says why. Measured: all
    1600 num_predict tokens spent on 6,964 characters of thinking and empty content, against
    375 tokens and 8.5s with think=False. Pass think=True only with a budget for BOTH.

    stats carries what the trace records.
    """
    body = {
        "model": model or MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        # Explicit, never left to the model's default — see the ⚠ note on MODEL.
        "think": think,
        "options": {"num_ctx": num_ctx, "temperature": temperature,
                    "num_predict": num_predict},
    }
    if schema is not None:
        body["format"] = schema

    resp = _post(body, model or MODEL, timeout)
    msg = resp.get("message", {})
    content = msg.get("content", "")
    # If a reasoning trace came back, the tokens it cost came out of num_predict. Record
    # it, so "the model returned no JSON" is never mistaken for a model defect.
    return content, _stats(resp, model or MODEL, num_ctx, content, len(msg.get("thinking") or ""))


def chat_tools(messages: list[dict], tools: list[dict] | None = None, *, think: bool,
               temperature: float, num_predict: int, fmt: dict | None = None,
               timeout: int = 600, on_part=None, should_stop=None) -> tuple[dict, dict]:
    """One bounded turn that may call tools. Returns ({content, tool_calls, raw_tool_calls},
    stats): tool_calls as [{name, arguments}], raw_tool_calls as the assistant message that goes
    back into the conversation carries them. The resident's one window, an explicit `think` and a
    capped `num_predict`, as every call here. `fmt` is a JSON schema for the reply, as chat()'s
    `schema`.

    With tools the call streams, raw (`_tools_raw`): `on_part` hears each tool call's parameters
    as they decode, and `should_stop()`, asked about once a second, abandons it (Stopped).
    Without them (the intent gate) it is one /api/chat call."""
    if tools:
        return _tools_raw(messages, tools, think=think, temperature=temperature, num_predict=num_predict,
                          fmt=fmt, timeout=timeout, on_part=on_part, should_stop=should_stop)
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "think": think,
        "options": {"num_ctx": RESIDENT_CTX, "temperature": temperature,
                    "num_predict": num_predict},
    }
    if tools:
        body["tools"] = tools
    if fmt is not None:
        body["format"] = fmt
    resp = _post(body, MODEL, timeout)
    msg = resp.get("message", {})
    content, raw = msg.get("content", ""), msg.get("tool_calls") or []
    thinking_chars = len(msg.get("thinking") or "")
    calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments") or {}}
             for c in raw]
    # A reply that is only tool calls is not one whose thinking ate the budget.
    stats = _stats(resp, MODEL, RESIDENT_CTX, content or bool(raw), thinking_chars)
    return {"content": content, "tool_calls": calls, "raw_tool_calls": raw}, stats


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
    """Go's json.Marshal: compact, `<>&` and U+2028/9 escaped, a map's keys sorted."""
    out = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort)
    return (out.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
               .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


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
_CALL_OPEN, _CALL_CLOSE, _VALUE_CLOSE = "<tool_call>", "</tool_call>", "</parameter>"


def _held(text: str, *tags: str) -> int:
    """How much of the end of `text` may be the start of one of `tags`: it waits for more."""
    return max((n for tag in tags for n in range(min(len(text), len(tag) - 1), 0, -1)
                if tag.startswith(text[-n:])), default=0)


class _Reader:
    """A reply as it decodes, read as Ollama's qwen3.5 parser reads one (model/parsers/qwen35.go and
    qwen3coder.go): thinking up to `</think>`, text, and `<tool_call>` blocks, each parameter's value
    typed by the tool's schema. Unlike that parser it hands a value on as it arrives:
    `on_part(kind, index, name, value)` hears ("call", i, tool, None), ("parameter", i, name, None),
    ("text", i, name, chunk), ("value", i, name, value) and ("end", i, None, None)."""

    def __init__(self, think: bool, tools: list[dict] | None, on_part=None):
        self.types = {t["function"]["name"]: {k: (v or {}).get("type") for k, v in
                                              ((t["function"].get("parameters") or {}).get("properties") or {}).items()}
                      for t in tools or ()}
        self.on_part = on_part or (lambda *_part: None)
        self.state, self.lead = ("think", True) if think else ("text", False)
        self.buf, self.text, self.thinking, self.calls = "", [], 0, []
        self.skip = False                     # the whitespace after </think>, however it arrives
        self.call = None                      # {"name", "arguments", "written"}: the call being read
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
        self.call, self.state = {"name": None, "arguments": {}, "written": ""}, "call"
        return True

    def _in_call(self) -> bool:
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
        found = _PARAMETER.search(self.buf)
        if found and (end < 0 or found.start() < end):
            self._eat(found.end())
            self.param, self.state = [found.group(1).strip(), "", True], "value"
            self.on_part("parameter", index, self.param[0], None)
            return True
        if end < 0:
            return False
        self._eat(end + len(_CALL_CLOSE))
        self.calls.append({"name": self.call["name"], "arguments": self.call["arguments"]})
        # Ollama trims what follows only within the same piece, and the next token is its own piece.
        self.call, self.state = None, "text"
        self.on_part("end", index, None, None)
        return True

    def _value(self) -> bool:
        index, (name, said, lead) = len(self.calls), self.param
        if lead:                               # one leading newline is not the value's
            if not self.buf:
                return False
            if self.buf[0] == "\n":
                self._eat(1)
            self.param[2] = False
        end, closed = self.buf.find(_VALUE_CLOSE), True
        if end < 0 and (end := self.buf.find(_CALL_CLOSE)) >= 0:
            closed = False                     # never closed: it ends with its call
        if end < 0:
            n = len(self.buf) - _held(self.buf, "\n" + _VALUE_CLOSE, _VALUE_CLOSE, "\n" + _CALL_CLOSE, _CALL_CLOSE)
            if n > 0:
                chunk = self._eat(n)
                self.param[1] += chunk
                self.on_part("text", index, name, chunk)
            return False
        value = said + self.buf[:end]
        value = value[:-1] if value.endswith("\n") else value     # nor one trailing newline
        if len(value) > len(said):
            self.on_part("text", index, name, value[len(said):])
        self._eat(end + (len(_VALUE_CLOSE) if closed else 0))
        typed = self._typed(self.call["name"], name, value.replace("\r\n", "\n").replace("\r", "\n"))
        self.call["arguments"][name] = typed
        self.param, self.state = None, "call"
        self.on_part("value", index, name, typed)
        return True

    def _typed(self, tool: str, name: str, value: str):
        """As Ollama types it: "null" is None; an array or object parameter is JSON if it parses."""
        if value.lower() == "null":
            return None
        kind = self.types.get(tool, {}).get(name)
        kinds = kind if isinstance(kind, list) else [kind]
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
        self.buf = ""
        return ("".join(self.text), self.thinking, self.calls,
                [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in self.calls])


def _tools_raw(messages: list[dict], tools: list[dict], *, think: bool, temperature: float, num_predict: int,
               fmt: dict | None, timeout: int, on_part, should_stop) -> tuple[dict, dict]:
    body = {"model": MODEL, "prompt": render(messages, tools, think), "raw": True, "stream": True,
            "images": [image for m in messages for image in m.get("images") or ()],
            "keep_alive": KEEP_ALIVE,
            # The prompt's own last line decides thinking; `think` is sent anyway, as on every call.
            "think": think,
            "options": {"num_ctx": RESIDENT_CTX, "temperature": temperature, "num_predict": num_predict}}
    if fmt is not None:
        body["format"] = fmt
    reader = _Reader(think, tools, on_part)
    final = _generate(body, reader.feed, should_stop, timeout)
    content, thinking_chars, calls, raw = reader.close()
    # A reply that is only tool calls is not one whose thinking ate the budget.
    stats = _stats(final, MODEL, RESIDENT_CTX, content or bool(calls), thinking_chars)
    return {"content": content, "tool_calls": calls, "raw_tool_calls": raw}, stats


def _generate(body: dict, on_chunk, should_stop, timeout: int) -> dict:
    """A raw /api/generate, streamed: each piece of the reply to `on_chunk`, the final chunk (the
    stats) returned. The lock spans the whole stream, as in `_stream`."""
    req = urllib.request.Request(OLLAMA_GENERATE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    final, asked = {}, time.monotonic()
    try:
        with _lock(), urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise LLMError(f"ollama failed mid-stream ({body['model']}): {str(data['error'])[:300]}")
                if data.get("response"):
                    on_chunk(data["response"])
                if data.get("done"):
                    final = data
                if should_stop is not None and time.monotonic() - asked >= STOP_CHECK_S:
                    asked = time.monotonic()
                    if should_stop():
                        raise Stopped
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"ollama rejected the call ({body['model']}): HTTP {e.code} — {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"ollama stream failed ({body['model']}): {e}") from e
    return final


def _post(body: dict, model: str, timeout: int) -> dict:
    req = urllib.request.Request(
        OLLAMA_CHAT, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with _lock():
            return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        # Ollama puts the real reason in the body; the status line alone is useless.
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"ollama rejected the call ({model}): HTTP {e.code} — {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"ollama call failed ({model}): {e}") from e


def _stream(body: dict, model: str, timeout: int, on_delta, on_thinking) -> tuple[str, int, dict]:
    """A streamed call: (content, thinking characters, the final chunk, which carries the stats)."""
    req = urllib.request.Request(
        OLLAMA_CHAT, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    parts: list[str] = []
    thinking_chars = 0
    final: dict = {}
    try:
        # The lock spans the whole iteration — see _CALL_LOCK. Releasing at first byte
        # would let another call interleave with this stream's own decode.
        with _lock(), urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                if not line.strip():
                    continue
                data = json.loads(line)
                msg = data.get("message", {})
                thinking = msg.get("thinking") or ""
                thinking_chars += len(thinking)
                if thinking and on_thinking is not None:
                    on_thinking(thinking)
                chunk = msg.get("content", "")
                if chunk:
                    parts.append(chunk)
                    if on_delta is not None:
                        on_delta(chunk)
                if data.get("done"):
                    final = data
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"ollama rejected the call ({model}): HTTP {e.code} — {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"ollama stream failed ({model}): {e}") from e
    return "".join(parts), thinking_chars, final


def _stats(resp: dict, model: str, num_ctx: int, content, thinking_chars: int) -> dict:
    """What the trace records of one call, the same for every call path."""
    prompt_tokens = resp.get("prompt_eval_count", 0)
    out_tokens = resp.get("eval_count", 0)
    prefill_s = resp.get("prompt_eval_duration", 0) / 1e9
    decode_s = resp.get("eval_duration", 0) / 1e9
    return {
        "model": model,
        "num_ctx": num_ctx,
        "prompt_tokens": prompt_tokens,
        "output_tokens": out_tokens,
        "prefill_tok_s": round(prompt_tokens / prefill_s) if prefill_s else None,
        "decode_tok_s": round(out_tokens / decode_s) if decode_s else None,
        "duration_s": round(resp.get("total_duration", 0) / 1e9, 2),
        # Ollama truncates the prompt to num_ctx without erroring. If these are
        # equal the prompt was probably cut off — the trace should show it.
        "ctx_saturated": prompt_tokens >= num_ctx,
        # "length" means the model hit num_predict instead of finishing. Under a
        # JSON grammar that usually means a repetition loop, and the JSON will be
        # truncated mid-string — the caller's parse will fail, loudly, which is
        # what we want. Recorded so the trace shows WHY.
        "done_reason": resp.get("done_reason"),
        "output_truncated": resp.get("done_reason") == "length",
        # Non-zero means the model reasoned. Under think=False it must be 0. Non-zero
        # AND empty content means thinking ate the output budget; the stats say so.
        "thinking_chars": thinking_chars,
        "thinking_ate_the_budget": bool(thinking_chars and not content),
    }


def chat_stream(messages: list[dict], model: str | None = None,
                num_ctx: int = RESIDENT_CTX, temperature: float = 0.0,
                num_predict: int = 1024, timeout: int = 600, think: bool = False,
                on_delta=None, on_thinking=None, schema: dict | None = None) -> tuple[str, dict]:
    """Streaming twin of chat(): same bounds and stats, with optional diagnostics.

    `on_delta(chunk)` fires per content chunk as it decodes; the full (content, stats)
    tuple returns when the stream completes. `on_thinking(chunk)` is opt-in, for the one UI
    that exposes the model's reasoning. A schema may be supplied by a caller that treats
    deltas as diagnostics and parses only the completed JSON.
    """
    body = {
        "model": model or MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        # Explicit, never left to the model's default — see the ⚠ note on MODEL.
        "think": think,
        "options": {"num_ctx": num_ctx, "temperature": temperature,
                    "num_predict": num_predict},
    }
    if schema is not None:
        body["format"] = schema
    content, thinking_chars, final = _stream(body, model or MODEL, timeout, on_delta, on_thinking)
    return content, _stats(final, model or MODEL, num_ctx, content, thinking_chars)


def chat_json(messages: list[dict], schema: dict, **kw) -> tuple[dict, dict]:
    """Schema-constrained call. Raises LLMError if the output won't parse."""
    content, stats = chat(messages, schema=schema, **kw)
    try:
        return json.loads(content), stats
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned non-JSON under a schema: {content[:200]!r}") from e


def chat_json_stream(messages: list[dict], schema: dict, **kw) -> tuple[dict, dict]:
    """Schema-constrained streaming call; only the completed content is parsed."""
    content, stats = chat_stream(messages, schema=schema, **kw)
    try:
        return json.loads(content), stats
    except json.JSONDecodeError as e:
        raise LLMError(f"model returned non-JSON under a streamed schema: {content[:200]!r}") from e


