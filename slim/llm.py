"""Chat client for llama.cpp's `llama-server`. The single place a local model gets called.

Two rules are enforced here rather than trusted to callers. `think` is ALWAYS set
explicitly, because qwen3.6 thinks by default. Structured calls pass a JSON schema
(`response_format`), so the decoder is constrained to valid JSON — but a constrained
decode still produces schema-valid nonsense, so the caller validates meaning, not syntax.

Why llama-server and not Ollama (2026-10-23): Ollama renders qwen3.6's prompt with its own Go
renderer instead of the model's chat template, and that renderer has open bugs in exactly the
thinking-then-tool-call shape an agent loop makes (ollama#14493, #16383). It also ignores
`preserve_thinking` (#16240). The Pi coding agent on Ollama looped until killed in 2 of 8 replays of
a real session; on llama-server, same weights, 8 of 8 finished in under 30 s. llama-server renders with the
template inside the GGUF, which is Qwen's own. One server now holds the weights for SLIM and the
agent, where Ollama plus a second server held two copies.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, nullcontext

from . import llm_render

# The server is started outside SLIM (README, "Requirements"): the qwen3.6 GGUF, its mmproj (the
# open note's images), `--jinja`, and a per-slot window of at least RESIDENT_CTX.
SERVER = "http://127.0.0.1:8080"
CHAT_URL = SERVER + "/v1/chat/completions"
TEMPLATE_URL = SERVER + "/apply-template"
COMPLETION_URL = SERVER + "/completion"
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
# ⚠ qwen3.6 THINKS BY DEFAULT, and its thinking tokens count against the output cap
#   (`max_tokens`/`n_predict`) while coming back in a separate field. Every call path therefore
#   sets `think` EXPLICITLY — never by inheriting the default. Left unset, the model spends its
#   whole output budget reasoning and returns content="", and the call dies at the first step.
#   `tests/test_llm_wire.py` pins it.
#
# The name of the weights: Ollama's `qwen3.6:35b` GGUF, the very file llama-server loads. The
# server serves one model and ignores the field; traces record it.
MODEL = "qwen3.6:35b"

# The window every input budget is derived from (copilot/context.py, copilot/turn.py). The server
# fixes the window per slot at its start, so this must equal its per-slot size; a call cannot
# change it. 256k since 2026-10-16 — THE OWNER'S CALL: twice the text per prompt, so a longer
# prefill and weaker recall of the middle on the biggest inputs.
#
# Ollama spawned one runner per (model, num_ctx), so every call had to use one size or reload
# 19 GB mid-query. llama-server has no such cliff, but the size stays one constant for the
# budgets. An overlong prompt is refused (HTTP 400, exceed_context_size_error), never truncated
# as Ollama truncated it, so the loud failure the big window traded away is back.
#
# ~16 MB of KV per 1k tokens, because qwen35moe is MoE with a 2048 embedding length. The
# weights dominate and the window is nearly free — none of which transfers to a dense model.
RESIDENT_CTX = 262144

# THE RECORDING PATH. All three calls a recording makes — `summarize`, `enrich`'s title and
# `suggest.build_card` — run here, on the resident's window.
#
# ⚠ AN ALIAS, NOT A NUMBER. Two constants that must be equal will drift; one that is defined
# as the other cannot.
RECORD_CTX = RESIDENT_CTX
# The recorder's three calls run on the RESIDENT (2026-08-25, their call).
RECORD_MODEL = MODEL

# The server's own thinking budget belongs to the coding agent that shares it. SLIM's bound on
# thinking is the output cap it always was: -1 turns the server's off for every SLIM call.
NO_THINKING_BUDGET = -1

# Temperature is split by JOB. Structured extraction keeps the 0.0 default, where
# determinism IS correctness. PROSE THE OWNER READS is a different job, and greedy decoding is
# why it reads canned: measured by observation 2026-07-20, the same note produced
# near-identical sentences across separate conversations.
# 0.7 is the conventional conversational value, not tuned here; the copilot's reply and
# `reflect` sample.
REPLY_TEMPERATURE = 0.7


class LLMError(RuntimeError):
    """The server unreachable, timed out, or returned unusable output."""


class Stopped(Exception):
    """The caller's should_stop() said stop while a streamed call ran: the call was abandoned."""


# --- the wire: SLIM's messages as the server reads them ------------------------------------------
#
# Callers build messages as they always have: `images` a list of base64 strings beside the text,
# `thinking` beside an assistant's content, a tool call's `arguments` an object. The server reads
# OpenAI's shapes: an image is a content part, reasoning is `reasoning_content`, arguments are a
# JSON string. Converted here, once, for every call path; replies come back the same way.

_IMAGE_TYPES = {"/9j/": "jpeg", "R0lG": "gif", "UklG": "webp"}      # base64 of each magic number
_STOPS = {"limit": "length", "eos": "stop", "word": "stop"}          # a raw call's stop_type


def _image(data: str) -> dict:
    kind = next((kind for head, kind in _IMAGE_TYPES.items() if data.startswith(head)), "png")
    return {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{data}"}}


def _wire(messages: list[dict]) -> list[dict]:
    """The messages as the chat template reads them. A message's images go before its text, in
    order: the order a raw call sends them in."""
    out = []
    for message in messages:
        wired = {"role": message["role"], "content": message.get("content") or ""}
        if message.get("images"):
            wired["content"] = [*map(_image, message["images"]), {"type": "text", "text": wired["content"]}]
        if message.get("thinking"):
            wired["reasoning_content"] = message["thinking"]
        if message.get("tool_calls"):
            wired["tool_calls"] = [{"type": "function", "function": {
                "name": call["function"]["name"],
                "arguments": json.dumps(call["function"].get("arguments") or {}, ensure_ascii=False)}}
                for call in message["tool_calls"]]
        out.append(wired)
    return out


def _calls(raw: list[dict]) -> list[dict]:
    """The server's tool calls as the conversation carries them: arguments an object again."""
    out = []
    for call in raw:
        arguments = call["function"].get("arguments") or {}
        out.append({"function": {"name": call["function"]["name"],
                                 "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments}})
    return out


def _body(messages: list[dict], *, model: str | None, think: bool, temperature: float,
          num_predict: int, schema: dict | None, stream: bool) -> dict:
    """A chat call's body, the same bounds on every path."""
    body = {
        "model": model or MODEL,
        "messages": _wire(messages),
        "stream": stream,
        "max_tokens": num_predict,
        "temperature": temperature,
        # Explicit, never left to the model's default — see the ⚠ note on MODEL.
        "chat_template_kwargs": {"enable_thinking": think},
        "thinking_budget_tokens": NO_THINKING_BUDGET,
    }
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
    return body


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

    `num_ctx` is the window the stats measure the prompt against; the server's is fixed at its
    start (RESIDENT_CTX). stats carries what the trace records.
    """
    body = _body(messages, model=model, think=think, temperature=temperature,
                 num_predict=num_predict, schema=schema, stream=False)
    resp = _post(CHAT_URL, body, model or MODEL, timeout)
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    # If a reasoning trace came back, the tokens it cost came out of num_predict. Record
    # it, so "the model returned no JSON" is never mistaken for a model defect.
    return content, _stats(resp.get("timings") or {}, choice.get("finish_reason"), model or MODEL,
                           num_ctx, content, len(msg.get("reasoning_content") or ""))


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
    Without them (the intent gate) it is one chat call."""
    if tools:
        return _tools_raw(messages, tools, think=think, temperature=temperature, num_predict=num_predict,
                          fmt=fmt, timeout=timeout, on_part=on_part, should_stop=should_stop)
    body = _body(messages, model=None, think=think, temperature=temperature,
                 num_predict=num_predict, schema=fmt, stream=False)
    resp = _post(CHAT_URL, body, MODEL, timeout)
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content, raw = msg.get("content") or "", _calls(msg.get("tool_calls") or [])
    thinking_chars = len(msg.get("reasoning_content") or "")
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in raw]
    # A reply that is only tool calls is not one whose thinking ate the budget.
    stats = _stats(resp.get("timings") or {}, choice.get("finish_reason"), MODEL, RESIDENT_CTX,
                   content or bool(raw), thinking_chars)
    return {"content": content, "tool_calls": calls, "raw_tool_calls": raw}, stats


# --- a call with tools: the template's prompt, the reply read as it decodes ----------------------
#
# The chat endpoint parses a tool call into JSON arguments. A RAW completion streams the model's
# own XML instead, which `llm_render._Reader` reads as it decodes: a parameter's text reaches the
# owner's draft as it arrives, and a `\frac` in it never passes through JSON. The prompt is the
# server's own rendering (`/apply-template`, the GGUF's chat template), so a raw call and a chat
# call over the same history share their cached prefix.

def _tools_raw(messages: list[dict], tools: list[dict], *, think: bool, temperature: float, num_predict: int,
               fmt: dict | None, timeout: int, on_part, should_stop) -> tuple[dict, dict]:
    rendered = _post(TEMPLATE_URL, {"messages": _wire(messages), "tools": tools,
                                    "chat_template_kwargs": {"enable_thinking": think}}, MODEL, timeout)
    images = [image for m in messages for image in m.get("images") or ()]
    # Each image is a media marker in the rendered prompt, in the order the images are sent.
    prompt = {"prompt_string": rendered["prompt"], "multimodal_data": images} if images else rendered["prompt"]
    body = {"prompt": prompt, "stream": True, "n_predict": num_predict, "temperature": temperature,
            "cache_prompt": True}
    if fmt is not None:
        body["json_schema"] = fmt
    reader = llm_render._Reader(think, tools, on_part)
    final = _generate(body, reader.feed, should_stop, timeout)
    content, thinking_chars, calls, raw = reader.close()
    # A reply that is only tool calls is not one whose thinking ate the budget.
    stats = _stats(final.get("timings") or {}, _STOPS.get(final.get("stop_type")), MODEL, RESIDENT_CTX,
                   content or bool(calls), thinking_chars)
    return {"content": content, "tool_calls": calls, "raw_tool_calls": raw}, stats


def _generate(body: dict, on_chunk, should_stop, timeout: int) -> dict:
    """A raw completion, streamed: each piece of the reply to `on_chunk`, the final event (the
    stats) returned. The lock spans the whole stream, as in `_stream`."""
    req = urllib.request.Request(COMPLETION_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    final, asked = {}, time.monotonic()
    try:
        with _lock(), urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.strip()
                if line.startswith(b"error:"):
                    raise LLMError(f"the server failed mid-stream ({MODEL}): "
                                   f"{line[len(b'error:'):].decode('utf-8', 'replace').strip()[:300]}")
                if not line.startswith(b"data:"):
                    continue
                data = json.loads(line[len(b"data:"):])
                if data.get("error"):
                    raise LLMError(f"the server failed mid-stream ({MODEL}): {str(data['error'])[:300]}")
                if data.get("content"):
                    on_chunk(data["content"])
                if data.get("stop"):
                    final = data
                if should_stop is not None and time.monotonic() - asked >= STOP_CHECK_S:
                    asked = time.monotonic()
                    if should_stop():
                        raise Stopped
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"the server rejected the call ({MODEL}): HTTP {e.code} — {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"the server's stream failed ({MODEL}): {e}") from e
    if not final:                              # a reply cut off is no reply, whatever it held
        raise LLMError(f"the server's stream ended before it was done ({MODEL})")
    return final


def _post(url: str, body: dict, model: str, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with _lock():
            return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        # The server puts the real reason in the body; the status line alone is useless.
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"the server rejected the call ({model}): HTTP {e.code} — {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"the server call failed ({model}): {e}") from e


def _stream(body: dict, model: str, timeout: int, on_delta, on_thinking) -> tuple[str, int, dict, str | None]:
    """A streamed chat call: (content, thinking characters, the timings, the finish reason)."""
    req = urllib.request.Request(
        CHAT_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    parts: list[str] = []
    thinking_chars = 0
    timings: dict = {}
    reason = None
    try:
        # The lock spans the whole iteration — see _CALL_LOCK. Releasing at first byte
        # would let another call interleave with this stream's own decode.
        with _lock(), urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                data = json.loads(line[len(b"data:"):])
                if data.get("error"):
                    raise LLMError(f"the server failed mid-stream ({model}): {str(data['error'])[:300]}")
                for choice in data.get("choices") or ():
                    delta = choice.get("delta") or {}
                    thinking = delta.get("reasoning_content") or ""
                    thinking_chars += len(thinking)
                    if thinking and on_thinking is not None:
                        on_thinking(thinking)
                    chunk = delta.get("content") or ""
                    if chunk:
                        parts.append(chunk)
                        if on_delta is not None:
                            on_delta(chunk)
                    reason = choice.get("finish_reason") or reason
                timings = data.get("timings") or timings
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"the server rejected the call ({model}): HTTP {e.code} — {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"the server's stream failed ({model}): {e}") from e
    return "".join(parts), thinking_chars, timings, reason


def _stats(timings: dict, reason: str | None, model: str, num_ctx: int, content, thinking_chars: int) -> dict:
    """What the trace records of one call, the same for every call path."""
    evaluated = timings.get("prompt_n", 0)
    prompt_tokens = evaluated + timings.get("cache_n", 0)     # the whole prompt, cached or not
    out_tokens = timings.get("predicted_n", 0)
    prefill_s = timings.get("prompt_ms", 0) / 1000
    decode_s = timings.get("predicted_ms", 0) / 1000
    return {
        "model": model,
        "num_ctx": num_ctx,
        "prompt_tokens": prompt_tokens,
        "output_tokens": out_tokens,
        "prefill_tok_s": round(evaluated / prefill_s) if prefill_s else None,
        "decode_tok_s": round(out_tokens / decode_s) if decode_s else None,
        "duration_s": round(prefill_s + decode_s, 2),
        # The server refuses a prompt over its window rather than truncating it, so this is true
        # only when this window and the server's disagree: the trace should show it.
        "ctx_saturated": prompt_tokens >= num_ctx,
        # "length" means the model hit num_predict instead of finishing. Under a
        # JSON grammar that usually means a repetition loop, and the JSON will be
        # truncated mid-string — the caller's parse will fail, loudly, which is
        # what we want. Recorded so the trace shows WHY.
        "done_reason": reason,
        "output_truncated": reason == "length",
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
    body = _body(messages, model=model, think=think, temperature=temperature,
                 num_predict=num_predict, schema=schema, stream=True)
    content, thinking_chars, timings, reason = _stream(body, model or MODEL, timeout, on_delta, on_thinking)
    return content, _stats(timings, reason, model or MODEL, num_ctx, content, thinking_chars)


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


