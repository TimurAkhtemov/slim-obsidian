"""Ollama chat client. The single place a local model gets called.

Two rules are enforced here rather than trusted to callers. `num_ctx` is ALWAYS
set explicitly, because Ollama's default is 4096 tokens and it truncates silently.
Structured calls pass a JSON schema to Ollama's `format`, so the decoder is
constrained to valid JSON — but a constrained decode still produces schema-valid
nonsense, so the caller validates meaning, not syntax.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, nullcontext

from . import llm_render

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


def _tools_raw(messages: list[dict], tools: list[dict], *, think: bool, temperature: float, num_predict: int,
               fmt: dict | None, timeout: int, on_part, should_stop) -> tuple[dict, dict]:
    body = {"model": MODEL, "prompt": llm_render.render(messages, tools, think), "raw": True, "stream": True,
            "images": [image for m in messages for image in m.get("images") or ()],
            "keep_alive": KEEP_ALIVE,
            # The prompt's own last line decides thinking; `think` is sent anyway, as on every call.
            "think": think,
            "options": {"num_ctx": RESIDENT_CTX, "temperature": temperature, "num_predict": num_predict}}
    if fmt is not None:
        body["format"] = fmt
    reader = llm_render._Reader(think, tools, on_part)
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
    if not final:                              # a reply cut off is no reply, whatever it held
        raise LLMError(f"ollama's stream ended before it was done ({body['model']})")
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


