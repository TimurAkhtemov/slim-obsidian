"""The scripted model the copilot's turn tests share (`test_run.py`, `test_loop.py`)."""

import json

from slim import llm
from slim.copilot import turn as copilot_turn


class Seen(list):
    """The calls a fake model was sent, and apart from them the intent gate's (`gate`)."""
    gate: list


def scripted(monkeypatch, *replies, intent=True):
    """A fake model: each call returns the next (content, [(tool, arguments)][, output tokens])
    and is recorded with what it was sent; a reply that is an exception is raised. A call that streams (`on_part`) hears its reply text as
    the raw reader says it, ("reply", -1, None, text). The intent gate's structured call says
    `intent` (True, False, a raw reply, or "error" for an Ollama error) and is recorded in
    `seen.gate`."""
    seen = Seen()
    seen.gate = []

    def chat_tools(messages, tools=None, **opts):
        if opts.get("fmt") is not None:
            seen.gate.append({"messages": [dict(m) for m in messages], "tools": tools, "opts": opts,
                              "held": getattr(llm._HELD, "on", False)})
            if intent == "error":
                raise llm.LLMError("ollama call failed: timed out")
            said = intent if isinstance(intent, str) else json.dumps({"change_note": intent})
            return {"content": said, "tool_calls": [], "raw_tool_calls": []}, {"output_tokens": 5}
        seen.append({"messages": [dict(m) for m in messages],
                     "tools": [t["function"]["name"] for t in tools or []], "opts": opts,
                     "held": getattr(llm._HELD, "on", False), "stream": False})
        if isinstance(replies[len(seen) - 1], Exception):
            raise replies[len(seen) - 1]
        content, calls, *tokens = replies[len(seen) - 1]
        if opts.get("on_part") and content:
            opts["on_part"]("reply", -1, None, content)
        return ({"content": content, "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                 "raw_tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]},
                {"duration_s": 1.0, "output_tokens": tokens[0] if tokens else 10})
    def chat_stream(messages, **opts):
        """Today's Ask path: a streamed answer with no tools."""
        seen.append({"messages": [dict(m) for m in messages], "tools": [], "opts": opts,
                     "held": getattr(llm._HELD, "on", False), "stream": True})
        content, _calls, *tokens = replies[len(seen) - 1]
        if opts.get("on_delta") and content:
            opts["on_delta"](content)
        return content, {"duration_s": 1.0, "output_tokens": tokens[0] if tokens else 10}
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    monkeypatch.setattr(copilot_turn.llm, "chat_stream", chat_stream)
    return seen
