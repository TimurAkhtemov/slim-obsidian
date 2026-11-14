"""The scripted model the copilot's turn tests share (`test_run.py`, `test_loop.py`)."""

from slim import llm
from slim.copilot import turn as copilot_turn


def scripted(monkeypatch, *replies):
    """A fake model: each call returns the next (content, [(tool, arguments)][, output tokens])
    and is recorded with what it was sent; a reply that is an exception is raised. A call that
    streams (`on_part`) hears its reply text as the raw reader says it, ("reply", -1, None, text)."""
    seen = []

    def chat_tools(messages, tools=None, **opts):
        seen.append({"messages": [dict(m) for m in messages],
                     "tools": [t["function"]["name"] for t in tools or []], "opts": opts,
                     "held": getattr(llm._HELD, "on", False)})
        if isinstance(replies[len(seen) - 1], Exception):
            raise replies[len(seen) - 1]
        content, calls, *tokens = replies[len(seen) - 1]
        if opts.get("on_part") and content:
            opts["on_part"]("reply", -1, None, content)
        return ({"content": content, "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                 "raw_tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]},
                {"duration_s": 1.0, "output_tokens": tokens[0] if tokens else 10})
    monkeypatch.setattr(copilot_turn.llm, "chat_tools", chat_tools)
    return seen
