"""The password pass: cue windows, the model that points at a secret, and the spans it names."""

import re
from collections.abc import Callable

from .. import llm, trace
from .pii_detect import _TURN_START, PLACEHOLDER

CUES = re.compile(
    r"\b(pass ?words?|pass ?codes?|pins?(?: number)?|door codes?|gate codes?|log ?in is|security codes?)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"[.?!]\s+|\n\s*\n")
_MAX_SECRET = 64
# Room for a 64-character secret plus its JSON framing: 64 tokens truncated the answer.
_ASK_TOKENS = 128
# llama-server unloads the model after 10 idle minutes and the first call reloads ~32 GB:
# 60 s turned a cold start into a `password` partial.
_ASK_TIMEOUT = 180
_SCHEMA = {
    "type": "object",
    "properties": {"secret": {"type": ["string", "null"], "maxLength": _MAX_SECRET}},
    "required": ["secret"],
}
_SYSTEM = (
    "You find secrets in a transcript excerpt. Return the exact password, passcode, PIN or access "
    "code as it appears in the text, copied character for character, or null if none is stated."
)


def ask_model(window: str) -> str | None:
    """The model points at a secret in `window`; it never writes text. Raises on failure."""
    out, _ = llm.chat_json(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": window}],
        _SCHEMA, think=False, num_predict=_ASK_TOKENS, timeout=_ASK_TIMEOUT,
    )
    secret = out.get("secret")
    return secret if isinstance(secret, str) else None


def _sentences(text: str) -> list[tuple[int, int]]:
    cuts = {0, len(text)}
    cuts.update(m.end() for m in _SENTENCE_END.finditer(text))
    cuts.update(m.start() for m in _TURN_START.finditer(text))
    edges = sorted(cuts)
    return [(a, b) for a, b in zip(edges, edges[1:]) if a < b]


def _windows(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence holding a cue plus up to two more, crossing at most one turn
    start: the password is often asked in one turn and answered in the next."""
    sents = _sentences(text)
    seen: set[int] = set()
    out = []
    for m in CUES.finditer(text):
        i = next(k for k, (a, b) in enumerate(sents) if a <= m.start() < b)
        if i in seen:
            continue
        seen.add(i)
        end, turns = sents[i][1], 0
        for a, b in sents[i + 1:i + 3]:
            turns += bool(_TURN_START.match(text, a))
            if turns > 1:
                break
            end = b
        out.append((sents[i][0], end))
    return out


_PLACEHOLDER = re.compile("|".join(re.escape(v) for v in PLACEHOLDER.values()))
# An answer made only of these is not a secret: "it", "the one", "okay".
_FUNCTION_WORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in is it its me my no not of on or "
    "our she so that the their them then there they this to us was we were what when where who will "
    "with you your yes okay ok sure it's that's what's there's here's i'm i'll i've you're we're they're "
    "he's she's don't can't isn't won't didn't doesn't let's".split()
)


def _answer_ok(secret: str | None) -> bool:
    return (
        bool(secret) and 3 <= len(secret) <= _MAX_SECRET and "[" not in secret and "]" not in secret
        and not all(w in _FUNCTION_WORDS for w in re.findall(r"[a-z0-9']+", secret.lower().replace("’", "'")))
    )


def _password_spans(
    text: str, find_password: Callable[[str], str | None]
) -> tuple[list[tuple[int, int]], bool]:
    """Spans of the model-named secret inside each cue window, and whether any call failed.

    Only a verbatim answer counts, as a whole word anywhere in the window (a secret is often said
    before its cue: "gavrell 68 is my password"), and never over a placeholder already in the text."""
    placeholders = [m.span() for m in _PLACEHOLDER.finditer(text)]
    spans: set[tuple[int, int]] = set()
    failed = False
    for a, b in _windows(text):
        window = text[a:b]
        try:
            secret = find_password(window)
        except Exception as exc:
            failed = True
            trace.record("pii", {"stage": "password", "status": "failed", "error": type(exc).__name__})
            continue
        if not _answer_ok(secret):
            continue
        pos = window.find(secret)
        while pos >= 0:
            s, e = a + pos, a + pos + len(secret)
            word = (s == 0 or not text[s - 1].isalnum()) and (e == len(text) or not text[e].isalnum())
            if word and all(e <= x or s >= y for x, y in placeholders):
                spans.add((s, e))
            pos = window.find(secret, pos + len(secret))
    kept: list[tuple[int, int]] = []
    for s in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if all(s[1] <= k[0] or s[0] >= k[1] for k in kept):
            kept.append(s)
    return sorted(kept), failed
