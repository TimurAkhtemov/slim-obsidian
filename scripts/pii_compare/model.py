"""The local model as a classifier of one marked number, called as `pii_password.ask_model` is:
through `slim.llm` only, thinking off, an enum schema, a capped answer, a timeout."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from slim import llm
from slim.capture import pii_detect as d

from .arms import Segment, placeholders

KINDS = ("ssn", "card", "account", "phone", "none")
_SCHEMA = {"type": "object", "properties": {"kind": {"type": "string", "enum": list(KINDS)}},
           "required": ["kind"]}
_TOKENS = 64      # 24 cut one answer off inside its JSON whitespace
_TIMEOUT = 180      # a cold start reloads the weights
TURNS_BEFORE, TURNS_AFTER = 3, 1
_MAX_WINDOW = 1600
OPEN, CLOSE = "⟦", "⟧"

SYSTEM = (
    "You read an excerpt of a phone-call transcript and say what ONE number in it is. The number is "
    f"marked {OPEN}like this{CLOSE}. The transcriber writes spoken numbers as digits, with commas where "
    "the speaker paused, so one number may be split into pieces and the mark may hold only one "
    "piece; say what the whole value that piece belongs to is. Turns are labelled **Me:** and "
    "**Them:**. Values already hidden show as [SSN], [CARD], [ACCOUNT], [EMAIL] or [ADDRESS].\n\n"
    "Answer with the kind of value, as the people on the call understand it:\n"
    "- ssn: a social security number, or the last four digits of one\n"
    "- card: a payment card number (credit or debit), not its expiry date or security code\n"
    "- account: a bank account number, a routing number, an IBAN or a sort code\n"
    "- phone: a phone number or an extension\n"
    "- none: anything else: a zip code, date, time, expiry, amount, balance, price, count, "
    "percentage, order / confirmation / tracking / claim / case / policy / reference / ticket "
    "number, member, customer or employee ID, security code, version, measurement\n\n"
    "Judge from what was asked and what was said, not from the digits alone. A question that asks "
    "for several things is usually answered in the order it asked."
)
PROMPT_VERSION = hashlib.sha256((SYSTEM + json.dumps(_SCHEMA)).encode()).hexdigest()[:12]


def window(text: str, base: list[d.Hit], seg: Segment) -> str:
    """The candidate's turn, up to TURNS_BEFORE turns before it and TURNS_AFTER after, with the
    earlier layer's hits as placeholders and the candidate marked."""
    marked = (placeholders(text[:seg.start], [h for h in base if h.end <= seg.start])
              + OPEN + text[seg.start:seg.end] + CLOSE)
    after = [d.Hit(h.start - seg.end, h.end - seg.end, h.cls) for h in base if h.start >= seg.end]
    tail = placeholders(text[seg.end:], after)
    full, at = marked + tail, len(marked)
    starts = [m.start() for m in d._TURN_START.finditer(full)]
    k = max((i for i, s in enumerate(starts) if s <= at), default=-1)
    lo = starts[max(0, k - TURNS_BEFORE)] if k >= 0 else 0
    hi = starts[k + 1 + TURNS_AFTER] if k + 1 + TURNS_AFTER < len(starts) else len(full)
    if hi - lo > _MAX_WINDOW:                       # never the marked turn itself
        lo = max(lo, min(at - _MAX_WINDOW // 2, starts[k] if k >= 0 else lo))
        hi = min(hi, max(at + _MAX_WINDOW // 2, at + 1))
    return full[lo:hi].strip()


class Classifier:
    """Asks the model, through a JSON cache on disk keyed by prompt and window. `use_cache=False`
    asks every time (stability runs) and still records each answer under a run key."""

    def __init__(self, cache: Path, *, use_cache: bool = True):
        self.path, self.use_cache = cache, use_cache
        self.cache = json.loads(cache.read_text()) if use_cache and cache.exists() else {}
        self.calls = self.cached = 0
        self.seconds: list[float] = []
        self.failures = 0
        self.errors: list[str] = []
        self._dirty = 0

    def key(self, prompt: str) -> str:
        return hashlib.sha256(f"{PROMPT_VERSION}\n{prompt}".encode()).hexdigest()

    def ask(self, prompt: str) -> str:
        k = self.key(prompt)
        if self.use_cache and k in self.cache:
            self.cached += 1
            return self.cache[k]
        t = time.monotonic()
        try:
            out, _ = llm.chat_json(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                _SCHEMA, think=False, num_predict=_TOKENS, timeout=_TIMEOUT)
            kind = out.get("kind") if out.get("kind") in KINDS else "none"
        except llm.LLMError as exc:
            self.failures += 1                 # a failed call masks nothing: counted, reported,
            self.errors.append(str(exc)[:160])  # never cached
            self.calls += 1
            return "none"
        self.seconds.append(time.monotonic() - t)
        self.calls += 1
        if self.use_cache:
            self.cache[k] = kind
            self._dirty += 1
            if self._dirty >= 20:
                self.save()
        return kind

    def save(self) -> None:
        if self._dirty:
            self.path.write_text(json.dumps(self.cache))
            self._dirty = 0

    def classify(self, text: str, base: list[d.Hit], seg: Segment) -> str:
        return self.ask(prompt(text, base, seg))


def prompt(text: str, base: list[d.Hit], seg: Segment) -> str:
    return (f"Excerpt:\n{window(text, base, seg)}\n\n"
            f"What is the marked number {OPEN}{text[seg.start:seg.end]}{CLOSE}?")
