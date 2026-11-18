"""The arms: the shipped rules, the shapes alone, and a classifier over the numbers left standing."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

from slim.capture import pii, pii_detect as d
from slim.capture.pii_normalise import normalise

CLASSES = ("ssn", "card", "account", "password", "email", "address")   # the shipped default
SCORED = ("ssn", "card", "account")
# The digit counts a value of each class can have: an SSN, its last four or a garbled 7-8; a card;
# an account, routing number or digit-only IBAN body.
POSSIBLE = {"ssn": {4, 7, 8, 9}, "card": set(range(12, 20)), "account": set(range(6, 35))}
LEAST = min(min(v) for v in POSSIBLE.values())


@contextmanager
def shapes_only():
    """The detector with its context layer off: only the cue-free shapes (dashed and spaced SSN,
    Luhn+brand card, IBAN, x@y email, address, phone shape). Every binding entry point returns
    nothing: cue before/after, the asked question, listed answers, last four, strong SSN cues,
    comma-read answers, cue-bound account tokens and spelled IBANs, cue-found emails."""
    never = re.compile(r"(?!)")
    patches = [
        (d._Scene, "bound", lambda self, cls, fits: []),
        (d._Scene, "before", lambda self, cls, a: False),
        (d._Scene, "after", lambda self, cls, b: False),
        (d, "_last_four_answers", lambda scene: []),
        (d, "_strong_ssns", lambda scene: []),
        (d, "_comma_read_answers", lambda scene: []),
        (d, "_SSN_LAST_FOUR", never),
        (d, "_glued_emails", lambda norm: []),
        (d, "_email_local", lambda norm, at, end: None),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    try:
        for obj, name, new in patches:
            setattr(obj, name, new)
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


def rules(text: str) -> list[d.Hit]:
    return pii.find(text, CLASSES)


def shapes(text: str) -> list[d.Hit]:
    with shapes_only():
        return pii.find(text, CLASSES)


@dataclass
class Segment:
    start: int          # original-text offsets
    end: int
    digits: int


@dataclass
class Candidate:
    """A standing number the earlier layer left alone: one or more segments of one run, in order.
    A run is the detector's `_RUN` (groups joined by one separator or ", "); a segment its
    `_SEGMENT` (a value read in groups, never across ", ")."""
    segments: list[Segment] = field(default_factory=list)

    @property
    def digits(self) -> int:
        return sum(s.digits for s in self.segments)


def _orig_span(idx: list[int], norm: str, a: int, b: int) -> tuple[int, int]:
    """As `pii_detect.find` maps a hit: a converted number phrase is never cut in half."""
    while a > 0 and idx[a - 1] == idx[a]:
        a -= 1
    q = b - 1
    while q + 1 < len(norm) and idx[q + 1] == idx[q]:
        q += 1
    return idx[a], idx[q + 1]


def candidates(text: str, hits: Iterable[d.Hit]) -> list[Candidate]:
    """The standing numbers outside `hits`, each run cut where a hit sits (inside a segment, at
    its digit groups: "4111 1111 1111 1111 9999 Elm Street" keeps the card part when the address
    took "9999"). A run with fewer digits than any secret class can hold is no candidate (code,
    not a model, says so)."""
    norm, idx = normalise(text)
    taken = [(h.start, h.end) for h in hits]

    def free(a: int, b: int) -> bool:
        return not any(a < y and x < b for x, y in taken)

    out = []
    for m in d._RUN.finditer(norm):
        if not d._alone(norm, *m.span()):
            continue
        cur = Candidate()
        for s in d._SEGMENT.finditer(m.group()):
            s0 = m.start() + s.start()
            a, b = _orig_span(idx, norm, s0, m.start() + s.end())
            if free(a, b):
                cur.segments.append(Segment(a, b, len(re.sub(r"\D", "", s.group()))))
                continue
            for g in d._GROUP.finditer(s.group()):           # the groups the hit left
                ga, gb = _orig_span(idx, norm, s0 + g.start(), s0 + g.end())
                if not free(ga, gb):
                    if cur.segments:
                        out.append(cur)
                    cur = Candidate()
                elif cur.segments and cur.segments[-1].end > a:     # the same segment's last group
                    last = cur.segments[-1]
                    cur.segments[-1] = Segment(last.start, gb, last.digits + len(g.group()))
                else:
                    cur.segments.append(Segment(ga, gb, len(g.group())))
        if cur.segments:
            out.append(cur)
    return [c for c in out if c.digits >= LEAST]


def placeholders(text: str, hits: Iterable[d.Hit]) -> str:
    out, pos = [], 0
    for h in sorted(hits, key=lambda h: h.start):
        out += [text[pos:h.start], d.PLACEHOLDER[h.cls]]
        pos = h.end
    return "".join(out) + text[pos:]


def apply(text: str, hits: Iterable[d.Hit]) -> str:
    return placeholders(text, hits)


def merge_labels(cand: Candidate, kinds: list[str], wanted=SCORED) -> list[d.Hit]:
    """Consecutive segments given the same class are one value, masked when the class is one to
    mask and its digit count is possible for the class; when the whole block's count is not, its
    longest leading part that is ("219099999, 10001" both called ssn masks the first), then the
    rest the same way. The span is code's, never text a classifier wrote."""
    hits, i = [], 0
    segs = cand.segments
    while i < len(segs):
        j = i
        while j + 1 < len(segs) and kinds[j + 1] == kinds[i]:
            j += 1
        k, p = kinds[i], i
        while k in wanted and p <= j:
            q = next((q for q in range(j, p - 1, -1)
                      if sum(s.digits for s in segs[p:q + 1]) in POSSIBLE[k]), None)
            if q is None:
                p += 1
                continue
            hits.append(d.Hit(segs[p].start, segs[q].end, k))
            p = q + 1
        i = j + 1
    return hits


Classify = Callable[[str, list[d.Hit], Segment], str]


def layered(text: str, base: list[d.Hit], classify: Classify) -> tuple[list[d.Hit], int]:
    """`base` hits, then each candidate segment classified and merged. Returns (hits, calls)."""
    added, calls = [], 0
    for cand in candidates(text, base):
        kinds = []
        for seg in cand.segments:
            kinds.append(classify(text, base, seg))
            calls += 1
        added += merge_labels(cand, kinds)
    return sorted(base + added, key=lambda h: h.start), calls
