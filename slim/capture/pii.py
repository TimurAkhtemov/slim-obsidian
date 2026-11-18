"""Masking of personal data in a recording's transcript, at the owner's request.

Spoken numbers ("five oh two") are first normalised to digits so patterns can be
matched on one shape; an offset map carries every hit back to the original text.
Detected values are only ever replaced by a class placeholder: they are never
stored, logged or traced.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .. import config, llm, trace

_TOKEN = re.compile(r"[A-Za-z]+|\d+|[^A-Za-z\d]+")

_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_DIGITS = {"zero": 0, "oh": 0, **_UNITS}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_REPEAT = {"double": 2, "triple": 3}


class _Tokens:
    def __init__(self, text):
        self.tok = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]

    def word(self, i):
        if i is None or i >= len(self.tok):
            return None
        t = self.tok[i][0]
        return t.lower() if t.isalpha() else None

    def next(self, i):
        """Index of the word after token i when only a space or hyphen lies between."""
        if i + 2 < len(self.tok) and self.tok[i + 1][0] in (" ", "-") and self.word(i + 2):
            return i + 2
        return None

    def small(self, i):
        """A number below 100 at word i: (value, index after it), or None."""
        w = self.word(i)
        if w in _UNITS:
            return _UNITS[w], i + 1
        if w in _TEENS:
            return _TEENS[w], i + 1
        if w in _TENS:
            j = self.next(i)
            if self.word(j) in _UNITS:
                return _TENS[w] + _UNITS[self.word(j)], j + 1
            return _TENS[w], i + 1
        return None

    def value(self, i):
        """One spoken number at word i: (digit string, index after it), or None."""
        w = self.word(i)
        if w in _REPEAT:
            j = self.next(i)
            if self.word(j) in _DIGITS:
                return str(_DIGITS[self.word(j)]) * _REPEAT[w], j + 1
            return None
        if w in _UNITS:
            j = self.next(i)
            if self.word(j) == "hundred":
                total, end = _UNITS[w] * 100, j + 1
                k = self.next(j)
                if self.word(k) == "and":
                    k = self.next(k)
                rest = self.small(k) if k is not None else None
                if rest:
                    total, end = total + rest[0], rest[1]
                return str(total), end
        if w in _DIGITS and w not in _UNITS:
            return str(_DIGITS[w]), i + 1
        small = self.small(i)
        if small:
            return str(small[0]), small[1]
        return None


def normalise(text: str) -> tuple[str, list[int]]:
    """Turn spoken number words into digits.

    Returns (norm, idx), idx of length len(norm) + 1: a span [a, b) of norm
    is [idx[a], idx[b]) of text. Every char of a converted group maps to the
    group's start; the char after it maps to the group's end.
    """
    t = _Tokens(text)
    out: list[str] = []
    idx: list[int] = []
    i = 0
    while i < len(t.tok):
        piece, start, end = t.tok[i]
        first = t.value(i) if piece.isalpha() else None
        if not first:
            out.append(piece)
            idx.extend(range(start, end))
            i += 1
            continue
        digits, i = first
        # keep joining number words separated by a single space
        while i < len(t.tok) and t.tok[i][0] == " ":
            more = t.value(i + 1) if i + 1 < len(t.tok) else None
            if not more:
                break
            digits += more[0]
            i = more[1]
        out.append(digits)
        idx.extend([start] * len(digits))
    idx.append(len(text))
    return "".join(out), idx


VERSION = "pii-1"
SECRET = frozenset({"ssn", "card", "account", "password"})
DEFAULT_CLASSES = config.PII_DEFAULT
PLACEHOLDER = {c: f"[{c.upper()}]" for c in ("ssn", "card", "account", "password", "email", "address", "phone")}


@dataclass(frozen=True)
class Hit:
    start: int  # offsets in the ORIGINAL text
    end: int
    cls: str


@dataclass
class Masked:
    text: str
    counts: dict[str, int]
    secrets: list[str]  # originals of secret-class hits, in text order; never persisted
    partial: bool = False


_SSN = re.compile(r"\b\d{3}[-\s,]+\d{2}[-\s,]+\d{4}\b")
_SSN_RUN = re.compile(r"\b\d{9}\b")
_SSN_CUE = re.compile(r"social|ssn|security number", re.I)
_DIGIT_RUN = re.compile(r"\d+(?:[ -]\d+)*")
_DASHED_SSN = re.compile(r"\d{3}-\d{2}-\d{4}")
_CARD_CUE = re.compile(r"card|visa|mastercard|amex|credit|debit", re.I)
_CARD_SHAPES = {(4, 4, 4, 4), (4, 4, 4, 4, 3), (4, 6, 5)}
_EMAIL_CUE = re.compile(r"e-?mail|address|reach|write|send|contact", re.I)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b")
_ROUTING = re.compile(r"routing(?:\s+(?:number|no\.?|#))?\s*:?\s*(\d{9})\b", re.I)
_ACCOUNT = re.compile(r"account\s+(?:number|no\.?|#)\s*:?\s*(\d+(?: \d+)*)\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SPOKEN_EMAIL = re.compile(
    r"\b(\w+(?: dot \w+)*) at \w+(?: dot \w+)* dot (?:com|org|net|edu|gov|io|co|us|uk|de|ca|me|info)\b",
    re.I,
)
_STREET = (
    "Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl|"
    "Terrace|Parkway|Pkwy|Circle|Highway|Hwy"
)
_ADDRESS = re.compile(
    rf"\b\d{{1,6}}(?:\s\d{{1,3}})?\s+(?:[A-Z][a-z]+\s+){{1,4}}(?:{_STREET})\b\.?"
    r"(?:,? (?i:apartment|apt|suite|unit|#) \w+)?(?:,? \d{5}(?:-\d{4})?\b)?"
)
_PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")


def _ssn_ok(digits: str) -> bool:
    return digits[:3] not in ("000", "666") and digits[0] != "9" and digits[3:5] != "00" and digits[5:] != "0000"


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _iban_ok(raw: str) -> bool:
    s = re.sub(r"\s", "", raw)
    return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def _aba_ok(d: str) -> bool:
    n = [int(c) for c in d]
    return (3 * (n[0] + n[3] + n[6]) + 7 * (n[1] + n[4] + n[7]) + (n[2] + n[5] + n[8])) % 10 == 0


def _cards(norm: str) -> list[tuple[int, int]]:
    """Whole runs of digit groups (single space or dash) that are 13-19 digits and pass Luhn,
    and are card-shaped or cued: ordinary number lists must not read as a card."""
    spans = []
    for run in _DIGIT_RUN.finditer(norm):
        groups = tuple(len(g) for g in re.findall(r"\d+", run.group()))
        digits = re.sub(r"\D", "", run.group())
        if not 13 <= len(digits) <= 19 or not _luhn(digits):
            continue
        shaped = len(groups) == 1 or groups in _CARD_SHAPES
        if shaped or _CARD_CUE.search(norm[max(0, run.start() - 60):run.start()]):
            spans.append((run.start(), run.end()))
    return spans


def _account_digits(m: re.Match) -> tuple[int, int] | None:
    """The cued digit groups of an account number, up to 17 digits, as a span."""
    end, total = m.start(1), 0
    for g in re.finditer(r"\d+", m.group(1)):
        total += len(g.group())
        if total > 17:
            break
        end = m.start(1) + g.end()
    return (m.start(1), end) if sum(c.isdigit() for c in m.string[m.start(1):end]) >= 6 else None


def _detect(norm: str, classes: set[str]) -> list[Hit]:
    """Hits in the normalised text."""
    hits: list[Hit] = []

    def add(cls, a, b):
        hits.append(Hit(a, b, cls))

    if "ssn" in classes:
        for m in _SSN.finditer(norm):
            cued = _SSN_CUE.search(norm[max(0, m.start() - 60):m.start()])
            if (cued or _DASHED_SSN.fullmatch(m.group())) and _ssn_ok(re.sub(r"\D", "", m.group())):
                add("ssn", m.start(), m.end())
        for m in _SSN_RUN.finditer(norm):
            if _SSN_CUE.search(norm[max(0, m.start() - 60):m.start()]) and _ssn_ok(m.group()):
                add("ssn", m.start(), m.end())
    if "card" in classes:
        for a, b in _cards(norm):
            add("card", a, b)
    if "account" in classes:
        for m in _IBAN.finditer(norm):
            if _iban_ok(m.group()):
                add("account", m.start(), m.end())
        for m in _ROUTING.finditer(norm):
            if _aba_ok(m.group(1)):
                add("account", m.start(1), m.end(1))
        for m in _ACCOUNT.finditer(norm):
            span = _account_digits(m)
            if span:
                add("account", *span)
    if "email" in classes:
        for m in _EMAIL.finditer(norm):
            add("email", m.start(), m.end())
        for m in _SPOKEN_EMAIL.finditer(norm):
            local = m.group(1)
            plain = " dot " in local or re.search(r"[\d_]", local)
            if plain or _EMAIL_CUE.search(norm[max(0, m.start() - 60):m.start()]):
                add("email", m.start(), m.end())
    if "address" in classes:
        for m in _ADDRESS.finditer(norm):
            add("address", m.start(), m.end())
    if "phone" in classes:
        for m in _PHONE.finditer(norm):
            add("phone", m.start(), m.end())
    return hits


def find(text: str, classes: Iterable[str]) -> list[Hit]:
    """Non-overlapping hits in the original text, sorted; the longest wins an overlap."""
    wanted = set(classes)
    norm, idx = normalise(text)
    mapped = []
    for h in _detect(norm, wanted):
        a, b = h.start, h.end
        # a converted number phrase is all-or-nothing: never cut one in half
        while a > 0 and idx[a - 1] == idx[a]:
            a -= 1
        q = b - 1
        while q + 1 < len(norm) and idx[q + 1] == idx[q]:
            q += 1
        mapped.append(Hit(idx[a], idx[q + 1], h.cls))
    kept: list[Hit] = []
    for h in sorted(mapped, key=lambda h: (-(h.end - h.start), h.start)):
        if all(h.end <= k.start or h.start >= k.end for k in kept):
            kept.append(h)
    return sorted(kept, key=lambda h: h.start)


CUES = re.compile(
    r"\b(pass ?words?|pass ?codes?|pins?(?: number)?|door codes?|gate codes?|log ?in is|security codes?)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"[.?!]\s+|\n\s*\n")
_TURN_START = re.compile(r"\*\*[^*\n]+:\*\*")
_MAX_SECRET = 64
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
        _SCHEMA, think=False, num_predict=_MAX_SECRET, timeout=60,
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
    """The sentence holding each cue plus the next, never crossing a turn start."""
    sents = _sentences(text)
    seen: set[int] = set()
    out = []
    for m in CUES.finditer(text):
        i = next(k for k, (a, b) in enumerate(sents) if a <= m.start() < b)
        if i in seen:
            continue
        seen.add(i)
        end = sents[i][1]
        if i + 1 < len(sents) and not _TURN_START.match(text, sents[i + 1][0]):
            end = sents[i + 1][1]
        out.append((sents[i][0], end))
    return out


def _password_spans(text: str, find_password: Callable[[str], str | None]) -> list[tuple[int, int]]:
    """Spans of the model-named secret inside each cue window; only verbatim answers count."""
    spans: set[tuple[int, int]] = set()
    for a, b in _windows(text):
        window = text[a:b]
        secret = find_password(window)
        if not secret or not 1 <= len(secret) <= _MAX_SECRET or secret not in window:
            continue
        pos = window.find(secret)
        while pos >= 0:
            spans.add((a + pos, a + pos + len(secret)))
            pos = window.find(secret, pos + len(secret))
    kept: list[tuple[int, int]] = []
    for s in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0])):
        if all(s[1] <= k[0] or s[0] >= k[1] for k in kept):
            kept.append(s)
    return sorted(kept)


def mask(
    text: str,
    classes: Iterable[str] | None = None,
    find_password: Callable[[str], str | None] | None = None,
) -> Masked:
    wanted = config.PII_MASK if classes is None else tuple(classes)
    hits = find(text, wanted)
    counts: dict[str, int] = {}
    found: list[tuple[int, str]] = []  # (position in the pattern-masked text, original)
    out: list[str] = []
    pos = 0
    length = 0
    for h in hits:
        out += [text[pos:h.start], PLACEHOLDER[h.cls]]
        length += h.start - pos
        pos = h.end
        counts[h.cls] = counts.get(h.cls, 0) + 1
        if h.cls in SECRET:
            found.append((length, text[h.start:h.end]))
        length += len(PLACEHOLDER[h.cls])
    out.append(text[pos:])
    masked = "".join(out)
    partial = False
    if "password" in wanted and find_password is not None:
        try:
            spans = _password_spans(masked, find_password)
        except Exception as exc:
            partial = True
            trace.record("pii", {"stage": "password", "status": "failed", "error": type(exc).__name__})
            spans = []
        if spans:
            parts, last = [], 0
            for a, b in spans:
                parts += [masked[last:a], PLACEHOLDER["password"]]
                found.append((a, masked[a:b]))
                last = b
            parts.append(masked[last:])
            masked = "".join(parts)
            counts["password"] = len(spans)
    secrets = [v for _, v in sorted(found, key=lambda f: f[0])]
    return Masked(masked, counts, secrets, partial)


def frontmatter(m: Masked) -> dict[str, object]:
    fm: dict[str, object] = {"pii_masked": [f"{c}:{n}" for c, n in sorted(m.counts.items())], "pii_by": VERSION}
    if m.partial:
        fm["pii_partial"] = "true"
    return fm
