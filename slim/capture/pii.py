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


_DASH_WORDS = {"dash", "hyphen", "minus"}


def normalise(text: str) -> tuple[str, list[int]]:
    """Turn spoken number words into digits, and a spoken dash between two numbers into "-".

    Returns (norm, idx), idx of length len(norm) + 1: a span [a, b) of norm
    is [idx[a], idx[b]) of text. Every char of a converted group maps to the
    group's start; the char after it maps to the group's end. A " dash " that became
    "-" maps to its leading space.
    """
    t = _Tokens(text)
    out: list[str] = []
    idx: list[int] = []
    i = 0

    def number_at(k):
        return k < len(t.tok) and (t.tok[k][0].isdigit() or t.value(k) is not None)

    while i < len(t.tok):
        piece, start, end = t.tok[i]
        first = t.value(i) if piece.isalpha() else None
        if first:
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
        else:
            out.append(piece)
            idx.extend(range(start, end))
            i += 1
            if not piece.isdigit():
                continue
        # "six four four dash zero four": a dash word between two numbers is "-"
        if (i + 3 < len(t.tok) and t.tok[i][0] == " " and t.word(i + 1) in _DASH_WORDS
                and t.tok[i + 2][0] == " " and number_at(i + 3)):
            out.append("-")
            idx.append(t.tok[i][1])
            i += 3
    idx.append(len(text))
    return "".join(out), idx


VERSION = "pii-1"
SECRET = frozenset({"ssn", "card", "account", "password"})
# Why masking is unfinished, as `pii_partial` names it. `password`: the password pass did not
# complete, which `slim mask` can redo from the text. `audio`: a secret may still be audible
# (a mute failed, or a secret could not be placed in token times), which a text rerun cannot fix.
PARTIAL_REASONS = ("audio", "password")
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
    partial: bool = False  # the password pass did not complete


_TURN_START = re.compile(r"\*\*[^*\n]+:\*\*")


def _cue(words: str) -> re.Pattern:
    return re.compile(rf"\b(?:{words})s?\b", re.I)


# Cues that make a number personal (R1). `_reach` decides whether one is close enough.
_SSN_CUE = _cue("social|ssn|security number")
_CARD_CUE = _cue("card|visa|mastercard|amex|american express|credit|debit")
_ACCOUNT_CUE = _cue("routing|account number|account no|acct|iban|bban|sort code")
_IBAN_CUE = _cue("iban")
_PHONE_CUE = _cue("phone|cell|mobile|call me|number is")
_REACH_BEFORE = 60   # may reach into the previous turn: a question, then its answer
_REACH_AFTER = 40    # "…, that's my social"; never past the next turn start


def _after(norm: str, b: int, cue: re.Pattern) -> bool:
    """A cue within 40 characters after position b, in the same turn."""
    after = norm[b:b + _REACH_AFTER]
    turn = _TURN_START.search(after)
    return bool(cue.search(after[:turn.start()] if turn else after))


def _reach(norm: str, a: int, b: int, cue: re.Pattern) -> bool:
    """A cue within 60 characters before [a, b), or within 40 after it in the same turn."""
    return bool(cue.search(norm[max(0, a - _REACH_BEFORE):a])) or _after(norm, b, cue)


# A maximal run of digit groups joined by a space, ", " or a dash. Every detector below judges
# a WHOLE run, so a secret is never read out of the middle of a longer list of numbers.
_GROUPS = re.compile(r"\d+(?:(?:, | |-)\d+)*")
# The same without commas: a card is never a sub-window of one of these (R3c).
_DIGIT_RUN = re.compile(r"\d+(?:[ -]\d+)*")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _shape(s: str) -> tuple[int, ...]:
    return tuple(len(g) for g in re.findall(r"\d+", s))


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


# --- ssn (R2) ---

_SSN_DASHED = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_SSN_SPACED = re.compile(r"\d{3} \d{2} \d{4}")


def _ssns(norm: str) -> list[tuple[int, int]]:
    # a. dashed 3-2-4: nothing else is written that way
    spans = [m.span() for m in _SSN_DASHED.finditer(norm) if _ssn_ok(_digits(m.group()))]
    for run in _GROUPS.finditer(norm):
        d = _digits(run.group())
        # b. exactly "741 05 7024", standing alone: how the transcriber writes a spoken SSN
        if _SSN_SPACED.fullmatch(run.group()) and _ssn_ok(d):
            spans.append(run.span())
        # c. a valid nine in any grouping, or d. a misheard 7-11 digits, with a cue in reach
        elif 7 <= len(d) <= 11 and (len(d) != 9 or _ssn_ok(d)) and _reach(norm, *run.span(), _SSN_CUE):
            spans.append(run.span())
    return spans


# --- card (R3) ---

# Issuer prefixes (inclusive ranges, compared on as many leading digits as the bound has)
# and the card lengths each brand issues.
_BRANDS = {
    "visa": ([(4, 4)], (13, 16, 19)),
    "mastercard": ([(51, 55), (2221, 2720)], (16,)),
    "amex": ([(34, 34), (37, 37)], (15,)),
    "discover": ([(6011, 6011), (644, 649), (65, 65)], (16, 17, 18, 19)),
    "diners": ([(300, 305), (36, 36), (38, 39)], (14,)),
    "jcb": ([(3528, 3589)], (16, 17, 18, 19)),
    "unionpay": ([(62, 62)], (16, 17, 18, 19)),
}
_CARD_SHAPES = {(4, 4, 4, 4), (4, 4, 4, 4, 3), (4, 6, 5)}   # how a card is read out in groups


def _branded(d: str) -> bool:
    return any(
        len(d) in lengths and any(lo <= int(d[:len(str(lo))]) <= hi for lo, hi in prefixes)
        for prefixes, lengths in _BRANDS.values()
    )


def _cards(norm: str) -> list[tuple[int, int]]:
    spans = []
    # a. Luhn and a brand: contiguous, or groups joined by spaces or dashes, any grouping ...
    for run in _DIGIT_RUN.finditer(norm):
        d = _digits(run.group())
        if _branded(d) and _luhn(d):
            spans.append(run.span())
    for run in _GROUPS.finditer(norm):
        d = _digits(run.group())
        # ... or joined by ", " only in the shapes a card is read out in
        if ", " in run.group() and _shape(run.group()) in _CARD_SHAPES and _branded(d) and _luhn(d):
            spans.append(run.span())
    # b. a card cue in reach: 12-19 digits are a card even when a misheard digit breaks Luhn
    for run in [*_DIGIT_RUN.finditer(norm), *_GROUPS.finditer(norm)]:
        if 12 <= len(_digits(run.group())) <= 19 and _reach(norm, *run.span(), _CARD_CUE):
            spans.append(run.span())
    return spans


# --- account (R4) ---

_IBAN = re.compile(r"(?:(?<=IBAN)|(?<![A-Za-z0-9]))[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
# An IBAN spelled out: single letters and digit groups ("G B 73, H W E C, 4532, ...").
_SPELLED = re.compile(r"(?<![A-Za-z0-9'’])(?:[A-Za-z]|\d+)(?:,? (?:[A-Za-z]|\d+))+(?![A-Za-z0-9])")
# One token holding letters, digits and dashes: "E11248142150", "ACCT-911155810".
_ACCOUNT_TOKEN = re.compile(r"(?<![\w-])[A-Za-z0-9][A-Za-z0-9-]{6,32}[A-Za-z0-9](?![\w-])")
_ROUTING = re.compile(r"\d{9}|\d{3} \d{3} \d{3}")
_FED_PREFIXES = [(0, 12), (21, 32), (61, 72), (80, 80)]   # Federal Reserve routing symbols


def _fed_prefix(d: str) -> bool:
    return any(lo <= int(d[:2]) <= hi for lo, hi in _FED_PREFIXES)


def _accounts(norm: str) -> list[tuple[int, int]]:
    # a. an IBAN that passes mod-97, "IBAN" glued in front or not
    spans = [m.span() for m in _IBAN.finditer(norm) if _iban_ok(m.group())]
    for run in _GROUPS.finditer(norm):
        d = _digits(run.group())
        # b. 6-17 digits with an account cue in reach
        if 6 <= len(d) <= 17 and _reach(norm, *run.span(), _ACCOUNT_CUE):
            spans.append(run.span())
        # c. a routing number standing alone: ABA checksum and a Federal Reserve prefix
        elif _ROUTING.fullmatch(run.group()) and _aba_ok(d) and _fed_prefix(d):
            spans.append(run.span())
    for m in _ACCOUNT_TOKEN.finditer(norm):
        if sum(c.isdigit() for c in m.group()) >= 6 and _reach(norm, *m.span(), _ACCOUNT_CUE):
            spans.append(m.span())
    for m in _SPELLED.finditer(norm):
        if sum(c.isalnum() for c in m.group()) >= 12 and _reach(norm, *m.span(), _IBAN_CUE):
            spans.append(m.span())
    return spans


# --- email (R5) ---

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_TLD = "com|org|net|edu|gov|io|co|us|uk|de|ca|me|info|ai|app|dev"
# " at example.org" as the transcriber writes it, or " at example dot org" as words
_AT_DOMAIN = re.compile(rf" at (?:[A-Za-z0-9-]+\.)+(?:{_TLD})\b| at (?:\w+ dot )+(?:{_TLD})\b", re.I)
_LOCAL_TOKEN = re.compile(r"[\w.+-]+$")
_LOCAL_DOTTED = re.compile(r"\w+(?: dot \w+)+$", re.I)
_EMAIL_BEFORE = re.compile(r"\be-?mail(?: address)?(?: is|:|'s)? ((?:\S+ ){0,4}\S+)$", re.I)
_EMAIL_AFTER = re.compile(r"\b(?:that's|that is|is) (?:my|the|his|her|their|your|our) e-?mail\b", re.I)
_SENTENCE_START = re.compile(r".*(?:[.?!]\s+|\n|\*\*[^*\n]+:\*\*\s*)", re.S)   # ends where one starts
# Words that never make a local part on their own: "it at amazon.com", "me at Starbucks.com".
_NOT_LOCAL = frozenset(
    "it me us them him her you that this there here up out in on over back look looked".split()
)


def _local_ok(local: str) -> bool:
    return not all(w.strip(".,'").lower() in _NOT_LOCAL for w in local.split())


def _email_local(norm: str, at: int, end: int) -> int | None:
    """Start of the local part in front of the " at " at `at`, or None when nothing says this is
    an address rather than a website mentioned after "at"."""
    lo = max(0, at - 200)
    head = norm[lo:at]
    starts = []
    m = _LOCAL_DOTTED.search(head)                       # "jane dot doe at …"
    if m:
        starts.append(m.start())
    m = _LOCAL_TOKEN.search(head)                        # "quarrel43 at …", "quarrel.telvane at …"
    if m and re.search(r"[A-Za-z]", m.group()) and re.search(r"[\d._]", m.group()) and _local_ok(m.group()):
        starts.append(m.start())
    m = _EMAIL_BEFORE.search(head)                       # "my email is vanholt at …"
    if m and _local_ok(m.group(1)):
        starts.append(m.start(1))
    if _after(norm, end, _EMAIL_AFTER):                  # "… at example.com, that's my email"
        begin = _SENTENCE_START.match(head)
        begin = begin.end() if begin else 0
        words = list(re.finditer(r"\S+", head[begin:]))[-3:]
        if words and _local_ok(" ".join(w.group() for w in words)):
            starts.append(begin + words[0].start())
    return lo + min(starts) if starts else None


def _emails(norm: str) -> list[tuple[int, int]]:
    spans = [m.span() for m in _EMAIL.finditer(norm)]
    for m in _AT_DOMAIN.finditer(norm):
        start = _email_local(norm, m.start(), m.end())
        if start is not None:
            spans.append((start, m.end()))
    return spans


# --- address, phone ---

_STREET = (
    "Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl|"
    "Terrace|Parkway|Pkwy|Circle|Highway|Hwy"
)
_ADDRESS = re.compile(
    rf"\b\d{{1,6}}(?:\s\d{{1,3}})?\s+(?:[A-Z][a-z]+\s+){{1,4}}(?:{_STREET})\b\.?"
    r"(?:,? (?i:apartment|apt|suite|unit|#) \w+)?(?:,? \d{5}(?:-\d{4})?\b)?"
)
# NXX NXX XXXX, area and exchange starting 2-9, separators a space, dash or dot
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\([2-9]\d{2}\) ?|[2-9]\d{2}[ .-])[2-9]\d{2}[ .-]\d{4}(?!\d)")
_PHONE_SHAPES = {(10,), (3, 7), (3, 3, 4)}


def _phones(norm: str) -> list[tuple[int, int]]:
    spans = [m.span() for m in _PHONE.finditer(norm)]
    for run in _GROUPS.finditer(norm):
        d = _digits(run.group())
        cued_shape = _shape(run.group()) in _PHONE_SHAPES and d[0] >= "2" and d[3] >= "2"
        if cued_shape and _reach(norm, *run.span(), _PHONE_CUE):
            spans.append(run.span())
    return spans


_DETECTORS = {
    "ssn": _ssns,
    "card": _cards,
    "account": _accounts,
    "phone": _phones,
    "address": lambda norm: [m.span() for m in _ADDRESS.finditer(norm)],
    "email": _emails,
}
# An exact tie goes to the class listed first (R9).
_RANK = {c: k for k, c in enumerate(_DETECTORS)}


def _detect(norm: str, classes: set[str]) -> list[Hit]:
    """Hits in the normalised text."""
    return [Hit(a, b, cls) for cls, detect in _DETECTORS.items() if cls in classes for a, b in detect(norm)]


def find(text: str, classes: Iterable[str]) -> list[Hit]:
    """Non-overlapping hits in the original text, sorted; the longest wins an overlap, and an
    exact tie goes to the higher class (ssn > card > account > phone > address > email)."""
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
    for h in sorted(mapped, key=lambda h: (-(h.end - h.start), h.start, _RANK[h.cls])):
        if all(h.end <= k.start or h.start >= k.end for k in kept):
            kept.append(h)
    return sorted(kept, key=lambda h: h.start)


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
    "with you your yes okay ok sure".split()
)


def _answer_ok(secret: str | None) -> bool:
    return (
        bool(secret) and 3 <= len(secret) <= _MAX_SECRET and "[" not in secret and "]" not in secret
        and not all(w in _FUNCTION_WORDS for w in re.findall(r"[a-z0-9']+", secret.lower()))
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
        spans, partial = _password_spans(masked, find_password)
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


def frontmatter(counts: dict[str, int], reasons: Iterable[str] = ()) -> dict[str, object]:
    """Flat keys for a recording: counts, version, and `pii_partial` only when unfinished."""
    fm: dict[str, object] = {"pii_masked": [f"{c}:{n}" for c, n in sorted(counts.items())], "pii_by": VERSION}
    unfinished = sorted(set(reasons) & set(PARTIAL_REASONS))
    if unfinished:
        fm["pii_partial"] = unfinished
    return fm


def spans(tokens, secrets: list[str]) -> tuple[list[tuple[float, float]], int]:
    """Audio time of each secret: `(seconds spans, unmatched count)`.

    `tokens` is `(text, start, end)` for the segment whose text was masked. Secrets arrive in
    text order, but EVERY occurrence of each distinct secret is muted: a secret left audible is the
    leak. Spans come back sorted and merged. Whitespace runs are collapsed on both sides;
    everything else must match exactly."""
    chars: list[str] = []
    owner: list[int] = []                 # chars[i] belongs to tokens[owner[i]]
    for k, tok in enumerate(tokens):
        for ch in tok[0]:
            if ch.isspace():
                if chars and chars[-1] == " ":
                    continue
                ch = " "
            chars.append(ch)
            owner.append(k)
    hay = "".join(chars)
    found: list[tuple[float, float]] = []
    missed = 0
    for secret in dict.fromkeys(" ".join(x.split()) for x in secrets):
        if not secret:
            continue
        at, hit = hay.find(secret), False
        while at >= 0:
            hit = True
            end = at + len(secret) - 1
            found.append((float(tokens[owner[at]][1]), float(tokens[owner[end]][2])))
            at = hay.find(secret, at + 1)
        missed += not hit
    merged: list[tuple[float, float]] = []
    for a, b in sorted(found):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, missed
