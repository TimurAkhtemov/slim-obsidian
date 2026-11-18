"""Pattern detection of personal data in normalised text: cue binding, checksums, tables, the
class detectors, and `find`, which maps hits back to the original text."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .pii_normalise import normalise

PLACEHOLDER = {c: f"[{c.upper()}]" for c in ("ssn", "card", "account", "password", "email", "address", "phone")}


@dataclass(frozen=True)
class Hit:
    start: int  # offsets in the ORIGINAL text
    end: int
    cls: str


_TURN_START = re.compile(r"\*\*[^*\n]+:\*\*")

# --- cue binding (N1) ---
#
# A number is masked on the strength of a cue only when it is BOUND to it: it is the first
# number after the cue with nothing but filler words between them (one turn label allowed:
# the question, then its answer), or it is directly followed by "that's my <cue>".

# Cue nouns per class, longest first, matched word-bounded on the whole text.
_CUE_NOUNS = {
    "ssn": "social security number|social security|social|ssn",
    "card": "credit card number|debit card number|card number|credit card|debit card|card|visa|mastercard|"
            "amex|american express",
    "account": "routing number|routing|account number|account no|account|acct|iban|bban|sort code",
    "iban": "iban",
    "phone": "phone number|phone|cell|mobile|number",
}
_CUE = {c: re.compile(rf"\b(?:{nouns})\b", re.I) for c, nouns in _CUE_NOUNS.items()}
_CUE_AFTER = {
    c: re.compile(
        r"\s*[,.]?\s*(?:and\s+)?(?:(?:that's|that is|it's|is|was)\s+)?(?:my|your|his|her|their|our|the)\s+"
        rf"(?:{nouns})\b", re.I)
    for c, nouns in _CUE_NOUNS.items()
}
_GAP_MAX = 60
_FILLER = frozenset(
    "is it's its it number no the my your okay ok sure yeah yes yep so um uh well here that's that "
    "would be and goes ahead right alright thanks please".split()
)


def _gap_ok(gap: str) -> bool:
    """Only filler words, punctuation and at most one turn label between a cue and its value."""
    if len(gap) > _GAP_MAX or re.search(r"\d", gap) or len(_TURN_START.findall(gap)) > 1:
        return False
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", _TURN_START.sub(" ", gap).lower().replace("’", "'"))
    return all(w in _FILLER for w in words)


def _binder(norm: str) -> Callable[[str, int, int], bool]:
    """bound(cls, a, b): is the value at [a, b) bound to a `cls` cue?"""
    cues = {c: [m.end() for m in rx.finditer(norm)] for c, rx in _CUE.items()}

    def bound(cls: str, a: int, b: int) -> bool:
        before = any(end <= a and _gap_ok(norm[end:a]) for end in cues[cls] if a - end <= _GAP_MAX)
        return before or bool(_CUE_AFTER[cls].match(norm, b))
    return bound


# --- numbers that stand alone (N2) ---

# A maximal run of digit groups joined by one separator: a space, ", ", "-", "." or "/".
_RUN = re.compile(r"\d+(?:(?:, |[ ./-])\d+)*")
_OPENERS = "([{\"'“‘"
_CLOSERS = ".,;:!?)]}\"'”’"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")


def _alone(norm: str, a: int, b: int) -> bool:
    """Not glued to a letter or to another digit group: "B219 09 9999", "12,219 09 9999" and the
    "989-3111" of "(987) 989-3111" are not alone."""
    before = a == 0 or norm[a - 1] in _OPENERS or (
        norm[a - 1].isspace() and not re.search(r"\d\)\s$", norm[max(0, a - 3):a]))
    after = b == len(norm) or norm[b].isspace() or (
        norm[b] in _CLOSERS and (b + 1 == len(norm) or norm[b + 1].isspace()))
    return before and after


def _runs(norm: str) -> list[re.Match]:
    return [m for m in _RUN.finditer(norm) if _alone(norm, *m.span())]


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
    s = re.sub(r"\s", "", raw).upper()
    return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def _aba_ok(d: str) -> bool:
    n = [int(c) for c in d]
    return (3 * (n[0] + n[3] + n[6]) + 7 * (n[1] + n[4] + n[7]) + (n[2] + n[5] + n[8])) % 10 == 0


Bound = Callable[[str, int, int], bool]

# --- ssn ---

_SSN_DASHED = re.compile(r"\d{3}-\d{2}-\d{4}")
_SSN_SPACED = re.compile(r"\d{3} \d{2} \d{4}")
# "741 05 7024" with no cue is how the transcriber writes a spoken SSN, and also how it writes
# "we had 300 25 2000 users". Measurement-gated: False switches the shape off.
SSN_SPACED_CUE_FREE = True


def _ssns(norm: str, bound: Bound) -> list[tuple[int, int]]:
    spans = []
    for run in _runs(norm):
        s, d = run.group(), _digits(run.group())
        if _SSN_DASHED.fullmatch(s) and _ssn_ok(d):
            spans.append(run.span())
        elif SSN_SPACED_CUE_FREE and _SSN_SPACED.fullmatch(s) and _ssn_ok(d):
            spans.append(run.span())
        # bound: 7-11 digits in up to 3 groups, whatever the range check says (a misheard digit)
        elif 7 <= len(d) <= 11 and len(_shape(s)) <= 3 and not _DATE.fullmatch(s) and bound("ssn", *run.span()):
            spans.append(run.span())
    return spans


# --- card ---

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


def _branded(d: str) -> bool:
    return any(
        len(d) in lengths and any(lo <= int(d[:len(str(lo))]) <= hi for lo, hi in prefixes)
        for prefixes, lengths in _BRANDS.values()
    )


def _cards(norm: str, bound: Bound) -> list[tuple[int, int]]:
    spans = []
    for run in _runs(norm):
        s, d, shape = run.group(), _digits(run.group()), _shape(run.group())
        card_groups = len(shape) <= 5 and min(shape) >= 3     # contiguous, or groups of 3+
        # cue-free: Luhn and a brand, groups joined by single spaces or dashes only
        if card_groups and not re.search(r"[,./]", s) and _branded(d) and _luhn(d):
            spans.append(run.span())
        # bound: 12-19 digits, no Luhn or brand needed (a misheard digit)
        elif card_groups and 12 <= len(d) <= 19 and bound("card", *run.span()):
            spans.append(run.span())
    return spans


# --- account ---

_IBAN = re.compile(
    r"(?:(?<=IBAN)|(?<![A-Za-z0-9]))[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b", re.I)
_IBAN_MIN = 15   # the shortest IBAN (Norway)
# An IBAN spelled out: single letters and digit groups ("G B 73, H W E C, 4532, ...").
_SPELLED = re.compile(r"(?<![A-Za-z0-9'’])(?:[A-Za-z]|\d+)(?:,? (?:[A-Za-z]|\d+))+(?![A-Za-z0-9])")
# One token holding letters, digits and dashes: "E11248142150", "ACCT-911155810" (all-digit
# tokens are runs, judged above).
_ACCOUNT_TOKEN = re.compile(r"(?<![\w-])[A-Za-z0-9][A-Za-z0-9-]{6,32}[A-Za-z0-9](?![\w-])")


def _iban(norm: str, m: re.Match) -> tuple[int, int] | None:
    """The longest prefix of the match, cut after a group of four, that passes mod-97: a number
    said right after an IBAN ("… 7034 2024") is not part of it."""
    ends, count = [], 0
    for k, ch in enumerate(m.group(), m.start()):
        if ch.isalnum():
            count += 1
            if count % 4 == 0 or k == m.end() - 1:
                ends.append((count, k + 1))
    for count, end in reversed(ends):
        if count >= _IBAN_MIN and _iban_ok(norm[m.start():end]):
            return m.start(), end
    return None


_SORT_CODE = re.compile(r"\d{2}[- ]\d{2}[- ]\d{2}")


def _account_shape(s: str) -> bool:
    """Contiguous, 2-5 groups of 3+ digits, or a sort code: a list holding a 1- or 2-digit group
    ("Routing was 675, 86, 415, 4, 643 packets") is not an account number."""
    shape = _shape(s)
    groups = len(shape) == 1 or (len(shape) <= 5 and min(shape) >= 3)
    return (groups and 6 <= sum(shape) <= 17 and not _DATE.fullmatch(s)) or bool(_SORT_CODE.fullmatch(s))


def _accounts(norm: str, bound: Bound) -> list[tuple[int, int]]:
    spans = [span for m in _IBAN.finditer(norm) if (span := _iban(norm, m))]
    for run in _runs(norm):
        if _account_shape(run.group()) and bound("account", *run.span()):
            spans.append(run.span())
    for m in _ACCOUNT_TOKEN.finditer(norm):
        token = m.group()
        lettered = re.search(r"[A-Za-z]", token) and sum(c.isdigit() for c in token) >= 6
        if lettered and _alone(norm, *m.span()) and bound("account", *m.span()):
            spans.append(m.span())
    for m in _SPELLED.finditer(norm):
        if sum(c.isalnum() for c in m.group()) >= 12 and bound("iban", *m.span()):
            spans.append(m.span())
    return spans


# --- email (N4) ---

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_TLD = "com|org|net|edu|gov|io|co|us|uk|de|ca|me|info|ai|app|dev"
# " at example.org" as the transcriber writes it, or " at example dot org" as words
_AT_DOMAIN = re.compile(rf" at (?:[A-Za-z0-9-]+\.)+(?:{_TLD})\b| at (?:\w+ dot )+(?:{_TLD})\b", re.I)
_LOCAL_TOKEN = re.compile(r"[\w.+-]+$")
_LOCAL_SPOKEN = re.compile(r"\w+(?: dot \w+)+$", re.I)
_LOCAL_NAME_DIGITS = re.compile(r"[A-Za-z]{3,}\d{1,4}")
# what may stand right before a "quarrel43": a sentence or turn start, or one of these words
_LOCAL_LEAD = re.compile(
    r"(?:[.?!]\s+|\n\s*|\*\*[^*\n]+:\*\*\s*|\b(?:it's|it is|is|use|try|to|at|email|e-mail)\s+|:\s*)$", re.I)
_EMAIL_BEFORE = re.compile(
    r"(?:\b(?:my|your|his|her|their|our|the) e-?mail(?: address)?((?: is|:|'s)?)|\be-?mail(?: address)?( is|:))"
    r" ((?:\S+ ){0,4}\S+)$", re.I)
_EMAIL_AFTER = re.compile(
    r"\s*[,.]?\s*(?:that's|that is)\s+(?:my|your|his|her|their|our)\s+e-?mail(?: address)?(?=\s*(?:[.,!?]|$))",
    re.I | re.M)
_SENTENCE_START = re.compile(r".*(?:[.?!]\s+|\n|\*\*[^*\n]+:\*\*\s*)", re.S)   # ends where one starts
# Words that never make a local part on their own: "it at amazon.com", "me at Starbucks.com".
_NOT_LOCAL = frozenset(
    "it me us them him her you that this there here up out in on over back look looked".split()
)


def _local_ok(local: str) -> bool:
    words = local.split()
    if len(words) == 1 and not re.search(r"[A-Za-z]", local):
        return False                                     # "$4.99 at Target.com", "3 at amazon.com"
    return not all(w.strip(".,'").lower() in _NOT_LOCAL for w in words)


def _dotted_name(token: str) -> bool:
    """"quarrel.telvane", "Faxmore.ster": dotted or underscored, every part 2+ letters."""
    parts = re.split(r"[._]", token)
    return len(parts) > 1 and all(len(re.findall(r"[A-Za-z]", p)) >= 2 for p in parts)


def _email_local(norm: str, at: int, end: int) -> int | None:
    """Start of the local part in front of the " at " at `at`, or None when nothing says this is
    an address rather than a website mentioned after "at"."""
    lo = max(0, at - 200)
    head = norm[lo:at]
    starts = []
    m = _LOCAL_SPOKEN.search(head)                       # "jane dot doe at …"
    if m:
        starts.append(m.start())
    m = _LOCAL_TOKEN.search(head)
    if m and _local_ok(m.group()):
        named = _LOCAL_NAME_DIGITS.fullmatch(m.group()) and (
            lo + m.start() == 0 or _LOCAL_LEAD.search(head[max(0, m.start() - 30):m.start()]))
        if _dotted_name(m.group()) or named:             # "quarrel.telvane at …", "It's quarrel43 at …"
            starts.append(m.start())
    m = _EMAIL_BEFORE.search(head)                       # "my email is vanholt at …"
    # "the email said to look at …": with no "is" or ":", only one word can be the local part
    if m and _local_ok(m.group(3)) and (m.group(1) or m.group(2) or " " not in m.group(3)):
        starts.append(m.start(3))
    if _EMAIL_AFTER.match(norm, end):                    # "… at example.com, that's my email"
        begin = _SENTENCE_START.match(head)
        begin = begin.end() if begin else 0
        words = list(re.finditer(r"\S+", head[begin:]))[-3:]
        if words and _local_ok(" ".join(w.group() for w in words)):
            starts.append(begin + words[0].start())
    return lo + min(starts) if starts else None


def _emails(norm: str, bound: Bound) -> list[tuple[int, int]]:
    spans = [m.span() for m in _EMAIL.finditer(norm)]
    for m in _AT_DOMAIN.finditer(norm):
        start = _email_local(norm, m.start(), m.end())
        if start is not None:
            spans.append((start, m.end()))
    return spans


# --- address, phone ---

_STREET = (
    "Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Place|Pl|"
    "Terrace|Parkway|Pkwy|Circle|Highway|Hwy|Trail|Trl|Loop|Ridge|Crescent|Square|Sq|Plaza|Alley|"
    "Pike|Turnpike|Expressway|Row"
)
_ADDRESS = re.compile(
    rf"\b\d{{1,6}}(?:\s\d{{1,3}})?\s+(?:[A-Z][a-z]+\s+){{1,4}}(?:{_STREET})\b\.?"
    r"(?:,? (?i:apartment|apt|suite|unit|#) \w+)?(?:,? \d{5}(?:-\d{4})?\b)?"
)
# NXX NXX XXXX, area and exchange starting 2-9, separators a space, dash or dot
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\([2-9]\d{2}\) ?|[2-9]\d{2}[ .-])[2-9]\d{2}[ .-]\d{4}(?!\d)")


def _phones(norm: str, bound: Bound) -> list[tuple[int, int]]:
    spans = [m.span() for m in _PHONE.finditer(norm)]
    for run in _runs(norm):
        if 10 <= len(_digits(run.group())) <= 11 and len(_shape(run.group())) <= 4 and bound("phone", *run.span()):
            spans.append(run.span())
    return spans


_DETECTORS = {
    "ssn": _ssns,
    "card": _cards,
    "account": _accounts,
    "phone": _phones,
    "address": lambda norm, bound: [m.span() for m in _ADDRESS.finditer(norm)],
    "email": _emails,
}
# An exact tie goes to the class listed first (R9).
_RANK = {c: k for k, c in enumerate(_DETECTORS)}


def _detect(norm: str, classes: set[str]) -> list[Hit]:
    """Hits in the normalised text."""
    bound = _binder(norm)
    return [Hit(a, b, cls) for cls, detect in _DETECTORS.items() if cls in classes for a, b in detect(norm, bound)]


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
