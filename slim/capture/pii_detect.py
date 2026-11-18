"""Pattern detection of personal data in normalised text: cue binding, checksums, tables, the
class detectors, and `find`, which maps hits back to the original text."""

import bisect
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

# --- cue binding ---
#
# A number is masked on the strength of a cue only when it is BOUND to it (N1): it is the
# first number after the cue with nothing but filler words between them, or it is directly
# followed by "that's my <cue>", or it answers a question that asked for the cue thing (N1c,
# P6). A value bound to a phone cue is a phone number and never another class (P2).

# Cue nouns per class, longest first, matched word-bounded on the whole text.
_CUE_NOUNS = {
    "ssn": "social security number|social security|social|ssn",
    "card": "credit card number|debit card number|card number|credit card|debit card|card|visa|mastercard|"
            "amex|american express",
    "account": "routing number|routing|account number|account no|account|acct|iban|bban|sort code",
    "iban": "iban",
    "phone": "phone number|phone|cell|mobile|callback|call back|best number|good number|number for|number on",
}
_CUE = {c: re.compile(rf"\b(?:{nouns})\b", re.I) for c, nouns in _CUE_NOUNS.items()}
_POSSESSIVE = "my|your|his|her|their|our|the"
# "644 04 8901. That's my social.", "12345678 for the account." The noun must end its clause:
# "888 655 8028 is the Social Security office" is not an SSN (P3).
_CUE_AFTER = {
    c: re.compile(
        r"\s*(?:[,.—–-]\s*)?(?:and\s+)?(?:(?:that's|that is|it's|it is|which is|is|was|would be)\s+)?(?:for\s+)?"
        rf"(?:{_POSSESSIVE})\s+(?:{nouns})(?=\s*(?:[.,!?;]|$))", re.I | re.M)
    for c, nouns in _CUE_NOUNS.items()
}
_GAP_MAX = 60
_FILLER = frozenset(
    "is it's its it number no the my your okay ok sure yeah yes yep so um uh well here that's that "
    "would be and goes ahead right alright thanks please".split()
)
# matched on the ORIGINAL text, so "one moment" is not read as a number (m2)
_FILLER_PHRASES = re.compile(
    r"\b(?:let me see|let me check|let me look|let me find it|hold on|just a second|just a sec|just a moment|"
    r"give me a second|bear with me|one second|one moment)\b", re.I)


def _gap_ok(gap: str) -> bool:
    """Only filler words, punctuation and at most one turn label between a cue and its value.
    `gap` is original text; a leading "'s" ("my social's …") reads as "is" (P7)."""
    if len(gap) > _GAP_MAX or len(_TURN_START.findall(gap)) > 1:
        return False
    gap = _FILLER_PHRASES.sub(" ", re.sub(r"^['’]s\b", " is", gap))
    if re.search(r"\d", gap):
        return False
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", _TURN_START.sub(" ", gap).lower().replace("’", "'"))
    return all(w in _FILLER for w in words)


# A cue in a question, or across a turn, must ask FOR its thing (P6): sentence-initial, or right
# after a determiner (one adjective allowed), or coordinated with a cue that does ("Routing and
# account number?") — and no other thing named before it ("the order number on the account").
_ASK_ADJECTIVES = "full|last|whole|complete|current|new|old|other|same"
_ASK_LEAD = {
    c: re.compile(
        r"\b(?:your|the|my|his|her|their|our|that|this|a|an|which|what)\s+"
        rf"(?:(?:{_ASK_ADJECTIVES}{'|best|good' if c == 'phone' else ''})\s+)?$", re.I)
    for c in _CUE_NOUNS
}   # "the best phone number": best|good only before a phone cue (ruling on P2b vs P6(i))
_COORDINATED = re.compile(r"\s*(?:,|and|or|, and|, or)\s*")
_OTHER_THINGS = (
    "order|confirmation|tracking|ticket|invoice|reference|ref|claim|case|policy|zip|postal|amount|balance|"
    "total|price|cost|date|year|years|birthday|born|expiry|expiration|how much|how many|when"
)
_PHONE_THINGS = "phone|cell|mobile|callback|best number|good number|number for|number on"
_OTHER = {
    c: re.compile(rf"\b(?:{_OTHER_THINGS}{'' if c == 'phone' else '|' + _PHONE_THINGS})\b", re.I) for c in _CUE
}
_SENTENCE_END = re.compile(r"[.?!]+(?=\s|$)")


def _asked_cues(norm: str, s: int, e: int) -> list[tuple[int, int, set[str]]]:
    """The cue nouns in sentence [s, e) that ask for their thing, in order: (start, end, classes)."""
    found: dict[tuple[int, int], set[str]] = {}
    for c, rx in _CUE.items():
        for m in rx.finditer(norm, s, e):
            found.setdefault(m.span(), set()).add(c)
    valid: list[tuple[int, int, set[str]]] = []
    for (a, b), classes in sorted(found.items()):
        pre = norm[s:a]
        coordinated = valid and _COORDINATED.fullmatch(norm[valid[-1][1]:a])
        asks = {c for c in classes if not _OTHER[c].search(pre)
                and (not pre.strip() or coordinated or _ASK_LEAD[c].search(pre))}
        if asks:
            valid.append((a, b, asks))
    return valid


def _sentences(norm: str, s: int, e: int) -> list[tuple[int, int]]:
    """Sentences of [s, e), each ending after its punctuation."""
    out, start = [], s
    for m in _SENTENCE_END.finditer(norm, s, e):
        out.append((start, m.end()))
        start = m.end()
    if norm[start:e].strip():
        out.append((start, e))
    return [(a + len(norm[a:b]) - len(norm[a:b].lstrip()), b) for a, b in out]


# --- numbers that stand alone (N2) ---

# A maximal run of digit groups joined by one separator: a space, ", ", "-", "." or "/". A run is
# a list of ", "-separated segments; a value ends at a comma (P1).
_RUN = re.compile(r"\d+(?:(?:, |[ ./-])\d+)*")
_SEGMENT = re.compile(r"\d+(?:[ ./-]\d+)*")
_OPENERS = "([{\"'“‘#"
_CLOSERS = ".,;:!?)]}\"'”’"


def _alone(norm: str, a: int, b: int) -> bool:
    """Not glued to a letter or to another digit group: "B219 09 9999", "12,219 09 9999" and the
    "989-3111" of "(987) 989-3111" are not alone."""
    before = a == 0 or norm[a - 1] in _OPENERS or (
        norm[a - 1].isspace() and not re.search(r"\d\)\s$", norm[max(0, a - 3):a]))
    after = b == len(norm) or norm[b].isspace() or (
        norm[b] in _CLOSERS and (b + 1 == len(norm) or norm[b + 1].isspace()))
    return before and after


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


# --- what a bound value is never (P2, P4) ---

_ZIP4 = re.compile(r"\d{5}[- ]\d{4}")
# written as a phone number, with separators: never an SSN or an account number
_PHONE_SHAPED = re.compile(r"(?:1[ .-]?)?(?:[2-9]\d{2}([ .-])[2-9]\d{2}\1\d{4}|[2-9]\d{2} [2-9]\d{6})")


def _year(y: str) -> bool:
    return len(y) == 2 or 1900 <= int(y) <= 2099


def _date(s: str) -> bool:
    """d/m/y or m/d/y (also . and -), y/m/d, and spaced with a four-digit year."""
    m = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{4}|\d{2})", s) or re.fullmatch(
        r"(\d{1,2})( )(\d{1,2}) (\d{4})", s)
    if m:
        a, b = int(m[1]), int(m[3])
        return ((1 <= a <= 12 and 1 <= b <= 31) or (1 <= b <= 12 and 1 <= a <= 31)) and _year(m[4])
    m = re.fullmatch(r"(\d{4})([./ -])(\d{1,2})\2(\d{1,2})", s)
    return bool(m) and _year(m[1]) and 1 <= int(m[3]) <= 12 and 1 <= int(m[4]) <= 31


def _not_a_value(s: str) -> bool:
    return _date(s) or bool(_ZIP4.fullmatch(s))


def _all_years(s: str) -> bool:
    return all(len(g) == 4 and 1900 <= int(g) <= 2099 for g in re.findall(r"\d+", s))


# The speaker says what the number is, and it is not the cue's thing (P5): "98919781 is the
# ticket", "821 688 dollars short".
_LABEL_AFTER = re.compile(
    r"\s*,?\s*(?:is|was|are|were)\s+(?:the|my|a|an|our|your|his|her|their)\s+([^.,!?;\n]*)", re.I)
_UNIT_AFTER = re.compile(
    r"\s*(?:dollars|bucks|cents|points|percent|years|people|times|views|followers|entries|lookups|packets|"
    r"users|months|days|hours|minutes|miles|items|units|of them|each|short|strong)\b", re.I)


class _Scene:
    """One text's runs, turns and cue positions, and the binding questions asked of them."""

    def __init__(self, norm: str, text: str, idx: list[int]):
        self.norm, self.text, self.idx = norm, text, idx
        self.runs = []                                   # (start, end, [segment spans])
        for m in _RUN.finditer(norm):
            if _alone(norm, *m.span()):
                segs = [(m.start() + s.start(), m.start() + s.end()) for s in _SEGMENT.finditer(m.group())]
                self.runs.append((m.start(), m.end(), segs))
        self.run_ends = [r[1] for r in self.runs]
        self.cues = {c: [m.span() for m in rx.finditer(norm)] for c, rx in _CUE.items()}
        self.cue_ends = {c: [b for _, b in spans] for c, spans in self.cues.items()}
        self.labels = [m.span() for m in _TURN_START.finditer(norm)]
        self.label_starts = [a for a, _ in self.labels]
        self._questions: dict[int, list[tuple[int, int, set[str]]]] = {}
        self.phone_bound = self.bound("phone", _phone_fits)

    def orig(self, a: int, b: int) -> str:
        return self.text[self.idx[a]:self.idx[b]]

    def turn(self, p: int) -> int:
        """Index of the turn holding position p, -1 before the first label."""
        return bisect.bisect_right(self.label_starts, p) - 1

    def turn_end(self, k: int) -> int:
        return self.labels[k + 1][0] if k + 1 < len(self.labels) else len(self.norm)

    def question(self, k: int) -> list[tuple[int, int, set[str]]]:
        """The cues that turn k asks for: its last sentence, or the one before it, ends with "?"."""
        if k not in self._questions:
            sents = _sentences(self.norm, self.labels[k][1], self.turn_end(k))
            asked: list[tuple[int, int, set[str]]] = []
            for s, e in sents[-1:] + sents[-2:-1]:
                if self.norm[s:e].rstrip().endswith("?"):
                    asked = _asked_cues(self.norm, s, e)
                    if asked:
                        break
            self._questions[k] = asked
        return self._questions[k]

    def asks_for(self, cls: str, cue: tuple[int, int]) -> bool:
        """The cue at `cue` asks for its thing in its own sentence (P6)."""
        k = self.turn(cue[0])
        start = self.labels[k][1] if k >= 0 else 0
        for a, b in _sentences(self.norm, start, self.turn_end(k)):
            if a <= cue[0] < b:
                return any((x, y) == cue and cls in cs for x, y, cs in _asked_cues(self.norm, a, b))
        return False

    def before(self, cls: str, a: int) -> bool:
        """Bound by a cue before the value, or by the question the value answers."""
        ends = self.cue_ends[cls]
        for j in range(bisect.bisect_left(ends, a - _GAP_MAX), bisect.bisect_right(ends, a)):
            end = ends[j]
            if not _gap_ok(self.orig(end, a)):
                continue
            if _TURN_START.search(self.norm, end, a) and not self.asks_for(cls, self.cues[cls][j]):
                continue
            return True
        return self.asked(cls, a)

    def asked(self, cls: str, a: int) -> bool:
        """The previous turn asked for `cls`, and this is the first number of the answer with only
        filler words before it — or the second, after a question that named two things."""
        k = self.turn(a)
        if k <= 0:
            return False
        q, body = self.question(k - 1), self.labels[k][1]
        if not q:
            return False
        if _gap_ok(self.orig(body, a)):
            return any(cls in cs for _, _, cs in q)
        r = bisect.bisect_right(self.run_ends, a) - 1        # the number before this one
        if len(q) < 2 or cls not in q[1][2] or r < 0 or self.runs[r][0] < body:
            return False
        prev = self.runs[r]
        return _gap_ok(self.orig(body, prev[0])) and _gap_ok(self.orig(prev[1], a))

    def after(self, cls: str, b: int) -> bool:
        return bool(_CUE_AFTER[cls].match(self.norm, b))

    def labelled(self, cls: str, b: int) -> bool:
        m = _LABEL_AFTER.match(self.norm, b)
        return bool(m and not _CUE[cls].search(m.group(1))) or bool(_UNIT_AFTER.match(self.norm, b))

    def bound(self, cls: str, fits: Callable[[str], bool]) -> list[tuple[int, int]]:
        """Values bound to a `cls` cue: the longest prefix of a run's segments that fits after a
        cue before it (unless the speaker labels it as something else), the longest suffix that
        fits before a cue after it."""
        spans = []
        for a0, b0, segs in self.runs:
            if self.before(cls, a0):
                for _, b in reversed(segs):
                    if fits(self.norm[a0:b]):
                        if not self.labelled(cls, b):
                            spans.append((a0, b))
                        break
            if self.after(cls, b0):
                for a, _ in segs:
                    if fits(self.norm[a:b0]):
                        spans.append((a, b0))
                        break
        return spans

    def segments(self):
        for _, _, segs in self.runs:
            for a, b in segs:
                yield a, b, self.norm[a:b]


# --- ssn ---

_SSN_DASHED = re.compile(r"\d{3}-\d{2}-\d{4}")
_SSN_SPACED = re.compile(r"\d{3} \d{2} \d{4}")
# "741 05 7024" with no cue is how the transcriber writes a spoken SSN, and also how it writes
# "we had 300 25 2000 users". Measurement-gated: False switches the shape off.
SSN_SPACED_CUE_FREE = True
# "the last four of my social are 8901" (m3); "four" is "4" after normalise
_SSN_LAST_FOUR = re.compile(
    rf"\blast 4(?: digits)? of (?:{_POSSESSIVE}) (?:{_CUE_NOUNS['ssn']})(?: is| are|:)? (\d{{4}})\b", re.I)


def _ssn_fits(s: str) -> bool:
    """Bound: 7-9 digits in up to 3 groups, whatever the range check says (a misheard digit)."""
    return 7 <= len(_digits(s)) <= 9 and len(_shape(s)) <= 3 and not _not_a_value(s) and not _PHONE_SHAPED.fullmatch(s)


def _ssns(scene: _Scene) -> list[tuple[int, int]]:
    spans = []
    for a, b, s in scene.segments():
        spaced = SSN_SPACED_CUE_FREE and _SSN_SPACED.fullmatch(s)
        if (_SSN_DASHED.fullmatch(s) or spaced) and _ssn_ok(_digits(s)):
            spans.append((a, b))
    for m in _SSN_LAST_FOUR.finditer(scene.norm):
        if _alone(scene.norm, *m.span(1)):
            spans.append(m.span(1))
    return spans + scene.bound("ssn", _ssn_fits)


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


def _brand_prefix(d: str) -> bool:
    return any(lo <= int(d[:len(str(lo))]) <= hi for prefixes, _ in _BRANDS.values() for lo, hi in prefixes)


def _branded(d: str) -> bool:
    return any(
        len(d) in lengths and any(lo <= int(d[:len(str(lo))]) <= hi for lo, hi in prefixes)
        for prefixes, lengths in _BRANDS.values()
    )


def _card_groups(s: str) -> bool:
    shape = _shape(s)
    return len(shape) <= 5 and min(shape) >= 3                # contiguous, or groups of 3+


def _card_fits(s: str) -> bool:
    """Bound: 12-19 digits with a brand prefix; Luhn and exact length not needed (a misheard
    digit), but "2007 2008 2009 2010" is years, not a card."""
    d = _digits(s)
    return 12 <= len(d) <= 19 and _card_groups(s) and _brand_prefix(d) and not _not_a_value(s) and not _all_years(s)


def _cards(scene: _Scene) -> list[tuple[int, int]]:
    spans = []
    for a, b, s in scene.segments():
        # cue-free: Luhn and a brand, groups joined by single spaces or dashes only
        if _card_groups(s) and not re.search(r"[./]", s) and _branded(_digits(s)) and _luhn(_digits(s)):
            spans.append((a, b))
    return spans + scene.bound("card", _card_fits)


# --- account ---

_IBAN = re.compile(
    r"(?:(?<=IBAN)|(?<![A-Za-z0-9]))[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b", re.I)
_IBAN_MIN = 15   # the shortest IBAN (Norway)
# An IBAN spelled out: single letters and digit groups ("G B 73, H W E C, 4532, ...").
_SPELLED = re.compile(r"(?<![A-Za-z0-9'’])(?:[A-Za-z]|\d+)(?:,? (?:[A-Za-z]|\d+))+(?![A-Za-z0-9])")
# One token holding letters, digits and dashes: "E11248142150", "ACCT-911155810" (all-digit
# tokens are runs, judged above).
_ACCOUNT_TOKEN = re.compile(r"(?<![\w-])[A-Za-z0-9][A-Za-z0-9-]{6,32}[A-Za-z0-9](?![\w-])")
_SORT_CODE = re.compile(r"\d{2}[- ]\d{2}[- ]\d{2}")


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


def _account_fits(s: str) -> bool:
    """Bound: contiguous, 2-5 groups of 3+ digits, or a sort code: a list holding a 1- or 2-digit
    group ("Routing was 675, 86, 415, 4, 643 packets") is not an account number."""
    shape = _shape(s)
    groups = len(shape) == 1 or (len(shape) <= 5 and min(shape) >= 3)
    if _not_a_value(s) or _all_years(s) or _PHONE_SHAPED.fullmatch(s):
        return False
    return (groups and 6 <= sum(shape) <= 17) or bool(_SORT_CODE.fullmatch(s))


def _accounts(scene: _Scene) -> list[tuple[int, int]]:
    norm = scene.norm
    spans = [span for m in _IBAN.finditer(norm) if (span := _iban(norm, m))]
    spans += scene.bound("account", _account_fits)
    for m in _ACCOUNT_TOKEN.finditer(norm):
        token = m.group()
        lettered = re.search(r"[A-Za-z]", token) and sum(c.isdigit() for c in token) >= 6
        if lettered and _alone(norm, *m.span()) and (
                (scene.before("account", m.start()) and not scene.labelled("account", m.end()))
                or scene.after("account", m.end())):
            spans.append(m.span())
    for m in _SPELLED.finditer(norm):
        # an IBAN starts with its two country letters: "1995, 1996, …" after "IBAN" is not one
        country = re.match(r"[A-Za-z],? [A-Za-z](?![A-Za-z0-9])", m.group())
        if country and sum(c.isalnum() for c in m.group()) >= 12 and (
                (scene.before("iban", m.start()) and not scene.labelled("iban", m.end()))
                or scene.after("iban", m.end())):
            spans.append(m.span())
    return spans


# --- email (N4) ---

# bounded parts: an unbounded local part backtracks quadratically on a long run of letters (m4)
_EMAIL = re.compile(r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63})+")
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


def _emails(scene: "_Scene") -> list[tuple[int, int]]:
    norm = scene.norm
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


def _phone_fits(s: str) -> bool:
    return 10 <= len(_digits(s)) <= 11 and len(_shape(s)) <= 4


def _phones(scene: _Scene) -> list[tuple[int, int]]:
    return [m.span() for m in _PHONE.finditer(scene.norm)] + scene.phone_bound


_DETECTORS = {
    "ssn": _ssns,
    "card": _cards,
    "account": _accounts,
    "phone": _phones,
    "address": lambda scene: [m.span() for m in _ADDRESS.finditer(scene.norm)],
    "email": _emails,
}
# An exact tie goes to the class listed first (R9).
_RANK = {c: k for k, c in enumerate(_DETECTORS)}


def _detect(norm: str, classes: set[str], text: str | None = None, idx: list[int] | None = None) -> list[Hit]:
    """Hits in the normalised text. `text` and `idx` (from `normalise`) let filler phrases be read
    in the original words; without them the normalised text stands in."""
    scene = _Scene(norm, norm if text is None else text, list(range(len(norm) + 1)) if idx is None else idx)
    hits = [Hit(a, b, cls) for cls, detect in _DETECTORS.items() if cls in classes for a, b in detect(scene)]
    # a value bound to a phone cue is a phone number, whether or not phones are masked (P2)
    phone = _merged(scene.phone_bound)
    starts = [a for a, _ in phone]

    def phone_number(h: Hit) -> bool:
        i = bisect.bisect_left(starts, h.end) - 1
        return i >= 0 and phone[i][1] > h.start
    return [h for h in hits if h.cls not in ("ssn", "card", "account") or not phone_number(h)]


def _merged(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def find(text: str, classes: Iterable[str]) -> list[Hit]:
    """Non-overlapping hits in the original text, sorted; the longest wins an overlap, and an
    exact tie goes to the higher class (ssn > card > account > phone > address > email)."""
    wanted = set(classes)
    norm, idx = normalise(text)
    mapped = []
    for h in _detect(norm, wanted, text, idx):
        a, b = h.start, h.end
        # a converted number phrase is all-or-nothing: never cut one in half
        while a > 0 and idx[a - 1] == idx[a]:
            a -= 1
        q = b - 1
        while q + 1 < len(norm) and idx[q + 1] == idx[q]:
            q += 1
        mapped.append(Hit(idx[a], idx[q + 1], h.cls))
    kept: list[Hit] = []                                      # disjoint, sorted by start
    starts: list[int] = []
    for h in sorted(mapped, key=lambda h: (-(h.end - h.start), h.start, _RANK[h.cls])):
        i = bisect.bisect_left(starts, h.start)
        if (i == 0 or kept[i - 1].end <= h.start) and (i == len(kept) or kept[i].start >= h.end):
            kept.insert(i, h)
            starts.insert(i, h.start)
    return kept
