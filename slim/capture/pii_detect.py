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
# The words that stand before a noun, as one list composed per use.
_PERSONAL = "my|your|his|her|their"         # "that's my social": one person's own
_POSSESSIVES = f"{_PERSONAL}|our"
_THE = f"{_POSSESSIVES}|the"                 # "my social", "the card number"
_DETERMINERS = f"{_THE}|that|this|a|an"     # "a callback number", "this account"

# --- cue binding ---
#
# A number is masked on the strength of a cue only when it is BOUND to it: it is the first
# number after the cue with nothing but filler words between them, or it is directly followed
# by "that's my <cue>", or it answers a question that asked for the cue thing. A value bound to
# a phone cue is a phone number and never another class.

# Cue nouns per class, longest first, matched word-bounded on the whole text.
_CARD_NAMES = "credit card|debit card|card|visa|mastercard|amex|american express"
# "the number on your card" asks for the card number, not a phone number
_NUMBER_ON_CARD = rf"number on (?:{_THE}) (?:{_CARD_NAMES})"
_CUE_NOUNS = {
    "ssn": "social security number|social security|social|ssn",
    "card": f"{_NUMBER_ON_CARD}|credit card number|debit card number|card number|{_CARD_NAMES}",
    "account": "routing number|routing|account number|account no|account|acct|iban|bban|sort code",
    "iban": "iban",
    "phone": f"phone number|phone|cell|mobile|callback number|call back number|callback|call back|best number|"
             f"good number|number for|"
             f"number on(?! (?:{_THE}) (?:{_CARD_NAMES})\\b)",
}
_CUE = {c: re.compile(rf"\b(?:{nouns})\b", re.I) for c, nouns in _CUE_NOUNS.items()}
# "644 04 8901. That's my social.", "12345678 for the account." The noun must end its clause:
# "888 655 8028 is the Social Security office" is not an SSN.
_CUE_AFTER = {
    c: re.compile(
        r"\s*(?:[,.—–-]\s*)?(?:and\s+)?(?:(?:that's|that is|it's|it is|which is|is|was|would be)\s+)?(?:for\s+)?"
        rf"(?:{_THE})\s+(?:{nouns})(?=\s*(?:[.,!?;]|$))", re.I | re.M)
    for c, nouns in _CUE_NOUNS.items()
}
_GAP_MAX = 60
_FILLER = frozenset(
    "is it's its it number no the my your okay ok sure yeah yes yep so um uh well here that's that "
    "would be and goes ahead right alright thanks please".split()
)
# matched on the ORIGINAL text, so "one moment" is not read as a number
_FILLER_PHRASES = re.compile(
    r"\b(?:let me see|let me check|let me look|let me find it|hold on|just a second|just a sec|just a moment|"
    r"give me a second|bear with me|one second|one moment|on file)\b", re.I)


def _gap_ok(gap: str) -> bool:
    """Only filler words, punctuation and at most one turn label between a cue and its value.
    `gap` is original text; a leading "'s" ("my social's …") reads as "is"."""
    if len(gap) > _GAP_MAX or len(_TURN_START.findall(gap)) > 1:
        return False
    gap = _FILLER_PHRASES.sub(" ", re.sub(r"^['’]s\b", " is", gap))
    if re.search(r"\d", gap):
        return False
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", _TURN_START.sub(" ", gap).lower().replace("’", "'"))
    return all(w in _FILLER for w in words)


# A cue in a question, or across a turn, must ask FOR its thing: sentence-initial, or right
# after a determiner (one adjective allowed), or coordinated with a cue that does ("Routing and
# account number?") — and no other thing named before it ("the order number on the account").
_ASK_ADJECTIVES = "full|last|whole|complete|current|new|old|other|same"
_ASK_LEAD = {
    c: re.compile(
        rf"\b(?:{_DETERMINERS}|which|what)\s+"
        rf"(?:(?:{_ASK_ADJECTIVES}{'|best|good' if c == 'phone' else ''})\s+)?$", re.I)
    for c in _CUE_NOUNS
}   # "the best phone number", "a good cell": best and good only ever describe a phone
_COORDINATED = re.compile(r"\s*(?:,|and|or|, and|, or)\s*")
# Another thing named in the cue's own clause: "your claim account number", "the order number on
# the account". Cue nouns themselves are blanked out first. An identifier is read like an account
# number, so a question naming one anywhere ("Can I get the order number and the account number?")
# may be answered with it; a zip, a date or an amount has a shape of its own and blocks only in
# the cue's clause ("Can I get your zip and your account number?" ⏎ "3744644286").
_IDENTIFIER = (r"order|confirmation|tracking|ticket|invoice|reference|ref|claim|case|policy|"
               r"employee id|member id|customer id|order id|id number|(?<!zip )(?<!postal )(?<!area )code")
_OTHER_ID = re.compile(rf"\b(?:{_IDENTIFIER})\b", re.I)
_OTHER = re.compile(
    rf"\b(?:{_IDENTIFIER}|zip|postal|amount|balance|total|price|cost|date|year|years|birthday|born|expiry|"
    r"expiration|how much|how many|when)\b", re.I)
# A cue noun that is the object of a preposition is a modifier, not the thing being given: "the
# phone number on the account", "your employee ID for the account". Cards are exempt: "put it on
# my visa, …" is how a card number is introduced.
_UNDER_PREPOSITION = re.compile(
    r"\b(?:on|for|to|with|from|in|at|about|into|onto|regarding|under|against)\s+"
    rf"(?:(?:{_DETERMINERS})\s+)?"
    rf"(?:(?:{_ASK_ADJECTIVES})\s+)?$", re.I)
_PREPOSITION_CLASSES = {"ssn", "account", "iban"}   # the classes a same-sentence cue is checked for


# A phone noun that governs the cue: "the phone number on the account", "a cell for my account".
# The number given is the phone's. "my phone bill", "my mobile account number" govern nothing.
_PHONE_GOVERNS = re.compile(
    r"\b(?:phone number|phone|cell|mobile|callback number|call back number|callback|call back|best number|"
    rf"good number|number)\s+(?:on|for|of|to|with)\s+(?:(?:{_DETERMINERS})\s+)?(?:(?:{_ASK_ADJECTIVES})\s+)?$", re.I)
# Where the cue's own clause begins: after a number, a comma, semicolon or colon, or a joining word.
_CLAUSE_CUT = re.compile(r"\d|[,;:]|\b(?:and|but|so)\b", re.I)


def _names_other(text: str, rx: re.Pattern) -> bool:
    if not rx.search(text):
        return False
    for cue in _CUE.values():
        text = cue.sub(lambda m: " " * len(m.group()), text)
    return bool(rx.search(text))


def _subject(norm: str, s: int, a: int, cls: str, preposition: bool = True, question: bool = False) -> bool:
    """The cue noun of `cls` at `a`, in the sentence starting at `s`, names the thing being given.
    `preposition`: whether a cue under a preposition is a modifier here (not when the number
    follows the cue directly: "on my account 80608689949"). `question`: an identifier named
    anywhere in the sentence blocks the cue ("Can I get the claim number, and then we'll pull up
    the account?" asks for the claim number)."""
    pre = norm[s:a]
    tail = pre if len(pre) <= 60 else pre[-60:].split(" ", 1)[-1]   # the phrase is short; whole words
    if preposition and cls in _PREPOSITION_CLASSES and _UNDER_PREPOSITION.search(tail):
        return False
    if cls != "phone" and _PHONE_GOVERNS.search(tail):
        return False
    # another thing named in the cue's own clause: "your claim account number", not "the order
    # number is 5512334, and my routing number is …"
    start = 0
    for cut in _CLAUSE_CUT.finditer(pre):
        start = cut.end()
    return not _names_other(pre[start:], _OTHER) and not (question and _names_other(pre, _OTHER_ID))
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
        asks = {c for c in classes if _subject(norm, s, a, c, question=True)
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


# --- numbers that stand alone ---

# A maximal run of digit groups joined by one separator: a space, ", ", "-", "." or "/". A run is
# a list of ", "-separated segments: "4111 1111 1111 1111, 09 28" is a card, then an expiry.
_RUN = re.compile(r"\d+(?:(?:, |[ ./-])\d+)*")
_SEGMENT = re.compile(r"\d+(?:[ ./-]\d+)*")
_GROUP = re.compile(r"\d+")
_OPENERS = "([{\"'“‘#*_`~<"
# What may follow a value before a space, the end or a turn label: "123-45-6789...", "(…6789),",
# "**…6789**" in a summary. A dash ends the value like a space unless a digit follows ("…6789—and").
_CLOSERS = ".,;:!?)]}\"'”’…*_`~>—–"
_DASHES = "—–"


def _alone(norm: str, a: int, b: int) -> bool:
    """Not glued to a letter or to another digit group: "B219 09 9999", "12,219 09 9999", "6789.5"
    and the "989-3111" of "(987) 989-3111" are not alone."""
    before = a == 0 or norm[a - 1] in _OPENERS or (
        norm[a - 1].isspace() and not re.search(r"\d\)\s$", norm[max(0, a - 3):a]))
    if not before:
        return False
    p = b
    while p < len(norm) and not norm[p].isspace() and not _TURN_START.match(norm, p):
        if norm[p] not in _CLOSERS:
            return False
        if norm[p] in _DASHES and p + 1 < len(norm) and not norm[p + 1].isdigit():
            return True
        p += 1
    return True


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


# --- what a bound value is never ---

_ZIP4 = re.compile(r"\d{5}-\d{4}")   # dashed only: a spaced 5-4 is also how an account is read
# A phone number as written: optional 1/+1, an area code (in brackets or not) and an exchange
# starting 2-9, separators a space, dash or dot. `_PHONE` finds one with no cue; `_PHONE_SHAPED`
# says a bound value is one (one separator throughout, or "344 5550101"), so it is never an SSN
# or an account number.
_NXX = r"[2-9]\d{2}"
_PHONE_LEAD = r"(?:\+?1[ .-]?)?"
_PHONE_FORM = rf"{_PHONE_LEAD}(?:\({_NXX}\) ?|{_NXX}[ .-]){_NXX}[ .-]\d{{4}}"
_PHONE_SHAPED = re.compile(
    rf"{_PHONE_LEAD}(?:\({_NXX}\) ?{_NXX}[ .-]\d{{4}}|{_NXX}([ .-]){_NXX}\1\d{{4}}|{_NXX} {_NXX}\d{{4}})")


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


# The speaker says what the number is, and it is not the cue's thing: "98919781 is the
# ticket", "821 688 dollars short".
_WHEN_AFTER = re.compile(r"\s*,?\s*(?:is|was)\s+when\b", re.I)   # "2024 was when I opened it"
_LABEL_AFTER = re.compile(
    rf"\s*,?\s*(?:is|was|are|were)\s+(?:{_THE}|a|an)\s+([^.,!?;\n]*)", re.I)
_UNIT_AFTER = re.compile(
    r"\s*(?:dollars|bucks|cents|points|percent|years|people|times|views|followers|entries|lookups|packets|"
    r"users|months|days|hours|minutes|miles|items|units|of them|each|short|strong|in the red|in the black|"
    r"over the limit|overdrawn)\b", re.I)


# The digits that make a complete value of each class: a comma-joined value stops growing there.
# A card is complete at 15 (an Amex), so "4242, 7844, 6476, 8778" is read whole. An account has
# no set length: leaking the tail of "123456, 7890" is worse than taking an unlabelled number.
CARD_COMPLETE = 15
_SENTENCE_REACH = 300
_LIST_SPLIT = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+", re.I)
# What a listed question may name besides a cue noun: "your zip and your account number?"
_LISTED_THING = re.compile(
    r"\b(?:zip|zip code|postal code|date of birth|birthday|dob|name|address|email|order number|claim number|"
    r"policy number|reference number|confirmation number|member id|employee id|customer id|amount|balance|"
    r"how much|how many)\b",
    re.I)
# Between two numbers of a listed answer: "10001 and 48213377", "10001, and then …", "10001. 4821…"
_LIST_GAP = re.compile(r"\s*,?\s*(?:and(?:\s+then)?|[.?!]+(?:\s+and(?:\s+then)?)?)\s*", re.I)
_COMPLETE = {"ssn": 9, "card": CARD_COMPLETE, "account": None, "iban": None, "phone": 10}
# The most digits a value of each class has: a bound value longer than this masks its longest
# leading part that is one (leaking a tail is better than leaving the whole).
_LONGEST = {"ssn": 9, "card": 19, "account": 17, "iban": 34, "phone": 11}


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
        self.run_starts = [r[0] for r in self.runs]
        self.cues = {c: [m.span() for m in rx.finditer(norm)] for c, rx in _CUE.items()}
        self.cue_ends = {c: [b for _, b in spans] for c, spans in self.cues.items()}
        self.labels = [m.span() for m in _TURN_START.finditer(norm)]
        self.label_starts = [a for a, _ in self.labels]
        self._questions: dict[int, list[tuple[int, int, set[str]]]] = {}
        self._asking: dict[int, list[tuple[int, int]]] = {}
        self._orders: dict[int, dict[int, set[str]]] = {}
        self.phone_bound = self.bound("phone", _phone_fits)

    def orig(self, a: int, b: int) -> str:
        return self.text[self.idx[a]:self.idx[b]]

    def turn(self, p: int) -> int:
        """Index of the turn holding position p, -1 before the first label."""
        return bisect.bisect_right(self.label_starts, p) - 1

    def turn_end(self, k: int) -> int:
        return self.labels[k + 1][0] if k + 1 < len(self.labels) else len(self.norm)

    def questions(self, k: int) -> list[tuple[int, int]]:
        """The sentences of turn k that may ask: its last sentence, or the one before a single
        trailing sentence, when it ends with "?"."""
        if k not in self._asking:
            sents = _sentences(self.norm, max(self.labels[k][1], self.turn_end(k) - 2 * _SENTENCE_REACH),
                               self.turn_end(k))
            self._asking[k] = [(s, e) for s, e in sents[-1:] + sents[-2:-1]
                               if self.norm[s:e].rstrip().endswith("?")]
        return self._asking[k]

    def question(self, k: int) -> list[tuple[int, int, set[str]]]:
        """The cues that turn k asks for."""
        if k not in self._questions:
            self._questions[k] = next(
                (asked for s, e in self.questions(k) if (asked := _asked_cues(self.norm, s, e))), [])
        return self._questions[k]

    def sentence_of(self, p: int) -> tuple[int, int]:
        """The sentence holding p, looked for no further than _SENTENCE_REACH either way (a text
        with no punctuation must not make every lookup read all of it)."""
        k = self.turn(p)
        lo = max(self.labels[k][1] if k >= 0 else 0, p - _SENTENCE_REACH)
        hi = min(self.turn_end(k), p + _SENTENCE_REACH)
        start = lo
        for m in _SENTENCE_END.finditer(self.norm, lo, p):
            start = m.end()
        end = _SENTENCE_END.search(self.norm, p, hi)
        start += len(self.norm[start:p]) - len(self.norm[start:p].lstrip())
        return start, end.end() if end else hi

    def asks_for(self, cls: str, cue: tuple[int, int]) -> bool:
        """The cue at `cue` asks for its thing in its own sentence."""
        a, b = self.sentence_of(cue[0])
        return any((x, y) == cue and cls in cs for x, y, cs in _asked_cues(self.norm, a, b))

    def before(self, cls: str, a: int) -> bool:
        """Bound by a cue before the value, or by the question the value answers."""
        ends = self.cue_ends[cls]
        for j in range(bisect.bisect_left(ends, a - _GAP_MAX), bisect.bisect_right(ends, a)):
            end = ends[j]
            if not _gap_ok(self.orig(end, a)):
                continue
            cue = self.cues[cls][j]
            if _TURN_START.search(self.norm, end, a):
                if not self.asks_for(cls, cue):
                    continue
            elif cls in _PREPOSITION_CLASSES:
                s, e = self.sentence_of(cue[0])
                if not _subject(self.norm, s, cue[0], cls,
                                preposition=bool(re.search(r"[A-Za-z]", self.orig(end, a))),
                                question=self.norm[s:e].rstrip().endswith("?")):
                    continue
            return True
        return self.asked(cls, a)

    def listed(self, k: int) -> list[set[str]] | None:
        """The two or three things turn k's question lists ("your zip code and your account
        number?"), in order: the classes each names ("last4" for the last four of a social, an
        empty set for a named thing that is no cue noun); None when it does not list. A part that
        names nothing ("Okay, and your social?") is no listed thing."""
        for s, e in self.questions(k):
            things = []
            for part in _LIST_SPLIT.split(self.norm[s:e].rstrip().rstrip("?")):
                if _LAST_FOUR_ASKED.search(part):
                    things.append({"last4"})
                elif cues := {c for c, rx in _CUE.items() if rx.search(part)}:
                    things.append(cues)
                elif _LISTED_THING.search(part):
                    things.append(set())
            if 2 <= len(things) <= 3:
                return things
        return None

    def ordinal(self, a: int) -> set[str] | None:
        """The listed thing the number starting at `a` answers, when the answer gives exactly as
        many numbers as the question listed. The numbers are runs: "3267, 333943" is one number
        read in pieces, and only "and" ("and then") or a sentence end separates two."""
        k = self.turn(a)
        if k <= 0:
            return None
        if k not in self._orders:
            order: dict[int, set[str]] = {}
            things = self.listed(k - 1)
            if things:
                body, end = self.labels[k][1], self.turn_end(k)
                numbers, prev = [], None
                i = bisect.bisect_left(self.run_starts, body)
                while i < len(self.runs) and self.runs[i][0] < end and len(numbers) <= len(things):
                    a0, b0, _ = self.runs[i]
                    if not (_gap_ok(self.orig(body, a0)) if prev is None
                            else _LIST_GAP.fullmatch(self.orig(prev, a0))):
                        break
                    numbers.append(a0)
                    prev, i = b0, i + 1
                if len(numbers) == len(things):
                    order = dict(zip(numbers, things))
            self._orders[k] = order
        return self._orders[k].get(a)

    def asked(self, cls: str, a: int) -> bool:
        """The previous turn asked for `cls`, and this is the first number of the answer with only
        filler words before it — or the second, after a question that named two things — or the
        question listed things and this number is in the place of a `cls` thing."""
        k = self.turn(a)
        if k <= 0:
            return False
        thing = self.ordinal(a)
        if thing is not None:
            return cls in thing
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
        if _WHEN_AFTER.match(self.norm, b):
            return True
        m = _LABEL_AFTER.match(self.norm, b)
        return bool(m and not _CUE[cls].search(m.group(1))) or bool(_UNIT_AFTER.match(self.norm, b))

    def bound(self, cls: str, fits: Callable[[str], bool]) -> list[tuple[int, int]]:
        """Values bound to a `cls` cue. A value read in comma-separated pieces takes the next piece
        only while it is not yet a complete value of its class, so "12345678, 300 dollars" stops
        at the account number; after a cue before it, it is dropped when the speaker labels it as
        something else."""
        need, most = _COMPLETE[cls], _LONGEST[cls]

        def short(a: int, b: int) -> bool:
            """Incomplete, and not yet past the longest value of the class (beyond it no piece
            can make one, and "12, 12, 12, …" must not be read whole at every length)."""
            n = len(_digits(self.norm[a:b])) if b - a <= 4 * most else most + 1
            return n <= most and (need is None or n < need)

        def leading(a0: int, segs: list[tuple[int, int]]) -> int | None:
            """End of the value starting the run: grown while incomplete, never into a piece the
            speaker labels ("…, 891 dollars"), then cut back to its longest leading part that fits."""
            k = 0
            while k + 1 < len(segs) and short(a0, segs[k][1]) and not self.labelled(cls, segs[k + 1][1]):
                k += 1
            b = next((segs[j][1] for j in range(k, -1, -1) if fits(self.norm[a0:segs[j][1]])), None)
            if b is None and len(_digits(self.norm[a0:segs[k][1]])) > most and not any(
                    _PHONE_SHAPED.fullmatch(self.norm[x:y]) or _not_a_value(self.norm[x:y]) for x, y in segs[:k + 1]):
                # too long for one value ("4111 1111 1111 1111 2222"): its longest leading whole
                # groups that are one, as for comma pieces
                ends, n = [], 0
                for m in _GROUP.finditer(self.norm, a0, segs[k][1]):
                    n += m.end() - m.start()
                    if n > most:
                        break
                    ends.append(m.end())
                b = next((e for e in reversed(ends) if fits(self.norm[a0:e])), None)
            return b if b is not None and not self.labelled(cls, b) else None

        spans = []
        for a0, b0, segs in self.runs:
            # a listed answer is bound only by its place in the list
            if self.asked(cls, a0) if self.ordinal(a0) is not None else self.before(cls, a0):
                b = leading(a0, segs)
                if b is not None:
                    spans.append((a0, b))
            if self.after(cls, b0):
                k = len(segs) - 1
                while k > 0 and short(segs[k][0], b0):
                    k -= 1
                a = next((segs[j][0] for j in range(k, len(segs)) if fits(self.norm[segs[j][0]:b0])), None)
                if a is not None:
                    spans.append((a, b0))
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
# "the last four of my social are 8901"; "four" is "4" after normalise
_LAST_FOUR_OF = rf"\blast 4(?: digits)? of (?:{_THE}) (?:{_CUE_NOUNS['ssn']})"
_SSN_LAST_FOUR = re.compile(rf"{_LAST_FOUR_OF}(?: is| are|:)? (\d{{4}})\b", re.I)


def _ssn_fits(s: str) -> bool:
    """Bound: 7-9 digits in up to 3 groups, whatever the range check says (a misheard digit)."""
    return 7 <= len(_digits(s)) <= 9 and len(_shape(s)) <= 3 and not _not_a_value(s) and not _PHONE_SHAPED.fullmatch(s)


_LAST_FOUR_ASKED = re.compile(rf"{_LAST_FOUR_OF}\b", re.I)


def _last_four_answers(scene: _Scene) -> list[tuple[int, int]]:
    """"Last four of your social?" ⏎ "Yeah, it's 4471.": exactly four digits, first in the answer."""
    spans = []
    for a, b, segs in scene.runs:
        thing = scene.ordinal(a)                             # a listed answer: "1/2/1980 and 4471"
        if thing is not None:
            if "last4" in thing and b - a == 4 and not scene.labelled("ssn", b):
                spans.append((a, b))
            continue
        k = scene.turn(a)
        if len(segs) != 1 or len(_digits(scene.norm[a:b])) != 4 or k <= 0 or b - a != 4:
            continue
        asked = any(_LAST_FOUR_ASKED.search(scene.norm, s, e) for s, e in scene.questions(k - 1))
        if asked and _gap_ok(scene.orig(scene.labels[k][1], a)) and not scene.labelled("ssn", b):
            spans.append((a, b))
    return spans


# A strong cue: an identifying phrase attached to the value ("265075.1253. That's my social.",
# "My social is 68581399."). Next to it, a number the transcriber garbled (fused with a list,
# a digit too many or too few) is still the SSN.
_STRONG_SSN_AFTER = re.compile(
    rf"\s*[,.]?\s*(?:that's|that is|it's|is)\s+(?:{_PERSONAL})\s+({_CUE_NOUNS['ssn']})(?=\s*(?:[.,!?;]|$))",
    re.I | re.M)
_STRONG_SSN_BEFORE = re.compile(rf"\b(?:{_PERSONAL})\s+({_CUE_NOUNS['ssn']})\s*(?:is\b|['’]s\b|:)", re.I)
# A benefit amount is not a social: "My social security is 1,840 a month." Seven digits or more
# with thousands commas and no cents is an SSN the transcriber wrote that way ("219,099,999").
_MONEY = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+\.\d{1,2}")


def _money(s: str) -> bool:
    return bool(_MONEY.fullmatch(s)) and ("." in s or len(_digits(s)) <= 6)


_RATE_AFTER = re.compile(
    r"\s*(?:a month|per month|monthly|a year|per year|annually|a week|per week|dollars|bucks)\b", re.I)
_DIGIT_TOKEN = re.compile(r"\d[\d.,-]*\d|\d")          # a number glued with . , or -
_PIECE_TO_HERE = re.compile(r"\d+(?:[ ./-]\d+)*$")      # groups up to here, never across ", "
# Ten digits after "social" is more often a phone number than a garbled SSN, even with no separators
_PHONE_DIGITS = re.compile(r"1?[2-9]\d{2}[2-9]\d{6}")


def _strong_value(s: str, noun: str) -> bool:
    """`noun` is the cue noun: a bare "social security" (no "number") is also how a benefit is
    named, so its value needs seven digits."""
    least = 7 if noun.lower() == "social security" else 4
    return (least <= len(_digits(s)) <= 12 and not _money(s) and not _PHONE_SHAPED.fullmatch(s)
            and not _PHONE_DIGITS.fullmatch(s) and not _date(s) and not _ZIP4.fullmatch(s))


def _strong_ssns(scene: _Scene) -> list[tuple[int, int]]:
    """Values held by a strong SSN cue: the whitespace-delimited number next to it, or the run of
    groups ending or starting there, whichever is longer, when it stands alone and is not a
    phone number, a date, a ZIP+4 or labelled as something else."""
    norm, spans = scene.norm, []

    def take(a: int, b: int, noun: str, check_label: bool) -> None:
        if (_alone(norm, a, b) and _strong_value(norm[a:b], noun) and not _RATE_AFTER.match(norm, b)
                and not (check_label and scene.labelled("ssn", b))):
            spans.append((a, b))

    for m in _DIGIT_TOKEN.finditer(norm):
        cue = None if m.start() and not norm[m.start() - 1].isspace() else _STRONG_SSN_AFTER.match(norm, m.end())
        if not cue:
            continue
        run = _PIECE_TO_HERE.search(norm, max(0, m.end() - 80), m.end())
        a = min(m.start(), run.start()) if run else m.start()
        take(a, m.end(), cue.group(1), check_label=False)
    for m in _STRONG_SSN_BEFORE.finditer(norm):
        p, words = m.end(), 0
        while True:                                          # at most two filler words
            gap = re.match(r"[\s,]*([A-Za-z']+)?", norm[p:p + 40])
            if gap.group(1) and gap.group(1).lower() in _FILLER and words < 2:
                p += gap.end()
                words += 1
                continue
            p += len(gap.group(0)) - len(gap.group(1) or "")
            break
        token, run = _DIGIT_TOKEN.match(norm, p), _SEGMENT.match(norm, p)
        if token:
            take(p, max(token.end(), run.end() if run else 0), m.group(1), check_label=True)
    return spans


def _ssns(scene: _Scene) -> list[tuple[int, int]]:
    spans = []
    for a, b, s in scene.segments():
        spaced = SSN_SPACED_CUE_FREE and _SSN_SPACED.fullmatch(s)
        if (_SSN_DASHED.fullmatch(s) or spaced) and _ssn_ok(_digits(s)):
            spans.append((a, b))
    for m in _SSN_LAST_FOUR.finditer(scene.norm):
        if _alone(scene.norm, *m.span(1)):
            spans.append(m.span(1))
    return spans + _last_four_answers(scene) + scene.bound("ssn", _ssn_fits) + _strong_ssns(scene)


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


def _issued(d: str, exact_length: bool) -> bool:
    """d starts with a brand's prefix, and (when asked) has one of that brand's lengths."""
    return any(
        (not exact_length or len(d) in lengths) and any(lo <= int(d[:len(str(lo))]) <= hi for lo, hi in prefixes)
        for prefixes, lengths in _BRANDS.values()
    )


def _brand_prefix(d: str) -> bool:
    return _issued(d, exact_length=False)


def _branded(d: str) -> bool:
    return _issued(d, exact_length=True)


def _card_groups(s: str) -> bool:
    """Contiguous, or up to five groups of 3+ digits: how a card or an account is read out."""
    shape = _shape(s)
    return len(shape) <= 5 and min(shape) >= 3


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
    groups = _card_groups(s)
    if _not_a_value(s) or _all_years(s) or _PHONE_SHAPED.fullmatch(s):
        return False
    # "7731 2209 54": one short group at either end of a space- or dash-read number (not a list)
    edge = (", " not in s and 2 <= len(shape) <= 5 and 8 <= sum(shape) <= 17
            and ((shape[0] <= 2 and min(shape[1:]) >= 3) or (shape[-1] <= 2 and min(shape[:-1]) >= 3)))
    return (groups and 6 <= sum(shape) <= 17) or edge or bool(_SORT_CODE.fullmatch(s))


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


# --- email ---

# bounded parts: an unbounded local part backtracks quadratically on a long run of letters
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
# "my email is …", "email: …": the determiner form may stand alone, the bare form needs is or ":"
_EMAIL_CUE = rf"(?:\b(?:{_THE}) e-?mail(?: address)?((?: is|:|'s)?)|\be-?mail(?: address)?( is|:))"
_EMAIL_BEFORE = re.compile(rf"{_EMAIL_CUE} ((?:\S+ ){{0,4}}\S+)$", re.I)
_EMAIL_AFTER = re.compile(
    rf"\s*[,.]?\s*(?:that's|that is)\s+(?:{_POSSESSIVES})\s+e-?mail(?: address)?(?=\s*(?:[.,!?]|$))",
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


# The transcriber sometimes drops the " at ": "Faxmore.steranatexample.com", "Vandrel.corvainat
# example.org". With an identifying cue, a token ending in a dotted TLD (and up to two name-like
# tokens before it) is the address; without one, a website is never an email.
_DOMAIN_TOKEN = re.compile(rf"(?<!\S)[\w.+-]*\.(?:{_TLD})\b", re.I)
_EMAIL_CUE_BEFORE = re.compile(rf"{_EMAIL_CUE}\s*$", re.I)


def _glued_emails(norm: str) -> list[tuple[int, int]]:
    spans = []
    for m in _DOMAIN_TOKEN.finditer(norm):
        start = m.start()
        for _ in range(2):
            prev = re.search(r"(\S+)\s+$", norm[max(0, start - 80):start])
            word = prev.group(1) if prev else ""
            if not re.search(r"[A-Za-z][._][A-Za-z]", word) or word[-1] in ".,?!:":
                break
            start -= len(prev.group(0))
        if _EMAIL_AFTER.match(norm, m.end()) or _EMAIL_CUE_BEFORE.search(norm[max(0, start - 60):start]):
            spans.append((start, m.end()))
    return spans


def _emails(scene: "_Scene") -> list[tuple[int, int]]:
    norm = scene.norm
    spans = [m.span() for m in _EMAIL.finditer(norm)]
    for m in _AT_DOMAIN.finditer(norm):
        start = _email_local(norm, m.start(), m.end())
        if start is not None:
            spans.append((start, m.end()))
    return spans + _glued_emails(norm)


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
_PHONE = re.compile(rf"(?<!\d){_PHONE_FORM}(?!\d)")


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
# Overlapping hits take the class listed first: the more harmful leak wins.
_RANK = {c: k for k, c in enumerate(_DETECTORS)}


def _detect(norm: str, classes: set[str], text: str | None = None, idx: list[int] | None = None) -> list[Hit]:
    """Hits in the normalised text. `text` and `idx` (from `normalise`) let filler phrases be read
    in the original words; without them the normalised text stands in."""
    scene = _Scene(norm, norm if text is None else text, list(range(len(norm) + 1)) if idx is None else idx)
    hits = [Hit(a, b, cls) for cls, detect in _DETECTORS.items() if cls in classes for a, b in detect(scene)]
    # a value bound to a phone cue is a phone number, whether or not phones are masked
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
    """Non-overlapping hits in the original text, sorted. Hits that overlap are masked as their
    union, under the highest class among them (ssn > card > account > phone > address > email):
    "219 09 9999 Main Street" must not keep five SSN digits readable beside an address."""
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
    kept: list[Hit] = []
    for h in sorted(mapped, key=lambda h: h.start):
        if kept and h.start < kept[-1].end:
            top = min(kept[-1].cls, h.cls, key=_RANK.__getitem__)
            kept[-1] = Hit(kept[-1].start, max(kept[-1].end, h.end), top)
        else:
            kept.append(h)
    return kept
