"""Synthetic call scripts: fake values that pass our validators, spoken renderings, templates."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


CLASSES = ("ssn", "card", "account", "password", "email", "address", "phone")
SECRET = ("ssn", "card", "account", "password")          # mirrors pii.SECRET
CUES = ("same_sentence", "previous_turn", "after_value", "none")
TEST_BINS = ("4111", "4242", "5555", "5105", "3782", "3714")

# Words that exist nowhere else: each item's distinctive token is built from one, so a scorer can
# tell "this value is still in the text" without guessing.
ROOTS = (
    "zorvath", "quillon", "brindel", "faxmore", "telvane", "morrask", "kestrin", "druvell",
    "ostrane", "pelwick", "varnholt", "gristen", "lomwell", "yarrick", "ombrel", "fennick",
    "quorrel", "talloway", "brasken", "wintrel", "salvorn", "dremmick", "corvane", "tamsker",
    "belvorn", "rustane", "orvick", "plimsey", "gavrell", "nolwick", "sterran", "wexmoor",
    "jorrin", "kalvant", "mirrow", "peskin", "vandrel", "zelmont", "harrock", "trevane",
)
STREETS = ("Street", "Avenue", "Road", "Lane", "Drive", "Court")
_WORD = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
         "6": "six", "7": "seven", "8": "eight", "9": "nine"}
_TEEN = {"10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
         "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen"}
_TENS = {"2": "twenty", "3": "thirty", "4": "forty", "5": "fifty",
         "6": "sixty", "7": "seventy", "8": "eighty", "9": "ninety"}

# The words that cue each class in the templates below (tests check a `previous_turn` answer
# carries none of them).
CUE_WORDS = {
    "ssn": re.compile(r"social|ssn|security", re.I),
    "card": re.compile(r"card|visa|mastercard|amex|credit|debit", re.I),
    "account": re.compile(r"routing|account|iban|bank", re.I),
    "password": re.compile(r"password|passcode|pin|code|log ?in", re.I),
    "email": re.compile(r"e-?mail|address|reach|write|send|contact", re.I),
    "address": re.compile(r"address|live|ship|send", re.I),
    "phone": re.compile(r"phone|number|call|cell|reach", re.I),
}


@dataclass(frozen=True)
class Value:
    cls: str
    kind: str
    written: str
    spoken: str
    token: str      # distinctive: present in the text exactly when the value is
    style: str


def _zero(rng) -> str:
    return rng.choice(("oh", "zero"))


def _say_group(digits: str, style: str, zero: str) -> str:
    """One run of digits, spoken. `grouped` is pairs after an optional leading digit: every pair
    opens with a tens, teen or "oh" word, so no tens word is ever followed by a unit word inside
    a group (that would read back as one number: "twenty" + "one" is 21)."""
    def one(d):
        return zero if d == "0" else _WORD[d]

    if style == "digits":
        return " ".join(one(d) for d in digits)
    if style == "double":
        words, i = [], 0
        while i < len(digits):
            j = i
            while j < len(digits) and digits[j] == digits[i]:
                j += 1
            n = j - i
            while n:
                take = 3 if n == 3 or n >= 5 else 2 if n >= 2 else 1
                words.append({1: "", 2: "double ", 3: "triple "}[take] + one(digits[i]))
                n -= take
            i = j
        return " ".join(words)
    chunks = [digits[:1]] if len(digits) % 2 else []
    chunks += [digits[k:k + 2] for k in range(len(digits) % 2, len(digits), 2)]
    words = []
    for c in chunks:
        if len(c) == 1:
            words.append(one(c))
        elif c[0] == "0":
            words.append(f"{zero} {one(c[1])}")
        elif c[0] == "1":
            words.append(_TEEN[c])
        elif c[1] == "0":
            words.append(_TENS[c[0]])
        else:
            words.append(f"{_TENS[c[0]]} {_WORD[c[1]]}")
    return " ".join(words)


def _say_groups(groups: list[str], rng, *, style: str | None = None, seps=(" ", ", ", " dash ")) -> tuple[str, str]:
    """Digit groups spoken with one style and separator. A plain space between groups only for
    digit-by-digit styles: after a group ending in a tens word it would join the next group."""
    style = style or rng.choice(("digits", "grouped", "double"))
    zero = _zero(rng)
    if style == "grouped":
        seps = tuple(s for s in seps if s != " ") or (", ",)
    sep = rng.choice(seps)
    return sep.join(_say_group(g, style, zero) for g in groups), style


def _luhn_digit(body: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return str((10 - total % 10) % 10)


def _rand_digits(rng, n: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(n))


def gen_value(cls: str, rng, roots: list[str] | None = None) -> Value:
    """One fake value of `cls`, valid by our validators, with a spoken rendering. `roots` is the
    pool of invented words still unused in this script (taken from as needed)."""
    roots = roots if roots is not None else list(ROOTS)

    def root():
        return roots.pop(rng.randrange(len(roots)))

    if cls == "ssn":
        area = rng.choice([a for a in range(1, 900) if a != 666])
        d = f"{area:03d}{rng.randint(1, 99):02d}{rng.randint(1, 9999):04d}"
        spoken, style = _say_groups([d[:3], d[3:5], d[5:]], rng)
        return Value(cls, "ssn", f"{d[:3]}-{d[3:5]}-{d[5:]}", spoken, d, style)
    if cls == "card":
        bin_ = rng.choice(TEST_BINS)
        amex = bin_.startswith("37")
        n = 15 if amex else 16
        body = bin_ + _rand_digits(rng, n - 5)
        d = body + _luhn_digit(body)
        shape = (4, 6, 5) if amex else (4, 4, 4, 4)
        groups, at = [], 0
        for g in shape:
            groups.append(d[at:at + g])
            at += g
        spoken, style = _say_groups(groups, rng)
        return Value(cls, "amex" if amex else "card", " ".join(groups), spoken, d, style)
    if cls == "account":
        kind = rng.choice(("routing", "account_number", "iban"))
        if kind == "routing":
            n = [0, rng.randint(1, 9)] + [rng.randrange(10) for _ in range(6)]
            s = 3 * (n[0] + n[3] + n[6]) + 7 * (n[1] + n[4] + n[7]) + (n[2] + n[5])
            d = "".join(map(str, n)) + str(-s % 10)
            spoken, style = _say_groups([d[:3], d[3:6], d[6:]], rng, seps=(" ", ", "))
            return Value(cls, kind, d, spoken, d, style)
        if kind == "account_number":
            d = str(rng.randint(1, 9)) + _rand_digits(rng, rng.randint(7, 11))
            groups = [d[k:k + 4] for k in range(0, len(d), 4)]
            spoken, style = _say_groups(groups, rng, seps=(" ", ", "))
            return Value(cls, kind, d, spoken, d, style)
        bank = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(4))
        bban = bank + _rand_digits(rng, 14)
        numeric = "".join(str(int(c, 36)) for c in bban + "GB00")
        compact = f"GB{98 - int(numeric) % 97:02d}{bban}"
        groups = [compact[k:k + 4] for k in range(0, len(compact), 4)]
        zero = _zero(rng)
        say = lambda g: " ".join(c if c.isalpha() else (zero if c == "0" else _WORD[c]) for c in g)
        spoken = rng.choice((" ", ", ")).join(say(g) for g in groups)
        return Value(cls, kind, " ".join(groups), spoken, compact, "spelled")
    if cls == "password":
        word, nn = root(), str(rng.randint(11, 99))
        style = rng.choice(("word", "word_digits", "spelled"))
        num = _say_group(nn, "grouped" if style == "word" else "digits", "oh")
        said = " ".join(word) if style == "spelled" else word
        return Value(cls, "password", f"{word}{nn}", f"{said} {num}", f"{word}{nn}", style)
    if cls == "email":
        first = root()
        style = rng.choice(("dotted", "digits", "plain"))
        domain = rng.choice(("com", "org"))
        if style == "dotted":
            second = root()
            local, said = f"{first}.{second}", f"{first} dot {second}"
        elif style == "digits":
            nn = str(rng.randint(11, 99))
            local = f"{first}{nn}"
            said = f"{first} {_say_group(nn, rng.choice(('digits', 'grouped')), 'oh')}"
        else:
            local, said = first, first
        return Value(cls, "email", f"{local}@example.{domain}", f"{said} at example dot {domain}",
                     first, style)
    if cls == "address":
        name = root().capitalize()
        num = str(rng.randint(10, 9999))
        style = rng.choice(("digits", "grouped"))
        street = f"{name} {rng.choice(STREETS)}"
        written, spoken = f"{num} {street}", f"{_say_group(num, style, 'oh')} {street}"
        extra = rng.choice(("", "apt", "zip"))
        if extra == "apt":
            n = str(rng.randint(2, 9))
            written, spoken = f"{written}, apartment {n}", f"{spoken}, apartment {_WORD[n]}"
        elif extra == "zip":
            z = str(rng.randint(10000, 99999))
            written, spoken = f"{written}, {z}", f"{spoken}, {_say_group(z, 'digits', 'oh')}"
        return Value(cls, "address", written, spoken, name, style)
    if cls == "phone":
        d = f"{rng.randint(2, 9)}{rng.randint(0, 9)}{rng.randint(1, 9)}55501{rng.randint(0, 99):02d}"
        spoken, style = _say_groups([d[:3], d[3:6], d[6:]], rng)
        return Value(cls, "phone", f"({d[:3]}) {d[3:6]}-{d[6:]}", spoken, d, style)
    raise ValueError(f"unknown class {cls!r}")


_LABEL = {"routing": "routing number", "account_number": "account number", "iban": "IBAN"}

# cls -> cue -> list of (question by the other speaker or None, the line with {v})
TEMPLATES = {
    "ssn": {
        "same_sentence": [(None, "Sure, my social is {v}."), (None, "My social security number is {v}.")],
        "previous_turn": [("Can I get your social security number?", "Yeah, it's {v}.")],
        "after_value": [(None, "{v}, that's my social."), (None, "{v} is my social security number.")],
        "none": [(None, "Take this down, {v}."), (None, "Okay, so it's {v}.")],
    },
    "card": {
        "same_sentence": [(None, "The card number is {v}."), (None, "Put it on my visa, {v}.")],
        "previous_turn": [("Which card do you want to use?", "Okay, {v}.")],
        "after_value": [(None, "{v}, that's the visa."), (None, "{v} is the card.")],
        "none": [(None, "Here you go, {v}."), (None, "Ready? {v}.")],
    },
    "account": {
        "same_sentence": [(None, "My {label} is {v}."), (None, "Sure, {label} {v}.")],
        "previous_turn": [("What's the {label}?", "It's {v}.")],
        "after_value": [(None, "{v}, that's the {label}.")],
        "none": [(None, "Here it is, {v}.")],
    },
    "password": {
        "same_sentence": [(None, "The password is {v}."), (None, "My login is {v}."),
                          (None, "The wifi passcode is {v}.")],
        "previous_turn": [("What's the password for the wifi?", "It's {v}.")],
        "after_value": [(None, "{v} is my password."), (None, "{v}, that's the passcode.")],
        "none": [(None, "Try {v}."), (None, "Just type {v}.")],
    },
    "email": {
        "same_sentence": [(None, "My email is {v}."), (None, "You can email me at {v}.")],
        "previous_turn": [("What's your email?", "It's {v}.")],
        "after_value": [(None, "{v}, that's my email.")],
        "none": [(None, "Use {v}."), (None, "It's {v}, all lowercase.")],
    },
    "address": {
        "same_sentence": [(None, "My address is {v}."), (None, "I live at {v}.")],
        "previous_turn": [("Where should we ship it?", "Sure, {v}.")],
        "after_value": [(None, "{v}, that's my address."), (None, "{v} is where I live.")],
        "none": [(None, "We're at {v}."), (None, "Meet me at {v}.")],
    },
    "phone": {
        "same_sentence": [(None, "My phone number is {v}."), (None, "Call me on my cell, {v}.")],
        "previous_turn": [("What's a good number for you?", "Sure, {v}.")],
        "after_value": [(None, "{v}, that's my cell.")],
        "none": [(None, "Okay, {v}.")],
    },
}


def _order(rng):
    return _say_group(_rand_digits(rng, rng.randint(5, 7)), "digits", rng.choice(("oh", "zero")))


def _triple(rng):
    groups = [_rand_digits(rng, 3), _rand_digits(rng, 2), _rand_digits(rng, 4)]
    return ", ".join(_say_group(g, rng.choice(("digits", "grouped")), "oh") for g in groups)


# kind -> (line template with {p}, phrase maker)
DISTRACTORS = {
    "time": ("Can we do {p}?", lambda r: r.choice(("two thirty", "half past four", "ten fifteen",
                                                  "quarter to six", "nine o'clock"))),
    "price": ("It was only {p}.", lambda r: r.choice(("four ninety nine", "twelve dollars",
                                                     "three fifty", "nineteen ninety five"))),
    "year": ("That was back in {p}.", lambda r: r.choice(("twenty nineteen", "two thousand eight",
                                                         "twenty twenty three"))),
    "order": ("The order number is {p}.", _order),
    "nine_one_one": ("I almost called {p}.", lambda r: "nine one one"),
    "page": ("It's on {p}.", lambda r: r.choice(("page two hundred and twelve", "page forty five",
                                                 "page three"))),
    "count": ("Okay, {p}, go.", lambda r: r.choice(("one, two, three", "one two three four five"))),
    "website": ("I ordered it at {p}.", lambda r: r.choice(("amazon dot com", "target dot com"))),
    "website_send": ("Can you send me the link? I bought it at {p}.",
                     lambda r: r.choice(("amazon dot com", "target dot com"))),
    "triple": ("The counts were {p}.", _triple),
    "votes": ("The vote was {p}.", lambda r: r.choice(("twelve to seven, with three abstaining",
                                                      "forty one to thirty"))),
}

SMALL_TALK = (
    "Hey, how's it going?", "Not bad, thanks. You?", "Pretty good. Busy week.",
    "Did you catch the game last night?", "No, I missed it.", "How was the trip?",
    "It was great, really relaxing.", "Okay, let's get started.", "Sounds good.",
    "The weather has been lovely.", "I know, finally some sun.", "Thanks so much.",
    "Anything else?", "No, I think that's everything.", "Great, talk soon.",
    "Sorry, can you repeat that?", "Perfect.",
)
BACKCHANNEL = ("Okay.", "Mm-hm.", "Right.", "Got it.")


@dataclass
class Item:
    cls: str
    kind: str
    written: str
    spoken: str
    token: str
    style: str
    cue: str
    speaker: str
    start: int = 0
    end: int = 0


@dataclass
class Distractor:
    kind: str
    text: str
    start: int = 0
    end: int = 0


@dataclass
class Script:
    sid: str
    lines: list[tuple[str, str]]
    items: list[Item]
    distractors: list[Distractor]
    transcript: str = ""


def _other(speaker: str) -> str:
    return "Them" if speaker == "Me" else "Me"


def _n_items(k: int, rng) -> int:
    return 0 if k % 6 == 5 else rng.randint(1, 3)


def _script(k: int, seed: int) -> Script:
    rng = random.Random(f"{seed}:{k}")
    roots = list(ROOTS)
    n_items = _n_items(k, rng)
    # Items are numbered across all scripts and walk the class x cue grid in order, so every
    # class meets every cue placement (28 cells) instead of a few correlated ones.
    first = sum(_n_items(q, random.Random(f"{seed}:{q}")) for q in range(k))
    blocks: list[list[tuple[str, str, object]]] = []    # (speaker, line, (start, end, ref) | None)
    items: list[Item] = []
    for j in range(n_items):
        g = first + j
        cls = CLASSES[g % len(CLASSES)]
        cue = CUES[(g // len(CLASSES)) % len(CUES)]
        v = gen_value(cls, rng, roots)
        speaker = rng.choice(("Me", "Them"))
        question, line = rng.choice(TEMPLATES[cls][cue])
        label = _LABEL.get(v.kind, "")
        item = Item(v.cls, v.kind, v.written, v.spoken, v.token, v.style, cue, speaker)
        items.append(item)
        before, after = line.format(v="\0", label=label).split("\0")
        block = []
        if question:
            block.append((_other(speaker), question.format(label=label), None))
        block.append((speaker, before + v.spoken + after, (len(before), len(before) + len(v.spoken), item)))
        blocks.append(block)
    distractors = []
    for kind in rng.sample(sorted(DISTRACTORS), rng.randint(2, 4)):
        template, make = DISTRACTORS[kind]
        phrase = make(rng)
        d = Distractor(kind, phrase)
        distractors.append(d)
        before, after = template.split("{p}")
        blocks.append([(rng.choice(("Me", "Them")), before + phrase + after,
                        (len(before), len(before) + len(phrase), d))])
    rng.shuffle(blocks)
    talk = list(SMALL_TALK)
    rng.shuffle(talk)
    opening = [[("Me", talk.pop(), None)], [("Them", talk.pop(), None)]]
    middle = []
    for b in blocks:
        middle.append(b)
        if rng.random() < 0.5:
            middle.append([(rng.choice(("Me", "Them")), talk.pop(), None)])
    closing = [[(rng.choice(("Me", "Them")), "Great, talk soon.", None)]]
    flat = [ln for b in opening + middle + closing for ln in b]
    # strictly alternating speakers, so every line is its own turn
    lines: list[tuple[str, str, object]] = []
    for ln in flat:
        if lines and lines[-1][0] == ln[0]:
            lines.append((_other(ln[0]), rng.choice(BACKCHANNEL), None))
        lines.append(ln)
    parts, pos = [], 0
    for speaker, text, mark in lines:
        head = f"**{speaker}:** "
        if mark:
            a, b, ref = mark
            ref.start, ref.end = pos + len(head) + a, pos + len(head) + b
        parts.append(head + text)
        pos += len(head) + len(text) + 2
    items.sort(key=lambda i: i.start)
    distractors.sort(key=lambda d: d.start)
    return Script(f"s{k + 1:02d}", [(s, t) for s, t, _ in lines], items, distractors, "\n\n".join(parts))


def generate(n: int = 30, seed: int = 7) -> list[Script]:
    """`n` call scripts, deterministic in `seed`; script k does not depend on `n`."""
    return [_script(k, seed) for k in range(n)]
