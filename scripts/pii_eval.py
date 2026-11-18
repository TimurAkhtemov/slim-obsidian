"""Measure PII masking on synthetic calls and public synthetic sets, in a throwaway vault.

    uv run python scripts/pii_eval.py text   --out DIR [--no-model] [--n 30] [--seed 7]
    uv run python scripts/pii_eval.py calls  --out DIR [--full 5] [--only s03] [--redo]
    uv run python scripts/pii_eval.py public --out DIR [--rows 2000]

Nothing here touches the owner's data. Every value is generated (valid-looking, never real) or
comes from a public synthetic dataset whose licence allows evaluation. `sandbox()` points
SLIM_VAULT, SLIM_DATA_DIR and SLIM_PROJECTS into DIR BEFORE anything from `slim` is imported
(config resolves its paths at import) and refuses to go on unless they resolved there.

Importing this module has no side effects: the generator and scorers are pure, and each mode's
heavy steps (say, Parakeet, the model, ffmpeg) arrive as callables so tests can fake them.
`text` and `calls` with the model are model jobs: one at a time on this Mac (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CLASSES = ("ssn", "card", "account", "password", "email", "address", "phone")
SECRET = ("ssn", "card", "account", "password")          # mirrors pii.SECRET
CUES = ("same_sentence", "previous_turn", "after_value", "none")
PLACEHOLDERS = re.compile(r"\[(SSN|CARD|ACCOUNT|PASSWORD|EMAIL|ADDRESS|PHONE)\]")
TEST_BINS = ("4111", "4242", "5555", "5105", "3782", "3714")
VOICE = {"Me": "Samantha", "Them": "Daniel"}
RECALL_GATE = 0.95
GARBLED_GATE = 0.8          # recall over values Parakeet did not write verbatim
SILENT_DBFS = -60.0           # a muted span must be quieter than this
SPEECH_DBFS = -45.0           # a line nobody muted must be louder than this
LEAD_SECONDS = 0.5

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


# --- values -------------------------------------------------------------------------------

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


# --- call scripts ---------------------------------------------------------------------------

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


# --- matching -------------------------------------------------------------------------------

_SAID_SEPARATOR = re.compile(r"\b(?:dash|hyphen)\b", re.I)   # "two one nine dash oh nine"


def _squash_map(text: str) -> tuple[str, list[int], list[int]]:
    """Lowercase letters and digits of the normalised text, each with its original span."""
    from slim.capture.pii import normalise

    norm, idx = normalise(text)
    said = {k for m in _SAID_SEPARATOR.finditer(norm) for k in range(*m.span())}
    chars, starts, ends = [], [], []
    for j, ch in enumerate(norm):
        c = ch.lower()
        if c.isascii() and c.isalnum() and j not in said:
            chars.append(c)
            starts.append(idx[j])
            ends.append(idx[j + 1])
    return "".join(chars), starts, ends


def squash(text: str) -> str:
    return _squash_map(text)[0]


def present(token: str, text: str) -> bool:
    """Is the value whose distinctive token this is still in `text`, in any spelling: digits or
    number words, spaced or spelled letters, any punctuation."""
    t = squash(token)
    return bool(t) and t in squash(text)


def locate(token: str, text: str) -> list[tuple[int, int]]:
    """Where `token` sits in `text` (original offsets), every occurrence."""
    t = squash(token)
    s, starts, ends = _squash_map(text)
    out, at = [], s.find(t) if t else -1
    while at >= 0:
        b = at + len(t)
        out.append((min(starts[at:b]), max(ends[at:b] + [starts[b - 1] + 1])))
        at = s.find(t, at + 1)
    return out


def _replaced(orig: str, masked: str) -> list[tuple[int, int, str]]:
    """Spans of `orig` that masking replaced, with the placeholder's class. Masking leaves every
    other byte alone, so `masked` is `orig` with some spans swapped for placeholders: align the
    literal pieces between placeholders."""
    pieces = PLACEHOLDERS.split(masked)
    lits, names = pieces[0::2], [n.lower() for n in pieces[1::2]]
    if not names:
        return []
    pattern = re.escape(lits[0]) + "".join("(.+?)" + re.escape(lit) for lit in lits[1:])
    m = re.fullmatch(pattern, orig, re.S)
    if m:
        return [(*m.span(g), names[g - 1]) for g in range(1, m.lastindex + 1)]
    out = []                       # not a pure substitution: fall back to a diff
    for op, a, b, _, _ in difflib.SequenceMatcher(None, orig, masked, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            out.append((a, b, "?"))
    return out


def replaced(orig: str, masked: str) -> list[tuple[int, int]]:
    return [(a, b) for a, b, _ in _replaced(orig, masked)]


def _line_at(text: str, pos: int) -> str:
    a = text.rfind("\n\n", 0, pos)
    b = text.find("\n\n", pos)
    return text[a + 2 if a >= 0 else 0:b if b >= 0 else len(text)]


def _covered(a: int, b: int, text: str, reps) -> bool:
    """Every letter and digit of text[a:b] lies under some replaced span."""
    return all(any(x <= k < y for x, y in reps) for k in range(a, b) if text[k].isalnum())


# --- text mode --------------------------------------------------------------------------------

def _empty(keys):
    return {k: {"n": 0, "masked": 0} for k in keys}


def _ratio(d):
    return round(d["masked"] / d["n"], 3) if d["n"] else None


def _gate_classes(classes, model: bool) -> list[str]:
    return [c for c in SECRET if c in classes and (model or c != "password")]


def _leftover(a: int, b: int, text: str, own) -> list[int]:
    """Positions of letters and digits in text[a:b] outside every `own` span."""
    return [k for k in range(a, b) if text[k].isalnum() and not any(x <= k < y for x, y in own)]


def _segments_outside(text: str, spans) -> list[str]:
    """`text` with `spans` cut out, squashed piece by piece (no run bridges a cut)."""
    out, pos = [], 0
    for a, b in sorted(spans):
        out.append(squash(text[pos:a]))
        pos = max(pos, b)
    out.append(squash(text[pos:]))
    return out


def _runs(s: str, n: int) -> set[str]:
    return {s[k:k + n] for k in range(len(s) - n + 1)}


def _fresh_runs(token_part: str, n: int, hay: str, elsewhere: list[str]) -> list[str]:
    """Runs of `n` characters of the token found in `hay` that cannot be explained by text found
    elsewhere in the original (a distractor "2019" holding "019")."""
    return sorted(r for r in _runs(token_part, n) if r in hay and not any(r in e for e in elsewhere))


def remnant(item: "Item", text: str, elsewhere: list[str]) -> str | None:
    """A readable trace of `item` in `text` in some other rendering: >= 4 consecutive digits of
    its digit string, >= 5 consecutive letters of its distinctive word, or a run of words that
    fuzzy-matches (>= 0.6) its spoken or written form. None when nothing is readable."""
    hay_text = PLACEHOLDERS.sub(" | ", text)
    hay = squash(hay_text)
    tok = squash(item.token)
    digits, letters = re.sub(r"\D", "", tok), re.sub(r"[^a-z]", "", tok)
    if len(digits) >= 4:
        runs = _fresh_runs(digits, 4, hay, elsewhere)
        if runs:
            return f"digits {runs[0]}"
    if len(letters) >= 5:
        runs = _fresh_runs(letters, 5, hay, elsewhere)
        if runs:
            return f"letters {runs[0]}"
    forms = {f for f in (squash(item.spoken), squash(item.written)) if f}
    for turn in hay_text.split("\n\n"):
        words = [w for w in re.split(r"\s+", turn) if w and "**" not in w and w != "|"]
        for size in range(1, min(8, len(words)) + 1):
            for k in range(len(words) - size + 1):
                window = squash(" ".join(words[k:k + size]))
                for f in forms:
                    if len(window) < len(f) / 2 or any(window in e for e in elsewhere):
                        continue
                    sm = difflib.SequenceMatcher(None, window, f)
                    if sm.quick_ratio() >= 0.6 and sm.ratio() >= 0.6:
                        return f"fuzzy {' '.join(words[k:k + size])!r}"
    return None


def score_text(scripts: list[Script], masked: dict[str, str], classes, *, model: bool) -> dict:
    """Recall per class and cue placement, false positives, distractors damaged, the gate.

    An item is masked only when every letter and digit of it lies under a placeholder; a mask
    reaching past its item (or masking an item outside `classes`) is a false positive, and
    damages any distractor its extra letters or digits touch."""
    classes = tuple(classes)
    by_class, by_cue = _empty(classes), _empty(CUES)
    misses, fps, placeholders, expected = [], [], Counter(), Counter()
    damaged = 0
    for s in scripts:
        t, m = s.transcript, masked[s.sid]
        reps3 = _replaced(t, m)
        reps = [(a, b) for a, b, _ in reps3]
        placeholders.update(p.lower() for p in PLACEHOLDERS.findall(m))
        scored = [i for i in s.items if i.cls in classes]
        for i in scored:
            expected[i.cls] += 1
            readable = present(i.token, m)
            gone = not readable and _covered(i.start, i.end, t, reps)
            for d in (by_class[i.cls], by_cue[i.cue]):
                d["n"] += 1
                d["masked"] += gone
            if not gone:
                misses.append({"script": s.sid, "class": i.cls, "kind": i.kind, "style": i.style,
                               "cue": i.cue, "result": "missed" if readable else "partial",
                               "context": _line_at(t, i.start), "value": i.spoken})
        own_all = [(i.start, i.end) for i in scored]
        for a, b, cls in reps3:
            own = [(x, y) for x, y in own_all if a < y and x < b]
            left = _leftover(a, b, t, own)
            if not left:
                continue
            hit = [d for d in s.distractors if any(d.start <= k < d.end for k in left)]
            damaged += len(hit)
            fps.append({"script": s.sid, "text": t[a:b], "context": _line_at(t, a),
                        "kind": "over-wide mask" if own else "false positive",
                        "leftover": "".join(t[k] for k in left),
                        "distractor": hit[0].kind if hit else None, "class": cls})
    gate_cls = _gate_classes(classes, model)
    n = sum(by_class[c]["n"] for c in gate_cls)
    secret_recall = sum(by_class[c]["masked"] for c in gate_cls) / n if n else None
    total = sum(d["n"] for d in by_class.values())
    no_evidence = [] if n else [f"no secret-class item ({','.join(gate_cls)}) in these scripts"]
    return {
        "mode": "text", "model": model, "classes": list(classes), "scripts": len(scripts),
        "items": total,
        "recall": round(sum(d["masked"] for d in by_class.values()) / total, 3) if total else None,
        "by_class": {c: {**d, "recall": _ratio(d)} for c, d in by_class.items()},
        "by_cue": {c: {**d, "recall": _ratio(d)} for c, d in by_cue.items()},
        "distractors": sum(len(s.distractors) for s in scripts),
        "distractors_damaged": damaged,
        "placeholders": {c: {"found": placeholders[c], "expected": expected[c]} for c in classes},
        "misses": misses, "false_positives": fps,
        "gate": {"ok": not no_evidence and damaged == 0 and not fps
                 and secret_recall >= RECALL_GATE,
                 "secret_classes": gate_cls,
                 "secret_recall": None if secret_recall is None else round(secret_recall, 3),
                 "distractors_damaged": damaged, "false_positives": len(fps),
                 "no_evidence": no_evidence},
    }


def _table(title: str, rows: dict) -> list[str]:
    out = [f"{title:<16}{'n':>5}{'masked':>8}{'recall':>8}"]
    for k, d in rows.items():
        r = "-" if d.get("recall") is None else f"{d['recall']:.2f}"
        out.append(f"  {k:<14}{d['n']:>5}{d['masked']:>8}{r:>8}")
    return out


def _gate_line(g: dict, extra: str = "") -> str:
    rec = "-" if g["secret_recall"] is None else f"{g['secret_recall']:.3f}"
    why = f"; NO EVIDENCE: {'; '.join(g['no_evidence'])}" if g["no_evidence"] else ""
    return (f"gate: {'ok' if g['ok'] else 'FAIL'} — secret recall {rec} over "
            f"{','.join(g['secret_classes'])} (needs >= {RECALL_GATE}), distractors damaged "
            f"{g['distractors_damaged']}, false positives {g['false_positives']} (need 0){extra}{why}")


def print_text_report(r: dict) -> None:
    print(f"\n== text: {r['scripts']} scripts, {r['items']} items, model={'yes' if r['model'] else 'no'}, "
          f"classes={','.join(r['classes'])}")
    print("An item counts as masked only when every letter and digit of it is under a placeholder.")
    print("\n".join(_table("class", r["by_class"])))
    print("\n".join(_table("cue placement", r["by_cue"])))
    ph = ", ".join(f"[{c.upper()}] {d['found']}/{d['expected']}" for c, d in r["placeholders"].items())
    print(f"placeholders found/expected: {ph}")
    print(f"distractors damaged: {r['distractors_damaged']} of {r['distractors']}")
    for m in r["misses"]:
        print(f"  MISS {m['script']} {m['class']}/{m['kind']} cue={m['cue']} style={m['style']} "
              f"({m['result']}): {m['context']}")
    for f in r["false_positives"]:
        where = f"distractor {f['distractor']}" if f["distractor"] else "other text"
        print(f"  FALSE POSITIVE {f['script']} [{f['class'].upper()}] {f['kind']} over {where}: "
              f"{f['text']!r} (unexplained: {f['leftover']!r}) in {f['context']}")
    print(_gate_line(r["gate"]))


def run_text(scripts, mask, classes, *, model: bool, log=print) -> dict:
    masked, partial = {}, []
    for s in scripts:
        t0 = time.perf_counter()
        m = mask(s.transcript)
        masked[s.sid] = m.text
        if m.partial:
            partial.append(s.sid)
        if model:
            log(f"  {s.sid}  {time.perf_counter() - t0:.1f}s")
    report = score_text(scripts, masked, classes, model=model)
    report["password_pass_failed"] = partial
    return report


# --- calls mode -------------------------------------------------------------------------------

def _bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        if Path(prefix, name).exists():
            return str(Path(prefix, name))
    raise SystemExit(f"{name} not found")


def _seconds(path: Path) -> float:
    out = subprocess.run([_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def audio_key(script: Script, seed: int) -> str:
    """What the synthesized audio depends on: the lines, the voices, the seed (gaps)."""
    blob = json.dumps([script.lines, VOICE, seed, LEAD_SECONDS, "synth-1"])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def synthesize(script: Script, dest: Path, seed: int = 7) -> dict:
    """Speak the script into a stereo 48 kHz opus `.webm`: Me (Samantha) LEFT, Them (Daniel)
    RIGHT, one line after another with a 0.3-0.6 s gap, never overlapping. Returns each line's
    time span. Line audio is cached beside `dest`."""
    parts = dest.parent / f"{script.sid}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{seed}:{script.sid}:gaps")
    args, graph, lines, t = [], [], [], LEAD_SECONDS
    for k, (speaker, text) in enumerate(script.lines):
        txt, aiff = parts / f"{k:02d}.txt", parts / f"{k:02d}.aiff"
        if not aiff.exists() or not txt.exists() or txt.read_text() != text:
            txt.write_text(text)
            subprocess.run(["say", "-v", VOICE[speaker], "-o", str(aiff), "-f", str(txt)], check=True)
        dur = _seconds(aiff)
        gap = round(rng.uniform(0.3, 0.6), 3)
        lines.append({"speaker": speaker, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + gap
        args += ["-i", str(aiff)]
        pan = "c0=c0|c1=0*c0" if speaker == "Me" else "c0=0*c0|c1=c0"
        delay = f",adelay={int(LEAD_SECONDS * 1000)}:all=1" if k == 0 else ""
        graph.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=mono,"
                     f"pan=stereo|{pan}{delay},apad=pad_dur={gap}[a{k}]")
    graph.append("".join(f"[a{k}]" for k in range(len(lines))) + f"concat=n={len(lines)}:v=0:a=1[out]")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_bin("ffmpeg"), "-y", "-nostdin", "-loglevel", "error", *args,
                    "-filter_complex", ";".join(graph), "-map", "[out]",
                    "-c:a", "libopus", "-b:a", "96k", "-ar", "48000", str(dest)], check=True)
    return {"lines": lines, "seconds": round(t, 3)}


def rms_dbfs(path: Path, a: float, b: float) -> float:
    """Loudest channel's RMS over [a, b] seconds, in dBFS. Raises when no audio could be read
    there: an unreadable span must fail a silence check, not pass it."""
    import numpy as np

    out = subprocess.run([_bin("ffmpeg"), "-nostdin", "-loglevel", "error", "-ss", f"{a:.3f}",
                          "-t", f"{max(b - a, 0.02):.3f}", "-i", str(path), "-f", "f32le", "-ac", "2",
                          "-ar", "48000", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(out[:len(out) - len(out) % 8], dtype="<f4").reshape(-1, 2)
    if not len(x):
        raise ValueError(f"no samples in {a:.2f}-{b:.2f} s")
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean(axis=0)).max())
    return 20 * float(np.log10(max(rms, 1e-10)))


def _form(snippet: str) -> str:
    has_digit = any(c.isdigit() for c in snippet)
    has_word = bool(re.search(r"\b(zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|"
                              r"eleven|twelve|teen|twenty|thirty|forty|fifty|sixty|seventy|"
                              r"eighty|ninety|double|triple)\b", snippet, re.I))
    return "mixed" if has_digit and has_word else "digits" if has_digit else "words"


def _heard(item: Item, line: str, raw: str) -> tuple[str, str]:
    """How Parakeet wrote this item: the located token's surroundings, else the raw sentence
    closest to the script line."""
    found = locate(item.token, raw)
    if found:
        a, b = found[0]
        a = max(raw.rfind(" ", 0, max(a - 1, 0)), raw.rfind("\n", 0, a)) + 1
        b2 = re.search(r"[.?!](\s|$)|\n", raw[b:])
        return raw[a:b + (b2.start() if b2 else len(raw) - b)].strip(), "transcribed"
    sentences = [x.strip() for x in re.split(r"(?<=[.?!])\s+|\n\n", raw) if x.strip()]
    best = max(sentences, key=lambda x: difflib.SequenceMatcher(None, x.lower(), line.lower()).ratio(),
               default="")
    return best, "not transcribed"


def rendered_span(item: Item, raw: str, span: tuple[int, int]) -> tuple[int, int]:
    """The item's value as Parakeet wrote it: the located token grown over neighbouring words
    while the whole grown region still reads as part of the value ("43 at example.org" after an
    email's name; "sixty eight sixty eight" before a street). Grows up to three words at a step,
    since a spoken number only reads back whole. Never over a speaker label or a turn break."""
    forms = [f for f in (squash(item.spoken), squash(item.written)) if f]
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", raw)]
    a, b = span
    inside = [k for k, (x, y) in enumerate(words) if x < b and a < y]
    if not inside:
        return span

    def joinable(k1, k2):            # words k1..k2 inclusive, with what lies between them
        piece = raw[words[k1][0]:words[k2][1]]
        return "**" not in piece and "\n\n" not in piece

    def reads(l, h):
        sq = squash(raw[words[l][0]:words[h][1]])
        return any(sq in f for f in forms)

    lo, hi = inside[0], inside[-1]
    grown = True
    while grown:
        grown = False
        for k in (1, 2, 3):
            if hi + k < len(words) and joinable(hi, hi + k) and reads(lo, hi + k):
                hi, grown = hi + k, True
                break
        for k in (1, 2, 3):
            if lo - k >= 0 and joinable(lo - k, lo) and reads(lo - k, hi):
                lo, grown = lo - k, True
                break
    return min(a, words[lo][0]), max(b, words[hi][1])


def score_call(script: Script, raw: str, masked: str, heard_after: str, classes, *,
               spans, reasons, manifest, measure, silenced: Path, model: bool = True) -> dict:
    """One call's numbers: items (transcribed, masked, remnant), false positives, audio checks.

    A transcribed item is masked only when every letter and digit of every place Parakeet wrote
    it is under a placeholder. An item Parakeet did not write verbatim is checked for a readable
    remnant in the masked text instead of being dropped. Audio: no run of 3 characters of a
    secret's token may come back from the muted copy unless the original says it elsewhere."""
    classes = tuple(classes)
    items, fps, garbled, audio_fail, renderings = [], [], [], [], []
    reps3 = _replaced(raw, masked)
    reps = [(a, b) for a, b, _ in reps3]
    located = {id(i): locate(i.token, raw) for i in script.items}
    rendered = {id(i): [rendered_span(i, raw, sp) for sp in located[id(i)]] for i in script.items}
    for i in script.items:
        line = next((t for _, t in script.lines if i.spoken in t), i.spoken)
        heard, status = _heard(i, line, raw)
        transcribed = bool(located[id(i)])
        elsewhere = _segments_outside(script.transcript, [(i.start, i.end)])
        if transcribed:
            readable = present(i.token, masked)
            covered = all(_covered(a, b, raw, reps) for a, b in located[id(i)])
            gone, rem = not readable and covered, None
            result = "masked" if gone else "missed" if readable else "partial"
        else:
            rem = remnant(i, masked, elsewhere)
            gone = rem is None
            result = "no remnant" if gone else "REMNANT"
        items.append({"class": i.cls, "kind": i.kind, "style": i.style, "cue": i.cue,
                      "transcribed": transcribed, "masked": gone, "result": result,
                      "remnant": rem, "spoken": i.spoken, "heard": heard,
                      "in_classes": i.cls in classes})
        renderings.append({"script": script.sid, "class": i.cls, "kind": i.kind, "style": i.style,
                           "spoken": i.spoken, "heard": heard, "status": status,
                           "form": _form(heard) if transcribed else "-"})
        if i.cls in _gate_classes(classes, model):
            raw_elsewhere = _segments_outside(raw, rendered[id(i)])
            leaked = _fresh_runs(squash(i.token), 3, squash(heard_after), raw_elsewhere)
            if leaked:
                audio_fail.append({"script": script.sid, "check": "retranscribed", "class": i.cls,
                                   "runs": leaked[:5], "heard": _heard(i, line, heard_after)[0]})
    own_all = [sp for i in script.items if i.cls in classes for sp in rendered[id(i)]]
    for a, b, cls in reps3:
        own = [(x, y) for x, y in own_all if a < y and x < b]
        left = _leftover(a, b, raw, own)
        if not left:
            continue
        entry = {"script": script.sid, "class": cls, "text": raw[a:b], "context": _line_at(raw, a),
                 "leftover": "".join(raw[k] for k in left),
                 "kind": "over-wide mask" if own else "false positive"}
        if not own:
            frag = squash(raw[a:b])
            near = max((difflib.SequenceMatcher(None, frag, squash(v)).ratio()
                        for i in script.items if i.cls in classes for v in (i.spoken, i.written)),
                       default=0.0)
            if near >= 0.6:
                garbled.append(entry)
                continue
        fps.append(entry)

    def level(a, b):
        try:
            return measure(silenced, a, b)
        except Exception as exc:                       # noqa: BLE001 - unreadable is a failure
            audio_fail.append({"script": script.sid, "check": "unmeasurable", "span": [a, b],
                               "error": str(exc)})
            return None

    for a, b in spans:
        got = level(a, b)
        if got is not None and got >= SILENT_DBFS:
            audio_fail.append({"script": script.sid, "check": "rms", "span": [a, b],
                               "dbfs": round(got, 1)})
    pad = 0.15
    for ln in manifest.get("lines", []):
        if any(ln["start"] < b + pad and a - pad < ln["end"] for a, b in spans):
            continue
        got = level(ln["start"], ln["end"])
        if got is not None and got < SPEECH_DBFS:
            audio_fail.append({"script": script.sid, "check": "speech", "line": ln,
                               "dbfs": round(got, 1)})
    if "audio" in reasons:
        audio_fail.append({"script": script.sid, "check": "unplaced",
                           "detail": "a secret could not be placed in token times"})
    return {"script": script.sid, "items": items, "false_positives": fps, "garbled_masked": garbled,
            "audio_failures": audio_fail, "renderings": renderings, "spans": [list(x) for x in spans],
            "reasons": sorted(reasons)}


def check_full(script: Script, note_text: str, result: dict, fragments: list[str],
               transcribed: set[str], classes, *, archived_levels=()) -> dict:
    """The filed note, its frontmatter, summary, title, index rows and archived audio, after
    `handle_record`. `archived_levels` is (span, dBFS or None) inside the spans the pipeline
    muted, measured on the archived file."""
    from slim.capture import pii
    from slim.recorder import note as nw
    from slim.vault.frontmatter import parse_frontmatter

    sec = nw.read_review_sections(note_text)
    fm, _ = parse_frontmatter(note_text)
    problems = []
    scored = [i for i in script.items if i.cls in classes and i.token in transcribed]
    for i in scored:
        if present(i.token, sec["transcript"]):
            problems.append(f"filed transcript keeps {i.cls}")
        if present(i.token, result.get("title") or ""):
            problems.append(f"title carries {i.cls}")
        if present(i.token, sec["summary"]) or present(i.token, result.get("summary") or ""):
            problems.append(f"summary carries {i.cls}")
        if any(present(i.token, f) for f in fragments):
            problems.append(f"an index fragment carries {i.cls}")
    counts, reasons = nw.read_pii(note_text)
    if counts != result["pii"]["masked"] or reasons != result["pii"]["partial"]:
        problems.append("frontmatter pii_masked/pii_partial differ from what the pipeline masked")
    found = Counter(p.lower() for p in PLACEHOLDERS.findall(sec["transcript"]))
    if dict(found) != counts:
        problems.append(f"placeholders in the transcript {dict(found)} != pii_masked {counts}")
    if fm.get("pii_by") != pii.VERSION:
        problems.append("pii_by missing or wrong")
    if not fragments:
        problems.append("the note has no index fragments")
    if any(c in SECRET for c in counts) and not archived_levels:
        problems.append("secrets were masked but the pipeline muted no audio span")
    for span, db in archived_levels:
        if db is None:
            problems.append(f"archived audio unreadable in {span}")
        elif db >= SILENT_DBFS:
            problems.append(f"archived audio at {db:.1f} dBFS in muted span {span}")
    return {"script": script.sid, "ok": not problems, "problems": problems,
            "note": result.get("note"), "title": result.get("title")}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detector_sha() -> str:
    """Results depend on the detector: a change to it rescores (but does not re-speak or
    re-transcribe) every script."""
    return _sha_file(REPO / "slim" / "capture" / "pii.py")[:16]


def run_calls(scripts: list[Script], out: Path, *, synth, transcribe, mask_segments, silence,
              measure, classes, model: bool = True, seed: int = 7, detector: str = "",
              full=None, full_n: int = 0, redo: bool = False, log=print) -> dict:
    """Every script: speak it, transcribe, mask, mute, re-transcribe, score.

    Caches, each keyed by what it depends on: audio by `audio_key` (lines, voices, seed); the
    first transcription by the audio's sha256; the result by seed, classes, model, detector and
    audio key. A result whose keys differ is recomputed, reusing the audio and the first
    transcription; `redo` drops all three."""
    out = Path(out)
    for sub in ("audio", "work", "results"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    results = []
    for n, s in enumerate(scripts):
        key = audio_key(s, seed)
        meta = {"seed": seed, "classes": list(classes), "model": model, "detector": detector,
                "audio_key": key, "full": bool(full is not None and n < full_n)}
        res_path = out / "results" / f"{s.sid}.json"
        audio_path = out / "audio" / f"{s.sid}.webm"
        manifest_path = audio_path.with_suffix(".json")
        if redo:
            for old in [res_path, audio_path, manifest_path, *(out / "work").glob(f"{s.sid}.*")]:
                old.unlink(missing_ok=True)
            shutil.rmtree(out / "audio" / f"{s.sid}.parts", ignore_errors=True)
        if res_path.exists():
            prior = json.loads(res_path.read_text())
            if prior.get("meta") == meta:
                results.append(prior)
                continue
            log(f"  {s.sid}: cached result is for {_diff(prior.get('meta') or {}, meta)}; rescoring")
        t0 = time.perf_counter()
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get("key") != key or not audio_path.exists():
            manifest = {**synth(s, audio_path), "key": key}
            manifest_path.write_text(json.dumps(manifest))
        audio_sha = _sha_file(audio_path)
        work = out / "work" / f"{s.sid}.webm"
        if not work.exists() or _sha_file(work) != audio_sha:
            shutil.copyfile(audio_path, work)
        first = out / "work" / f"{s.sid}.first.json"
        cached = json.loads(first.read_text()) if first.exists() else {}
        if cached.get("audio_sha") == audio_sha:
            got = cached["got"]
        else:
            got = transcribe(work)
            first.write_text(json.dumps({"audio_sha": audio_sha, "got": got}))
        raw = str(got.get("text") or "")
        masked_gots, counts, reasons, seg_spans = mask_segments([dict(got)])
        masked = str(masked_gots[0].get("text") or "")
        spans = [tuple(x) for x in seg_spans[0]]
        silenced = out / "work" / f"{s.sid}.silenced.webm"
        shutil.copyfile(work, silenced)
        silence(silenced, spans)
        heard_after = str(transcribe(silenced).get("text") or "")
        res = score_call(s, raw, masked, heard_after, classes, spans=spans, reasons=reasons,
                         manifest=manifest, measure=measure, silenced=silenced, model=model)
        res["meta"] = meta
        res["counts"] = counts
        res["masked"] = masked
        if meta["full"]:
            transcribed = {i.token for i in s.items if locate(i.token, raw)}
            res["full"] = full(s, audio_path, work, transcribed)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        (out / "results" / f"{s.sid}.raw.md").write_text(raw)
        (out / "results" / f"{s.sid}.masked.md").write_text(masked)
        (out / "results" / f"{s.sid}.silenced.md").write_text(heard_after)
        res_path.write_text(json.dumps(res, indent=1))
        log(f"  {s.sid}  {res['seconds']:.1f}s  items={len(s.items)} spans={len(spans)} "
            f"audio_fail={len(res['audio_failures'])}")
        results.append(res)
    return aggregate_calls(results, classes, model=model)


def _diff(old: dict, new: dict) -> str:
    keys = [k for k in new if old.get(k) != new[k]]
    return "other " + ", ".join(keys) if keys else "another run"


def aggregate_calls(results: list[dict], classes, *, model: bool = True) -> dict:
    """Totals over scripts. Recall counts transcribed items; items Parakeet did not write
    verbatim are scored apart (`garbled`: masked = no readable remnant). The gate needs
    evidence: some transcribed secret item, and at most half of the secret items untranscribed."""
    classes = tuple(classes)
    by_class, by_cue, garbled_rows = _empty(classes), _empty(CUES), _empty(classes)
    misses, remnants, not_transcribed = [], [], 0
    gate_cls = _gate_classes(classes, model)
    secret_total = secret_untranscribed = 0
    untranscribed_secrets = []
    for res in results:
        for i in res["items"]:
            if i["class"] not in classes:
                continue
            if i["class"] in gate_cls:
                secret_total += 1
            if not i["transcribed"]:
                not_transcribed += 1
                row = garbled_rows[i["class"]]
                row["n"] += 1
                row["masked"] += bool(i["masked"])
                if i["class"] in gate_cls:
                    secret_untranscribed += 1
                    untranscribed_secrets.append(f"{res['script']}:{i['class']}")
                if not i["masked"]:
                    remnants.append({"script": res["script"], **{k: i[k] for k in
                                     ("class", "kind", "style", "cue", "spoken", "heard", "remnant")}})
                continue
            for d in (by_class[i["class"]], by_cue[i["cue"]]):
                d["n"] += 1
                d["masked"] += bool(i["masked"])
            if not i["masked"]:
                misses.append({"script": res["script"], **{k: i[k] for k in
                               ("class", "kind", "style", "cue", "spoken", "heard", "result")}})
    fps = [f for r in results for f in r["false_positives"]]
    audio_fail = [f for r in results for f in r["audio_failures"]]
    fulls = [r["full"] for r in results if "full" in r]
    n = sum(by_class[c]["n"] for c in gate_cls)
    secret_recall = sum(by_class[c]["masked"] for c in gate_cls) / n if n else None
    gn = sum(garbled_rows[c]["n"] for c in gate_cls)
    garbled_recall = sum(garbled_rows[c]["masked"] for c in gate_cls) / gn if gn else None
    no_evidence = []
    if not n:
        no_evidence.append("no secret-class item was transcribed")
    if secret_total and secret_untranscribed > secret_total / 2:
        no_evidence.append(f"{secret_untranscribed} of {secret_total} secret items not transcribed: "
                           + ", ".join(untranscribed_secrets))
    ok = (not no_evidence and not fps and secret_recall >= RECALL_GATE
          and (garbled_recall is None or garbled_recall >= GARBLED_GATE)
          and not audio_fail and all(f["ok"] for f in fulls))
    return {
        "mode": "calls", "model": model, "scripts": len(results), "classes": list(classes),
        "by_class": {c: {**d, "recall": _ratio(d)} for c, d in by_class.items()},
        "by_cue": {c: {**d, "recall": _ratio(d)} for c, d in by_cue.items()},
        "garbled": {c: {**d, "recall": _ratio(d)} for c, d in garbled_rows.items()},
        "not_transcribed": not_transcribed, "misses": misses, "remnants": remnants,
        "false_positives": fps, "distractors_damaged": len(fps),
        "garbled_masked": [g for r in results for g in r.get("garbled_masked", [])],
        "audio_failures": audio_fail,
        "renderings": [x for r in results for x in r["renderings"]],
        "full": fulls,
        "partial": {r["script"]: r["reasons"] for r in results if r.get("reasons")},
        "seconds": round(sum(r.get("seconds", 0) for r in results), 1),
        "gate": {"ok": ok, "secret_classes": gate_cls,
                 "secret_recall": None if secret_recall is None else round(secret_recall, 3),
                 "garbled_secret_recall": None if garbled_recall is None else round(garbled_recall, 3),
                 "distractors_damaged": len(fps), "false_positives": len(fps),
                 "audio_failures": len(audio_fail),
                 "full_failures": sum(not f["ok"] for f in fulls), "no_evidence": no_evidence},
    }


def print_calls_report(r: dict) -> None:
    print(f"\n== calls: {r['scripts']} scripts, model={'yes' if r['model'] else 'no'}, "
          f"classes={','.join(r['classes'])}, {r['seconds']}s of work")
    print("How Parakeet wrote each spoken form:")
    print(f"  {'script':<7}{'class':<9}{'style':<12}{'form':<8}spoken -> heard")
    for x in r["renderings"]:
        print(f"  {x['script']:<7}{x['class']:<9}{x['style']:<12}{x['form']:<8}"
              f"{x['spoken']!r} -> {x['heard']!r}")
    print("Transcribed items (Parakeet wrote the exact value); masked = every letter and digit "
          "under a placeholder:")
    print("\n".join(_table("class", r["by_class"])))
    print("\n".join(_table("cue placement", r["by_cue"])))
    print(f"Not transcribed verbatim ({r['not_transcribed']}); masked = no readable remnant "
          f"(>= 4 digits, >= 5 letters, or a fuzzy match >= 0.6) left in the masked text:")
    print("\n".join(_table("garbled", r["garbled"])))
    for m in r["misses"]:
        print(f"  MISS {m['script']} {m['class']}/{m['kind']} cue={m['cue']} ({m['result']}): "
              f"heard {m['heard']!r}")
    for m in r["remnants"]:
        print(f"  REMNANT {m['script']} {m['class']}/{m['kind']} cue={m['cue']}: {m['remnant']}; "
              f"heard {m['heard']!r}")
    for f in r["false_positives"]:
        print(f"  FALSE POSITIVE {f['script']} [{f['class'].upper()}] {f['kind']}: {f['text']!r} "
              f"(unexplained: {f['leftover']!r}) in {f['context']}")
    for g in r["garbled_masked"]:
        print(f"  masked a garbled value {g['script']}: {g['text']!r}")
    for f in r["audio_failures"]:
        print(f"  AUDIO {f['script']} {f['check']}: "
              + json.dumps({k: v for k, v in f.items() if k not in ('script', 'check')}))
    for f in r["full"]:
        print(f"  FULL {f['script']} {'ok' if f['ok'] else 'FAIL'}: title {f['title']!r} "
              + "; ".join(f["problems"]))
    if r["partial"]:
        print(f"pii_partial reasons: {r['partial']}")
    g = r["gate"]
    grec = "-" if g["garbled_secret_recall"] is None else f"{g['garbled_secret_recall']:.3f}"
    print(_gate_line(g, f", garbled secret recall {grec} (needs >= {GARBLED_GATE}), audio "
                        f"failures {g['audio_failures']}, full-pipeline failures {g['full_failures']}"))


# --- public mode ------------------------------------------------------------------------------

PERMISSIVE = {"apache-2.0", "mit", "cc-by-4.0", "cc0-1.0"}
DATASETS = (
    ("gretelai/synthetic_pii_finance_multilingual", "test"),
    ("gretelai/gretel-pii-masking-en-v1", "test"),
    ("ai4privacy/pii-masking-300k", "validation"),
)
LABELS = {
    "ssn": "ssn", "credit_card_number": "card", "iban": "account", "bban": "account",
    "bank_routing_number": "account", "account_number": "account", "email": "email",
    "street_address": "address", "address": "address", "phone_number": "phone",
}
PUBLIC_CLASSES = ("ssn", "card", "account", "email", "address", "phone")


def _finance(row: dict):
    if row.get("language") != "English":
        return None
    text = row["generated_text"]
    gold = [(int(s["start"]), int(s["end"]), LABELS.get(s["label"]), s["label"])
            for s in json.loads(row.get("pii_spans") or "[]")]
    # a span past the end of its text is the dataset's error, not a value we could find
    return text, [g for g in gold if 0 <= g[0] < g[1] <= len(text)]


def _en_v1(row: dict):
    text = row["text"]
    gold = []
    for e in ast.literal_eval(row.get("entities") or "[]"):
        value, label = str(e.get("entity") or ""), (e.get("types") or [""])[0]
        if not value:
            continue
        for m in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", text):
            gold.append((m.start(), m.end(), LABELS.get(label), label))
    gold.sort()
    return text, gold


ADAPTERS = {
    "gretelai/synthetic_pii_finance_multilingual": _finance,
    "gretelai/gretel-pii-masking-en-v1": _en_v1,
}


def _invalid(cls: str, label: str, value: str) -> bool:
    """The gold value fails our own validator: rejecting it is right, not a miss."""
    from slim.capture import pii

    d = re.sub(r"\D", "", value)
    if cls == "ssn":
        return len(d) != 9 or not pii._ssn_ok(d)
    if cls == "card":
        return not 13 <= len(d) <= 19 or not pii._luhn(d)
    if label == "iban":
        s = re.sub(r"\s", "", value)
        return not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{8,30}", s) or not pii._iban_ok(s)
    if label == "bank_routing_number":
        return len(d) != 9 or not pii._aba_ok(d)
    if cls == "email":
        return "@" not in value            # a template slot such as "[Shipper's Email]"
    if cls == "phone":
        return len(d) < 7
    return False


def score_public(docs, find) -> dict:
    """Recall (every letter and digit of a gold value under some hit) and precision (a hit of class
    C overlapping a gold span mapped to C), per class; misses split into invalid gold and true."""
    stats = {c: {"gold": 0, "found": 0, "found_valid": 0, "touched": 0, "invalid": 0,
                 "true_miss": 0, "hits": 0,
                 "tp_hits": 0, "misses": [], "false_positives": []} for c in PUBLIC_CLASSES}
    for text, gold in docs:
        hits = find(text)
        spans = [(h.start, h.end) for h in hits]
        for a, b, cls, label in gold:
            if cls not in stats:
                continue
            st = stats[cls]
            st["gold"] += 1
            touched = any(x < b and a < y for x, y in spans)
            st["touched"] += touched
            bad = _invalid(cls, label, text[a:b])
            st["invalid"] += bad
            if touched and _covered(a, b, text, spans):
                st["found"] += 1
                st["found_valid"] += not bad
                continue
            st["true_miss"] += not bad
            if not bad and len(st["misses"]) < 10:
                st["misses"].append({"label": label, "value": text[a:b],
                                     "context": text[max(0, a - 40):b + 40].replace("\n", " ")})
        for h in hits:
            if h.cls not in stats:
                continue
            st = stats[h.cls]
            st["hits"] += 1
            over = [g for g in gold if g[0] < h.end and h.start < g[1]]
            if any(g[2] == h.cls for g in over):
                st["tp_hits"] += 1
            elif len(st["false_positives"]) < 10:
                st["false_positives"].append({"text": text[h.start:h.end],
                                              "gold_label": over[0][3] if over else None})
    for st in stats.values():
        st["recall"] = round(st["found"] / st["gold"], 3) if st["gold"] else None
        valid = st["gold"] - st["invalid"]
        st["recall_valid"] = round(st["found_valid"] / valid, 3) if valid else None
        st["precision"] = round(st["tp_hits"] / st["hits"], 3) if st["hits"] else None
    return {"by_class": stats}


def licence(dataset: str, fetch) -> tuple[str, bool]:
    info = fetch(f"https://huggingface.co/api/datasets/{dataset}")
    lic = (info.get("cardData") or {}).get("license")
    if not lic:
        tags = [t.split(":", 1)[1] for t in info.get("tags") or [] if t.startswith("license:")]
        lic = tags[0] if len(tags) == 1 else tags or None
    names = lic if isinstance(lic, list) else [lic] if lic else []
    ok = bool(names) and all(str(n).lower() in PERMISSIVE for n in names)
    return (",".join(map(str, names)) or "unknown"), ok


_LAST_REQUEST = [0.0]


def http_json(url: str):
    """GET JSON with a generic User-Agent, one request a second, backing off on 429 and 5xx
    (Retry-After when given)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "pii-eval/1"})
    for attempt in range(6):
        time.sleep(max(0.0, 1.0 - (time.monotonic() - _LAST_REQUEST[0])))
        _LAST_REQUEST[0] = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as exc:                       # noqa: BLE001 - retried, then raised
            code = getattr(exc, "code", None)
            if attempt == 5 or (code is not None and code < 500 and code != 429):
                raise
            after = (getattr(exc, "headers", None) or {}).get("Retry-After") if code else None
            wait = float(after) if after and str(after).isdigit() else min(60.0, 5.0 * 2 ** attempt)
            print(f"  HTTP {code or type(exc).__name__}: retrying in {wait:.0f}s")
            time.sleep(wait)


def _fetch_failed(report, entry, exc, where: str, log) -> None:
    code = getattr(exc, "code", None)
    if code is not None:
        entry["error"] = f"HTTP {code} at {where}"
        log(f"HTTP {code} at {where}: stopped this dataset (cached pages are kept; rerun resumes)")
    else:
        report["network"] = False
        entry["error"] = f"network unavailable ({type(exc).__name__}) at {where}"
        log(entry["error"])


def run_public(out: Path, *, rows: int, fetch, find, log=print) -> dict:
    """Each candidate dataset whose licence is permissive: up to `rows` English rows through
    `find`. Pages are cached under `<out>/public/`."""
    report = {"mode": "public", "network": True, "rows": rows, "datasets": {}}
    for dataset, split in DATASETS:
        entry = report["datasets"].setdefault(dataset, {"used": False})
        try:
            lic, ok = licence(dataset, fetch)
        except OSError as exc:
            _fetch_failed(report, entry, exc, f"{dataset} licence", log)
            if not report["network"]:
                log("network unavailable: public mode skipped")
                return report
            continue
        entry["licence"] = lic
        if not ok:
            entry["reason"] = f"licence {lic!r} is not one of {sorted(PERMISSIVE)}"
            log(f"skip {dataset}: {entry['reason']}")
            continue
        adapter = ADAPTERS.get(dataset)
        if adapter is None:
            entry["reason"] = "no adapter for its format"
            log(f"skip {dataset}: {entry['reason']}")
            continue
        cache = out / "public" / dataset.replace("/", "__")
        cache.mkdir(parents=True, exist_ok=True)
        docs, offset = [], 0
        while len(docs) < rows:
            page_path = cache / f"{split}-{offset}.json"
            if page_path.exists():
                page = json.loads(page_path.read_text())
            else:
                url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
                       f"&config=default&split={split}&offset={offset}&length=100")
                try:
                    page = fetch(url)
                except OSError as exc:
                    _fetch_failed(report, entry, exc, f"{dataset} offset {offset}", log)
                    break
                if not isinstance(page, dict) or "rows" not in page:
                    err = page.get("error") if isinstance(page, dict) else type(page).__name__
                    entry["error"] = f"no rows at {dataset} offset {offset}: {str(err)[:120]}"
                    log(entry["error"] + " (not cached)")
                    break
                page_path.write_text(json.dumps(page))
            got = page.get("rows") or []
            for r in got:
                doc = adapter(r.get("row") or {})
                if doc and len(docs) < rows:
                    docs.append(doc)
            offset += 100
            total = page.get("num_rows_total")
            if len(got) < 100 or (total is not None and offset >= total):
                break
        entry.update({"used": True, "split": split, "rows": len(docs), **score_public(docs, find)})
        log(f"{dataset} ({lic}, {split}): {len(docs)} English rows")
    return report


def print_public_report(r: dict) -> None:
    print("\n== public (written text, patterns only; no speech)")
    print("recall: every letter and digit of a gold value under a hit of ANY class (an account "
          "masked as [CARD] counts as found); valid-rec: found among values that pass our "
          "validators; invalid: gold that fails them (counted whether found or not); "
          "precision: hits of class C over a gold span mapped to C")
    for name, d in r["datasets"].items():
        if not d.get("used"):
            print(f"{name}: skipped — {d.get('reason', 'not reached')}")
            continue
        print(f"{name}: licence {d['licence']}, {d['rows']} rows")
        if d.get("error"):
            print(f"  incomplete: {d['error']}")
        print(f"  {'class':<9}{'gold':>6}{'found':>7}{'touched':>8}{'recall':>8}{'valid-rec':>10}"
              f"{'invalid':>9}{'true-miss':>10}{'hits':>6}{'precision':>11}")
        for c, st in d["by_class"].items():
            f = lambda v: "-" if v is None else f"{v:.2f}"
            print(f"  {c:<9}{st['gold']:>6}{st['found']:>7}{st['touched']:>8}{f(st['recall']):>8}"
                  f"{f(st['recall_valid']):>10}{st['invalid']:>9}{st['true_miss']:>10}"
                  f"{st['hits']:>6}{f(st['precision']):>11}")
        for c, st in d["by_class"].items():
            for m in st["misses"]:
                print(f"    miss {c} ({m['label']}): {m['value']!r} … {m['context']!r}")
            for fp in st["false_positives"]:
                print(f"    false positive {c}: {fp['text']!r} (gold label: {fp['gold_label']})")


# --- sandbox ----------------------------------------------------------------------------------

def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def check_out(out: Path, *, home: Path | None = None, repo: Path = REPO) -> Path:
    """`out`, resolved, or exit: never inside a vault, iCloud, the real brain DB or this repo."""
    home = (home or Path.home()).resolve()
    out = Path(out).expanduser().resolve()
    refuse = [
        (home / "Library" / "Mobile Documents", "under iCloud Drive"),
        (home / "Library" / "Application Support" / "slim", "inside the real brain DB folder"),
        (Path(repo).resolve(), "inside the repository"),
    ]
    for root, why in refuse:
        if _inside(out, root.resolve()):
            raise SystemExit(f"refusing --out: it is {why}")
    for p in (out, *out.parents):
        if (p / ".obsidian").is_dir():
            raise SystemExit("refusing --out: it is inside an Obsidian vault")
    return out


def write_projects(path: Path, classes) -> None:
    lines = ['owner: "Alex"', "projects:", "  work:",
             '    description: "Day job — meetings and calls."', "  school:",
             '    description: "Coursework — lectures and readings."']
    if classes is not None:
        lines += ["pii:", f"  mask: [{', '.join(classes)}]"]
    path.write_text("\n".join(lines) + "\n")


def sandbox(out: Path, *, classes=None, home: Path | None = None, repo: Path = REPO):
    """Point SLIM at `<out>` and prove it: returns `slim.config`, or exits having done nothing."""
    out = check_out(out, home=home, repo=repo)
    vault, data, projects = out / "vault", out / "data", out / "projects.yaml"
    os.environ["SLIM_VAULT"] = str(vault)
    os.environ["SLIM_DATA_DIR"] = str(data)
    os.environ["SLIM_PROJECTS"] = str(projects)
    for d in (vault / "Notes" / "work", vault / "Notes" / "school",
              vault / "Attachments" / "_incoming", data):
        d.mkdir(parents=True, exist_ok=True)
    write_projects(projects, classes)
    import slim.config as config

    for name in ("VAULT", "DATA_DIR", "PROJECTS_YAML", "DB_PATH", "TRACE_DIR"):
        if not _inside(Path(getattr(config, name)).resolve(), out):
            raise SystemExit(f"refusing to run: slim.config.{name} resolves outside --out "
                             "(was slim imported before the sandbox was set?)")
    return config


# --- entry ------------------------------------------------------------------------------------

def _write(out: Path, mode: str, report: dict) -> None:
    (out / f"report-{mode}.json").write_text(json.dumps(report, indent=1, default=str))


def _classes(arg: str | None):
    if arg is None:
        return None
    if arg == "all":
        return CLASSES
    got = tuple(c.strip() for c in arg.split(",") if c.strip())
    bad = [c for c in got if c not in CLASSES]
    if bad:
        raise SystemExit(f"unknown classes: {bad}")
    return got


def _full_runner(config, *, pipeline=None, measure=rms_dbfs):
    """`handle_record` on a staged copy, then `check_full`, including the archived audio's
    level inside the spans the pipeline itself muted (seen through `_silence_segments`)."""
    import sqlite3

    def run(script: Script, audio_path: Path, work: Path, transcribed: set[str]) -> dict:
        pl = pipeline
        if pl is None:
            from slim.recorder import pipeline as pl
        rel = f"Attachments/_incoming/eval{script.sid}-001.webm"
        staged = config.VAULT / rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(audio_path, staged)
        cache = work.with_suffix(work.suffix + ".transcript.json")
        if cache.exists():               # same bytes, so the pipeline reuses this transcript
            shutil.copyfile(cache, staged.with_suffix(staged.suffix + ".transcript.json"))
        muted: list = []
        real = pl._silence_segments

        def spy(segments, seg_spans):
            muted.append([[tuple(x) for x in sp] for sp in seg_spans])
            return real(segments, seg_spans)

        pl._silence_segments = spy
        try:
            result = pl.handle_record(audio_rel=rel, notes_md="", declared_type="meeting",
                                      title="", vault=config.VAULT)
        except Exception as exc:                       # noqa: BLE001 - scored as a failure
            return {"script": script.sid, "ok": False, "title": None, "note": None,
                    "problems": [f"handle_record raised {type(exc).__name__}"]}
        finally:
            pl._silence_segments = real
        note_text = (config.VAULT / result["note"]).read_text(encoding="utf-8")
        con = sqlite3.connect(config.DB_PATH)
        try:
            fragments = [r[0] for r in con.execute(
                "SELECT f.text FROM fragments f JOIN sources s ON s.id = f.source_id "
                "WHERE s.path = ? AND s.deleted = 0", (result["note"],))]
        finally:
            con.close()
        archived = config.VAULT / result["audio"]
        levels = []
        for a, b in (muted[-1][0] if muted and muted[-1] else []):
            try:
                levels.append(([a, b], measure(archived, a, b)))
            except Exception:                          # noqa: BLE001 - unreadable fails the check
                levels.append(([a, b], None))
        return check_full(script, note_text, result, fragments, transcribed,
                          config.PII_MASK, archived_levels=levels)
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("text", "calls", "public"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--only", action="append", help="script id (repeatable)")
    ap.add_argument("--no-model", action="store_true", help="patterns only: no password pass")
    ap.add_argument("--classes", help="comma list or 'all' (default: the shipped default)")
    ap.add_argument("--full", type=int, default=5, help="calls: run handle_record on the first N")
    ap.add_argument("--redo", action="store_true", help="calls: rescore scripts already scored")
    ap.add_argument("--rows", type=int, default=2000, help="public: English rows per dataset")
    a = ap.parse_args(argv)

    config = sandbox(a.out, classes=_classes(a.classes))
    out = a.out.resolve()
    from slim.capture import pii

    classes = tuple(config.PII_MASK)
    if a.mode == "public":
        report = run_public(out, rows=a.rows, fetch=http_json,
                            find=lambda t: pii.find(t, PUBLIC_CLASSES))
        print_public_report(report)
        _write(out, "public", report)
        return 0

    scripts = generate(a.n, a.seed)
    if a.only:
        scripts = [s for s in scripts if s.sid in set(a.only)]
    (out / "scripts.json").write_text(json.dumps([asdict(s) for s in scripts], indent=1))
    model = not a.no_model
    if model:
        print("model job: check nothing else is loading a model on this Mac (ps, ollama ps, "
              "the llama-server log) before this runs")
    if a.mode == "text":
        find = pii.ask_model if model else None
        report = run_text(scripts, lambda t: pii.mask(t, classes, find_password=find), classes,
                          model=model)
        print_text_report(report)
    else:
        from slim.capture import audio, transcribe as tmod
        from slim.recorder import pipeline

        report = run_calls(
            scripts, out,
            synth=lambda s, dest: synthesize(s, dest, a.seed),
            transcribe=lambda p: pipeline._transcribe_segment(Path(p), tmod)[0],
            mask_segments=lambda gots: pipeline._mask_segments(gots, model=model),
            silence=audio.silence, measure=rms_dbfs, classes=classes, model=model,
            seed=a.seed, detector=detector_sha(),
            full=_full_runner(config) if a.full and model else None, full_n=a.full,
            redo=a.redo)
        print_calls_report(report)
    _write(out, a.mode, report)
    return 0 if report["gate"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
