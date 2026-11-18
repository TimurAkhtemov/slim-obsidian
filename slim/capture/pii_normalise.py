"""Spoken numbers to digits, with an offset map back to the original text."""

import re

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
