"""Masking of personal data in a recording's transcript, at the owner's request.

Spoken numbers ("five oh two") are first normalised to digits so patterns can be
matched on one shape; an offset map carries every hit back to the original text.
Detected values are only ever replaced by a class placeholder: they are never
stored, logged or traced.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

# Re-exported so callers, the evaluation and tests reach the detector's parts (and patch llm and
# trace) through one module.
from .. import config, llm, trace  # noqa: F401
from .pii_detect import PLACEHOLDER, Hit, _iban_ok, _luhn, _ssn_ok, find  # noqa: F401
from .pii_normalise import normalise  # noqa: F401
from .pii_password import CUES, _password_spans, ask_model  # noqa: F401

VERSION = "pii-1"
SECRET = frozenset({"ssn", "card", "account", "password"})
# Why masking is unfinished, as `pii_partial` names it. `password`: the password pass did not
# complete, which `slim mask` can redo from the text. `audio`: a secret may still be audible
# (a mute failed, or a secret could not be placed in token times), which a text rerun cannot fix.
PARTIAL_REASONS = ("audio", "password")


@dataclass
class Masked:
    text: str
    counts: dict[str, int]
    secrets: list[str]  # originals of secret-class hits, in text order; never persisted
    partial: bool = False  # the password pass did not complete


def mask(
    text: str,
    classes: Iterable[str] | None = None,
    find_password: Callable[[str], str | None] | None = None,
    should_stop: Callable[[], None] | None = None,
) -> Masked:
    """`should_stop` is called between password calls and raises to abandon the masking (a
    cancelled recording)."""
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
        spans, partial = _password_spans(masked, find_password, should_stop)
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
    leak. A match is whole words: "tiger" does not mute "tigers", nor "4471" a longer number.
    Spans come back sorted and merged. Whitespace runs are collapsed on both sides; everything
    else must match exactly."""
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
            end = at + len(secret) - 1
            if (at == 0 or not hay[at - 1].isalnum()) and (end + 1 == len(hay) or not hay[end + 1].isalnum()):
                hit = True
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
