"""Matching and scoring: is a value still readable, what masking replaced, the text and calls scorers.

Known strictness: in `calls`, one mask over two adjacent values Parakeet garbled excuses only
one of them, so the other counts as a false positive. Generated scripts put each item in its
own turn, where an excuse cannot reach across.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import Counter
from pathlib import Path

from .gen import CUES, Item, SECRET, TRIPLE_KINDS, Script


PLACEHOLDERS = re.compile(r"\[(SSN|CARD|ACCOUNT|PASSWORD|EMAIL|ADDRESS|PHONE)\]")
RECALL_GATE = 0.95
GARBLED_GATE = 0.8          # recall over values Parakeet did not write verbatim
SILENT_DBFS = -60.0           # a muted span must be quieter than this
SPEECH_DBFS = -45.0           # a line nobody muted must be louder than this
VANISHED_BELOW = 0.35         # a muted transcript's best turn reading this little like the line: gone


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


class NotSubstitution(ValueError):
    """The masked text is not the original with spans swapped for known placeholders."""


NOT_SUBSTITUTION = "not a placeholder substitution"


def _replaced(orig: str, masked: str) -> list[tuple[int, int, str]]:
    """Spans of `orig` that masking replaced, with the placeholder's class. Masking leaves every
    other byte alone, so `masked` is `orig` with some spans swapped for placeholders: align the
    literal pieces between placeholders. Anything else (a rewrite, "XXXX" over a value) raises
    `NotSubstitution`: the instrument never credits a change it cannot explain."""
    pieces = PLACEHOLDERS.split(masked)
    lits, names = pieces[0::2], [n.lower() for n in pieces[1::2]]
    if not names:
        if masked != orig:
            raise NotSubstitution(NOT_SUBSTITUTION)
        return []
    pattern = re.escape(lits[0]) + "".join("(.+?)" + re.escape(lit) for lit in lits[1:])
    m = re.fullmatch(pattern, orig, re.S)
    if not m:
        raise NotSubstitution(NOT_SUBSTITUTION)
    return [(*m.span(g), names[g - 1]) for g in range(1, m.lastindex + 1)]


def replaced(orig: str, masked: str) -> list[tuple[int, int]]:
    return [(a, b) for a, b, _ in _replaced(orig, masked)]


def _line_at(text: str, pos: int) -> str:
    a = text.rfind("\n\n", 0, pos)
    b = text.find("\n\n", pos)
    return text[a + 2 if a >= 0 else 0:b if b >= 0 else len(text)]


def _covered(a: int, b: int, text: str, reps) -> bool:
    """Every letter and digit of text[a:b] lies under some replaced span."""
    return all(any(x <= k < y for x, y in reps) for k in range(a, b) if text[k].isalnum())


def _empty(keys):
    return {k: {"n": 0, "masked": 0} for k in keys}


def _ratio(d):
    return round(d["masked"] / d["n"], 3) if d["n"] else None


GATE_RULE = (f"gate rule: secret-class recall >= {RECALL_GATE} over CUED items (same_sentence, "
             "previous_turn, after_value); items with no cue are shape-only, reported, not gated; "
             "zero false positives, every letter and digit counted, except the digits of an "
             "ssn-shaped triple itself (a known trade-off), reported apart and not gated")


def _cued_recall(records, gate_cls) -> tuple[float | None, int, dict]:
    """(recall over cued secret items, how many, the shape-only row) from (cls, cue, masked)."""
    cued = [m for c, q, m in records if c in gate_cls and q != "none"]
    shape = [m for c, q, m in records if c in gate_cls and q == "none"]
    row = {"n": len(shape), "masked": sum(shape)}
    row["recall"] = _ratio(row)
    return (sum(cued) / len(cued) if cued else None), len(cued), row


def _triple_positions(text: str, dspans) -> set[int]:
    """The characters of every ssn-shaped triple's own number: the digits of an `ssn_triple`
    line, every letter and digit of a spoken `triple` phrase (which is only the number)."""
    out = set()
    for kind, x, y in dspans:
        if kind in TRIPLE_KINDS:
            out |= {k for k in range(x, y)
                    if text[k].isdigit() or (kind != "ssn_triple" and text[k].isalnum())}
    return out


def _damage(text: str, a: int, b: int, own, excused, dspans):
    """Per character: the letters and digits of text[a:b] that no item's own span, no excused
    garbled value and no triple's own number explains (the false positive), every distractor
    those touch, and the triples whose own number was masked (reported, not gated)."""
    tri = _triple_positions(text, dspans)
    left = [k for k in range(a, b) if text[k].isalnum() and k not in tri
            and not any(x <= k < y for x, y in [*own, *excused])]
    kinds = [kind for kind, x, y in dspans if any(x <= k < y for k in left)]
    masked_triples = [(kind, x) for kind, x, y in dspans if kind in TRIPLE_KINDS
                      and any(k in tri and x <= k < y for k in range(a, b))]
    return left, kinds, masked_triples


def _gate_classes(classes, model: bool) -> list[str]:
    return [c for c in SECRET if c in classes and (model or c != "password")]


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


def _count(hay: str, run: str) -> int:
    """Occurrences of `run` in `hay`, overlapping ones included."""
    n, at = 0, hay.find(run)
    while at >= 0:
        n, at = n + 1, hay.find(run, at + 1)
    return n


def _fresh_runs(token_part: str, n: int, hay: str, elsewhere: list[str]) -> list[str]:
    """Runs of `n` characters of the token that `hay` holds MORE often than the rest of the
    original accounts for. A distractor "2019" excuses one "019", not two: a run the other lines
    explain once and `hay` repeats is still this item's."""
    return sorted(r for r in _runs(token_part, n)
                  if _count(hay, r) > sum(_count(e, r) for e in elsewhere))


def elsewhere_in_script(script: "Script") -> list[str]:
    """The script with EVERY item's value cut out, squashed piece by piece: what may explain a
    run without it being an item's. Other items' values never excuse anything."""
    return _segments_outside(script.transcript, [(i.start, i.end) for i in script.items])


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
                    if len(window) < len(f) / 2 or (
                            _count(hay, window) <= sum(_count(e, window) for e in elsewhere)):
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
    triples: set = set()
    records = []
    errors = []
    for s in scripts:
        t, m = s.transcript, masked[s.sid]
        try:
            reps3 = _replaced(t, m)
        except NotSubstitution as exc:
            errors.append({"script": s.sid, "error": str(exc)})
            reps3 = []                 # nothing is credited: every item scores as a miss
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
            records.append((i.cls, i.cue, gone))
            if not gone:
                misses.append({"script": s.sid, "class": i.cls, "kind": i.kind, "style": i.style,
                               "cue": i.cue, "result": "missed" if readable else "partial",
                               "context": _line_at(t, i.start), "value": i.spoken})
        own_all = [(i.start, i.end) for i in scored]
        for a, b, cls in reps3:
            own = [(x, y) for x, y in own_all if a < y and x < b]
            dspans = [(d.kind, d.start, d.end) for d in s.distractors]
            left, hit, tri = _damage(t, a, b, own, [], dspans)
            triples |= {(s.sid, *x) for x in tri}
            if not left:
                continue
            damaged += len(hit)
            fps.append({"script": s.sid, "text": t[a:b], "context": _line_at(t, a),
                        "kind": "over-wide mask" if own else "false positive",
                        "leftover": "".join(t[k] for k in left),
                        "distractor": hit[0] if hit else None, "distractors": hit, "class": cls})
    gate_cls = _gate_classes(classes, model)
    secret_recall, n, shape = _cued_recall(records, gate_cls)
    gated = fps
    triples = len(triples)
    total = sum(d["n"] for d in by_class.values())
    no_evidence = [] if n else [f"no cued secret-class item ({','.join(gate_cls)}) in these scripts"]
    return {
        "mode": "text", "model": model, "classes": list(classes), "scripts": len(scripts),
        "items": total,
        "recall": round(sum(d["masked"] for d in by_class.values()) / total, 3) if total else None,
        "by_class": {c: {**d, "recall": _ratio(d)} for c, d in by_class.items()},
        "by_cue": {c: {**d, "recall": _ratio(d)} for c, d in by_cue.items()},
        "distractors": sum(len(s.distractors) for s in scripts),
        "distractors_damaged": damaged, "ssn_shaped_triples": triples,
        "shape_only": shape,
        "placeholders": {c: {"found": placeholders[c], "expected": expected[c]} for c in classes},
        "misses": misses, "false_positives": fps,
        "errors": errors,
        "gate": {"ok": not no_evidence and not gated and not errors
                 and secret_recall >= RECALL_GATE, "errors": len(errors),
                 "rule": GATE_RULE, "secret_classes": gate_cls,
                 "secret_recall": None if secret_recall is None else round(secret_recall, 3),
                 "distractors_damaged": damaged, "false_positives": len(gated),
                 "ssn_shaped_triples": triples, "no_evidence": no_evidence},
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
    return (f"gate: {'ok' if g['ok'] else 'FAIL'} — cued secret recall {rec} over "
            f"{','.join(g['secret_classes'])} (needs >= {RECALL_GATE}), distractors damaged "
            f"{g['distractors_damaged']}, false positives {g['false_positives']} (need 0; "
            f"ssn-shaped triples masked {g['ssn_shaped_triples']}, not gated){extra}{why}")


def print_text_report(r: dict) -> None:
    print(f"\n== text: {r['scripts']} scripts, {r['items']} items, model={'yes' if r['model'] else 'no'}, "
          f"classes={','.join(r['classes'])}")
    print(GATE_RULE)
    print("An item counts as masked only when every letter and digit of it is under a placeholder.")
    print("\n".join(_table("class", r["by_class"])))
    print("\n".join(_table("cue placement", r["by_cue"])))
    print("\n".join(_table("secret, no cue", {"no cue — shape only": r["shape_only"]})))
    ph = ", ".join(f"[{c.upper()}] {d['found']}/{d['expected']}" for c, d in r["placeholders"].items())
    print(f"placeholders found/expected: {ph}")
    print(f"distractors damaged: {r['distractors_damaged']} of {r['distractors']}")
    print(f"ssn-shaped triples masked (their own digits only): {r['ssn_shaped_triples']} "
          "(known trade-off, not gated)")
    for e in r.get("errors", []):
        print(f"  ERROR {e['script']}: {e['error']} (fails the gate)")
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


_PIECE_CUT = re.compile(r"(?<=[.?!])\s|\n\n|\*\*[^*\n]+:\*\*")


def _raw_turns(raw: str) -> list[tuple[int, int]]:
    edges = [0, *(m.end() for m in re.finditer(r"\n\n", raw)), len(raw) + 2]
    return [(x, y - 2) for x, y in zip(edges, edges[1:])]


def _turn_match(item: Item, script: Script, text: str, turns) -> tuple[tuple[int, int], float]:
    """The turn of `text` that holds the script line `item` is on, and how well it reads as that
    line: Parakeet keeps the order and wording of lines, not their exact spelling or turn breaks,
    so turns are matched by content, not counted."""
    line = squash(next((t for _, t in script.lines if item.spoken in t), item.spoken))
    scored = [(difflib.SequenceMatcher(None, squash(text[x:y]), line, autojunk=False).ratio(), (x, y))
              for x, y in turns]
    best = max(scored, default=(0.0, (0, 0)))
    return best[1], best[0]


def _turn_of(item: Item, script: Script, raw: str, turns) -> tuple[int, int]:
    return _turn_match(item, script, raw, turns)[0]


_TURN_CUT = re.compile(r"\n\n|\*\*[^*\n]+:\*\*")


def _blocks(sq: str, f: str):
    return [bl for bl in difflib.SequenceMatcher(None, sq, f, autojunk=False).get_matching_blocks()
            if bl.size]


def _best_window(raw, words, f, *, anchor=None):
    """The word run that reads best as `f` (>= 0.6, at least half its length): any run, or one
    that starts at the first word (`anchor="start"`) or ends at the last (`anchor="end"`).
    Returns (ratio, start, end, f_start, f_end) or None."""
    best = None
    n = len(words)
    for i in range(n):
        if anchor == "start" and i:
            break
        for j in range(i, min(i + 12, n)):
            if anchor == "end" and j != n - 1:
                continue
            ws, we = words[i][0], words[j][1]
            sq = squash(raw[ws:we])
            if not f or not sq or len(sq) < len(f) / 2:
                continue
            r = difflib.SequenceMatcher(None, sq, f, autojunk=False).ratio()
            if r >= 0.6 and (best is None or (r, we - ws) > (best[0], best[2] - best[1])):
                bl = _blocks(sq, f)
                best = (r, ws, we, bl[0].b, bl[-1].b + bl[-1].size)
    return best


def _aligned(raw: str, a: int, b: int, forms) -> tuple[int, int] | None:
    """The stretch of raw[a:b] excused as a value Parakeet garbled, or None. `forms` is
    (form, turn span). The best word run inside one sentence that reads as a form (>= 0.6, at
    least half its length), inside that value's own turn; then grown, sentence by sentence but
    never past a turn start, over neighbouring sentences whose opening (or closing) words read
    as the REST of the same form in order ("6691 Yerrick Road. 26430." for an address with its
    ZIP). Everything else in the replaced span stays a false positive."""
    cuts = sorted({a, b, *(m.start() + a for m in _PIECE_CUT.finditer(raw[a:b])),
                   *(m.end() + a for m in _PIECE_CUT.finditer(raw[a:b]))})
    pieces = []
    for x, y in zip(cuts, cuts[1:]):
        words = [(m.start() + x, m.end() + x) for m in re.finditer(r"\S+", raw[x:y])
                 if "**" not in m.group()]
        if words:
            pieces.append(words)

    def joined(p, q):                  # pieces p < q with no turn start between them
        return not _TURN_CUT.search(raw[pieces[p][-1][1]:pieces[q][0][0]])

    best = None
    for k, words in enumerate(pieces):
        for f, (lo, hi) in forms:
            got = _best_window(raw, words, f)
            if got and lo <= got[1] and got[2] <= hi and (
                    best is None or (got[0], got[2] - got[1]) > (best[0][0], best[0][2] - best[0][1])):
                best = (got, k, f, (lo, hi))
    if best is None:
        return None
    (_, ws, we, f_start, f_end), k, f, (lo, hi) = best
    q = k
    while f_end < len(f) and q + 1 < len(pieces) and joined(q, q + 1):
        nxt = _best_window(raw, pieces[q + 1], f[f_end:], anchor="start")
        if not nxt or nxt[2] > hi:
            break
        we, f_end, q = nxt[2], f_end + nxt[4], q + 1
    q = k
    while f_start > 0 and q > 0 and joined(q - 1, q):
        prev = _best_window(raw, pieces[q - 1], f[:f_start], anchor="end")
        if not prev or prev[1] < lo:
            break
        ws, f_start, q = prev[1], prev[3], q - 1
    return ws, we


def _distractor_spans(script: Script, raw: str) -> list[tuple[str, int, int]]:
    """Where Parakeet wrote each distractor (any spelling squash reads alike)."""
    return [(d.kind, x, y) for d in script.distractors for x, y in locate(d.text, raw)]


def _resembles(script: Script, raw: str, left: list[int]) -> list[str]:
    """For damage on no located distractor: the distractor it resembles (>= 0.6), if any."""
    text = squash("".join(raw[k] for k in left))
    near = [(difflib.SequenceMatcher(None, text, squash(d.text)).ratio(), d.kind)
            for d in script.distractors]
    best = max(near, default=(0.0, None))
    return [best[1]] if best[0] >= 0.6 else []


def score_call(script: Script, raw: str, masked: str, heard_after: str, classes, *,
               spans, reasons, manifest, measure, silenced: Path, model: bool = True) -> dict:
    """One call's numbers: items (transcribed, masked, remnant), false positives, audio checks.

    A transcribed item is masked only when every letter and digit of every place Parakeet wrote
    it is under a placeholder. An item Parakeet did not write verbatim is checked for a readable
    remnant in the masked text instead of being dropped. Audio: no run of 3 characters of a
    secret's token may come back from the muted copy unless the original says it elsewhere."""
    classes = tuple(classes)
    items, fps, garbled, audio_fail, renderings = [], [], [], [], []
    error = None
    try:
        reps3 = _replaced(raw, masked)
    except NotSubstitution as exc:
        error, reps3 = str(exc), []    # nothing is credited; the gate fails on the error
    reps = [(a, b) for a, b, _ in reps3]
    located = {id(i): locate(i.token, raw) for i in script.items}
    rendered = {id(i): [rendered_span(i, raw, sp) for sp in located[id(i)]] for i in script.items}
    for i in script.items:
        line = next((t for _, t in script.lines if i.spoken in t), i.spoken)
        heard, status = _heard(i, line, raw)
        transcribed = bool(located[id(i)])
        elsewhere = elsewhere_in_script(script)
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
        audible = None
        if i.cls in _gate_classes(classes, model):
            # Only inside the muted transcript's turn that holds this item's own line: runs
            # elsewhere in the call are other text's (another item left audible, a distractor).
            # A turn that vanished because it was muted reads as nothing like the line: a pass.
            # Within the turn, the rest of the item's own line may explain a run.
            (x, y), fit = _turn_match(i, script, heard_after, _raw_turns(heard_after))
            rest = next((t for _, t in script.lines if i.spoken in t), i.spoken).replace(i.spoken, " | ")
            hay = squash(heard_after[x:y]) if fit >= VANISHED_BELOW else ""
            leaked = _fresh_runs(squash(i.token), 3, hay, [squash(p) for p in rest.split("|")])
            audible = bool(leaked)
            if leaked:
                audio_fail.append({"script": script.sid, "check": "retranscribed", "class": i.cls,
                                   "cue": i.cue, "runs": leaked[:5], "heard": heard_after[x:y].strip()})
        items[-1]["audible"] = audible
    own_all = [sp for i in script.items if i.cls in classes for sp in rendered[id(i)]]
    dspans = _distractor_spans(script, raw)
    # Only a value Parakeet did NOT write verbatim may excuse a mask as "garbled", and only inside
    # its own turn: a value written verbatim already has its own span.
    turns = _raw_turns(raw)
    forms = [(squash(v), _turn_of(i, script, raw, turns))
             for i in script.items if i.cls in classes and not located[id(i)]
             for v in (i.spoken, i.written)]
    triples: set = set()
    for a, b, cls in reps3:
        own = [(x, y) for x, y in own_all if a < y and x < b]
        part = None if own else _aligned(raw, a, b, forms)
        excused = [part] if part else []
        left, hit, tri = _damage(raw, a, b, own, excused, dspans)
        triples |= set(tri)
        if part:
            garbled.append({"script": script.sid, "class": cls, "text": raw[part[0]:part[1]],
                            "context": _line_at(raw, a)})
        if not left:
            continue
        hit = hit or _resembles(script, raw, left)
        fps.append({"script": script.sid, "class": cls, "text": raw[a:b],
                    "context": _line_at(raw, a), "leftover": "".join(raw[k] for k in left),
                    "kind": "over-wide mask" if own else "beyond a garbled value" if part
                    else "false positive",
                    "distractor": hit[0] if hit else None, "distractors": hit})

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
            "reasons": sorted(reasons), "ssn_shaped_triples": len(triples), "error": error}


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
    records, garbled_records = [], []
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
                garbled_records.append((i["class"], i["cue"], bool(i["masked"])))
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
            records.append((i["class"], i["cue"], bool(i["masked"])))
            if not i["masked"]:
                misses.append({"script": res["script"], **{k: i[k] for k in
                               ("class", "kind", "style", "cue", "spoken", "heard", "result")}})
    errors = [{"script": r["script"], "error": r["error"]} for r in results if r.get("error")]
    fps = [f for r in results for f in r["false_positives"]]
    triples = sum(r.get("ssn_shaped_triples", 0) for r in results)
    gated = fps
    damaged = [k for f in fps for k in f.get("distractors") or []]
    audio_all = [f for r in results for f in r["audio_failures"]]
    audio_fail = [f for f in audio_all if f.get("cue") != "none"]       # span checks carry no cue
    audio_no_cue = [f for f in audio_all if f.get("cue") == "none"]
    cued = [i for r in results for i in r["items"]
            if i["class"] in gate_cls and i["class"] in classes and i["cue"] != "none"]
    headline = {"text": [sum(bool(i["masked"]) for i in cued), len(cued)],
                "audio": [sum(i.get("audible") is False for i in cued), len(cued)]}
    fulls = [r["full"] for r in results if "full" in r]
    secret_recall, n, shape = _cued_recall(records, gate_cls)
    garbled_recall, _, garbled_shape = _cued_recall(garbled_records, gate_cls)
    no_evidence = []
    if not n:
        no_evidence.append("no cued secret-class item was transcribed")
    if secret_total and secret_untranscribed > secret_total / 2:
        no_evidence.append(f"{secret_untranscribed} of {secret_total} secret items not transcribed: "
                           + ", ".join(untranscribed_secrets))
    ok = (not no_evidence and not gated and not errors and secret_recall >= RECALL_GATE
          and (garbled_recall is None or garbled_recall >= GARBLED_GATE)
          and not audio_fail and all(f["ok"] for f in fulls))
    return {
        "mode": "calls", "model": model, "scripts": len(results), "classes": list(classes),
        "by_class": {c: {**d, "recall": _ratio(d)} for c, d in by_class.items()},
        "by_cue": {c: {**d, "recall": _ratio(d)} for c, d in by_cue.items()},
        "garbled": {c: {**d, "recall": _ratio(d)} for c, d in garbled_rows.items()},
        "not_transcribed": not_transcribed, "misses": misses, "remnants": remnants,
        "false_positives": fps, "distractors_damaged": len(damaged),
        "ssn_shaped_triples": triples, "shape_only": shape, "garbled_shape_only": garbled_shape,
        "garbled_masked": [g for r in results for g in r.get("garbled_masked", [])],
        "audio_failures": audio_fail, "audio_no_cue": audio_no_cue, "headline": headline,
        "renderings": [x for r in results for x in r["renderings"]],
        "full": fulls,
        "partial": {r["script"]: r["reasons"] for r in results if r.get("reasons")},
        "seconds": round(sum(r.get("seconds", 0) for r in results), 1),
        "errors": errors,
        "gate": {"ok": ok, "rule": GATE_RULE, "secret_classes": gate_cls, "errors": len(errors),
                 "secret_recall": None if secret_recall is None else round(secret_recall, 3),
                 "garbled_secret_recall": None if garbled_recall is None else round(garbled_recall, 3),
                 "distractors_damaged": len(damaged), "false_positives": len(gated),
                 "ssn_shaped_triples": triples,
                 "audio_failures": len(audio_fail),
                 "full_failures": sum(not f["ok"] for f in fulls), "no_evidence": no_evidence},
    }


def print_calls_report(r: dict) -> None:
    print(f"\n== calls: {r['scripts']} scripts, model={'yes' if r['model'] else 'no'}, "
          f"classes={','.join(r['classes'])}, {r['seconds']}s of work")
    h = r["headline"]
    print(f"cued secrets masked in text: {h['text'][0]} of {h['text'][1]}")
    print(f"cued secrets silent in audio: {h['audio'][0]} of {h['audio'][1]}")
    print(GATE_RULE + "; the audio check gates cued items too")
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
    print("\n".join(_table("secret, no cue", {"no cue — shape only": r["shape_only"],
                                               "garbled, no cue": r["garbled_shape_only"]})))
    for e in r.get("errors", []):
        print(f"  ERROR {e['script']}: {e['error']} (fails the gate)")
    for m in r["misses"]:
        print(f"  MISS {m['script']} {m['class']}/{m['kind']} cue={m['cue']} ({m['result']}): "
              f"heard {m['heard']!r}")
    for m in r["remnants"]:
        print(f"  REMNANT {m['script']} {m['class']}/{m['kind']} cue={m['cue']}: {m['remnant']}; "
              f"heard {m['heard']!r}")
    for f in r["false_positives"]:
        where = f"distractor {f['distractor']}" if f.get("distractor") else "other text"
        print(f"  FALSE POSITIVE {f['script']} [{f['class'].upper()}] {f['kind']} over {where}: "
              f"{f['text']!r} (unexplained: {f['leftover']!r}) in {f['context']}")
    print(f"false positives: {len(r['false_positives'])}; distractors damaged: "
          f"{r['distractors_damaged']}; ssn-shaped triples masked: {r['ssn_shaped_triples']} "
          "(known trade-off, not gated)")
    for g in r["garbled_masked"]:
        print(f"  masked a garbled value {g['script']}: {g['text']!r}")
    for f in r["audio_failures"]:
        print(f"  AUDIO {f['script']} {f['check']}: "
              + json.dumps({k: v for k, v in f.items() if k not in ('script', 'check')}))
    for f in r["audio_no_cue"]:
        print(f"  no cue — audible, not gated: {f['script']} {f['class']} runs {f['runs']} "
              f"heard {f['heard']!r}")
    for f in r["full"]:
        print(f"  FULL {f['script']} {'ok' if f['ok'] else 'FAIL'}: title {f['title']!r} "
              + "; ".join(f["problems"]))
    if r["partial"]:
        print(f"pii_partial reasons: {r['partial']}")
    g = r["gate"]
    grec = "-" if g["garbled_secret_recall"] is None else f"{g['garbled_secret_recall']:.3f}"
    print(_gate_line(g, f", garbled secret recall {grec} (needs >= {GARBLED_GATE}), audio "
                        f"failures {g['audio_failures']}, full-pipeline failures {g['full_failures']}"))
