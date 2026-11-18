"""The PII evaluation script's pure parts: generator, scorers, sandbox guard, and every mode
driven with fakes. No model, no `say`, no network."""

import json
import random
import re
from pathlib import Path

import pytest

from scripts.pii_eval import calls as ev_calls
from scripts.pii_eval import gen as ev_gen
from scripts.pii_eval import public as ev_public
from scripts.pii_eval import sandbox as ev_sandbox
from scripts.pii_eval import score as ev_score
from slim.capture import pii


def _digits(s):
    return re.sub(r"\D", "", s)


def _norm_digits(spoken):
    return _digits(pii.normalise(spoken)[0])


NUMERIC = {"ssn", "card", "phone"}


@pytest.mark.parametrize("cls", ev_gen.CLASSES)
def test_values_pass_validators_and_round_trip(cls):
    rng = random.Random(1)
    for _ in range(300):
        v = ev_gen.gen_value(cls, rng)
        d = _digits(v.written)
        if cls == "ssn":
            assert len(d) == 9 and pii._ssn_ok(d)
        elif cls == "card":
            assert pii._luhn(d) and d[:4] in ev_gen.TEST_BINS
            groups = tuple(len(g) for g in re.findall(r"\d+", v.written))
            assert groups in {(4, 4, 4, 4), (4, 6, 5)}
        elif cls == "account":
            if v.kind == "iban":
                assert pii._iban_ok(v.written)
            elif v.kind == "routing":
                assert len(d) == 9 and pii._aba_ok(d)
            else:
                assert 8 <= len(d) <= 12
        elif cls == "email":
            assert re.fullmatch(r"[a-z0-9.]+@example\.(com|org)", v.written)
        elif cls == "phone":
            assert len(d) == 10 and d[3:8] == "55501"
        # the spoken rendering says the same value
        if cls in NUMERIC or v.kind in ("routing", "account_number"):
            assert _norm_digits(v.spoken) == d, v.spoken
        assert ev_score.present(v.token, v.spoken), (v.token, v.spoken)
        assert ev_score.present(v.token, v.written), (v.token, v.written)


def test_spoken_styles_cover_what_people_say():
    rng = random.Random(3)
    spoken = [ev_gen.gen_value(c, rng).spoken for c in ev_gen.CLASSES for _ in range(200)]
    joined = " | ".join(spoken)
    for phrase in (" oh ", " zero ", "double ", " dash ", " dot ", " at example dot "):
        assert phrase in joined, phrase
    assert re.search(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|eleven|twelve)\b", joined)
    assert re.search(r"\b[a-z] [a-z] [a-z] [a-z]\b", joined)          # a spelled-out password
    styles = {ev_gen.gen_value("ssn", rng).style for _ in range(200)}
    assert {"digits", "grouped", "double"} <= styles


def test_generator_is_seeded_and_deterministic():
    a, b, c = ev_gen.generate(30, seed=7), ev_gen.generate(30, seed=7), ev_gen.generate(30, seed=8)
    assert [s.transcript for s in a] == [s.transcript for s in b]
    assert [s.transcript for s in a] != [s.transcript for s in c]


def test_scripts_shape():
    scripts = ev_gen.generate(30, seed=7)
    assert len(scripts) == 30 and len({s.sid for s in scripts}) == 30
    assert sum(not s.items for s in scripts) >= 5
    cells = {(i.cls, i.cue) for s in scripts for i in s.items}
    assert cells == {(c, q) for c in ev_gen.CLASSES for q in ev_gen.CUES}      # every class meets every cue
    for s in scripts:
        assert 0 <= len(s.items) <= 3 and 3 <= len(s.distractors) <= 6
        assert {sp for sp, _ in s.lines} == {"Me", "Them"}
        t = s.transcript
        assert t.startswith("**") and "**Me:**" in t and "**Them:**" in t
        for x in [*s.items, *s.distractors]:
            assert t[x.start:x.end] == (x.spoken if hasattr(x, "spoken") else x.text)


def test_every_cue_word_family_and_line_appears_in_the_default_scripts():
    scripts = ev_gen.generate(30, seed=7)
    kinds = {d.kind for s in scripts for d in s.distractors}
    for family, lines in ev_gen.CUE_FAMILIES.items():
        assert {k for k, _ in lines} <= kinds, family
    assert ev_gen.SSN_TRIPLE in kinds
    per_script = [sum(d.kind in {k for f in ev_gen.CUE_FAMILIES.values() for k, _ in f}
                      for d in s.distractors) for s in scripts]
    assert set(per_script) == {1, 2}
    assert [s.transcript for s in scripts] == [s.transcript for s in ev_gen.generate(30, seed=7)]


def test_tokens_are_distinctive():
    for s in ev_gen.generate(30, seed=7):
        t = s.transcript
        for i in s.items:
            assert ev_score.present(i.token, t)
            gone = t[:i.start] + "[X]" + t[i.end:]
            assert not ev_score.present(i.token, gone), (s.sid, i.token)


def test_previous_turn_answer_carries_no_cue():
    seen = 0
    for s in ev_gen.generate(30, seed=7):
        for i in s.items:
            if i.cue != "previous_turn":
                continue
            seen += 1
            turn_start = s.transcript.rfind("**", 0, i.start)
            answer = s.transcript[turn_start:s.transcript.find("\n\n", i.end) % (len(s.transcript) + 1)]
            assert not ev_gen.CUE_WORDS[i.cls].search(answer.replace(i.spoken, "")), answer
            before = s.transcript[:turn_start]
            assert ev_gen.CUE_WORDS[i.cls].search(before[before.rfind("**Me:**") if i.speaker == "Them"
                                                       else before.rfind("**Them:**"):])
    assert seen




# --- scorers: text --------------------------------------------------------------------------
# Every scorer test builds its "masked" text by hand or with a fake masker: nothing here depends
# on what the detector currently catches.

@pytest.fixture(autouse=True)
def _pin_mask_classes(monkeypatch):
    """Results never depend on a local config/projects.yaml."""
    from slim import config
    monkeypatch.setattr(config, "PII_MASK", config.PII_DEFAULT)


def test_replaced_finds_masked_ranges():
    orig = "**Me:** my social is two one nine. ok."
    masked = "**Me:** my social is [SSN]. ok."
    assert ev_score.replaced(orig, masked) == [(orig.index("two"), orig.index(". ok"))]
    assert ev_score.replaced(orig, orig) == []


def _mask_spans(text, spans):
    out, pos = [], 0
    for a, b, cls in sorted(spans):
        out += [text[pos:a], pii.PLACEHOLDER[cls]]
        pos = b
    return "".join(out + [text[pos:]])


def _perfect(script, classes=ev_gen.CLASSES):
    return _mask_spans(script.transcript, [(i.start, i.end, i.cls) for i in script.items
                                           if i.cls in classes])


SCRIPTS = ev_gen.generate(30, seed=7)


def _with(pred):
    return next(s for s in SCRIPTS if any(pred(i) for i in s.items))


def test_score_text_perfect_identity_and_damage():
    classes = ev_gen.CLASSES
    perfect = ev_score.score_text(SCRIPTS, {s.sid: _perfect(s) for s in SCRIPTS}, classes, model=True)
    assert perfect["recall"] == 1.0 and perfect["distractors_damaged"] == 0
    assert perfect["gate"]["ok"] and not perfect["misses"] and not perfect["false_positives"]
    assert sum(v["n"] for v in perfect["by_cue"].values()) == sum(len(s.items) for s in SCRIPTS)

    nothing = ev_score.score_text(SCRIPTS, {s.sid: s.transcript for s in SCRIPTS}, classes, model=True)
    assert nothing["recall"] == 0.0 and not nothing["gate"]["ok"]
    assert len(nothing["misses"]) == sum(len(s.items) for s in SCRIPTS)
    assert {"script", "class", "cue", "context"} <= set(nothing["misses"][0])

    s = SCRIPTS[0]
    d = next(d for d in s.distractors if d.kind not in ev_gen.TRIPLE_KINDS)
    damaged = {x.sid: _perfect(x) for x in SCRIPTS}
    damaged[s.sid] = _mask_spans(s.transcript, [(i.start, i.end, i.cls) for i in s.items]
                                 + [(d.start, d.end, "card")])
    got = ev_score.score_text(SCRIPTS, damaged, classes, model=True)
    assert got["distractors_damaged"] == 1 and not got["gate"]["ok"]
    assert got["false_positives"][0]["script"] == s.sid


def test_partly_masked_value_is_a_miss():
    # "[SSN], nine nine nine nine": 4 of 9 digits still readable
    s = _with(lambda i: i.cls == "ssn")
    item = next(i for i in s.items if i.cls == "ssn")
    cut = item.start + len(item.spoken) // 2
    while s.transcript[cut].isalnum():
        cut += 1
    others = [(i.start, i.end, i.cls) for i in s.items if i is not item]
    masked = {x.sid: _perfect(x) for x in SCRIPTS}
    masked[s.sid] = _mask_spans(s.transcript, others + [(item.start, cut, "ssn")])
    got = ev_score.score_text(SCRIPTS, masked, ev_gen.CLASSES, model=True)
    assert got["by_class"]["ssn"]["masked"] == got["by_class"]["ssn"]["n"] - 1
    miss = [m for m in got["misses"] if m["script"] == s.sid and m["class"] == "ssn"]
    assert miss and miss[0]["result"] == "partial"
    assert got["gate"]["secret_recall"] < 1.0


def test_partial_whose_token_is_gone_still_misses():
    # The digit run masked, the bank letters left ("G B nine four E K S C [CARD]"): the compact
    # token is gone from the text, but letters of the value are still readable.
    s = _with(lambda i: i.kind == "iban")
    item = next(i for i in s.items if i.kind == "iban")
    bank = re.search(r"(?:[A-Z],? ){3}[A-Z]", item.spoken[4:])
    at = item.start + 4 + bank.end()
    others = [(i.start, i.end, i.cls) for i in s.items if i is not item]
    masked = {x.sid: _perfect(x) for x in SCRIPTS}
    masked[s.sid] = _mask_spans(s.transcript, others + [(at, item.end, "card")])
    got = ev_score.score_text(SCRIPTS, masked, ev_gen.CLASSES, model=True)
    miss = [m for m in got["misses"] if m["script"] == s.sid and m["kind"] == "iban"]
    assert miss and miss[0]["result"] == "partial"
    assert got["by_class"]["account"]["masked"] == got["by_class"]["account"]["n"] - 1


def test_over_wide_mask_damages_what_it_swallows():
    # An [SSN] that runs past its item, over the turn break, the next label and a distractor.
    s = next(x for x in SCRIPTS
             if any(i.cls == "ssn" and any(d.start > i.end for d in x.distractors) for i in x.items))
    item = next(i for i in s.items if i.cls == "ssn")
    d = next(d for d in s.distractors if d.start > item.end)
    swallowed = [i for i in s.items if item.end <= i.start < d.end]
    others = [(i.start, i.end, i.cls) for i in s.items if i is not item and i not in swallowed]
    masked = {x.sid: _perfect(x) for x in SCRIPTS}
    masked[s.sid] = _mask_spans(s.transcript, others + [(item.start, d.end, "ssn")])
    got = ev_score.score_text(SCRIPTS, masked, ev_gen.CLASSES, model=True)
    fp = [f for f in got["false_positives"] if f["script"] == s.sid]
    assert fp and fp[0]["kind"] == "over-wide mask" and got["distractors_damaged"] >= 1
    assert not got["gate"]["ok"]


def test_value_outside_classes_masked_as_another_class_is_a_false_positive():
    s = _with(lambda i: i.cls == "phone")
    phone = next(i for i in s.items if i.cls == "phone")
    classes = tuple(c for c in ev_gen.CLASSES if c != "phone")
    masked = {x.sid: _perfect(x, classes) for x in SCRIPTS}
    masked[s.sid] = _mask_spans(s.transcript, [(i.start, i.end, i.cls) for i in s.items
                                               if i.cls in classes] + [(phone.start, phone.end, "account")])
    got = ev_score.score_text(SCRIPTS, masked, classes, model=True)
    assert [f["class"] for f in got["false_positives"]] == ["account"]
    assert not got["gate"]["ok"]


def test_score_text_without_model_leaves_password_out_of_the_gate():
    classes = ev_gen.CLASSES
    masked = {s.sid: _perfect(s, tuple(c for c in classes if c != "password")) for s in SCRIPTS}
    got = ev_score.score_text(SCRIPTS, masked, classes, model=False)
    assert got["by_class"]["password"]["masked"] == 0
    assert got["gate"]["ok"] and "password" not in got["gate"]["secret_classes"]


def test_text_gate_is_on_cued_items_only():
    # every secret said with no cue left unmasked: reported in its own row, the gate holds
    masked = {}
    for s in SCRIPTS:
        masked[s.sid] = _mask_spans(s.transcript, [(i.start, i.end, i.cls) for i in s.items
                                                   if not (i.cls in pii.SECRET and i.cue == "none")])
    got = ev_score.score_text(SCRIPTS, masked, ev_gen.CLASSES, model=True)
    assert got["shape_only"]["n"] > 0 and got["shape_only"]["masked"] == 0
    assert got["gate"]["ok"] and got["gate"]["secret_recall"] == 1.0
    assert "CUED" in got["gate"]["rule"]


def test_text_ssn_shaped_triple_damage_is_reported_not_gated():
    s = next(x for x in SCRIPTS if any(d.kind == ev_gen.SSN_TRIPLE for d in x.distractors))
    d = next(d for d in s.distractors if d.kind == ev_gen.SSN_TRIPLE)
    num = re.search(r"\d+ \d+ \d+", d.text)
    masked = {x.sid: _perfect(x) for x in SCRIPTS}
    masked[s.sid] = _mask_spans(s.transcript, [(i.start, i.end, i.cls) for i in s.items]
                                + [(d.start + num.start(), d.start + num.end(), "ssn")])
    got = ev_score.score_text(SCRIPTS, masked, ev_gen.CLASSES, model=True)
    assert got["ssn_shaped_triples"] == 1 and got["distractors_damaged"] == 0
    assert got["gate"]["ok"] and got["gate"]["false_positives"] == 0


def test_text_gate_needs_evidence():
    quiet = [s for s in SCRIPTS if not s.items]
    got = ev_score.score_text(quiet, {s.sid: s.transcript for s in quiet}, ev_gen.CLASSES, model=True)
    assert not got["gate"]["ok"] and got["gate"]["no_evidence"]


def test_classes_not_masked_are_not_scored():
    got = ev_score.score_text(SCRIPTS, {s.sid: s.transcript for s in SCRIPTS}, ("ssn",), model=True)
    assert set(got["by_class"]) == {"ssn"}


def test_text_report_names_every_miss(capsys):
    got = ev_score.score_text(SCRIPTS, {s.sid: s.transcript for s in SCRIPTS}, ev_gen.CLASSES, model=True)
    ev_score.print_text_report(got)
    out = capsys.readouterr().out
    assert "same_sentence" in out and "MISS" in out


def test_remnant_finds_other_renderings():
    s = _with(lambda i: i.cls == "ssn")
    item = next(i for i in s.items if i.cls == "ssn")
    d = item.token
    elsewhere = ev_score._segments_outside(s.transcript, [(item.start, item.end)])
    assert ev_score.remnant(item, f"**Me:** it's {d[:3]}, O {d[4]}, {d[5:]}.", elsewhere)
    assert ev_score.remnant(item, "**Me:** it's [SSN].", elsewhere) is None
    word = _with(lambda i: i.cls == "password")
    pw = next(i for i in word.items if i.cls == "password")
    root = re.sub(r"\d", "", pw.token)
    out = ev_score._segments_outside(word.transcript, [(pw.start, pw.end)])
    assert ev_score.remnant(pw, f"**Them:** It's {root.capitalize()} 12.", out)


# --- calls, end to end with fakes -------------------------------------------------------

class FakeStudio:
    """`say` + Parakeet + ffmpeg stand-ins: the audio file is a JSON list of (speaker, word, t0,
    t1), silencing drops words inside spans, transcription returns what is left. `render` maps a
    spoken phrase to what "Parakeet" writes instead."""

    def __init__(self, render=None):
        self.transcribed = 0
        self.synths = 0
        self.render = render or {}

    def synth(self, script, dest: Path):
        self.synths += 1
        words, t, lines = [], 0.5, []
        for speaker, text in script.lines:
            for said, heard in self.render.items():
                text = text.replace(said, heard)
            start = t
            for w in text.split():
                words.append([speaker, w, t, t + 0.3])
                t += 0.35
            lines.append({"speaker": speaker, "start": start, "end": t})
            t += 0.4
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(words))
        return {"lines": lines, "seconds": t}

    def transcribe(self, path: Path):
        self.transcribed += 1
        words = json.loads(Path(path).read_text())
        turns = []
        for speaker, w, *_ in words:
            if turns and turns[-1][0] == speaker:
                turns[-1][1].append(w)
            else:
                turns.append((speaker, [w]))
        text = "\n\n".join(f"**{s}:** {' '.join(ws)}" for s, ws in turns)
        return {"text": text, "speakers_by": "channels", "solo_speaker": "",
                "tokens": [[" " + w, a, b] for _, w, a, b in words]}

    def silence(self, path: Path, spans, pad=0.15):
        words = json.loads(Path(path).read_text())
        keep = [w for w in words if not any(a - pad <= w[2] and w[3] <= b + pad for a, b in spans)]
        Path(path).write_text(json.dumps(keep))

    def measure(self, path: Path, a: float, b: float) -> float:
        words = json.loads(Path(path).read_text())
        return -20.0 if any(w[2] < b and w[3] > a for w in words) else -120.0


class FakeMasker:
    """`_mask_segments` stand-in that masks exactly the script's values found verbatim in the
    text. `how` bends it: "partial" masks the first half of each value, "wide" runs to the end
    of the turn, "short_mute" mutes only the first half of a value's words."""

    def __init__(self, scripts, how="exact", classes=ev_gen.CLASSES):
        self.items = [i for s in scripts for i in s.items if i.cls in classes]
        self.how = how

    def __call__(self, gots):
        got = gots[0]
        text, tokens = got["text"], got.get("tokens") or []
        spans, counts, secrets, cuts = [], {}, [], []
        for i in self.items:
            at = text.find(i.spoken)
            if at < 0:
                continue
            end = at + len(i.spoken)
            if self.how == "partial":
                end = at + len(" ".join(i.spoken.split()[:len(i.spoken.split()) // 2]))
            elif self.how == "wide":
                nxt = text.find("\n\n", end)
                end = nxt if nxt >= 0 else len(text)
            cuts.append((at, end, i.cls))
            counts[i.cls] = counts.get(i.cls, 0) + 1
            if i.cls in pii.SECRET:
                words = text[at:end].split()
                if self.how == "short_mute":
                    words = words[:max(1, len(words) // 2)]
                secrets.append(words)
        def bare(w):
            return w.strip().rstrip(".,?!")

        for words in secrets:
            words = [bare(w) for w in words]
            seq = [bare(t[0]) for t in tokens]
            for k in range(len(seq) - len(words) + 1):
                if seq[k:k + len(words)] == words:
                    spans.append((tokens[k][1], tokens[k + len(words) - 1][2]))
        masked = _mask_spans(text, cuts)
        return [{**got, "text": masked}], counts, set(), [sorted(spans)]


def _kw(studio, masker, **extra):
    return dict(synth=studio.synth, transcribe=studio.transcribe, mask_segments=masker,
                silence=studio.silence, measure=studio.measure, classes=ev_gen.CLASSES,
                detector="d1", **extra)


def test_calls_scores_with_fakes_and_resumes(tmp_path, capsys):
    scripts = SCRIPTS[:8]
    studio = FakeStudio()
    kw = _kw(studio, FakeMasker(scripts))
    report = ev_calls.run_calls(scripts, tmp_path, **kw)
    assert report["scripts"] == 8 and report["gate"]["ok"], report["gate"]
    assert report["by_class"]["ssn"]["recall"] == 1.0 and not report["audio_failures"]
    for s in scripts:
        res = json.loads((tmp_path / "results" / f"{s.sid}.json").read_text())
        assert (tmp_path / "results" / f"{s.sid}.raw.md").read_text().startswith("**")
        assert "masked" in res and "audio_failures" in res and res["meta"]["detector"] == "d1"
    assert report["renderings"] and {"class", "spoken", "heard"} <= set(report["renderings"][0])
    before = studio.transcribed
    ev_calls.run_calls(scripts, tmp_path, **kw)
    assert studio.transcribed == before                        # resumed: nothing redone
    ev_calls.run_calls(scripts[:1], tmp_path, redo=True, **kw)
    assert studio.transcribed == before + 2                    # raw + silenced
    assert scripts[0].sid in capsys.readouterr().out


def test_calls_partial_mask_is_a_miss(tmp_path):
    scripts = [_with(lambda i: i.cls == "card")]
    report = ev_calls.run_calls(scripts, tmp_path, **_kw(FakeStudio(), FakeMasker(scripts, "partial")))
    assert report["by_class"]["card"]["masked"] == 0 and report["misses"]
    assert report["misses"][0]["result"] in ("partial", "missed") and not report["gate"]["ok"]


def test_calls_over_wide_mask_is_a_false_positive(tmp_path):
    scripts = [s for s in SCRIPTS if any(i.cls == "card" and i.cue == "after_value" for i in s.items)][:1]
    assert scripts
    report = ev_calls.run_calls(scripts, tmp_path, **_kw(FakeStudio(), FakeMasker(scripts, "wide")))
    assert any(f["kind"] == "over-wide mask" for f in report["false_positives"])
    assert not report["gate"]["ok"]


def test_calls_value_in_another_rendering_is_a_remnant(tmp_path):
    s = _with(lambda i: i.cls == "ssn")
    item = next(i for i in s.items if i.cls == "ssn")
    d = item.token
    studio = FakeStudio(render={item.spoken: f"{d[:3]}, O {d[4]}, {d[5:]}"})   # not the exact token
    report = ev_calls.run_calls([s], tmp_path, **_kw(studio, FakeMasker([s])))
    assert report["garbled"]["ssn"]["n"] == 1 and report["garbled"]["ssn"]["masked"] == 0
    assert report["remnants"] and report["remnants"][0]["class"] == "ssn"


def test_calls_gate_fails_without_evidence(tmp_path):
    s = next(x for x in SCRIPTS if any(i.cls in pii.SECRET for i in x.items))
    studio = FakeStudio(render={i.spoken: "mumble" for i in s.items})
    report = ev_calls.run_calls([s], tmp_path, **_kw(studio, FakeMasker([s])))
    assert report["gate"]["secret_recall"] is None
    assert not report["gate"]["ok"] and report["gate"]["no_evidence"]
    assert report["not_transcribed"] == len(s.items)


def test_calls_short_mute_leaks_in_the_retranscription(tmp_path):
    s = _with(lambda i: i.cls == "card")
    report = ev_calls.run_calls([s], tmp_path, **_kw(FakeStudio(), FakeMasker([s], "short_mute")))
    leaks = [f for f in report["audio_failures"] if f["check"] == "retranscribed"]
    assert "card" in {f["class"] for f in leaks} and not report["gate"]["ok"]


def test_calls_audio_check_fails_when_a_secret_stays_loud(tmp_path):
    s = _with(lambda i: i.cls in ("card", "ssn"))
    kw = _kw(FakeStudio(), FakeMasker([s]))
    kw["silence"] = lambda p, spans, pad=0.15: None
    report = ev_calls.run_calls([s], tmp_path, **kw)
    checks = {f["check"] for f in report["audio_failures"]}
    assert {"rms", "retranscribed"} <= checks and not report["gate"]["ok"]


def test_calls_unreadable_audio_fails(tmp_path):
    s = _with(lambda i: i.cls in ("card", "ssn"))
    kw = _kw(FakeStudio(), FakeMasker([s]))

    def broken(path, a, b):
        raise ValueError("no samples")
    kw["measure"] = broken
    report = ev_calls.run_calls([s], tmp_path, **kw)
    assert any(f["check"] == "unmeasurable" for f in report["audio_failures"])
    assert not report["gate"]["ok"]


def test_calls_without_model_leaves_password_out_of_the_gate(tmp_path):
    scripts = SCRIPTS[:8]
    no_pw = tuple(c for c in ev_gen.CLASSES if c != "password")
    report = ev_calls.run_calls(scripts, tmp_path, model=False,
                          **_kw(FakeStudio(), FakeMasker(scripts, classes=no_pw)))
    assert "password" not in report["gate"]["secret_classes"] and report["gate"]["ok"], report["gate"]


def test_calls_stale_caches_are_rescored_reusing_audio_and_transcript(tmp_path):
    scripts = SCRIPTS[:3]
    studio = FakeStudio()
    ev_calls.run_calls(scripts, tmp_path, **_kw(studio, FakeMasker(scripts)))
    synths, transcribed = studio.synths, studio.transcribed
    # a detector change: rescored, same audio, same first transcription; only the muted copy
    ev_calls.run_calls(scripts, tmp_path, **{**_kw(studio, FakeMasker(scripts)), "detector": "d2"})
    assert studio.synths == synths and studio.transcribed == transcribed + 3
    # other classes: no KeyError, rescored
    kw = _kw(studio, FakeMasker(scripts, classes=("ssn", "card")))
    kw["classes"] = ("ssn", "card")
    report = ev_calls.run_calls(scripts, tmp_path, **kw)
    assert set(report["by_class"]) == {"ssn", "card"} and studio.synths == synths
    # another seed: different scripts under the same ids, so new audio
    other = ev_gen.generate(3, seed=8)
    ev_calls.run_calls(other, tmp_path, seed=8, **_kw(studio, FakeMasker(other)))
    assert studio.synths == synths + 3


# Parakeet's real output for s01 (seed 7) from a sandbox run: digits, "IBANGB73" glued on.
S01_RAW = """**Me:** It was great, really relaxing.

**Them:** How was the trip?

**Me:** I ordered it at Amazon.com.

**Them:** Sure. IBANGB73 HWEC 4532 6581 9231 44.

**Me:** Okay.

**Them:** Sure. My social is 741 05 7024.

**Me:** Not bad. Thanks. You? Put

**Them:** it on my visa. 4111 7962 7393 8077.

**Me:** Okay.

**Them:** That was back in 2019.

**Me:** The counts were 194, 34, 67, 34.

**Them:** Great. Talk soon."""


def test_score_call_on_real_parakeet_output(tmp_path):
    s01 = SCRIPTS[0]
    assert s01.sid == "s01" and {i.kind for i in s01.items} == {"iban", "ssn", "card"}
    masked = S01_RAW.replace("741 05 7024", "[SSN]").replace("4111 7962 7393 8077", "[CARD]")
    after = masked.replace(" [SSN]", "").replace(" [CARD]", "")
    res = ev_score.score_call(s01, S01_RAW, masked, after, ev_gen.CLASSES, spans=[(10.0, 12.0)],
                        reasons=set(), manifest={}, measure=lambda p, a, b: -120.0,
                        silenced=tmp_path / "x")
    got = {i["kind"]: (i["transcribed"], i["result"]) for i in res["items"]}
    assert got == {"iban": (True, "missed"), "ssn": (True, "masked"), "card": (True, "masked")}
    assert not res["false_positives"]
    assert [f["class"] for f in res["audio_failures"]] == ["account"]     # the IBAN is audible
    # the card masked as "4111 [CARD]": a partial, and "4111" stays readable
    half = S01_RAW.replace("7962 7393 8077", "[CARD]").replace("741 05 7024", "[SSN]")
    res = ev_score.score_call(s01, S01_RAW, half, after, ev_gen.CLASSES, spans=[], reasons=set(),
                        manifest={}, measure=lambda p, a, b: -120.0, silenced=tmp_path / "x")
    assert {i["kind"]: i["result"] for i in res["items"]}["card"] == "partial"


def _mini(lines, item, distractor=None):
    """A hand-made one-item script: `item` is (cls, token, written, spoken), placed in `lines`."""
    text = "\n\n".join(f"**{sp}:** {t}" for sp, t in lines)
    cls, token, written, spoken = item
    at = text.index(spoken)
    it = ev_gen.Item(cls, cls, written, spoken, token, "digits", "same_sentence", "Them",
                     at, at + len(spoken))
    ds = []
    if distractor:
        kind, phrase = distractor
        d_at = text.index(phrase)
        ds = [ev_gen.Distractor(kind, phrase, d_at, d_at + len(phrase))]
    return ev_gen.Script("m01", list(lines), [it], ds, text)


def _call(script, raw, masked, after, tmp_path):
    return ev_score.score_call(script, raw, masked, after, ev_gen.CLASSES, spans=[], reasons=set(),
                               manifest={}, measure=lambda p, a, b: -120.0,
                               silenced=tmp_path / "x")


SSN_219 = ("ssn", "219099999", "219-09-9999", "two one nine oh nine nine nine nine nine")


def test_audio_check_covers_a_secret_parakeet_garbled(tmp_path):
    # The probe: raw "219 09 999" (a digit lost, so the value is not located), masked [SSN],
    # and the muted copy's transcript still says "219 09 999".
    s = _mini([("Me", "Hi."), ("Them", f"Sure. My social is {SSN_219[3]}.")], SSN_219)
    raw = "**Me:** Hi.\n\n**Them:** Sure. My social is 219 09 999."
    masked = "**Me:** Hi.\n\n**Them:** Sure. My social is [SSN]."
    res = _call(s, raw, masked, raw, tmp_path)
    assert not res["items"][0]["transcribed"]
    assert [f["check"] for f in res["audio_failures"]] == ["retranscribed"]
    report = ev_score.aggregate_calls([res], ev_gen.CLASSES)
    assert not report["gate"]["ok"]
    # muted for real: nothing comes back
    after = "**Me:** Hi.\n\n**Them:** Sure. My social is."
    assert not _call(s, raw, masked, after, tmp_path)["audio_failures"]


def test_a_distractor_excuses_its_own_digits_once(tmp_path):
    # The count "one two three four five" and the SSN 123-45-6789; Parakeet wrote only "one two
    # three four five" for the SSN and nothing was masked: 5 of 9 digits are readable.
    s = _mini([("Me", "Okay, one two three four five, go."),
               ("Them", "Sure, my social is one two three four five six seven eight nine.")],
              ("ssn", "123456789", "123-45-6789", "one two three four five six seven eight nine"),
              distractor=("count", "one two three four five"))
    raw = "**Me:** Okay, one two three four five, go.\n\n**Them:** Sure, one two three four five."
    res = _call(s, raw, raw, raw, tmp_path)
    assert res["items"][0]["result"] == "REMNANT", res["items"][0]
    # the distractor alone, once: explained, no remnant
    alone = "**Me:** Okay, one two three four five, go.\n\n**Them:** Sure, [SSN]."
    res = _call(s, raw.replace("Sure, one two three four five.", "Sure, [SSN]."), alone, alone,
                tmp_path)
    assert res["items"][0]["result"] == "no remnant"


def test_another_items_value_does_not_excuse_a_remnant():
    s = _mini([("Me", "Card four two four two."), ("Them", f"My social is {SSN_219[3]}.")], SSN_219)
    other = ev_gen.Item("card", "card", "4242", "four two four two", "4242", "digits", "none", "Me",
                        s.transcript.index("four two"), s.transcript.index("four two") + 17)
    s.items.append(other)
    elsewhere = ev_score.elsewhere_in_script(s)
    assert not any("4242" in e for e in elsewhere) and not any("219099" in e for e in elsewhere)


def test_garbled_excuse_covers_the_value_not_its_neighbour(tmp_path):
    s = _mini([("Me", "Hi."), ("Them", f"Sure. {SSN_219[3]}, nine one one.")], SSN_219,
              distractor=("nine_one_one", "nine one one"))
    raw = "**Me:** Hi.\n\n**Them:** Sure. 219 09 998, 911."          # garbled: not located
    masked = "**Me:** Hi.\n\n**Them:** Sure. [SSN]."
    res = _call(s, raw, masked, masked, tmp_path)
    assert not res["items"][0]["transcribed"]
    assert res["garbled_masked"] and res["false_positives"]
    fp = res["false_positives"][0]
    assert fp["kind"] == "beyond a garbled value" and fp["leftover"] and fp["distractor"] == "nine_one_one"
    report = ev_score.aggregate_calls([res], ev_gen.CLASSES)
    assert report["distractors_damaged"] == 1 and report["gate"]["false_positives"] == 1


def test_calls_false_positive_off_any_distractor_is_not_damage(tmp_path, capsys):
    s = _mini([("Me", "Hello there."), ("Them", f"Sure. {SSN_219[3]}.")], SSN_219)
    raw = "**Me:** Hello there.\n\n**Them:** Sure. 219-09-9999."
    masked = "**Me:** [EMAIL] there.\n\n**Them:** Sure. [SSN]."
    res = _call(s, raw, masked, masked, tmp_path)
    report = ev_score.aggregate_calls([res], ev_gen.CLASSES)
    assert report["gate"]["false_positives"] == 1 and report["distractors_damaged"] == 0
    ev_score.print_calls_report(report)
    assert "false positives: 1; distractors damaged: 0" in capsys.readouterr().out


def _result(items, fps=()):
    return {"script": "m01", "items": items, "false_positives": list(fps), "audio_failures": [],
            "renderings": [], "reasons": [], "garbled_masked": []}


def _it(cls, cue, masked, transcribed=True):
    return {"class": cls, "kind": cls, "style": "digits", "cue": cue, "transcribed": transcribed,
            "masked": masked, "result": "masked" if masked else "missed", "remnant": None,
            "spoken": "x", "heard": "x", "in_classes": True}


def test_calls_gate_is_on_cued_items_only():
    items = [_it("ssn", "same_sentence", True), _it("card", "previous_turn", True),
             _it("account", "none", False), _it("card", "none", False)]
    got = ev_score.aggregate_calls([_result(items)], ev_gen.CLASSES)
    assert got["gate"]["ok"] and got["shape_only"] == {"n": 2, "masked": 0, "recall": 0.0}
    got = ev_score.aggregate_calls([_result(items + [_it("ssn", "after_value", False)])],
                                   ev_gen.CLASSES)
    assert not got["gate"]["ok"]
    only_none = ev_score.aggregate_calls([_result([_it("ssn", "none", True)])], ev_gen.CLASSES)
    assert not only_none["gate"]["ok"] and only_none["gate"]["no_evidence"]


def test_calls_ssn_shaped_triple_damage_is_reported_not_gated(tmp_path):
    s = _mini([("Me", "We had 123 45 6789 users."), ("Them", f"My social is {SSN_219[3]}.")],
              SSN_219, distractor=(ev_gen.SSN_TRIPLE, "We had 123 45 6789 users."))
    raw = "**Me:** We had 123 45 6789 users.\n\n**Them:** My social is 219-09-9999."
    masked = "**Me:** We had [SSN] users.\n\n**Them:** My social is [SSN]."
    res = _call(s, raw, masked, masked, tmp_path)
    assert not res["false_positives"] and res["ssn_shaped_triples"] == 1
    got = ev_score.aggregate_calls([res], ev_gen.CLASSES)
    assert got["ssn_shaped_triples"] == 1 and got["gate"]["false_positives"] == 0
    assert got["gate"]["ok"], got["gate"]


@pytest.mark.parametrize("raw_tail, must_leave", [
    ("Sure. 219 09 998. I paid 99.", ["Ipaid99"]),          # past a sentence end
    # "21 99" is under half the value's length: nothing here reads as it, nothing is excused
    ("Sure. Room 21 99 is mine.", ["Room", "2199", "ismine"]),
])
def test_garbled_excuse_stops_at_the_sentence_and_the_stretch(tmp_path, raw_tail, must_leave):
    s = _mini([("Me", "Hi."), ("Them", f"Sure. {SSN_219[3]}.")], SSN_219)
    raw = f"**Me:** Hi.\n\n**Them:** {raw_tail}"
    masked = "**Me:** Hi.\n\n**Them:** Sure. [SSN]."
    res = _call(s, raw, masked, masked, tmp_path)
    assert res["false_positives"], res
    assert all(m in res["false_positives"][0]["leftover"] for m in must_leave)


def test_garbled_excuse_never_crosses_a_turn_into_a_distractor(tmp_path):
    s = _mini([("Them", f"Sure. {SSN_219[3]}."), ("Me", "It was only four ninety nine.")], SSN_219,
              distractor=("price", "four ninety nine"))
    raw = "**Them:** Sure. 219 09 99.\n\n**Me:** It was only 4.99."
    masked = "**Them:** Sure. [SSN]."
    res = _call(s, raw, masked, masked, tmp_path)
    fp = res["false_positives"][0]
    assert "price" in fp["distractors"] and "Itwasonly499" in fp["leftover"]
    report = ev_score.aggregate_calls([res], ev_gen.CLASSES)
    assert report["distractors_damaged"] >= 1 and not report["gate"]["ok"]


TRIPLE_LINE = "We had 123 45 6789 users."


def test_text_triple_exemption_is_its_nine_digits_only():
    lines = [("Them", f"My social is {SSN_219[3]}."), ("Me", TRIPLE_LINE), ("Them", "Great, talk soon.")]
    s = _mini(lines, SSN_219, distractor=(ev_gen.SSN_TRIPLE, TRIPLE_LINE))
    t, item = s.transcript, s.items[0]
    num = t.index("123 45 6789")
    cases = {
        "digits only": [(item.start, item.end, "ssn"), (num, num + 11, "ssn")],
        "into the next turn": [(item.start, item.end, "ssn"), (num, len(t) - 1, "ssn")],
        "from the item through it": [(item.start, len(t) - 1, "ssn")],
    }
    got = {k: ev_score.score_text([s], {s.sid: _mask_spans(t, v)}, ev_gen.CLASSES, model=True)
           for k, v in cases.items()}
    assert got["digits only"]["gate"]["ok"] and got["digits only"]["ssn_shaped_triples"] == 1
    for k in ("into the next turn", "from the item through it"):
        assert not got[k]["gate"]["ok"] and got[k]["false_positives"], k
        assert "talk" in got[k]["false_positives"][0]["leftover"], k
    assert ev_gen.SSN_TRIPLE in got["from the item through it"]["false_positives"][0]["distractors"]


def test_calls_one_mask_over_two_distractors_damages_both(tmp_path):
    counts = "The counts were 1234, 5678, 2468, 1357."
    s = _mini([("Them", f"My social is {SSN_219[3]}."), ("Me", TRIPLE_LINE), ("Them", counts)],
              SSN_219, distractor=(ev_gen.SSN_TRIPLE, TRIPLE_LINE))
    at = s.transcript.index(counts)
    s.distractors.append(ev_gen.Distractor("plain_counts", counts, at, at + len(counts)))
    raw = f"**Them:** My social is 219-09-9999.\n\n**Me:** {TRIPLE_LINE}\n\n**Them:** {counts}"
    masked = "**Them:** My social is [SSN].\n\n**Me:** We had [SSN]."
    res = _call(s, raw, masked, masked, tmp_path)
    fp = res["false_positives"][0]
    assert set(fp["distractors"]) == {ev_gen.SSN_TRIPLE, "plain_counts"}
    report = ev_score.aggregate_calls([res], ev_gen.CLASSES)
    assert report["distractors_damaged"] == 2 and not report["gate"]["ok"]


def _windows(text):
    """Every stretch of consecutive digit groups in the normalised text, as (groups, digits)."""
    norm = pii.normalise(text)[0]
    for run in re.finditer(r"\d+(?:[ ,\-]+\d+)*", norm):
        groups = re.findall(r"\d+", run.group())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups) + 1):
                yield tuple(len(g) for g in groups[i:j]), "".join(groups[i:j])


def test_no_distractor_is_a_valid_secret_over_200_seeds():
    bad = []
    for seed in range(200):
        for s in ev_gen.generate(30, seed=seed):
            for d in s.distractors:
                if d.kind in ev_gen.TRIPLE_KINDS:
                    continue
                for shape, digits in _windows(d.text):
                    if (12 <= len(digits) <= 19 and pii._luhn(digits)
                            or len(digits) == 9 and pii._aba_ok(digits)
                            or shape == (3, 2, 4) and pii._ssn_ok(digits)):
                        bad.append((seed, s.sid, d.kind, d.text))
    assert not bad, bad[:5]


def test_with_phone_masked_no_distractor_is_phone_shaped():
    bad = [(seed, d.kind, d.text) for seed in range(200)
           for s in ev_gen.generate(30, seed=seed, phone=True) for d in s.distractors
           if pii._PHONE.search(pii.normalise(d.text)[0])]
    assert not bad, bad[:5]
    assert ev_gen.generate(30, seed=7, phone=True) == ev_gen.generate(30, seed=7, phone=True)


def test_a_verbatim_value_cannot_excuse_a_look_alike_distractor(tmp_path):
    # seed 28 s16: SSN 255-53-0179 written verbatim; "Call (563) 555-0112 about the account
    # number." with its number masked read >= 0.6 like the SSN and was excused.
    s = next(x for x in ev_gen.generate(30, seed=28) if x.sid == "s16")
    item = s.items[0]
    d = next(d for d in s.distractors if d.kind == "account_call")
    assert item.token == "255530179" and "555-0112" in d.text
    raw = f"**Them:** My social is 255-53-0179.\n\n**Me:** {d.text}"
    masked = "**Them:** My social is [SSN].\n\n**Me:** Call (563) [ACCOUNT] about the account number."
    mini = ev_gen.Script("s16", [("Them", "x"), ("Me", d.text)], [item], [d], s.transcript)
    res = _call(mini, raw, masked, masked, tmp_path)
    assert res["items"][0]["transcribed"] and not res["garbled_masked"]
    assert res["false_positives"] and res["false_positives"][0]["distractors"] == ["account_call"]


def test_a_garbled_value_excuses_only_inside_its_own_turn(tmp_path):
    s = _mini([("Them", f"My social is {SSN_219[3]}."), ("Me", "Order 219 09 998, thanks.")],
              SSN_219, distractor=("plain_order", "Order 219 09 998, thanks."))
    raw = "**Them:** My social is 219 09 99.\n\n**Me:** Order 219 09 998, thanks."
    in_turn = "**Them:** My social is [SSN].\n\n**Me:** Order 219 09 998, thanks."
    res = _call(s, raw, in_turn, in_turn, tmp_path)
    assert res["garbled_masked"] and not res["false_positives"]
    other_turn = "**Them:** My social is 219 09 99.\n\n**Me:** Order [SSN], thanks."
    res = _call(s, raw, other_turn, other_turn, tmp_path)
    assert not res["garbled_masked"] and res["false_positives"][0]["distractors"] == ["plain_order"]


def test_a_rewrite_that_is_not_a_placeholder_substitution_fails(tmp_path):
    rewrite = {x.sid: re.sub(r"\[[A-Z]+\]", "XXXX", _perfect(x)) for x in SCRIPTS}
    got = ev_score.score_text(SCRIPTS, rewrite, ev_gen.CLASSES, model=True)
    assert got["errors"] and got["errors"][0]["error"] == ev_score.NOT_SUBSTITUTION
    assert not got["gate"]["ok"] and got["gate"]["secret_recall"] == 0.0
    s = _mini([("Me", "Hi."), ("Them", f"My social is {SSN_219[3]}.")], SSN_219)
    raw = "**Me:** Hi.\n\n**Them:** My social is 219-09-9999."
    res = _call(s, raw, raw.replace("219-09-9999", "XXXX"), raw, tmp_path)
    assert res["error"] == ev_score.NOT_SUBSTITUTION and not res["items"][0]["masked"]
    report = ev_score.aggregate_calls([res], ev_gen.CLASSES)
    assert not report["gate"]["ok"] and report["gate"]["errors"] == 1
    with pytest.raises(ev_score.NotSubstitution):
        ev_score.replaced(raw, raw.replace("[", "").replace("Hi", "Ho"))


def test_check_full_and_full_runner_with_fakes(tmp_path):
    import sqlite3
    from types import SimpleNamespace

    s = _with(lambda i: i.cls == "card")
    card = next(i for i in s.items if i.cls == "card")
    studio = FakeStudio()
    vault = tmp_path / "vault"
    db = tmp_path / "slim.db"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE sources (id TEXT, path TEXT, deleted INTEGER DEFAULT 0);"
                      "CREATE TABLE fragments (id INTEGER PRIMARY KEY, source_id TEXT, text TEXT);")
    con.commit()
    con.close()

    def make_pipeline(mute: bool, leak_title: bool):
        pl = SimpleNamespace()

        def silence_segments(segments, seg_spans):
            if mute:
                for seg, sp in zip(segments, seg_spans):
                    studio.silence(seg, sp)
            return True

        def handle_record(*, audio_rel, notes_md, declared_type, title, vault):
            seg = vault / audio_rel
            got = studio.transcribe(seg)
            (masked,), counts, reasons, spans = FakeMasker([s])([got])
            pl._silence_segments([seg], spans)
            archived = vault / "Attachments" / "Recordings" / "a.webm"
            archived.parent.mkdir(parents=True, exist_ok=True)
            seg.replace(archived)
            note = vault / "Notes" / "work" / "n.md"
            note.parent.mkdir(parents=True, exist_ok=True)
            listed = ", ".join(f"{c}:{n}" for c, n in sorted(counts.items()))
            note.write_text(f"---\npii_masked: [{listed}]\npii_by: {pii.VERSION}\n---\n\n"
                            f"## Transcript\n\n{masked['text']}\n")
            c = sqlite3.connect(db)
            c.execute("INSERT INTO sources VALUES ('x', 'Notes/work/n.md', 0)")
            c.execute("INSERT INTO fragments (source_id, text) VALUES ('x', ?)", (masked["text"],))
            c.commit()
            c.close()
            return {"note": "Notes/work/n.md", "audio": "Attachments/Recordings/a.webm",
                    "title": card.spoken if leak_title else "A call", "summary": "",
                    "pii": {"masked": counts, "partial": []}}

        pl._silence_segments = silence_segments
        pl.handle_record = handle_record
        return pl

    config = SimpleNamespace(VAULT=vault, DB_PATH=db, PII_MASK=ev_gen.CLASSES)
    audio = tmp_path / "audio.webm"
    studio.synth(s, audio)
    work = tmp_path / "work.webm"
    work.write_text(audio.read_text())
    tokens = {i.token for i in s.items}

    ok = ev_calls._full_runner(config, pipeline=make_pipeline(True, False), measure=studio.measure)(
        s, audio, work, tokens)
    assert ok["ok"], ok["problems"]
    bad = ev_calls._full_runner(config, pipeline=make_pipeline(False, True), measure=studio.measure)(
        s, audio, work, tokens)
    assert not bad["ok"]
    assert any("title carries card" in p for p in bad["problems"])
    assert any("archived audio" in p for p in bad["problems"])


# --- public ----------------------------------------------------------------------------

FINANCE_PAGE = {
    "rows": [
        {"row": {"language": "English",
                 "generated_text": "Send to jo.ann@example.com, SSN 219-09-9999, card 4111 1111 1111 1111, "
                                   "bad card 4111 1111 1111 1112, IBAN GB82 WEST 1234 5698 7654 32.",
                 "pii_spans": json.dumps([
                     {"start": 8, "end": 26, "label": "email"},
                     {"start": 32, "end": 43, "label": "ssn"},
                     {"start": 50, "end": 69, "label": "credit_card_number"},
                     {"start": 80, "end": 99, "label": "credit_card_number"},
                     {"start": 106, "end": 133, "label": "iban"},
                 ])}},
        {"row": {"language": "German", "generated_text": "x", "pii_spans": "[]"}},
    ],
}
EN_V1_PAGE = {
    "rows": [
        {"row": {"text": "Reach me at 12 Quorrel Street or kim@example.org, not 1234 or 7234.",
                 "entities": "[{'entity': '12 Quorrel Street', 'types': ['street_address']}, "
                             "{'entity': 'kim@example.org', 'types': ['email']}, "
                             "{'entity': '234', 'types': ['account_number']}]"}},
    ],
}


class Hit:
    def __init__(self, start, end, cls):
        self.start, self.end, self.cls = start, end, cls


def _hits_at(text, pieces):
    """A fake detector: the given (substring, class) pairs, wherever they occur."""
    def find(t):
        return [Hit(m.start(), m.end(), cls) for piece, cls in pieces
                for m in re.finditer(re.escape(piece), t)]
    return find


def test_adapters_map_labels_onto_our_classes():
    docs = [d for r in FINANCE_PAGE["rows"]
            if (d := ev_public.ADAPTERS["gretelai/synthetic_pii_finance_multilingual"](r["row"]))]
    assert len(docs) == 1
    text, gold = docs[0]
    assert {g[2] for g in gold} == {"email", "ssn", "card", "account"}
    assert text[gold[1][0]:gold[1][1]] == "219-09-9999"
    text, gold = ev_public.ADAPTERS["gretelai/gretel-pii-masking-en-v1"](EN_V1_PAGE["rows"][0]["row"])
    # "234" is only ever inside longer numbers here: never tagged
    assert [(text[a:b], c) for a, b, c, _ in gold] == [("12 Quorrel Street", "address"),
                                                      ("kim@example.org", "email")]


def test_score_public_splits_invalid_gold_from_true_misses():
    text, gold = ev_public.ADAPTERS["gretelai/synthetic_pii_finance_multilingual"](FINANCE_PAGE["rows"][0]["row"])
    find = _hits_at(text, [("219-09-9999", "ssn"), ("4111 1111 1111 1111", "card"),
                           ("jo.ann@example.com", "email"), ("4111 1111 1111 1112", "card")])
    got = ev_public.score_public([(text, gold)], find)
    card = got["by_class"]["card"]
    # the invalid card was "found" too: valid recall stays a ratio over valid values
    assert card["gold"] == 2 and card["found"] == 2 and card["invalid"] == 1
    assert card["true_miss"] == 0 and card["recall_valid"] == 1.0
    assert got["by_class"]["account"]["true_miss"] == 1
    assert got["by_class"]["email"]["precision"] == 1.0


def test_valid_recall_never_exceeds_one():
    text = "a 4111 1111 1111 1112 b 4111 1111 1111 1111"
    gold = [(2, 21, "card", "credit_card_number"), (24, 43, "card", "credit_card_number")]
    got = ev_public.score_public([(text, gold)], _hits_at(text, [("4111 1111 1111 1112", "card")]))
    card = got["by_class"]["card"]
    assert card["found"] == 1 and card["recall_valid"] == 0.0 <= 1


def test_licence_check_and_cached_fetch(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "api/datasets/" in url:
            return {"cardData": {"license": "apache-2.0" if "gretel" in url else "other"}}
        return FINANCE_PAGE if "finance" in url else EN_V1_PAGE

    assert ev_public.licence("gretelai/x", fetch) == ("apache-2.0", True)
    assert ev_public.licence("ai4privacy/pii-masking-300k", fetch) == ("other", False)
    find = _hits_at("", [])
    report = ev_public.run_public(tmp_path, rows=10, fetch=fetch, find=find, log=lambda *a: None)
    assert report["datasets"]["ai4privacy/pii-masking-300k"]["used"] is False
    assert report["datasets"]["gretelai/synthetic_pii_finance_multilingual"]["used"] is True
    n = len(calls)
    ev_public.run_public(tmp_path, rows=10, fetch=fetch, find=find, log=lambda *a: None)
    assert all("datasets-server" not in u for u in calls[n:])          # pages came from cache
    assert any(tmp_path.joinpath("public").rglob("*.json"))


def test_public_error_body_is_not_cached(tmp_path):
    def fetch(url):
        if "api/datasets/" in url:
            return {"cardData": {"license": "apache-2.0"}}
        return {"error": "The dataset viewer is not available"}
    report = ev_public.run_public(tmp_path, rows=10, fetch=fetch, find=lambda t: [], log=lambda *a: None)
    d = report["datasets"]["gretelai/synthetic_pii_finance_multilingual"]
    assert "no rows" in d["error"] and not any(tmp_path.joinpath("public").rglob("*.json"))


def test_public_without_network_says_so(tmp_path):
    def fetch(url):
        raise OSError("network down")
    lines = []
    report = ev_public.run_public(tmp_path, rows=10, fetch=fetch, find=lambda t: [],
                           log=lambda *a: lines.append(" ".join(map(str, a))))
    assert report["network"] is False and any("network" in l for l in lines)


def test_public_http_error_is_not_a_network_outage(tmp_path):
    class Limited(OSError):
        code = 429

    def fetch(url):
        if "api/datasets/" in url:
            return {"cardData": {"license": "apache-2.0"}}
        raise Limited("too many requests")
    report = ev_public.run_public(tmp_path, rows=10, fetch=fetch, find=lambda t: [], log=lambda *a: None)
    d = report["datasets"]["gretelai/synthetic_pii_finance_multilingual"]
    assert report["network"] is True and "HTTP 429" in d["error"]


def test_public_counts_letters_and_digits_and_rejects_slots():
    text = "mail [kim@example.org] or [Shipper's Email Address] now"
    a = text.index("[kim")
    gold = [(a, a + len("[kim@example.org]"), "email", "email"),
            (text.index("[Ship"), text.index(" now"), "email", "email")]
    got = ev_public.score_public([(text, gold)], _hits_at(text, [("kim@example.org", "email")]))
    got = got["by_class"]["email"]
    assert got["found"] == 1 and got["invalid"] == 1 and got["true_miss"] == 0


# --- sandbox guard ---------------------------------------------------------------------

def _snapshot(root: Path) -> dict:
    """Every path under root with its kind, size and mtime: a refusal must change none."""
    return {str(p): (p.is_dir(), p.lstat().st_size, p.lstat().st_mtime_ns)
            for p in [root, *root.rglob("*")]}


def _guard(out, tmp_path, discovered=None):
    return ev_sandbox.guard(out, home=tmp_path / "home", discovered=lambda: discovered, env={})


@pytest.mark.parametrize("where", ["icloud", "brain", "git", "obsidian", "obsidian_part", "discovered"])
def test_guard_refuses_and_touches_nothing(tmp_path, where):
    home = tmp_path / "home"
    places = {
        "icloud": home / "Library" / "Mobile Documents" / "x",
        "brain": home / "Library" / "Application Support" / "slim" / "x",
        "git": tmp_path / "checkout" / "deep" / "x",
        "obsidian": tmp_path / "v" / "sub" / "x",
        "obsidian_part": tmp_path / "w" / ".obsidian" / "x",
        "discovered": tmp_path / "real-vault" / "x",
    }
    (tmp_path / "checkout" / ".git").mkdir(parents=True)
    (tmp_path / "v" / ".obsidian").mkdir(parents=True)
    (tmp_path / "w" / ".obsidian").mkdir(parents=True)
    (tmp_path / "real-vault").mkdir()
    for p in places.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit):
        _guard(places[where], tmp_path, discovered=(tmp_path / "real-vault").resolve())
    assert _snapshot(tmp_path) == before


def test_guard_refuses_another_checkouts_config_and_keeps_its_projects_yaml(tmp_path):
    checkout = tmp_path / "other"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "config").mkdir()
    real = checkout / "config" / "projects.yaml"
    real.write_text('owner: "Someone"\n')
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit):
        ev_sandbox.sandbox(checkout / "config", home=tmp_path / "home", discovered=lambda: None,
                           env={})
    assert _snapshot(tmp_path) == before and real.read_text() == 'owner: "Someone"\n'
    # the same folder outside any git tree: refused for having no marker
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "projects.yaml").write_text('owner: "Someone"\n')
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="not a pii_eval output directory"):
        _guard(plain, tmp_path)
    assert _snapshot(tmp_path) == before


def test_guard_follows_a_vault_symlink_into_an_obsidian_vault(tmp_path):
    target = tmp_path / "somebodys-vault"
    (target / ".obsidian").mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    (out / ev_sandbox.MARKER).write_text("x")
    (out / "vault").symlink_to(target)
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="vault"):
        _guard(out, tmp_path)
    assert _snapshot(tmp_path) == before and sorted(p.name for p in target.iterdir()) == [".obsidian"]


def test_guard_expands_the_home_once_and_returns_the_one_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    before = _snapshot(tmp_path)
    got = ev_sandbox.guard("~/eval", home=tmp_path / "home", discovered=lambda: None, env={})
    assert got == (tmp_path / "home" / "eval").resolve() and _snapshot(tmp_path) == before


def test_guard_accepts_its_own_directory_and_refuses_a_foreign_projects_yaml(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / ev_sandbox.MARKER).write_text("x")
    ev_sandbox.write_projects(out / "projects.yaml", None)
    assert _guard(out, tmp_path) == out.resolve()
    (out / "projects.yaml").write_text('owner: "Someone"\n')
    with pytest.raises(SystemExit, match="projects.yaml"):
        _guard(out, tmp_path)


def test_sandbox_refuses_when_config_resolved_elsewhere_before_writing(tmp_path, monkeypatch):
    # In this process slim.config is already imported with other paths: refuse, write nothing.
    for name in ("SLIM_VAULT", "SLIM_DATA_DIR", "SLIM_PROJECTS"):
        monkeypatch.delenv(name, raising=False)           # restored after the test
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="imported before"):
        ev_sandbox.sandbox(tmp_path / "out", home=tmp_path / "home", discovered=lambda: None,
                           env={})
    assert _snapshot(tmp_path) == before and "SLIM_VAULT" not in __import__("os").environ


def test_sandbox_projects_yaml_is_generic(tmp_path):
    ev_sandbox.write_projects(tmp_path / "projects.yaml", ev_gen.CLASSES)
    text = (tmp_path / "projects.yaml").read_text()
    assert text.startswith(ev_sandbox.PROJECTS_HEADER) and 'owner: "Alex"' in text and "phone" in text


def test_discovered_vault_loads_vaultpath_without_importing_slim(tmp_path):
    import subprocess
    import sys
    code = ("import sys; from scripts.pii_eval import sandbox as s; v = s.discovered_vault(); "
            "print(v); print(any(m == 'slim' or m.startswith('slim.') for m in sys.modules))")
    env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("SLIM_")}
    env["SLIM_VAULT"] = str(tmp_path / "v")
    out = subprocess.run([sys.executable, "-c", code], cwd=ev_sandbox.REPO, env=env,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[0] == str((tmp_path / "v").resolve()) and out[1] == "False"


def _plant(tmp_path):
    """A marked --out, and somebody's vault with a note in it."""
    out = tmp_path / "out"
    (out / "vault").mkdir(parents=True)
    (out / ev_sandbox.MARKER).write_text("x")
    target = tmp_path / "somebodys-vault"
    (target / ".obsidian").mkdir(parents=True)
    (target / "Notes").mkdir()
    (target / "Notes" / "note.md").write_text("theirs\n")
    return out, target


@pytest.mark.parametrize("planted", ["notes_dir", "scripts_json"])
def test_guard_refuses_a_symlink_anywhere_below_out(tmp_path, planted):
    out, target = _plant(tmp_path)
    if planted == "notes_dir":
        (out / "vault" / "Notes").symlink_to(target / "Notes")
    else:
        (out / "scripts.json").symlink_to(target / "Notes" / "note.md")
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="contains a symlink"):
        _guard(out, tmp_path)
    assert _snapshot(tmp_path) == before
    assert (target / "Notes" / "note.md").read_text() == "theirs\n"


def test_guard_refuses_a_hard_link_below_out(tmp_path):
    import os
    out, target = _plant(tmp_path)
    note = target / "Notes" / "note.md"
    os.link(note, out / "scripts.json")
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit, match="hard-linked"):
        _guard(out, tmp_path)
    assert _snapshot(tmp_path) == before and note.read_bytes() == b"theirs\n"


@pytest.mark.parametrize("where", ["home", "home_parent", "desktop", "documents", "no_parent",
                                   "caller_data", "caller_vault_itself"])
def test_guard_refuses_home_icloud_folders_typos_and_caller_paths(tmp_path, where):
    home = tmp_path / "home"
    for d in ("Desktop", "Documents"):
        (home / d).mkdir(parents=True)
    (tmp_path / "data-they-set").mkdir()
    (tmp_path / "vault-they-set").mkdir()
    env = {"SLIM_DATA_DIR": str(tmp_path / "data-they-set"),
           "SLIM_VAULT": str(tmp_path / "vault-they-set")}
    out = {"home": home, "home_parent": tmp_path, "desktop": home / "Desktop" / "eval",
           "documents": home / "Documents" / "eval", "no_parent": tmp_path / "typo" / "deeper",
           "caller_data": tmp_path / "data-they-set" / "eval",
           "caller_vault_itself": tmp_path / "vault-they-set"}[where]
    before = _snapshot(tmp_path)
    with pytest.raises(SystemExit):
        ev_sandbox.guard(out, home=home, discovered=lambda: None, env=env)
    assert _snapshot(tmp_path) == before


def test_cli_prints_the_resolved_out_before_creating_it(tmp_path):
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLIM_")}
    out = tmp_path / "a" / ".." / "eval"
    run = subprocess.run([sys.executable, "-m", "scripts.pii_eval", "text", "--no-model", "--n", "2",
                          "--out", str(out)], cwd=ev_sandbox.REPO, env=env, capture_output=True,
                         text=True)
    assert run.stdout.splitlines()[0] == f"--out {(tmp_path / 'eval').resolve()}", run.stderr
    assert (tmp_path / "eval" / ev_sandbox.MARKER).is_file() and not (tmp_path / "a").exists()
