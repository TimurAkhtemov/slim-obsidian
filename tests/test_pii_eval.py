"""The PII evaluation script's pure parts: generator, scorers, sandbox guard, and every mode
driven with fakes. No model, no `say`, no network."""

import json
import random
import re
from pathlib import Path

import pytest

from scripts import pii_eval as ev
from slim.capture import pii
from slim.recorder import pipeline


def _digits(s):
    return re.sub(r"\D", "", s)


def _norm_digits(spoken):
    return _digits(pii.normalise(spoken)[0])


NUMERIC = {"ssn", "card", "phone"}


@pytest.mark.parametrize("cls", ev.CLASSES)
def test_values_pass_validators_and_round_trip(cls):
    rng = random.Random(1)
    for _ in range(300):
        v = ev.gen_value(cls, rng)
        d = _digits(v.written)
        if cls == "ssn":
            assert len(d) == 9 and pii._ssn_ok(d)
        elif cls == "card":
            assert pii._luhn(d) and d[:4] in ev.TEST_BINS
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
        assert ev.present(v.token, v.spoken), (v.token, v.spoken)
        assert ev.present(v.token, v.written), (v.token, v.written)


def test_spoken_styles_cover_what_people_say():
    rng = random.Random(3)
    spoken = [ev.gen_value(c, rng).spoken for c in ev.CLASSES for _ in range(200)]
    joined = " | ".join(spoken)
    for phrase in (" oh ", " zero ", "double ", " dash ", " dot ", " at example dot "):
        assert phrase in joined, phrase
    assert re.search(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|eleven|twelve)\b", joined)
    assert re.search(r"\b[a-z] [a-z] [a-z] [a-z]\b", joined)          # a spelled-out password
    styles = {ev.gen_value("ssn", rng).style for _ in range(200)}
    assert {"digits", "grouped", "double"} <= styles


def test_generator_is_seeded_and_deterministic():
    a, b, c = ev.generate(30, seed=7), ev.generate(30, seed=7), ev.generate(30, seed=8)
    assert [s.transcript for s in a] == [s.transcript for s in b]
    assert [s.transcript for s in a] != [s.transcript for s in c]


def test_scripts_shape():
    scripts = ev.generate(30, seed=7)
    assert len(scripts) == 30 and len({s.sid for s in scripts}) == 30
    assert sum(not s.items for s in scripts) >= 5
    cells = {(i.cls, i.cue) for s in scripts for i in s.items}
    assert cells == {(c, q) for c in ev.CLASSES for q in ev.CUES}      # every class meets every cue
    for s in scripts:
        assert 0 <= len(s.items) <= 3 and 2 <= len(s.distractors) <= 4
        assert {sp for sp, _ in s.lines} == {"Me", "Them"}
        t = s.transcript
        assert t.startswith("**") and "**Me:**" in t and "**Them:**" in t
        for x in [*s.items, *s.distractors]:
            assert t[x.start:x.end] == (x.spoken if hasattr(x, "spoken") else x.text)


def test_tokens_are_distinctive():
    for s in ev.generate(30, seed=7):
        t = s.transcript
        for i in s.items:
            assert ev.present(i.token, t)
            gone = t[:i.start] + "[X]" + t[i.end:]
            assert not ev.present(i.token, gone), (s.sid, i.token)


def test_previous_turn_answer_carries_no_cue():
    seen = 0
    for s in ev.generate(30, seed=7):
        for i in s.items:
            if i.cue != "previous_turn":
                continue
            seen += 1
            turn_start = s.transcript.rfind("**", 0, i.start)
            answer = s.transcript[turn_start:s.transcript.find("\n\n", i.end) % (len(s.transcript) + 1)]
            assert not ev.CUE_WORDS[i.cls].search(answer.replace(i.spoken, "")), answer
            before = s.transcript[:turn_start]
            assert ev.CUE_WORDS[i.cls].search(before[before.rfind("**Me:**") if i.speaker == "Them"
                                                       else before.rfind("**Them:**"):])
    assert seen


def test_replaced_finds_masked_ranges():
    orig = "**Me:** my social is two one nine. ok."
    masked = "**Me:** my social is [SSN]. ok."
    assert ev.replaced(orig, masked) == [(orig.index("two"), orig.index(". ok"))]
    assert ev.replaced(orig, orig) == []


def _perfect(script):
    t, out, pos = script.transcript, [], 0
    for i in sorted(script.items, key=lambda i: i.start):
        out += [t[pos:i.start], pii.PLACEHOLDER[i.cls]]
        pos = i.end
    return "".join(out + [t[pos:]])


def test_score_text_perfect_identity_and_damage():
    scripts = ev.generate(30, seed=7)
    classes = ev.CLASSES
    perfect = ev.score_text(scripts, {s.sid: _perfect(s) for s in scripts}, classes, model=True)
    assert perfect["recall"] == 1.0 and perfect["distractors_damaged"] == 0
    assert perfect["gate"]["ok"] and not perfect["misses"]
    assert sum(v["n"] for v in perfect["by_cue"].values()) == sum(len(s.items) for s in scripts)

    nothing = ev.score_text(scripts, {s.sid: s.transcript for s in scripts}, classes, model=True)
    assert nothing["recall"] == 0.0 and not nothing["gate"]["ok"]
    assert len(nothing["misses"]) == sum(len(s.items) for s in scripts)
    miss = nothing["misses"][0]
    assert {"script", "class", "cue", "context"} <= set(miss)

    s = scripts[0]
    d = s.distractors[0]
    damaged = {x.sid: _perfect(x) for x in scripts}
    damaged[s.sid] = _perfect(s).replace(d.text, "[CARD]", 1)
    got = ev.score_text(scripts, damaged, classes, model=True)
    assert got["distractors_damaged"] == 1 and not got["gate"]["ok"]
    assert got["false_positives"][0]["script"] == s.sid


def test_score_text_without_model_leaves_password_out_of_the_gate():
    scripts = ev.generate(30, seed=7)
    masked = {}
    for s in scripts:
        keep = [i for i in s.items if i.cls == "password"]
        t = _perfect(s)
        for i in keep:                          # no model: every password survives
            t = t.replace("[PASSWORD]", i.spoken, 1)
        masked[s.sid] = t
    got = ev.score_text(scripts, masked, ev.CLASSES, model=False)
    assert got["by_class"]["password"]["masked"] == 0
    assert got["gate"]["ok"] and "password" not in got["gate"]["secret_classes"]


def test_classes_not_masked_are_not_scored():
    scripts = ev.generate(30, seed=7)
    got = ev.score_text(scripts, {s.sid: s.transcript for s in scripts}, ("ssn",), model=True)
    assert set(got["by_class"]) == {"ssn"}


def test_text_report_names_every_miss(capsys):
    scripts = ev.generate(30, seed=7)
    got = ev.score_text(scripts, {s.sid: s.transcript for s in scripts}, ev.CLASSES, model=True)
    ev.print_text_report(got)
    out = capsys.readouterr().out
    assert "same_sentence" in out and "MISS" in out


# --- sandbox guard ---------------------------------------------------------------------

def test_out_inside_a_vault_or_icloud_or_the_repo_is_refused(tmp_path):
    vault = tmp_path / "v"
    (vault / ".obsidian").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ev.check_out(vault / "eval", home=tmp_path / "home", repo=tmp_path / "repo")
    icloud = tmp_path / "home" / "Library" / "Mobile Documents" / "x"
    with pytest.raises(SystemExit):
        ev.check_out(icloud, home=tmp_path / "home", repo=tmp_path / "repo")
    with pytest.raises(SystemExit):
        ev.check_out(tmp_path / "repo" / "out", home=tmp_path / "home", repo=tmp_path / "repo")
    ok = tmp_path / "scratch" / "eval"
    assert ev.check_out(ok, home=tmp_path / "home", repo=tmp_path / "repo") == ok.resolve()


def test_sandbox_refuses_when_config_resolved_elsewhere(tmp_path, monkeypatch):
    # In this process slim.config is already imported with other paths: the guard must catch it.
    for name in ("SLIM_VAULT", "SLIM_DATA_DIR", "SLIM_PROJECTS"):
        monkeypatch.delenv(name, raising=False)           # restored after the test
    with pytest.raises(SystemExit):
        ev.sandbox(tmp_path / "out", home=tmp_path / "home", repo=tmp_path / "repo")


def test_sandbox_projects_yaml_is_generic(tmp_path):
    ev.write_projects(tmp_path / "projects.yaml", ev.CLASSES)
    text = (tmp_path / "projects.yaml").read_text()
    assert 'owner: "Alex"' in text and "phone" in text


# --- calls, end to end with fakes -------------------------------------------------------

class FakeStudio:
    """`say` + Parakeet + ffmpeg stand-ins: the audio file is a JSON list of (speaker, word, t0,
    t1), silencing drops words inside spans, transcription returns what is left."""

    def __init__(self):
        self.transcribed = 0

    def synth(self, script, dest: Path):
        words, t, lines = [], 0.5, []
        for speaker, text in script.lines:
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


def _finder(scripts):
    secrets = [i.spoken for s in scripts for i in s.items if i.cls == "password"]
    return lambda window: next((x for x in secrets if x in window), None)


def test_calls_scores_with_fakes_and_resumes(tmp_path, monkeypatch, capsys):
    scripts = ev.generate(8, seed=7)
    monkeypatch.setattr(pii, "ask_model", _finder(scripts))
    studio = FakeStudio()
    kw = dict(synth=studio.synth, transcribe=studio.transcribe,
              mask_segments=lambda gots: pipeline._mask_segments(gots, model=True),
              silence=studio.silence, measure=studio.measure, classes=ev.CLASSES)
    report = ev.run_calls(scripts, tmp_path, **kw)
    assert set(report["by_class"]) <= set(ev.CLASSES)
    assert report["scripts"] == 8
    for s in scripts:
        res = json.loads((tmp_path / "results" / f"{s.sid}.json").read_text())
        assert (tmp_path / "results" / f"{s.sid}.raw.md").read_text().startswith("**")
        assert "masked" in res and "audio_failures" in res
    assert report["renderings"] and {"class", "spoken", "heard"} <= set(report["renderings"][0])
    # every secret the text pass masked is gone from the re-transcribed audio
    audio_fail = [f for f in report["audio_failures"] if f["check"] == "retranscribed"]
    masked_secrets = [m for m in report["misses"] if m["class"] in pii.SECRET]
    assert len(audio_fail) <= len(masked_secrets)
    calls_before = studio.transcribed
    ev.run_calls(scripts, tmp_path, **kw)
    assert studio.transcribed == calls_before                    # resumed: nothing redone
    ev.run_calls(scripts[:1], tmp_path, redo=True, **kw)
    assert studio.transcribed == calls_before + 2                # raw + silenced
    out = capsys.readouterr().out
    assert scripts[0].sid in out


def test_calls_counts_untranscribed_values_apart(tmp_path, monkeypatch):
    scripts = [s for s in ev.generate(30, seed=7) if s.items][:3]
    monkeypatch.setattr(pii, "ask_model", lambda w: None)
    studio = FakeStudio()
    garble = lambda p: {**studio.transcribe(p), "text": "**Me:** something else entirely",
                        "tokens": [[" something", 0.1, 0.2]]}
    report = ev.run_calls(scripts, tmp_path, synth=studio.synth, transcribe=garble,
                          mask_segments=lambda g: pipeline._mask_segments(g, model=True),
                          silence=studio.silence, measure=studio.measure, classes=ev.CLASSES)
    assert report["not_transcribed"] == sum(len(s.items) for s in scripts)
    assert all(v["n"] == 0 for v in report["by_class"].values())


def test_calls_audio_check_fails_when_a_secret_stays_loud(tmp_path, monkeypatch):
    scripts = [s for s in ev.generate(30, seed=7)
               if pii.mask(s.transcript, ev.CLASSES).secrets][:1]
    assert scripts
    monkeypatch.setattr(pii, "ask_model", lambda w: None)
    studio = FakeStudio()
    report = ev.run_calls(scripts, tmp_path, synth=studio.synth, transcribe=studio.transcribe,
                          mask_segments=lambda g: pipeline._mask_segments(g, model=True),
                          silence=lambda p, s, pad=0.15: None, measure=studio.measure,
                          classes=ev.CLASSES)
    checks = {f["check"] for f in report["audio_failures"]}
    assert {"rms", "retranscribed"} <= checks and not report["gate"]["ok"]


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
        {"row": {"text": "Reach me at 12 Quorrel Street or kim@example.org.",
                 "entities": "[{'entity': '12 Quorrel Street', 'types': ['street_address']}, "
                             "{'entity': 'kim@example.org', 'types': ['email']}]"}},
    ],
}


def test_adapters_map_labels_onto_our_classes():
    docs = [d for r in FINANCE_PAGE["rows"]
            if (d := ev.ADAPTERS["gretelai/synthetic_pii_finance_multilingual"](r["row"]))]
    assert len(docs) == 1
    text, gold = docs[0]
    assert {g[2] for g in gold} == {"email", "ssn", "card", "account"}
    assert text[gold[1][0]:gold[1][1]] == "219-09-9999"
    text, gold = ev.ADAPTERS["gretelai/gretel-pii-masking-en-v1"](EN_V1_PAGE["rows"][0]["row"])
    assert [(text[a:b], c) for a, b, c, _ in gold] == [("12 Quorrel Street", "address"),
                                                      ("kim@example.org", "email")]


def test_score_public_splits_invalid_gold_from_true_misses():
    text, gold = ev.ADAPTERS["gretelai/synthetic_pii_finance_multilingual"](FINANCE_PAGE["rows"][0]["row"])
    got = ev.score_public([(text, gold)], lambda t: pii.find(t, ev.CLASSES))
    card = got["by_class"]["card"]
    assert card["gold"] == 2 and card["found"] == 1 and card["invalid"] == 1 and card["true_miss"] == 0
    assert got["by_class"]["ssn"]["recall"] == 1.0
    assert got["by_class"]["email"]["precision"] == 1.0


def test_licence_check_and_cached_fetch(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "api/datasets/" in url:
            return {"cardData": {"license": "apache-2.0" if "gretel" in url else "other"}}
        return FINANCE_PAGE if "finance" in url else EN_V1_PAGE

    assert ev.licence("gretelai/x", fetch) == ("apache-2.0", True)
    assert ev.licence("ai4privacy/pii-masking-300k", fetch) == ("other", False)
    report = ev.run_public(tmp_path, rows=10, fetch=fetch,
                           find=lambda t: pii.find(t, ev.CLASSES), log=lambda *a: None)
    assert report["datasets"]["ai4privacy/pii-masking-300k"]["used"] is False
    assert report["datasets"]["gretelai/synthetic_pii_finance_multilingual"]["used"] is True
    n = len(calls)
    ev.run_public(tmp_path, rows=10, fetch=fetch, find=lambda t: pii.find(t, ev.CLASSES),
                  log=lambda *a: None)
    assert all("datasets-server" not in u for u in calls[n:])          # pages came from cache
    assert any(tmp_path.joinpath("public").rglob("*.json"))


def test_public_without_network_says_so(tmp_path):
    def fetch(url):
        raise OSError("network down")
    lines = []
    report = ev.run_public(tmp_path, rows=10, fetch=fetch, find=lambda t: [],
                           log=lambda *a: lines.append(" ".join(map(str, a))))
    assert report["network"] is False and any("network" in l for l in lines)


def test_public_http_error_is_not_a_network_outage(tmp_path):
    class Limited(OSError):
        code = 429

    def fetch(url):
        if "api/datasets/" in url:
            return {"cardData": {"license": "apache-2.0"}}
        raise Limited("too many requests")
    report = ev.run_public(tmp_path, rows=10, fetch=fetch, find=lambda t: [], log=lambda *a: None)
    d = report["datasets"]["gretelai/synthetic_pii_finance_multilingual"]
    assert report["network"] is True and "HTTP 429" in d["error"]


def test_public_counts_letters_and_digits_and_rejects_slots():
    text = "mail [kim@example.org] or [Shipper's Email Address] now"
    a = text.index("[kim")
    gold = [(a, a + len("[kim@example.org]"), "email", "email"),
            (text.index("[Ship"), text.index(" now"), "email", "email")]
    got = ev.score_public([(text, gold)], lambda t: pii.find(t, ev.CLASSES))["by_class"]["email"]
    assert got["found"] == 1 and got["invalid"] == 1 and got["true_miss"] == 0
