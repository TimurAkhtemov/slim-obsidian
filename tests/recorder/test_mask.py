"""`slim mask`: recordings already in the vault, masked in place. Synthetic values only."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from slim import config
from slim.capture import pii
from slim.recorder import mask, note as note_writer, state
from slim.vault.frontmatter import parse_frontmatter

NOW = datetime(2026, 8, 4, 14, 30, tzinfo=timezone.utc)
SSN = "219-09-9999"
EMAIL = "someone@example.com"
NOTES = f"- typed by the owner: {SSN} stays as typed\n- second line  "
TRANSCRIPT = f"**Me:** my number is {SSN}.\n\n**Them:** mail it to {EMAIL} please."


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(config, "PII_MASK", config.PII_DEFAULT)
    monkeypatch.setattr(config, "PII_SKIP", (), raising=False)
    monkeypatch.setattr(pii, "ask_model", lambda window: None)


@pytest.fixture
def indexed(monkeypatch):
    calls = []
    monkeypatch.setattr(state, "index_now",
                        lambda vault, rel, **kw: (calls.append((rel, kw)), {"note": rel})[1])
    return calls


def _recorded(vault: Path, rel: str, *, transcript=TRANSCRIPT, notes=NOTES, summary="",
              after="", pii_fm=None) -> str:
    text = note_writer.render_note(
        title="Call", when=NOW, type_tag="meeting", transcript=transcript, notes_md=notes,
        summary_md=summary, topics=[], audio_rel="Attachments/Recordings/a.webm",
        filed_by_slim=True, pii=pii_fm) + after
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _legacy(vault: Path, rel: str, extra_fm="") -> str:
    """A recording filed before the slim-meeting block existed: sections at top level."""
    text = ("---\ntitle: \"Old call\"\ntype: meeting\norigin: recorded\n" + extra_fm + "---\n\n"
            "<!-- slim:summary prompt=\"record-v1\" -->\n## Summary\n\nWrite to " + EMAIL + ".\n"
            "<!-- /slim:summary -->\n\n## Notes\n\n" + NOTES + "\n## Transcript\n\n"
            + TRANSCRIPT + "\n\ntrailing words  \n")
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _body(text: str) -> str:
    _, skip = parse_frontmatter(text)
    return "".join(text.splitlines(keepends=True)[skip:])


def test_masks_transcript_only(tmp_path, indexed):
    rel = "Notes/work/call.md"
    before = _recorded(tmp_path, rel, after="\nTheir own words after the block.\n")
    out = mask.mask_note(tmp_path, rel)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert out["changed"] is True and out["counts"] == {"email": 1, "ssn": 1}
    assert "[SSN]." in text and "[EMAIL] please" in text
    assert note_writer.read_review_sections(text)["notes"] == NOTES.strip()
    assert NOTES in text            # the owner's typing, byte for byte, value included
    # Outside the two spans, nothing moved: not the labels, not the block, not what follows it.
    expected = _body(before).replace(f"is {SSN}.", "is [SSN].").replace(EMAIL, "[EMAIL]")
    assert _body(text) == expected
    fm, _ = parse_frontmatter(text)
    assert fm["pii_masked"] == ["email:1", "ssn:1"] and fm["pii_by"] == pii.VERSION
    assert fm["pii_partial"] == ["audio"]            # masked in text only: the audio still says it
    assert indexed == [(rel, {"sweep": False})]


def test_the_summary_is_masked_too(tmp_path, indexed):
    rel = "Notes/work/call.md"
    before = _recorded(tmp_path, rel, summary=f"They asked for mail at {EMAIL}.\n\nMore.",
                       after="\nafter the block\n")
    out = mask.mask_note(tmp_path, rel)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert EMAIL not in text
    assert _body(text) == _body(before).replace(f"is {SSN}.", "is [SSN].").replace(EMAIL, "[EMAIL]")
    assert note_writer.read_review_sections(text)["summary"] == "They asked for mail at [EMAIL].\n\nMore."
    assert NOTES in text
    assert out["counts"] == {"email": 2, "ssn": 1}
    assert parse_frontmatter(text)[0]["pii_masked"] == ["email:2", "ssn:1"]


def test_a_legacy_note_is_masked_in_place(tmp_path, indexed):
    rel = "Notes/work/old.md"
    before = _legacy(tmp_path, rel)
    out = mask.mask_note(tmp_path, rel)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert out["counts"] == {"email": 2, "ssn": 1}
    assert SSN in text and text.count(SSN) == 1 and NOTES in text   # only the Notes copy
    assert EMAIL not in text
    assert text.endswith("**Them:** mail it to [EMAIL] please.\n\ntrailing words  \n")
    assert _body(text) == _body(before).replace(f"is {SSN}.", "is [SSN].").replace(EMAIL, "[EMAIL]")


def test_second_run_changes_nothing(tmp_path, indexed):
    rel = "Notes/work/call.md"
    _recorded(tmp_path, rel, summary=f"Mail {EMAIL}.")
    mask.mask_note(tmp_path, rel)
    first = (tmp_path / rel).read_bytes()
    out = mask.mask_note(tmp_path, rel)
    assert out["changed"] is False and out["skipped"] == "already masked"
    assert (tmp_path / rel).read_bytes() == first
    assert len(indexed) == 1


def test_a_forced_rerun_changes_no_bytes(tmp_path, indexed):
    for rel, make in (("Notes/a/new.md", _recorded), ("Notes/a/old.md", _legacy)):
        make(tmp_path, rel)
        mask.mask_note(tmp_path, rel)
        first = (tmp_path / rel).read_bytes()
        out = mask.mask_note(tmp_path, rel, force=True)
        assert out["changed"] is False and out["counts"] == {} and not out.get("skipped")
        assert (tmp_path / rel).read_bytes() == first


def test_a_note_with_nothing_to_mask_gains_only_the_version(tmp_path, indexed):
    rel = "Notes/work/quiet.md"
    before = _recorded(tmp_path, rel, transcript="**Me:** nothing personal here.")
    out = mask.mask_note(tmp_path, rel)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    assert out["changed"] is True and out["counts"] == {}
    assert _body(text) == _body(before)
    fm, _ = parse_frontmatter(text)
    assert fm["pii_masked"] == [] and fm["pii_by"] == pii.VERSION


PASSWORD_TRANSCRIPT = "**Me:** the wifi password is tango42 for guests."


def _fails(window):
    raise RuntimeError("model down")


def test_partial_note_is_rerun(tmp_path, indexed, monkeypatch):
    rel = "Notes/work/wifi.md"
    _recorded(tmp_path, rel, transcript=PASSWORD_TRANSCRIPT + f" {SSN}")
    monkeypatch.setattr(pii, "ask_model", _fails)
    out = mask.mask_note(tmp_path, rel)
    fm, _ = parse_frontmatter((tmp_path / rel).read_text(encoding="utf-8"))
    assert out["partial"] == ["audio", "password"] and fm["pii_partial"] == ["audio", "password"]
    assert fm["pii_by"] == pii.VERSION and fm["pii_masked"] == ["ssn:1"]

    monkeypatch.setattr(pii, "ask_model", lambda window: "tango42" if "tango42" in window else None)
    out = mask.mask_note(tmp_path, rel)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(text)
    assert out["changed"] is True and out["partial"] == ["audio"] and out["counts"] == {"password": 1}
    assert "tango42" not in text and "password is [PASSWORD] for guests" in text
    assert fm["pii_partial"] == ["audio"]            # `password` done; `audio` is never removed
    assert fm["pii_masked"] == ["password:1", "ssn:1"]


def test_a_rerun_keeps_the_audio_reason(tmp_path, indexed, monkeypatch):
    rel = "Notes/work/wifi.md"
    _recorded(tmp_path, rel, transcript=PASSWORD_TRANSCRIPT,
              pii_fm=pii.frontmatter({"ssn": 1}, ["audio", "password"]))
    monkeypatch.setattr(pii, "ask_model", lambda window: "tango42")
    out = mask.mask_note(tmp_path, rel)
    fm, _ = parse_frontmatter((tmp_path / rel).read_text(encoding="utf-8"))
    assert out["partial"] == ["audio"] and fm["pii_partial"] == ["audio"]
    assert fm["pii_masked"] == ["password:1", "ssn:1"]
    again = mask.mask_note(tmp_path, rel)
    assert again["skipped"] == "already masked" and again["partial"] == ["audio"]


def test_a_legacy_partial_flag_is_the_password_pass(tmp_path, indexed, monkeypatch):
    rel = "Notes/work/wifi.md"
    _legacy(tmp_path, rel, extra_fm="pii_masked: [email:1]\npii_by: pii-1\npii_partial: true\n")
    out = mask.mask_note(tmp_path, rel)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(text)
    assert out["changed"] is True and fm["pii_partial"] == ["audio"]   # was `true`; SSN masked


def test_a_failed_password_pass_on_a_done_note_adds_the_reason(tmp_path, indexed, monkeypatch):
    rel = "Notes/work/wifi.md"
    _recorded(tmp_path, rel, transcript=PASSWORD_TRANSCRIPT, pii_fm=pii.frontmatter({}, ["audio"]))
    monkeypatch.setattr(pii, "ask_model", _fails)
    out = mask.mask_note(tmp_path, rel, force=True)
    assert out["partial"] == ["audio", "password"]
    fm, _ = parse_frontmatter((tmp_path / rel).read_text(encoding="utf-8"))
    assert fm["pii_partial"] == ["audio", "password"]


def test_an_older_version_is_rerun(tmp_path, indexed):
    rel = "Notes/work/call.md"
    _recorded(tmp_path, rel, pii_fm={"pii_masked": ["ssn:3"], "pii_by": "pii-0"})
    out = mask.mask_note(tmp_path, rel)
    fm, _ = parse_frontmatter((tmp_path / rel).read_text(encoding="utf-8"))
    assert out["changed"] is True and fm["pii_by"] == pii.VERSION
    assert fm["pii_masked"] == ["email:1", "ssn:4"]          # merged with what it said


def test_dry_run_writes_nothing(tmp_path, indexed):
    rel = "Notes/work/call.md"
    _recorded(tmp_path, rel)
    before = (tmp_path / rel).read_bytes()
    out = mask.mask_note(tmp_path, rel, dry_run=True)
    assert out["counts"] == {"email": 1, "ssn": 1} and out["changed"] is False
    assert (tmp_path / rel).read_bytes() == before and indexed == []


def test_notes_that_cannot_be_masked_are_reported_not_raised(tmp_path, indexed):
    (tmp_path / "Notes/w").mkdir(parents=True)
    (tmp_path / "Notes/w/typed.md").write_text("---\ntitle: t\n---\nmine\n")
    (tmp_path / "Notes/w/bare.md").write_text("no frontmatter at all\n")
    (tmp_path / "Notes/w/broken.md").write_text("---\norigin: recorded\nnever closed\n")
    (tmp_path / "Notes/w/empty.md").write_text("---\norigin: recorded\n---\n\nno words\n")
    got = {rel: mask.mask_note(tmp_path, f"Notes/w/{rel}")["skipped"]
           for rel in ("typed.md", "bare.md", "broken.md", "empty.md")}
    assert got == {"typed.md": "not a recording", "bare.md": "no frontmatter",
                   "broken.md": "no frontmatter", "empty.md": "no transcript"}
    assert indexed == []


def test_masking_off_touches_nothing(tmp_path, indexed, monkeypatch):
    rel = "Notes/work/call.md"
    _recorded(tmp_path, rel)
    before = (tmp_path / rel).read_bytes()
    monkeypatch.setattr(config, "PII_MASK", ())
    assert mask.mask_note(tmp_path, rel)["skipped"] == "masking is off"
    assert (tmp_path / rel).read_bytes() == before


def test_an_index_failure_is_returned(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "index_now", lambda vault, rel, **kw: {"error": "db locked"})
    rel = "Notes/work/call.md"
    _recorded(tmp_path, rel)
    out = mask.mask_note(tmp_path, rel)
    assert out["changed"] is True and out["index_error"] == "db locked"


def test_a_note_edited_while_masking_is_masked_as_it_now_stands(tmp_path, indexed, monkeypatch):
    """The CLI's lock is not the server's: an append landing while the password pass runs must
    not be overwritten by the text read before it."""
    rel = "Notes/work/wifi.md"
    _recorded(tmp_path, rel, transcript=PASSWORD_TRANSCRIPT)
    path = tmp_path / rel
    appended = []

    def appends_once(window):
        if not appended:
            appended.append(1)
            path.write_text(note_writer.append_transcript(
                path.read_text(encoding="utf-8"), f"**Them:** also {EMAIL}"), encoding="utf-8")
        return None

    monkeypatch.setattr(pii, "ask_model", appends_once)
    out = mask.mask_note(tmp_path, rel)
    text = path.read_text(encoding="utf-8")
    assert out["changed"] is True and "**Them:** also [EMAIL]" in text and EMAIL not in text


def test_recordings_skips_configured_prefixes_and_typed_notes(tmp_path, monkeypatch):
    for rel in ("Notes/a/one.md", "Notes/ab/two.md", "Notes/some-folder/x.md",
                "Notes/some-folder/deep/y.md", "Journal/day.md",
                "Attachments/copy.md", "_Reflections/r.md"):
        _recorded(tmp_path, rel)
    (tmp_path / "Notes/a/typed.md").write_text("---\ntitle: mine\n---\nwords\n")
    (tmp_path / "Notes/a/bare.md").write_text("words\n")
    monkeypatch.setattr(config, "PII_SKIP", ("Notes/some-folder", "Notes/a"))
    assert mask.recordings(tmp_path) == ["Journal/day.md", "Notes/ab/two.md"]
    monkeypatch.setattr(config, "PII_SKIP", ())
    assert mask.recordings(tmp_path) == [
        "Journal/day.md", "Notes/a/one.md", "Notes/ab/two.md",
        "Notes/some-folder/deep/y.md", "Notes/some-folder/x.md"]


def test_an_explicit_note_under_a_skipped_prefix_still_runs(tmp_path, indexed, monkeypatch):
    rel = "Notes/some-folder/x.md"
    _recorded(tmp_path, rel)
    monkeypatch.setattr(config, "PII_SKIP", ("Notes/some-folder",))
    assert mask.mask_note(tmp_path, rel)["changed"] is True


def test_the_masked_value_leaves_the_index(tmp_path, monkeypatch):
    """No stub of `index_now`: the real ingest on a tmp DB, with the embedder stubbed."""
    from slim import db
    from slim.index import embed, ingest
    monkeypatch.setattr(embed, "embed_source", lambda con, sid: {"embedded": 0})
    rel = "Notes/work/call.md"
    _recorded(tmp_path, rel, notes="- typed, nothing personal")   # their notes are never masked
    con = db.connect()
    ingest.ingest_note(con, tmp_path, rel)
    assert con.execute("SELECT count(*) FROM fragments WHERE text LIKE ?",
                       (f"%{SSN}%",)).fetchone()[0] == 1
    con.close()
    out = mask.mask_note(tmp_path, rel)
    assert "index_error" not in out
    con = db.connect()
    rows = [r["text"] for r in con.execute("SELECT text FROM fragments")]
    con.close()
    assert rows and not any("9999" in t or EMAIL in t for t in rows)
    assert any("[SSN]" in t for t in rows)
    con = db.connect()
    hits = con.execute("SELECT count(*) FROM fragments_fts WHERE fragments_fts MATCH '9999'")
    assert hits.fetchone()[0] == 0
    con.close()


# --- fix round 1: splices only, refuse what cannot stay byte-exact ---------------------------

FM = "---\ntitle: t\norigin: recorded\n---\n"
SUM = (f"<!-- slim:summary prompt=\"record-v2\" -->\n## Summary\n\nMail {EMAIL} today.\n"
       "<!-- /slim:summary -->\n")
TR = f"## Transcript\n\n**Me:** my number is {SSN}.\n"

SHAPES = {
    "legacy, title and embed above the summary":
        FM + "\n# My title\n\n![[a.webm]]\n\n" + SUM + "\n## Notes\n\n- mine\n\n" + TR,
    "legacy, owner prose above the summary": FM + "\nSome prose of mine.\n\n" + SUM + "\n" + TR,
    "an owner line between the summary and Notes":
        FM + "\n````slim-meeting\n" + SUM + "an owner line\n\n## Notes\n\n- mine\n" + TR + "````\n",
    "owner text after the summary, no Notes heading":
        FM + "\n````slim-meeting\n" + SUM + "\nowner text, no heading\n" + TR + "````\n",
    "Notes with no blank line under the heading":
        FM + "\n````slim-meeting\n" + SUM + "\n## Notes\n- mine\n" + TR + "````\n",
    "trailing spaces and blank lines after the transcript":
        FM + "\n````slim-meeting\n" + SUM + "\n" + TR + "  \n\n\n````\n",
    "five-backtick fence, opener info, closer at EOF":
        FM + "\n`````slim-meeting x\n" + SUM + "\n" + TR + "`````",
    "an extra blank line under ## Summary":
        FM + "\n````slim-meeting\n" + SUM.replace("## Summary\n\n", "## Summary\n\n\n")
        + "\n" + TR + "````\nafter\n",
}


def _without_pii_lines(text: str) -> str:
    _, skip = parse_frontmatter(text)
    lines = text.splitlines(keepends=True)
    return "".join([l for l in lines[:skip] if not l.startswith("pii_")] + lines[skip:])


@pytest.mark.parametrize("shape", SHAPES)
def test_only_the_masked_spans_and_pii_lines_change(tmp_path, indexed, shape):
    rel = "Notes/w/shape.md"
    before = SHAPES[shape]
    (tmp_path / "Notes/w").mkdir(parents=True)
    (tmp_path / rel).write_text(before, encoding="utf-8")
    out = mask.mask_note(tmp_path, rel)
    after = (tmp_path / rel).read_text(encoding="utf-8")
    assert out["changed"] is True and out["counts"] == {"email": 1, "ssn": 1}, out
    assert _without_pii_lines(after) == before.replace(EMAIL, "[EMAIL]").replace(SSN, "[SSN]")
    assert parse_frontmatter(after)[0]["pii_by"] == pii.VERSION


def test_a_note_with_cr_line_endings_is_refused_untouched(tmp_path, indexed):
    rel = "Notes/w/crlf.md"
    raw = SHAPES["legacy, owner prose above the summary"].replace("\n", "\r\n").encode()
    (tmp_path / "Notes/w").mkdir(parents=True)
    (tmp_path / rel).write_bytes(raw)
    out = mask.mask_note(tmp_path, rel)
    assert out["error"] == "CR line endings" and out["changed"] is False
    assert (tmp_path / rel).read_bytes() == raw and indexed == []


def test_a_written_note_keeps_its_exact_bytes(tmp_path, indexed):
    """Through real file I/O: nothing is translated on the way in or out."""
    rel = "Notes/w/bytes.md"
    before = SHAPES["trailing spaces and blank lines after the transcript"].encode()
    (tmp_path / "Notes/w").mkdir(parents=True)
    (tmp_path / rel).write_bytes(before)
    mask.mask_note(tmp_path, rel)
    after = (tmp_path / rel).read_bytes()
    assert _without_pii_lines(after.decode()).encode() == \
        before.replace(EMAIL.encode(), b"[EMAIL]").replace(SSN.encode(), b"[SSN]")


def test_a_splice_that_moves_other_bytes_is_refused(tmp_path, indexed, monkeypatch):
    rel = "Notes/w/call.md"
    _recorded(tmp_path, rel, summary=f"Mail {EMAIL}.")
    before = (tmp_path / rel).read_bytes()
    real = note_writer.replace_summaries
    monkeypatch.setattr(note_writer, "replace_summaries",
                        lambda text, summaries: real(text, summaries).replace("- typed", "- TYPED"))
    out = mask.mask_note(tmp_path, rel)
    assert out["error"] == "integrity" and out["changed"] is False
    assert (tmp_path / rel).read_bytes() == before and indexed == []


def test_a_note_that_keeps_changing_is_a_failure_not_a_skip(tmp_path, indexed, monkeypatch):
    rel = "Notes/w/busy.md"
    _recorded(tmp_path, rel, transcript=PASSWORD_TRANSCRIPT)
    path = tmp_path / rel

    def always_appends(window):
        path.write_text(path.read_text(encoding="utf-8") + " more", encoding="utf-8")
        return None

    monkeypatch.setattr(pii, "ask_model", always_appends)
    out = mask.mask_note(tmp_path, rel)
    assert out["error"] == "changed while masking" and out["changed"] is False
    assert "pii_by" not in path.read_text(encoding="utf-8") and indexed == []


def test_a_draft_or_excluded_note_is_never_written(tmp_path, indexed):
    for rel in ("Notes/_unfiled/rec.slim-draft.md", "Attachments/copy.md"):
        _recorded(tmp_path, rel)
        before = (tmp_path / rel).read_bytes()
        out = mask.mask_note(tmp_path, rel)
        assert out["skipped"] == "not indexed: draft or excluded folder"
        assert (tmp_path / rel).read_bytes() == before
    assert indexed == []


def test_recordings_leaves_out_a_draft_still_recording(tmp_path):
    _recorded(tmp_path, "Notes/_unfiled/rec.slim-draft.md")
    _recorded(tmp_path, "Journal/rec.slim-draft.md")
    _recorded(tmp_path, "Notes/_unfiled/filed.md")
    assert mask.recordings(tmp_path) == ["Notes/_unfiled/filed.md"]


# --- fix round 2 -----------------------------------------------------------------------------

def test_a_frontmatter_writer_that_touches_another_key_is_refused(tmp_path, indexed, monkeypatch):
    rel = "Notes/w/call.md"
    _recorded(tmp_path, rel)
    before = (tmp_path / rel).read_bytes()
    real = note_writer.set_pii_frontmatter
    monkeypatch.setattr(note_writer, "set_pii_frontmatter", lambda text, counts, reasons: real(
        text, counts, reasons).replace('title: "Call"', 'title: "Other"'))
    out = mask.mask_note(tmp_path, rel)
    assert out["error"] == "integrity" and (tmp_path / rel).read_bytes() == before
    assert indexed == []


def test_the_guard_ignores_only_the_pii_keys_and_their_list_lines():
    head = "---\ntitle: t\npii_masked:\n  - ssn:1\npii_by: pii-0\naudio: a\n---\nbody\n"
    assert mask._head_without_pii(head) == ["---\n", "title: t\n", "audio: a\n", "---\n"]


ONE_SUMMARY = ('<!-- slim:summary prompt="record-v2" -->\n## Summary\n\nMail {} today.\n'
               "<!-- /slim:summary -->\n")


def test_every_summary_block_is_masked(tmp_path, indexed):
    other = "other@example.com"
    rel = "Notes/w/two.md"
    before = (FM + "\n" + ONE_SUMMARY.format(EMAIL) + "\n" + ONE_SUMMARY.format(other) + "\n" + TR)
    (tmp_path / "Notes/w").mkdir(parents=True)
    (tmp_path / rel).write_text(before, encoding="utf-8")
    out = mask.mask_note(tmp_path, rel)
    after = (tmp_path / rel).read_text(encoding="utf-8")
    assert out["counts"] == {"email": 2, "ssn": 1}
    assert _without_pii_lines(after) == \
        before.replace(EMAIL, "[EMAIL]").replace(other, "[EMAIL]").replace(SSN, "[SSN]")


@pytest.mark.parametrize("broken", [
    "<!-- slim:summary prompt=\"x\" -->\n## Summary\n\nMail {} today, never closed.\n\n",
    ONE_SUMMARY + "<!-- slim:summary prompt=\"x\" -->\n## Summary\n\nand {} again\n\n",
])
def test_a_summary_the_splice_cannot_see_refuses_the_note(tmp_path, indexed, broken):
    rel = "Notes/w/broken.md"
    raw = (FM + "\n````slim-meeting\n" + broken.format(EMAIL, EMAIL) + TR + "````\n").encode()
    (tmp_path / "Notes/w").mkdir(parents=True)
    (tmp_path / rel).write_bytes(raw)
    out = mask.mask_note(tmp_path, rel)
    assert out["error"] == "summary block" and out["changed"] is False
    assert (tmp_path / rel).read_bytes() == raw and indexed == []


def test_on_disk_takes_the_disk_spelling():
    tree = {".": ["Notes", "x"], "Notes": ["w", "W2", "Ab", "aB"], "Notes/w": ["c.md", "C.md"],
            "Notes/Ab": ["n.md"]}

    def listdir(path):
        rel = Path(path).relative_to("/v").as_posix()
        return tree[rel]

    v = Path("/v")
    assert mask.on_disk(v, "notes/W/c.md", listdir) == "Notes/w/c.md"     # exact name wins
    assert mask.on_disk(v, "NOTES/w2", listdir) == "Notes/W2"
    with pytest.raises(FileNotFoundError):
        mask.on_disk(v, "Notes/ab/n.md", listdir)                          # two case matches
    with pytest.raises(FileNotFoundError):
        mask.on_disk(v, "Notes/w/missing.md", listdir)


# --- the audio reason: a text-only run never silenced the audio ---------------------------------

def test_a_secret_masked_in_text_leaves_the_audio_reason(tmp_path, indexed):
    rel = "Notes/w/old.md"
    _recorded(tmp_path, rel, transcript=f"**Me:** my number is {SSN}.")
    out = mask.mask_note(tmp_path, rel)
    text = (tmp_path / rel).read_text(encoding="utf-8")
    fm, _ = parse_frontmatter(text)
    assert out["partial"] == ["audio"] and fm["pii_partial"] == ["audio"]
    first = (tmp_path / rel).read_bytes()
    again = mask.mask_note(tmp_path, rel)
    assert again["skipped"] == "already masked" and again["partial"] == ["audio"]
    forced = mask.mask_note(tmp_path, rel, force=True)
    assert forced["changed"] is False and forced["partial"] == ["audio"]
    assert (tmp_path / rel).read_bytes() == first


def test_only_non_secret_classes_add_no_reason(tmp_path, indexed):
    rel = "Notes/w/mail.md"
    _recorded(tmp_path, rel, transcript=f"**Them:** mail it to {EMAIL} please.")
    out = mask.mask_note(tmp_path, rel)
    fm, _ = parse_frontmatter((tmp_path / rel).read_text(encoding="utf-8"))
    assert out["counts"] == {"email": 1} and out["partial"] == [] and "pii_partial" not in fm


def test_a_password_found_on_a_rerun_trades_password_for_audio(tmp_path, indexed, monkeypatch):
    """Its span was never muted at capture, so the audio still says it."""
    rel = "Notes/w/wifi.md"
    _recorded(tmp_path, rel, transcript=PASSWORD_TRANSCRIPT,
              pii_fm=pii.frontmatter({}, ["password"]))
    monkeypatch.setattr(pii, "ask_model", lambda window: "tango42")
    out = mask.mask_note(tmp_path, rel)
    fm, _ = parse_frontmatter((tmp_path / rel).read_text(encoding="utf-8"))
    assert out["counts"] == {"password": 1} and out["partial"] == ["audio"]
    assert fm["pii_partial"] == ["audio"]


def test_a_dry_run_reports_the_audio_reason_it_would_write(tmp_path, indexed):
    rel = "Notes/w/old.md"
    _recorded(tmp_path, rel, transcript=f"**Me:** my number is {SSN}.")
    before = (tmp_path / rel).read_bytes()
    out = mask.mask_note(tmp_path, rel, dry_run=True)
    assert out["partial"] == ["audio"] and (tmp_path / rel).read_bytes() == before
