"""REFLECT — deterministic journal selection, the quiet-night guardrail, wikilink safety, the
note writer, ingestion exclusion, and the privacy contract.

The dense model is MOCKED throughout, so no Ollama is needed and the module's OWN logic (window
math, skips, link sanitizing, frontmatter, resume-from-disk) is pinned deterministically. Journal
BODIES here are entirely SYNTHETIC — the real rule that the agent never reads journal content
applies to the code and to these tests alike. Reflection quality on real journals is the owner's read,
not a unit test.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from slim import db, ingest, reflect
from slim.chunk import parse_frontmatter


@pytest.fixture
def con(tmp_path):
    connection = db.connect(tmp_path / "brain.db")
    yield connection
    connection.close()


def _journal(con, sid, date, body="a synthetic entry", title=None, folder="Journal"):
    """Insert one journal source + its fragment(s). `body` may be a list → multiple seq fragments."""
    path = f"{folder}/{date}--{sid}.md"
    con.execute(
        "INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
        "VALUES (?, ?, ?, 'journal', ?, 'h')",
        (sid, path, title or path, date))
    parts = body if isinstance(body, list) else [body]
    for seq, text in enumerate(parts):
        con.execute(
            "INSERT INTO fragments (source_id, seq, heading_path, text, content_hash) "
            "VALUES (?, ?, '§', ?, 'fh')", (sid, seq, text))
    con.commit()


class _FakeChat:
    """Stand-in for llm.chat. Records the calls (so dry-run can assert it was NOT invoked and other
    tests can inspect the prompt) and returns configurable content + a stats dict."""
    def __init__(self, content="You wrote a little this week."):
        self.content = content
        self.calls = []

    def __call__(self, messages, **kw):
        self.calls.append({"messages": messages, "kw": kw})
        return self.content, {"model": kw.get("model", "test-model"), "output_tokens": 42,
                              "duration_s": 1.0, "ctx_saturated": False}


@pytest.fixture
def fake_chat(monkeypatch):
    fc = _FakeChat()
    monkeypatch.setattr(reflect.llm, "chat", fc)
    return fc


NOW = datetime(2026, 7, 23, 4, 0, 0)   # a fixed nightly-shaped run time


# --------------------------------------------------------------------------
# deterministic selection
# --------------------------------------------------------------------------

def test_selects_journals_in_window_ordered_by_date(con):
    _journal(con, "a", "2026-07-10")
    _journal(con, "b", "2026-07-15")
    _journal(con, "c", "2026-07-20")
    _journal(con, "d", "2026-07-23")
    got = reflect._journals(con, on_or_after="2026-07-15")
    assert [e.authored_at for e in got] == ["2026-07-15", "2026-07-20", "2026-07-23"]
    assert [e.stem for e in got] == ["2026-07-15--b", "2026-07-20--c", "2026-07-23--d"]


def test_only_journals_are_selected(con):
    _journal(con, "j", "2026-07-20")
    # a meeting note on the same date must never be pulled into a reflection
    con.execute(
        "INSERT INTO sources (id, path, title, type, authored_at, current_hash) VALUES "
        "('m', 'Notes/school/2026-07-20 sync.md', 't', 'meeting-note', '2026-07-20', 'h')")
    con.commit()
    got = reflect._journals(con, on_or_after="2026-07-01")
    assert [e.source_id for e in got] == ["j"]


def test_body_is_fragments_joined_in_seq_order(con):
    _journal(con, "a", "2026-07-20", body=["first", "second", "third"])
    got = reflect._journals(con, on_or_after="2026-07-01")
    assert got[0].body == "first\n\nsecond\n\nthird"


# --------------------------------------------------------------------------
# guardrail 1 — quiet nights stay quiet
# --------------------------------------------------------------------------

def test_empty_window_skips_writes_nothing_and_traces(con, fake_chat, tmp_path):
    # nothing on/after the --days=1 cutoff (2026-07-22)
    _journal(con, "old", "2026-07-01")
    result = reflect.reflect(con, vault=tmp_path, days=1, now=NOW)
    assert result.skipped and result.reason == "quiet"
    assert not fake_chat.calls, "a quiet night must not call the model"
    assert not (tmp_path / reflect.NAMESPACE).exists() or \
        not list((tmp_path / reflect.NAMESPACE).glob("*.md"))
    rec = json.loads(_last_trace_line(tmp_path))
    assert rec["kind"] == "reflect" and rec["skipped"] is True


def test_min_entries_floor_skips(con, fake_chat, tmp_path):
    _journal(con, "a", "2026-07-22")
    _journal(con, "b", "2026-07-23")
    result = reflect.reflect(con, vault=tmp_path, days=7, min_entries=3, now=NOW)
    assert result.skipped and result.reason == "below-floor"
    assert result.focus_count == 2
    assert not fake_chat.calls


# --------------------------------------------------------------------------
# window resumption — "since last reflection", widening across gaps
# --------------------------------------------------------------------------

def test_since_last_reflection_advances_and_widens(con, fake_chat, tmp_path):
    _journal(con, "a", "2026-07-18")
    _journal(con, "b", "2026-07-20")
    first = reflect.reflect(con, vault=tmp_path, now=datetime(2026, 7, 21, 4, 0))
    assert first.window_end == "2026-07-20" and first.focus_count == 2
    assert first.path and (tmp_path / first.path).exists()

    # a quiet next day: no new journals → skip, and the window does NOT advance
    quiet = reflect.reflect(con, vault=tmp_path, now=datetime(2026, 7, 22, 4, 0))
    assert quiet.skipped

    # then two new entries arrive: the window resumed from 07-20 (exclusive) and WIDENED over the gap
    _journal(con, "c", "2026-07-23")
    _journal(con, "d", "2026-07-24")
    third = reflect.reflect(con, vault=tmp_path, now=datetime(2026, 7, 25, 4, 0))
    assert third.window_start == "2026-07-20"
    assert [s for s in third.sources] == ["2026-07-23--c", "2026-07-24--d"]
    assert third.focus_count == 2


def test_first_run_defaults_to_last_week(con, fake_chat, tmp_path):
    _journal(con, "old", "2026-07-01")     # outside a 7-day window from 07-23
    _journal(con, "new", "2026-07-20")     # inside
    result = reflect.reflect(con, vault=tmp_path, now=NOW)
    assert result.window_start == "2026-07-16"   # 07-23 minus 7 days
    assert result.sources == ["2026-07-20--new"]


# --------------------------------------------------------------------------
# CONTEXT — read-only continuity, excludes FOCUS, respects the caps
# --------------------------------------------------------------------------

def test_context_is_prior_entries_excludes_focus_and_respects_lookback(con, fake_chat, tmp_path):
    _journal(con, "way-before", "2026-07-05")   # >14 days before earliest focus → excluded
    _journal(con, "ctx1", "2026-07-08")         # within lookback → context
    _journal(con, "ctx2", "2026-07-20")         # within lookback → context
    _journal(con, "focus1", "2026-07-21")       # focus (days=2 from 07-23 → since 07-21)
    _journal(con, "focus2", "2026-07-22")       # focus
    result = reflect.reflect(con, vault=tmp_path, days=2, now=NOW)
    assert result.focus_count == 2
    assert result.context_count == 2
    # the model prompt must render focus stems only in the FOCUS block, never in the CONTEXT block
    # (the trailing "Wikilinks you may use" allow-list DOES name them — scope to the context body)
    user = fake_chat.calls[0]["messages"][1]["content"]
    focus_block, rest = user.split("CONTEXT ENTRIES", 1)
    context_block = rest.split("Wikilinks you may use", 1)[0]
    assert "focus1" in focus_block and "focus2" in focus_block
    assert "focus1" not in context_block and "focus2" not in context_block
    assert "ctx1" in context_block and "way-before" not in context_block


def test_context_capped_at_max_entries(con, fake_chat, tmp_path, monkeypatch):
    # 40 context entries all before the focus window; only the most recent CONTEXT_MAX_ENTRIES survive
    for day in range(1, 41):
        _journal(con, f"c{day}", f"2026-06-{day:02d}" if day <= 30 else f"2026-07-{day - 30:02d}")
    _journal(con, "focus", "2026-07-20")
    monkeypatch.setattr(reflect, "CONTEXT_LOOKBACK_DAYS", 365)   # make all 40 eligible
    result = reflect.reflect(con, vault=tmp_path, days=1, now=datetime(2026, 7, 20, 4, 0))
    assert result.context_count == reflect.CONTEXT_MAX_ENTRIES


# --------------------------------------------------------------------------
# wikilink safety
# --------------------------------------------------------------------------

def test_wikilinks_canonicalized_to_short_alias_invented_ones_stripped(con, tmp_path, monkeypatch):
    _journal(con, "real", "2026-07-22", title="Voice memo note")
    _journal(con, "real2", "2026-07-23")     # no title → de-slugged stem label
    content = ("You returned to [[2026-07-22--real]] and to [[made-up-note]], "
               "and [[2026-07-23--real2|whatever the model called it]] closed it.")
    monkeypatch.setattr(reflect.llm, "chat",
                        lambda messages, **kw: (content, {"model": "gemma4:31b"}))
    result = reflect.reflect(con, vault=tmp_path, days=7, now=NOW)
    written = (tmp_path / result.path).read_text()
    # a bare valid link gains the note's title as its short label...
    assert "[[2026-07-22--real|Voice memo note]]" in written
    # ...and a model-chosen alias is REPLACED by the canonical de-slugged label (compactness wins)
    assert "[[2026-07-23--real2|real2]]" in written
    assert "whatever the model called it" not in written
    # the long raw stem never renders on its own
    assert "[[2026-07-22--real]]" not in written
    assert "[[made-up-note]]" not in written and "made-up-note" in written
    assert "_Reflected on: [[2026-07-22--real|Voice memo note]], [[2026-07-23--real2|real2]]_" in written


# --------------------------------------------------------------------------
# the writer — frontmatter, atomic create-only with suffix
# --------------------------------------------------------------------------

def test_note_has_code_stamped_frontmatter(con, fake_chat, tmp_path):
    _journal(con, "a", "2026-07-22")
    _journal(con, "b", "2026-07-23")
    result = reflect.reflect(con, vault=tmp_path, days=7, now=NOW)
    fm, _ = parse_frontmatter((tmp_path / result.path).read_text())
    assert fm["type"] == "reflection"
    assert fm["generated_by"] == "slim reflect"
    assert fm["window_end"] == "2026-07-23"
    assert fm["authored_at"] == "2026-07-23"
    assert fm["model"] == reflect.llm.MODEL
    assert fm["sources"] == ["2026-07-22--a", "2026-07-23--b"]
    assert result.path == "_Reflections/2026-07-23--reflection.md"


def test_same_date_rerun_suffixes_never_overwrites(con, fake_chat, tmp_path):
    _journal(con, "a", "2026-07-23")
    first = reflect.reflect(con, vault=tmp_path, days=7, now=NOW)
    second = reflect.reflect(con, vault=tmp_path, days=7, now=NOW)   # same window → same date
    assert first.path == "_Reflections/2026-07-23--reflection.md"
    assert second.path == "_Reflections/2026-07-23--reflection-2.md"
    assert (tmp_path / first.path).exists() and (tmp_path / second.path).exists()


# --------------------------------------------------------------------------
# ingestion exclusion — the derived artifact must never be indexed
# --------------------------------------------------------------------------

def test_reflections_folder_is_excluded_from_ingestion(tmp_path):
    (tmp_path / "Journal").mkdir()
    (tmp_path / "Journal" / "2026-07-23--x.md").write_text("a real source")
    (tmp_path / reflect.NAMESPACE).mkdir()
    (tmp_path / reflect.NAMESPACE / "2026-07-23--reflection.md").write_text("derived output")
    eligible = {str(p.relative_to(tmp_path)) for p in ingest.eligible_files(tmp_path)}
    assert "Journal/2026-07-23--x.md" in eligible
    assert not any(reflect.NAMESPACE in p for p in eligible)


# --------------------------------------------------------------------------
# privacy — no journal (or reflection) prose in the operational trace
# --------------------------------------------------------------------------

def test_trace_never_contains_body_or_reflection_prose(con, tmp_path, monkeypatch):
    _journal(con, "a", "2026-07-23", body="SENTINEL_PRIVATE_JOURNAL_BODY")
    monkeypatch.setattr(
        reflect.llm, "chat",
        lambda messages, **kw: ("SENTINEL_REFLECTION_PROSE about you", {"model": "gemma4:31b"}))
    reflect.reflect(con, vault=tmp_path, days=7, now=NOW)
    raw = _trace_file(tmp_path).read_text()
    assert "SENTINEL_PRIVATE_JOURNAL_BODY" not in raw
    assert "SENTINEL_REFLECTION_PROSE" not in raw
    assert '"kind": "reflect"' in raw


# --------------------------------------------------------------------------
# dry-run — pure preview
# --------------------------------------------------------------------------

def test_dry_run_makes_no_model_call_and_no_write(con, fake_chat, tmp_path):
    _journal(con, "a", "2026-07-22")
    _journal(con, "b", "2026-07-23")
    result = reflect.reflect(con, vault=tmp_path, days=7, now=NOW, dry_run=True)
    assert result.dry_run and not result.skipped
    assert result.focus_count == 2
    assert result.path == "_Reflections/2026-07-23--reflection.md"   # what it WOULD write
    assert not fake_chat.calls
    assert not (tmp_path / reflect.NAMESPACE).exists()


# --------------------------------------------------------------------------
# helpers for reading the isolated trace (conftest points TRACE_DIR at a tmp path)
# --------------------------------------------------------------------------

def _trace_file(_tmp_path) -> Path:
    files = sorted(reflect.trace.TRACE_DIR.glob("*.jsonl"))
    assert files, "expected a trace file to be written"
    return files[-1]


def _last_trace_line(tmp_path) -> str:
    return _trace_file(tmp_path).read_text().splitlines()[-1]
