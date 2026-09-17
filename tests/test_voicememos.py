"""Tests for the copy-only Voice Memos sweep (P2 / R4).

The rules under test are the ones that make it safe to run every five minutes forever AND safe
to point at Apple's app data: idempotent by SHA-256, copy-only (the container is never
mutated), loud (never silent) when refused, and the title db opened read-only + immutable.

Never invokes real ASR or ffmpeg — the sweep is pure file copying; transcription is a separate
stage tested in test_inbox.
"""
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from slim import cli, inbox, trace, voicememos


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake Voice Memos container + a throwaway journal inbox. Never touches the real ones."""
    container = tmp_path / "Recordings"
    container.mkdir()
    journal = tmp_path / "vault" / "Inbox" / "Journal"
    journal.mkdir(parents=True)

    monkeypatch.setattr(voicememos, "MANIFEST", tmp_path / "data" / "voicememos-swept.json")
    monkeypatch.setattr(inbox, "INBOX", {"journal": journal})
    return SimpleNamespace(container=container, journal=journal)


def _write_title_db(container, mapping: dict[str, str]) -> None:
    con = sqlite3.connect(str(container / voicememos.TITLE_DB))
    con.execute(
        "CREATE TABLE ZCLOUDRECORDING (ZPATH TEXT, ZCUSTOMLABEL TEXT, ZENCRYPTEDTITLE TEXT)")
    for path, title in mapping.items():
        con.execute("INSERT INTO ZCLOUDRECORDING VALUES (?, ?, ?)", (path, title, None))
    con.commit()
    con.close()


# --- copy + idempotency -------------------------------------------------------------------

def test_copies_new_audio_with_hash_suffix(env):
    src = env.container / "20260719 090000.m4a"
    src.write_bytes(b"memo one")
    (r,) = voicememos.sweep(container=env.container)

    digest = hashlib.sha256(b"memo one").hexdigest()
    assert r.status == "copied"
    assert r.dest.name == f"20260719 090000--{digest[:8]}.m4a"   # stem fallback, no db title
    assert r.dest.parent == env.journal
    assert r.dest.read_bytes() == b"memo one"

    rec = trace.last(1)[0]
    assert rec["kind"] == "voicememo_sweep" and rec["status"] == "ok"
    assert rec["copied"] == 1 and rec["skipped"] == 0


def test_resweep_skips_via_manifest(env):
    (env.container / "a.m4a").write_bytes(b"aaa")
    voicememos.sweep(container=env.container)
    names_after_first = sorted(p.name for p in env.journal.iterdir())

    (r,) = voicememos.sweep(container=env.container)
    assert r.status == "skipped"
    assert sorted(p.name for p in env.journal.iterdir()) == names_after_first  # no re-copy


def test_a_re_export_of_the_same_recording_does_not_duplicate_it(env):
    """MEASURED IN PRODUCTION 2026-07-20, and it would have flooded the vault.

    The macOS Shortcut does not produce byte-stable exports: exporting the SAME recording
    twice yields different bytes (re-encode / fresh metadata), so content-addressed dedup
    saw a brand-new recording every time and transcribed it again. Four of the owner's memos
    had two notes each before this was caught — and with the launchd timer running every
    five minutes and "Overwrite" on, it would have produced a fresh duplicate of every memo
    on every tick.

    Identity therefore cannot be the exported file's hash. Voice Memos' own filename stem
    ("20260715 211049-9F98935C") carries its stable recording id and survives re-export, so
    a stem already swept is the same recording no matter what bytes arrive.
    """
    src = env.container / "20260715 211049-9F98935C.m4a"
    src.write_bytes(b"first export of this recording")
    (first,) = voicememos.sweep(container=env.container)
    assert first.status == "copied"

    # Same recording, re-exported: identical name, different bytes.
    src.write_bytes(b"second export, re-encoded, same recording")
    (second,) = voicememos.sweep(container=env.container)

    assert second.status == "skipped", "a re-export must not read as a new recording"
    assert len(list(env.journal.iterdir())) == 1, "exactly one copy in the inbox"


def test_a_genuinely_new_recording_still_sweeps(env):
    """The stem guard must not become a blanket 'skip everything' — a different recording
    has a different stem and still arrives."""
    (env.container / "20260715 211049-9F98935C.m4a").write_bytes(b"one")
    voicememos.sweep(container=env.container)
    (env.container / "20260716 084500-AABBCCDD.m4a").write_bytes(b"two")

    results = voicememos.sweep(container=env.container)

    copied = [r for r in results if r.status == "copied"]
    assert len(copied) == 1 and "AABBCCDD" in copied[0].source.name
    assert len(list(env.journal.iterdir())) == 2


def test_container_files_are_untouched_after_sweep(env):
    src = env.container / "a.m4a"
    src.write_bytes(b"precious recording")
    before_bytes = src.read_bytes()
    before = src.stat()

    voicememos.sweep(container=env.container)

    assert src.exists()
    assert src.read_bytes() == before_bytes
    after = src.stat()
    assert after.st_mtime == before.st_mtime and after.st_size == before.st_size
    # nothing new was written INTO the container
    assert sorted(p.name for p in env.container.iterdir()) == ["a.m4a"]


def test_same_title_different_bytes_get_distinct_collision_safe_names(env):
    _write_title_db(env.container, {"a.m4a": "Same Title", "b.m4a": "Same Title"})
    (env.container / "a.m4a").write_bytes(b"one")
    (env.container / "b.m4a").write_bytes(b"two")

    dests = {r.dest.name for r in voicememos.sweep(container=env.container)
             if r.status == "copied"}
    assert len(dests) == 2                       # the hash suffix keeps them apart
    assert all(name.startswith("Same-Title--") for name in dests)


def test_a_quiet_sweep_writes_no_trace(env):
    """The timer ticks every five minutes, so a sweep that copied nothing is the OVERWHELMING
    majority of them — 288 identical "found 0" records a day, burying the ones that say
    something happened. A trace exists to record an event; nothing happening is not one
    (2026-09-03). Refusals still trace: those ARE events."""
    assert voicememos.sweep(container=env.container) == []
    assert trace.last(1) == []

    # And a sweep that only re-saw what it already had is just as quiet.
    (env.container / "a.m4a").write_bytes(b"aaa")
    voicememos.sweep(container=env.container)
    quiet_from = len(trace.last(50))
    (r,) = voicememos.sweep(container=env.container)
    assert r.status == "skipped"
    assert len(trace.last(50)) == quiet_from


def test_dry_run_copies_nothing(env):
    (env.container / "a.m4a").write_bytes(b"aaa")
    (r,) = voicememos.sweep(container=env.container, dry_run=True)
    assert r.status == "copied" and r.detail == "(dry run)"
    assert list(env.journal.iterdir()) == []
    assert not voicememos.MANIFEST.exists()


# --- loud refusals ------------------------------------------------------------------------

def test_permission_denied_is_an_authored_refusal_and_traced(env, monkeypatch):
    from pathlib import Path
    real_iterdir = Path.iterdir

    def denied(self):
        if self == env.container:
            raise PermissionError(1, "Operation not permitted")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", denied)
    (r,) = voicememos.sweep(container=env.container)
    assert r.status == "refused"
    assert "Operation not permitted" in r.detail and "Grant access" in r.detail
    rec = trace.last(1)[0]
    assert rec["kind"] == "voicememo_sweep" and rec["status"] == "refused"
    assert rec["reason"] == "permission"


def test_missing_container_is_an_authored_refusal(tmp_path):
    (r,) = voicememos.sweep(container=tmp_path / "not-synced")
    assert r.status == "refused"
    assert "does not exist" in r.detail
    rec = trace.last(1)[0]
    assert rec["reason"] == "missing-container"


# --- title db handling --------------------------------------------------------------------

def test_title_from_db_becomes_the_filename(env):
    _write_title_db(env.container, {"rec1.m4a": "Morning walk idea"})
    (env.container / "rec1.m4a").write_bytes(b"aaa")
    copied = [r for r in voicememos.sweep(container=env.container) if r.status == "copied"]
    assert len(copied) == 1
    assert copied[0].dest.name.startswith("Morning-walk-idea--")


def test_corrupt_title_db_falls_back_and_sweep_still_succeeds(env):
    (env.container / voicememos.TITLE_DB).write_bytes(b"this is not a database")
    (env.container / "a.m4a").write_bytes(b"aaa")

    titles, detail = voicememos._titles(env.container)
    assert titles == {}                          # undecodable -> empty, never raises

    copied = [r for r in voicememos.sweep(container=env.container) if r.status == "copied"]
    assert len(copied) == 1
    assert copied[0].dest.name.startswith("a--")  # stem fallback


def test_absent_title_db_returns_empty(env):
    titles, detail = voicememos._titles(env.container)
    assert titles == {} and detail == "db-missing"


def test_title_db_is_opened_read_only_and_immutable(env, monkeypatch):
    (env.container / voicememos.TITLE_DB).write_bytes(b"x")
    captured = {}

    def spy(database, *args, **kwargs):
        captured["uri"] = database
        captured["uri_kw"] = kwargs.get("uri")
        raise voicememos.sqlite3.OperationalError("blocked")

    monkeypatch.setattr(voicememos.sqlite3, "connect", spy)
    titles, _ = voicememos._titles(env.container)

    assert titles == {}                          # a blocked db never breaks the sweep
    expected = f"file:{env.container / voicememos.TITLE_DB}?mode=ro&immutable=1"
    assert captured["uri"] == expected
    assert captured["uri_kw"] is True


# --- CLI wiring: sweep -> process -> (index only if written, not dry-run) ------------------

def _written():
    return SimpleNamespace(status="written", note=None, detail="",
                           audio=SimpleNamespace(name="x.m4a"), transcript=None)


def test_memos_cli_runs_sweep_then_process_then_indexes_when_written(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cli, "_require_vault_dir", lambda: None)
    monkeypatch.setattr(cli.db, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "require_vault_identity", lambda _con: None)
    monkeypatch.setattr(voicememos, "sweep",
                        lambda container, dry_run: (calls.append("sweep"), [])[1])
    monkeypatch.setattr(inbox, "process",
                        lambda **k: (calls.append("process"), [_written()])[1])
    monkeypatch.setattr(cli, "cmd_ingest", lambda args: calls.append("ingest"))

    cli.cmd_memos(SimpleNamespace(container=None, dry_run=False))
    assert calls == ["sweep", "process", "ingest"]


def test_memos_cli_skips_index_when_nothing_written(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cli, "_require_vault_dir", lambda: None)
    monkeypatch.setattr(cli.db, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "require_vault_identity", lambda _con: None)
    monkeypatch.setattr(voicememos, "sweep",
                        lambda container, dry_run: (calls.append("sweep"), [])[1])
    monkeypatch.setattr(inbox, "process",
                        lambda **k: (calls.append("process"), [])[1])
    monkeypatch.setattr(cli, "cmd_ingest", lambda args: calls.append("ingest"))

    cli.cmd_memos(SimpleNamespace(container=None, dry_run=False))
    assert calls == ["sweep", "process"]         # no ingest


def test_memos_cli_dry_run_never_indexes(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cli, "_require_vault_dir", lambda: None)
    monkeypatch.setattr(cli.db, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "require_vault_identity", lambda _con: None)
    monkeypatch.setattr(voicememos, "sweep",
                        lambda container, dry_run: (calls.append("sweep"), [])[1])
    monkeypatch.setattr(inbox, "process",
                        lambda **k: (calls.append("process"), [_written()])[1])
    monkeypatch.setattr(cli, "cmd_ingest", lambda args: calls.append("ingest"))

    cli.cmd_memos(SimpleNamespace(container=None, dry_run=True))
    assert "ingest" not in calls
