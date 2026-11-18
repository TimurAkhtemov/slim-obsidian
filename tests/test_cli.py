"""What the CLI commands print: `slim status` is the one command that reports state it would
refuse elsewhere."""

from types import SimpleNamespace

import pytest

from slim import cli, db
from slim.vault import identity
from slim.index.ingest import ingest
from tests.index.test_ingest import counts
from tests.vault.test_identity import stub_ingest, use_vault


def test_cmd_status_missing_vault(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "non_existent_vault"
    use_vault(monkeypatch, missing)

    from slim import db
    con = db.connect(":memory:")
    monkeypatch.setattr(db, "connect", lambda *a, **kw: con)
    monkeypatch.setattr(cli, "unembedded_count", lambda con: 0)

    cli.cmd_status(SimpleNamespace())
    assert "directory not found" in capsys.readouterr().out


def _status(monkeypatch, tmp_path, capsys, *, pinned=None, populated=False):
    """Run `slim status` against a vault at tmp_path/Vault and return the `pin:` line."""
    out = _status_out(monkeypatch, tmp_path, capsys, pinned=pinned, populated=populated)
    return next(line for line in out.splitlines() if line.startswith("pin:"))


def _status_out(monkeypatch, tmp_path, capsys, *, pinned=None, populated=False):
    """Run `slim status` against a vault at tmp_path/Vault and return everything it printed."""
    from slim import db
    vault = tmp_path / "Vault"
    vault.mkdir(exist_ok=True)
    con = stub_ingest(monkeypatch, tmp_path, vault)
    if populated:
        con.execute("INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
                    "VALUES ('a', 'Notes/a.md', 'a', 'note', '2026-09-01', 'hash')")
    if pinned is not None:
        identity.pin_vault(pinned)
    monkeypatch.setattr(db, "connect", lambda *a, **kw: con)
    cli.cmd_status(SimpleNamespace())
    return capsys.readouterr().out


def test_status_reports_a_pin_that_matches(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys, pinned=tmp_path / "Vault", populated=True)
    assert "⚠" not in line


def test_status_reports_an_index_built_from_another_vault(monkeypatch, tmp_path, capsys):
    """The one command you run to see what state SLIM is in has to say so — and still print
    the counts, rather than refusing the way a writing command does."""
    line = _status(monkeypatch, tmp_path, capsys, pinned=tmp_path / "Elsewhere", populated=True)
    assert "⚠" in line and "Elsewhere" in line and "--vault-moved" in line
    assert identity.pinned_vault() == tmp_path / "Elsewhere", "reporting never re-pins"


def test_status_reports_a_populated_index_with_no_pin(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys, populated=True)
    assert "⚠" in line and "--vault-moved" in line
    assert identity.pinned_vault() is None, "reporting never adopts"


def test_status_on_a_new_index_is_not_a_warning(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys)
    assert "⚠" not in line
    assert identity.pinned_vault() is None


def test_status_names_recordings_that_never_filed(monkeypatch, tmp_path, capsys):
    """A stranded recording is a meeting that never got filed; status is where they look."""
    incoming = tmp_path / "Vault" / "Attachments" / "_incoming"
    incoming.mkdir(parents=True)
    for name in ["20260924T184435-66c7d3cc-001.webm", "20260924T184435-66c7d3cc-001.webm.transcript.json",
                 "20260925T090000-aaaaaaaa-001.webm", "20260925T090000-aaaaaaaa-002.webm",
                 "20260827T070252-a742b2b3-001.webm.transcript.json",
                 "20260910T165802-0bbc930f-001.webm.transcript.json"]:
        (incoming / name).write_bytes(b"x")

    out = _status_out(monkeypatch, tmp_path, capsys)

    assert "stranded:  2 recordings in Attachments/_incoming never filed" in out
    assert "20260924T184435-66c7d3cc  2026-09-24" in out
    assert "20260925T090000-aaaaaaaa  2026-09-25  2 segments" in out
    assert "orphaned:  2 cached transcripts whose audio is gone" in out


def test_status_is_quiet_when_nothing_is_stranded(monkeypatch, tmp_path, capsys):
    out = _status_out(monkeypatch, tmp_path, capsys)
    assert "stranded:  none" in out
    assert "orphaned" not in out


def test_status_reports_the_newest_reflection(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "Vault" / "_Reflections"
    folder.mkdir(parents=True)
    for name in ["2026-08-04--reflection.md", "2026-08-22--reflection.md",
                 "2026-08-22--reflection-2.md", "2026-07-30--reflection.md", "notes.md"]:
        (folder / name).write_text("", encoding="utf-8")

    assert "reflected: 2026-08-22" in _status_out(monkeypatch, tmp_path, capsys)


def test_status_says_when_nothing_was_ever_reflected(monkeypatch, tmp_path, capsys):
    assert "reflected: never" in _status_out(monkeypatch, tmp_path, capsys)


@pytest.fixture(name="vault")
def _one_note_vault(tmp_path):
    v = tmp_path / "vault"
    (v / "Projects/Demo").mkdir(parents=True)
    (v / "Projects/Demo/note.md").write_text(
        "---\ntitle: \"Demo Note\"\ntype: meeting-note\ndate: 2026-07-01\n---\n\n"
        "# Demo Note\n\n## Decisions\n\n"
        "We decided to use SQLite with FTS5 for the index.\n")
    return v


@pytest.fixture(name="con")
def _db(tmp_path):
    return db.connect(tmp_path / "test.db")


def test_status_reports_fragments_without_a_current_vector(con, vault, monkeypatch, capsys):
    """`embedded:` counts rows in `embeddings`, which includes STALE vectors, so after a content
    pass it can equal `fragments:` while retrieval runs on a fraction of the corpus (measured
    2026-08-04: 2,802 of 3,403). `slim status` must show the counter that catches it."""
    from slim import cli
    ingest(con, vault)
    fragments = counts(con)["fragments"]
    assert fragments > 0
    monkeypatch.setattr(cli.db, "connect", lambda *a, **k: con)
    cli.cmd_status(None)
    out = capsys.readouterr().out
    assert f"unembedded: {fragments}" in out
    assert "slim ingest" in out


# --- slim mask ----------------------------------------------------------------------------

def _run_mask(monkeypatch, *argv):
    import sys
    monkeypatch.setattr(sys, "argv", ["slim", "mask", *argv])
    cli.main()


@pytest.fixture
def mask_vault(tmp_path, monkeypatch):
    from slim import config
    from slim.capture import pii
    from slim.recorder import state
    from tests.recorder.test_mask import _recorded
    v = tmp_path / "vault"
    _recorded(v, "Notes/w/a.md")
    _recorded(v, "Notes/w/b.md", transcript="**Me:** nothing here.")
    monkeypatch.setattr(cli, "VAULT", v)
    monkeypatch.setattr(config, "PII_MASK", config.PII_DEFAULT)
    monkeypatch.setattr(config, "PII_SKIP", ())
    monkeypatch.setattr(pii, "ask_model", lambda window: None)
    monkeypatch.setattr(identity, "require_vault_dir", lambda: None)
    monkeypatch.setattr(identity, "require_vault_identity", lambda con, **kw: None)
    monkeypatch.setattr(state, "index_now", lambda vault, rel, **kw: {"note": rel})
    compacted = []
    monkeypatch.setattr(db, "compact", lambda con: compacted.append(1))
    return v, compacted


def test_mask_with_no_notes_exits_non_zero(monkeypatch, mask_vault):
    with pytest.raises(SystemExit) as e:
        _run_mask(monkeypatch)
    assert e.value.code not in (0, None)


def test_mask_all_prints_counts_never_values(monkeypatch, mask_vault, capsys):
    v, compacted = mask_vault
    _run_mask(monkeypatch, "--all")
    out = capsys.readouterr().out
    assert out.splitlines() == ["Notes/w/a.md  email:1 ssn:1", "Notes/w/b.md  (unchanged)",
                                "1 notes masked, 1 unchanged, 0 skipped"]
    assert "219-09-9999" not in out and "@" not in out
    assert compacted == [1]
    _run_mask(monkeypatch, "--all")
    out = capsys.readouterr().out
    assert out.splitlines() == ["Notes/w/a.md  (already masked)", "Notes/w/b.md  (already masked)",
                                "0 notes masked, 0 unchanged, 2 skipped"]
    assert compacted == [1]                  # nothing changed, nothing to compact


def test_mask_dry_run_writes_and_compacts_nothing(monkeypatch, mask_vault, capsys):
    v, compacted = mask_vault
    before = (v / "Notes/w/a.md").read_bytes()
    _run_mask(monkeypatch, "Notes/w/a.md", "--dry-run")
    out = capsys.readouterr().out
    assert out.splitlines() == ["Notes/w/a.md  email:1 ssn:1",
                                "dry run: 1 notes masked, 0 unchanged, 0 skipped"]
    assert (v / "Notes/w/a.md").read_bytes() == before and compacted == []


def test_mask_off_reads_no_note(monkeypatch, mask_vault, capsys):
    from slim import config
    from slim.recorder import mask
    monkeypatch.setattr(config, "PII_MASK", ())
    monkeypatch.setattr(mask, "mask_note", lambda *a, **k: pytest.fail("read a note"))
    monkeypatch.setattr(mask, "recordings", lambda *a, **k: pytest.fail("listed notes"))
    with pytest.raises(SystemExit) as e:
        _run_mask(monkeypatch, "--all")
    assert e.value.code not in (0, None)
    assert "masking is off: pii.mask is empty in projects.yaml" in capsys.readouterr().out


def test_mask_says_when_a_note_was_not_reindexed(monkeypatch, mask_vault, capsys):
    from slim.recorder import state
    monkeypatch.setattr(state, "index_now", lambda vault, rel, **kw: {"error": "locked"})
    with pytest.raises(SystemExit) as e:
        _run_mask(monkeypatch, "--all")
    assert e.value.code not in (0, None)
    out = capsys.readouterr().out
    assert "Notes/w/a.md  email:1 ssn:1\n  ⚠ not re-indexed: locked\n" in out
    assert "1 notes masked, 1 unchanged, 0 skipped" in out


def test_mask_survives_a_bad_note_and_a_busy_index(monkeypatch, mask_vault, capsys):
    import sqlite3
    v, _ = mask_vault
    (v / "Notes/w/c.md").write_bytes(b"---\norigin: recorded\n---\n\xff\xfe\n")

    def busy(con):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "compact", busy)
    with pytest.raises(SystemExit) as e:
        _run_mask(monkeypatch, "Notes/w/c.md", "Notes/w/a.md", "--force")
    assert e.value.code not in (0, None)          # the bad note is an error
    out = capsys.readouterr().out
    assert "Notes/w/c.md  (error: UnicodeDecodeError)" in out
    assert "Notes/w/a.md  email:1 ssn:1" in out
    assert "⚠ index not compacted (database in use)" in out
    assert "1 notes masked, 0 unchanged, 1 skipped" in out


def test_mask_marks_partial_notes(monkeypatch, mask_vault, capsys):
    from slim.capture import pii
    from tests.recorder.test_mask import _recorded
    v, _ = mask_vault
    _recorded(v, "Notes/w/p.md", transcript="**Me:** the password is tango42.")

    def down(window):
        raise RuntimeError("down")

    monkeypatch.setattr(pii, "ask_model", down)
    _run_mask(monkeypatch, "Notes/w/p.md")
    assert capsys.readouterr().out.splitlines()[0] == "Notes/w/p.md  (unchanged) (partial: password)"
