"""What the CLI commands print: `slim status` is the one command that reports state it would
refuse elsewhere."""

from types import SimpleNamespace

import pytest

from slim import cli, db
from slim.memos import inbox, voicememos
from slim.vault import identity
from slim.index.ingest import ingest
from tests.test_ingest import counts
from tests.test_vaultidentity import stub_ingest, use_vault


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


# --- CLI wiring: sweep -> process -> (index only if written, not dry-run) ------------------

def _written():
    return SimpleNamespace(status="written", note=None, detail="",
                           audio=SimpleNamespace(name="x.m4a"), transcript=None)


def test_memos_cli_runs_sweep_then_process_then_indexes_when_written(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(identity, "require_vault_dir", lambda: None)
    monkeypatch.setattr(cli.db, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(identity, "require_vault_identity", lambda _con: None)
    monkeypatch.setattr(voicememos, "sweep",
                        lambda container, dry_run: (calls.append("sweep"), [])[1])
    monkeypatch.setattr(inbox, "process",
                        lambda **k: (calls.append("process"), [_written()])[1])
    monkeypatch.setattr(cli, "cmd_ingest", lambda args: calls.append("ingest"))

    cli.cmd_memos(SimpleNamespace(container=None, dry_run=False))
    assert calls == ["sweep", "process", "ingest"]


def test_memos_cli_skips_index_when_nothing_written(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(identity, "require_vault_dir", lambda: None)
    monkeypatch.setattr(cli.db, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(identity, "require_vault_identity", lambda _con: None)
    monkeypatch.setattr(voicememos, "sweep",
                        lambda container, dry_run: (calls.append("sweep"), [])[1])
    monkeypatch.setattr(inbox, "process",
                        lambda **k: (calls.append("process"), [])[1])
    monkeypatch.setattr(cli, "cmd_ingest", lambda args: calls.append("ingest"))

    cli.cmd_memos(SimpleNamespace(container=None, dry_run=False))
    assert calls == ["sweep", "process"]         # no ingest


def test_memos_cli_dry_run_never_indexes(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(identity, "require_vault_dir", lambda: None)
    monkeypatch.setattr(cli.db, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(identity, "require_vault_identity", lambda _con: None)
    monkeypatch.setattr(voicememos, "sweep",
                        lambda container, dry_run: (calls.append("sweep"), [])[1])
    monkeypatch.setattr(inbox, "process",
                        lambda **k: (calls.append("process"), [_written()])[1])
    monkeypatch.setattr(cli, "cmd_ingest", lambda args: calls.append("ingest"))

    cli.cmd_memos(SimpleNamespace(container=None, dry_run=True))
    assert "ingest" not in calls
