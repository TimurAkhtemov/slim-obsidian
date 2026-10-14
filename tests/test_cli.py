"""What the CLI commands print: `slim status` is the one command that reports state it would
refuse elsewhere."""

from types import SimpleNamespace

from slim import cli, vaultidentity
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
        vaultidentity.pin_vault(pinned)
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
    assert vaultidentity.pinned_vault() == tmp_path / "Elsewhere", "reporting never re-pins"


def test_status_reports_a_populated_index_with_no_pin(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys, populated=True)
    assert "⚠" in line and "--vault-moved" in line
    assert vaultidentity.pinned_vault() is None, "reporting never adopts"


def test_status_on_a_new_index_is_not_a_warning(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys)
    assert "⚠" not in line
    assert vaultidentity.pinned_vault() is None


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
