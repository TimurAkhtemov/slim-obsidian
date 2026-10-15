"""The index is pinned to the vault it was built from: every command that can write refuses a
different or unidentified vault before it acts, and says so in the trace."""

from types import SimpleNamespace

import pytest

from slim import server as chat_mod, cli, config
from slim.vault import identity


def use_vault(monkeypatch, vault):
    """The CLI reports the vault and vault.identity guards it; each reads it at import."""
    monkeypatch.setattr(cli, "VAULT", vault)
    monkeypatch.setattr(identity, "VAULT", vault)


def stub_ingest(monkeypatch, tmp_path, vault):
    """Drive cmd_ingest without a real vault, DB or embedder."""
    from slim import db
    con = db.connect(":memory:")
    use_vault(monkeypatch, vault)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cli.db, "connect", lambda *a, **kw: con)
    monkeypatch.setattr(cli.ingest_mod, "ingest", lambda con, verbose=False: {"removed": 0})
    monkeypatch.setattr(cli.embed_mod, "embed_missing", lambda con, verbose=True: {})
    monkeypatch.setattr(cli, "unembedded_count", lambda con: 0)
    monkeypatch.setattr(cli.trace, "record", lambda *a, **kw: None)
    return con


def test_cmd_ingest_missing_vault_names_a_shell_export(monkeypatch, tmp_path, capsys):
    """`.env` is not read for CLI configuration, so naming it would leave the user stuck."""
    missing = tmp_path / "non_existent_vault"
    use_vault(monkeypatch, missing)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_ingest(SimpleNamespace(verbose=False))
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "Vault directory not found" in out
    assert "export SLIM_VAULT=" in out
    assert ".env" not in out


def test_first_ingest_pins_the_vault_it_indexed(monkeypatch, tmp_path):
    vault = tmp_path / "Vault"
    vault.mkdir()
    stub_ingest(monkeypatch, tmp_path, vault)

    cli.cmd_ingest(SimpleNamespace(verbose=False))

    assert identity.pinned_vault() == vault


def test_existing_unpinned_index_refuses_before_the_removal_sweep(
        monkeypatch, tmp_path, capsys):
    """The first run after this feature lands has a database but no pin. That is not a new
    index, and silently adopting discovery would leave the exact destructive upgrade gap the
    pin exists to close."""
    vault = tmp_path / "Discovered"
    vault.mkdir()
    monkeypatch.delenv("SLIM_VAULT", raising=False)
    con = stub_ingest(monkeypatch, tmp_path, vault)
    con.execute(
        "INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
        "VALUES ('old', 'Journal/old.md', 'old', 'journal', '2026-09-01', 'hash')")
    con.commit()
    calls = []
    monkeypatch.setattr(cli.ingest_mod, "ingest",
                        lambda con, verbose=False: calls.append(1) or {"removed": 0})

    with pytest.raises(SystemExit) as exc:
        cli.cmd_ingest(SimpleNamespace(verbose=False))

    assert exc.value.code == 1
    assert calls == []
    assert identity.pinned_vault() is None
    assert "existing index has no vault pin" in capsys.readouterr().out


def test_explicit_env_cannot_adopt_an_existing_unpinned_index(monkeypatch, tmp_path, capsys):
    """An env-named vault must not silently adopt an unpinned populated index: the plugin
    always sets SLIM_VAULT, so the old bypass let opening any vault pin it — and the next
    ingest sweep would delete every source from the real vault."""
    vault = tmp_path / "Named"
    vault.mkdir()
    monkeypatch.setenv("SLIM_VAULT", str(vault))
    con = stub_ingest(monkeypatch, tmp_path, vault)
    con.execute(
        "INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
        "VALUES ('old', 'Journal/old.md', 'old', 'journal', '2026-09-01', 'hash')")
    con.commit()

    with pytest.raises(SystemExit) as exc:
        cli.cmd_ingest(SimpleNamespace(verbose=False))

    assert exc.value.code == 1
    assert identity.pinned_vault() is None
    assert "--vault-moved" in capsys.readouterr().out


def test_existing_unpinned_index_can_explicitly_adopt_the_current_vault(monkeypatch, tmp_path):
    vault = tmp_path / "Moved"
    vault.mkdir()
    con = stub_ingest(monkeypatch, tmp_path, vault)
    con.execute(
        "INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
        "VALUES ('old', 'Journal/old.md', 'old', 'journal', '2026-09-01', 'hash')")
    con.commit()

    cli.cmd_ingest(SimpleNamespace(verbose=False, vault_moved=True))

    assert identity.pinned_vault() == vault


def test_ingest_pins_before_embedding_can_fail(monkeypatch, tmp_path):
    """ingest() commits before embedding. A model outage must not leave that committed index
    without the identity needed to protect its next removal sweep."""
    vault = tmp_path / "Vault"
    vault.mkdir()
    stub_ingest(monkeypatch, tmp_path, vault)
    monkeypatch.setattr(cli.embed_mod, "embed_missing",
                        lambda con, verbose=True: (_ for _ in ()).throw(RuntimeError("offline")))

    with pytest.raises(RuntimeError, match="offline"):
        cli.cmd_ingest(SimpleNamespace(verbose=False))

    assert identity.pinned_vault() == vault


def test_ingest_refuses_a_vault_the_index_was_not_built_from(monkeypatch, tmp_path, capsys):
    """The removal sweep deletes every source not found under the current vault. Discovery can
    now change that path with nobody asking, so a mismatch must stop the run, not wipe the DB."""
    first = tmp_path / "First"
    first.mkdir()
    stub_ingest(monkeypatch, tmp_path, first)
    cli.cmd_ingest(SimpleNamespace(verbose=False))

    second = tmp_path / "Second"
    second.mkdir()
    use_vault(monkeypatch, second)
    calls = []
    monkeypatch.setattr(cli.ingest_mod, "ingest",
                        lambda con, verbose=False: calls.append(1) or {"removed": 0})

    with pytest.raises(SystemExit) as exc:
        cli.cmd_ingest(SimpleNamespace(verbose=False))
    assert exc.value.code == 1
    assert calls == []                       # refused BEFORE the sweep, not after

    out = capsys.readouterr().out
    assert str(first) in out and str(second) in out
    assert "--vault-moved" in out
    assert identity.pinned_vault() == first       # the pin is not moved by a refusal


def test_vault_moved_repoints_the_pin(monkeypatch, tmp_path):
    first = tmp_path / "First"
    first.mkdir()
    stub_ingest(monkeypatch, tmp_path, first)
    cli.cmd_ingest(SimpleNamespace(verbose=False))

    second = tmp_path / "Second"
    second.mkdir()
    use_vault(monkeypatch, second)

    cli.cmd_ingest(SimpleNamespace(verbose=False, vault_moved=True))

    assert identity.pinned_vault() == second


def test_memos_refuses_a_mismatched_vault_before_copying(monkeypatch, tmp_path):
    from slim.memos import inbox, voicememos

    first = tmp_path / "First"
    first.mkdir()
    second = tmp_path / "Second"
    second.mkdir()
    con = stub_ingest(monkeypatch, tmp_path, first)
    identity.pin_vault(first)
    use_vault(monkeypatch, second)
    calls = []
    monkeypatch.setattr(voicememos, "sweep", lambda **kw: calls.append("sweep") or [])
    monkeypatch.setattr(inbox, "process", lambda **kw: calls.append("process") or [])

    with pytest.raises(SystemExit):
        cli.cmd_memos(SimpleNamespace(container=None, dry_run=False))

    assert calls == []
    con.close()


def test_reflect_refuses_a_mismatched_vault_before_the_model_or_write(monkeypatch, tmp_path):
    from slim import reflect

    first = tmp_path / "First"
    first.mkdir()
    second = tmp_path / "Second"
    second.mkdir()
    con = stub_ingest(monkeypatch, tmp_path, first)
    identity.pin_vault(first)
    use_vault(monkeypatch, second)
    calls = []
    monkeypatch.setattr(reflect, "reflect", lambda *a, **kw: calls.append(1))

    with pytest.raises(SystemExit):
        cli.cmd_reflect(SimpleNamespace(days=None, min_entries=1, dry_run=False))

    assert calls == []
    con.close()


def test_chat_refuses_a_mismatched_vault_before_serving_writes(monkeypatch, tmp_path):
    first = tmp_path / "First"
    first.mkdir()
    second = tmp_path / "Second"
    second.mkdir()
    con = stub_ingest(monkeypatch, tmp_path, first)
    identity.pin_vault(first)
    use_vault(monkeypatch, second)
    calls = []
    monkeypatch.setattr(chat_mod, "serve", lambda *a, **kw: calls.append(1))

    with pytest.raises(SystemExit):
        cli.cmd_chat(SimpleNamespace(parent_pid=None, port=7546, reload=False,
                                     host="127.0.0.1"))

    assert calls == []
    con.close()


# --- refusals are traced (finding #8) -------------------------------------------------------

def test_vault_dir_missing_is_traced(monkeypatch, tmp_path):
    use_vault(monkeypatch, tmp_path / "gone")
    traced = []
    monkeypatch.setattr(cli.trace, "record", lambda kind, data: traced.append((kind, data)))
    with pytest.raises(SystemExit):
        identity.require_vault_dir()
    assert any(k == "vault_identity" and d["reason"] == "vault_dir_missing" for k, d in traced)


def test_unpinned_populated_refusal_is_traced(monkeypatch, tmp_path):
    vault = tmp_path / "V"
    vault.mkdir()
    con = stub_ingest(monkeypatch, tmp_path, vault)
    con.execute(
        "INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
        "VALUES ('s1', 'x.md', 't', 'note', '2026-01-01', 'h')")
    con.commit()
    traced = []
    monkeypatch.setattr(cli.trace, "record", lambda kind, data: traced.append((kind, data)))
    with pytest.raises(SystemExit):
        cli.cmd_ingest(SimpleNamespace(verbose=False))
    assert any(k == "vault_identity" and d["reason"] == "unpinned_populated" for k, d in traced)


def test_pinned_mismatch_refusal_is_traced(monkeypatch, tmp_path):
    first = tmp_path / "A"
    first.mkdir()
    second = tmp_path / "B"
    second.mkdir()
    con = stub_ingest(monkeypatch, tmp_path, first)
    identity.pin_vault(first)
    use_vault(monkeypatch, second)
    traced = []
    monkeypatch.setattr(cli.trace, "record", lambda kind, data: traced.append((kind, data)))
    with pytest.raises(SystemExit):
        cli.cmd_ingest(SimpleNamespace(verbose=False))
    assert any(k == "vault_identity" and d["reason"] == "pinned_mismatch" for k, d in traced)
