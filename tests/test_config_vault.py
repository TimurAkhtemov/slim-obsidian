"""Where the vault IS. Discovery replaced a hardcoded path, so every consumer of that path —
the CLI, the server's health report, the privacy guard, the backup — has to agree on it.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from slim import chat as chat_mod, cli, config, vaultpath

HOOK = Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "private-content-guard.py"


def registry_at(home: Path, vaults: dict) -> None:
    """Write an Obsidian app registry under a fake home."""
    app_support = home / "Library" / "Application Support" / "obsidian"
    app_support.mkdir(parents=True, exist_ok=True)
    (app_support / "obsidian.json").write_text(json.dumps({"vaults": vaults}), encoding="utf-8")


# --- discovery ------------------------------------------------------------------------------

def test_discover_vault_from_env(monkeypatch, tmp_path):
    custom_vault = tmp_path / "MyCustomVault"
    custom_vault.mkdir()
    monkeypatch.setenv("SLIM_VAULT", str(custom_vault))

    assert vaultpath.discover_vault() == custom_vault.resolve()


def test_discover_vault_from_obsidian_registry(monkeypatch, tmp_path):
    monkeypatch.delenv("SLIM_VAULT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    active_vault = tmp_path / "ActiveVault"
    active_vault.mkdir()
    older_vault = tmp_path / "OlderVault"
    older_vault.mkdir()
    registry_at(tmp_path, {
        "v1": {"path": str(older_vault), "ts": 1000, "open": False},
        "v2": {"path": str(active_vault), "ts": 2000, "open": True},
        # A vault that has been deleted from disk is not a candidate, however recent.
        "v3": {"path": str(tmp_path / "DeletedVault"), "ts": 3000, "open": True},
    })

    assert vaultpath._vault_from_obsidian_registry() == active_vault.resolve()


def test_icloud_accepts_a_directory_that_holds_dot_obsidian(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    icloud = tmp_path / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
    vault = icloud / "MyIcloudVault"
    (vault / ".obsidian").mkdir(parents=True)

    assert vaultpath._vault_from_icloud() == vault.resolve()


def test_icloud_refuses_a_container_with_no_vault_in_it(monkeypatch, tmp_path):
    """The container existing is not proof a vault is in it. Returning an arbitrary folder
    would pass every existence check and then let `ingest` mark the real vault removed."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    icloud = tmp_path / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
    (icloud / "Some Other Folder").mkdir(parents=True)
    (icloud / ".Trash").mkdir()

    assert vaultpath._vault_from_icloud() is None


def test_discover_vault_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("SLIM_VAULT", raising=False)
    monkeypatch.setattr(vaultpath, "_vault_from_obsidian_registry", lambda: None)
    monkeypatch.setattr(vaultpath, "_vault_from_icloud", lambda: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert vaultpath.discover_vault() == tmp_path / "Documents" / "Obsidian Vault"


def test_vaultpath_runs_as_a_script(tmp_path):
    """`scripts/backup.sh` and the README install block ask the module for the path, with the
    SYSTEM python — so it must run standalone, with no package import and no third party."""
    vault = tmp_path / "Vault"
    vault.mkdir()
    out = subprocess.run(
        [sys.executable, str(Path(vaultpath.__file__)), ],
        env={"SLIM_VAULT": str(vault), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(vault.resolve())


# --- the CLI --------------------------------------------------------------------------------

def stub_ingest(monkeypatch, tmp_path, vault):
    """Drive cmd_ingest without a real vault, DB or embedder."""
    from slim import db
    con = db.connect(":memory:")
    monkeypatch.setattr(cli, "VAULT", vault)
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
    monkeypatch.setattr(cli, "VAULT", missing)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_ingest(SimpleNamespace(verbose=False))
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert "Vault directory not found" in out
    assert "export SLIM_VAULT=" in out
    assert ".env" not in out


def test_cmd_status_missing_vault(monkeypatch, tmp_path, capsys):
    missing = tmp_path / "non_existent_vault"
    monkeypatch.setattr(cli, "VAULT", missing)

    from slim import db
    con = db.connect(":memory:")
    monkeypatch.setattr(db, "connect", lambda *a, **kw: con)
    monkeypatch.setattr(cli, "unembedded_count", lambda con: 0)

    cli.cmd_status(SimpleNamespace())
    assert "directory not found" in capsys.readouterr().out


def _status(monkeypatch, tmp_path, capsys, *, pinned=None, populated=False):
    """Run `slim status` against a vault at tmp_path/Vault and return the `pin:` line."""
    from slim import db
    vault = tmp_path / "Vault"
    vault.mkdir(exist_ok=True)
    con = stub_ingest(monkeypatch, tmp_path, vault)
    if populated:
        con.execute("INSERT INTO sources (id, path, title, type, authored_at, current_hash) "
                    "VALUES ('a', 'Notes/a.md', 'a', 'note', '2026-09-01', 'hash')")
    if pinned is not None:
        cli.pin_vault(pinned)
    monkeypatch.setattr(db, "connect", lambda *a, **kw: con)
    cli.cmd_status(SimpleNamespace())
    return next(line for line in capsys.readouterr().out.splitlines() if line.startswith("pin:"))


def test_status_reports_a_pin_that_matches(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys, pinned=tmp_path / "Vault", populated=True)
    assert "⚠" not in line


def test_status_reports_an_index_built_from_another_vault(monkeypatch, tmp_path, capsys):
    """The one command you run to see what state SLIM is in has to say so — and still print
    the counts, rather than refusing the way a writing command does."""
    line = _status(monkeypatch, tmp_path, capsys, pinned=tmp_path / "Elsewhere", populated=True)
    assert "⚠" in line and "Elsewhere" in line and "--vault-moved" in line
    assert cli.pinned_vault() == tmp_path / "Elsewhere", "reporting never re-pins"


def test_status_reports_a_populated_index_with_no_pin(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys, populated=True)
    assert "⚠" in line and "--vault-moved" in line
    assert cli.pinned_vault() is None, "reporting never adopts"


def test_status_on_a_new_index_is_not_a_warning(monkeypatch, tmp_path, capsys):
    line = _status(monkeypatch, tmp_path, capsys)
    assert "⚠" not in line
    assert cli.pinned_vault() is None


def test_first_ingest_pins_the_vault_it_indexed(monkeypatch, tmp_path):
    vault = tmp_path / "Vault"
    vault.mkdir()
    stub_ingest(monkeypatch, tmp_path, vault)

    cli.cmd_ingest(SimpleNamespace(verbose=False))

    assert cli.pinned_vault() == vault


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
    assert cli.pinned_vault() is None
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
    assert cli.pinned_vault() is None
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

    assert cli.pinned_vault() == vault


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

    assert cli.pinned_vault() == vault


def test_ingest_refuses_a_vault_the_index_was_not_built_from(monkeypatch, tmp_path, capsys):
    """The removal sweep deletes every source not found under the current vault. Discovery can
    now change that path with nobody asking, so a mismatch must stop the run, not wipe the DB."""
    first = tmp_path / "First"
    first.mkdir()
    stub_ingest(monkeypatch, tmp_path, first)
    cli.cmd_ingest(SimpleNamespace(verbose=False))

    second = tmp_path / "Second"
    second.mkdir()
    monkeypatch.setattr(cli, "VAULT", second)
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
    assert cli.pinned_vault() == first       # the pin is not moved by a refusal


def test_vault_moved_repoints_the_pin(monkeypatch, tmp_path):
    first = tmp_path / "First"
    first.mkdir()
    stub_ingest(monkeypatch, tmp_path, first)
    cli.cmd_ingest(SimpleNamespace(verbose=False))

    second = tmp_path / "Second"
    second.mkdir()
    monkeypatch.setattr(cli, "VAULT", second)

    cli.cmd_ingest(SimpleNamespace(verbose=False, vault_moved=True))

    assert cli.pinned_vault() == second


def test_memos_refuses_a_mismatched_vault_before_copying(monkeypatch, tmp_path):
    from slim import inbox, voicememos

    first = tmp_path / "First"
    first.mkdir()
    second = tmp_path / "Second"
    second.mkdir()
    con = stub_ingest(monkeypatch, tmp_path, first)
    cli.pin_vault(first)
    monkeypatch.setattr(cli, "VAULT", second)
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
    cli.pin_vault(first)
    monkeypatch.setattr(cli, "VAULT", second)
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
    cli.pin_vault(first)
    monkeypatch.setattr(cli, "VAULT", second)
    calls = []
    monkeypatch.setattr(chat_mod, "serve", lambda *a, **kw: calls.append(1))

    with pytest.raises(SystemExit):
        cli.cmd_chat(SimpleNamespace(parent_pid=None, port=7546, reload=False,
                                     host="127.0.0.1"))

    assert calls == []
    con.close()


# --- refusals are traced (finding #8) -------------------------------------------------------

def test_vault_dir_missing_is_traced(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "VAULT", tmp_path / "gone")
    traced = []
    monkeypatch.setattr(cli.trace, "record", lambda kind, data: traced.append((kind, data)))
    with pytest.raises(SystemExit):
        cli._require_vault_dir()
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
    cli.pin_vault(first)
    monkeypatch.setattr(cli, "VAULT", second)
    traced = []
    monkeypatch.setattr(cli.trace, "record", lambda kind, data: traced.append((kind, data)))
    with pytest.raises(SystemExit):
        cli.cmd_ingest(SimpleNamespace(verbose=False))
    assert any(k == "vault_identity" and d["reason"] == "pinned_mismatch" for k, d in traced)


# --- the server -----------------------------------------------------------------------------

def test_health_reports_the_serving_vault(monkeypatch, tmp_path):
    """The plugin sends vault-RELATIVE paths and reuses whatever is on 7546. A server that
    resolves them against a different vault has to be visible from the outside."""
    vault = tmp_path / "ServedVault"
    monkeypatch.setattr(config, "VAULT", vault)

    assert chat_mod.code_state()["vault"] == str(vault)


# --- the privacy guard ----------------------------------------------------------------------

def guard(event: dict, home: Path):
    return subprocess.run(
        [sys.executable, str(HOOK)], input=json.dumps(event),
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True)


def test_guard_protects_the_discovered_vault(tmp_path):
    """The motivating case: the vault moves, SLIM_VAULT is unset. A guard still pointing at
    the old hardcoded path matches nothing and passes journal transcripts through."""
    vault = tmp_path / "Moved Vault"
    (vault / "Journal").mkdir(parents=True)
    (vault / ".obsidian").mkdir()
    registry_at(tmp_path, {"v1": {"path": str(vault), "ts": 1, "open": True}})

    out = guard({"tool_name": "Read",
                 "tool_input": {"file_path": str(vault / "Journal" / "2026-09-10.md")}}, tmp_path)
    assert out.returncode == 2
    assert "private" in out.stderr


def test_guard_still_allows_the_rest_of_the_vault(tmp_path):
    vault = tmp_path / "Moved Vault"
    (vault / "Capture").mkdir(parents=True)
    (vault / ".obsidian").mkdir()
    registry_at(tmp_path, {"v1": {"path": str(vault), "ts": 1, "open": True}})

    out = guard({"tool_name": "Read",
                 "tool_input": {"file_path": str(vault / "Capture" / "meeting.md")}}, tmp_path)
    assert out.returncode == 0


def test_guard_fails_closed_when_shared_discovery_cannot_load(tmp_path):
    """A broken checkout must not silently restore the obsolete path and expose the real
    Journal while claiming the guard is active."""
    hook = tmp_path / "repo" / ".claude" / "hooks" / HOOK.name
    hook.parent.mkdir(parents=True)
    shutil.copy2(HOOK, hook)  # deliberately do not copy slim/vaultpath.py

    out = subprocess.run(
        [sys.executable, str(hook)], input=json.dumps({"tool_name": "Read",
                                                       "tool_input": {"file_path": "/tmp/x"}}),
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)

    assert out.returncode == 2
    assert "could not determine the vault" in out.stderr


# --- the backup -----------------------------------------------------------------------------

def test_backup_loads_the_env_override_before_discovering_the_vault(tmp_path):
    """launchd supplies no SLIM_VAULT. The script's .env is its persistent configuration, so
    the override in that file must be present before vaultpath.py is asked."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "slim").mkdir()
    shutil.copy2(Path(__file__).resolve().parent.parent / "scripts" / "backup.sh",
                 repo / "scripts" / "backup.sh")
    shutil.copy2(Path(vaultpath.__file__), repo / "slim" / "vaultpath.py")
    vault = tmp_path / "Configured Vault"
    vault.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    restic = fake_bin / "restic"
    restic.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    restic.chmod(0o755)
    (repo / ".env").write_text(
        f'export PATH="{fake_bin}:/usr/bin:/bin"\n'
        f'export SLIM_VAULT="{vault}"\n'
        'export RESTIC_REPOSITORY="test"\n'
        'export RESTIC_PASSWORD="test"\n', encoding="utf-8")

    out = subprocess.run(
        ["/bin/bash", str(repo / "scripts" / "backup.sh")],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=True)

    assert f"backup: vault={vault}" in out.stdout


def test_backup_stops_before_pruning_when_the_vault_shrinks(tmp_path):
    """Under launchd bash cannot read the vault, so restic's own file count is the only sign
    of evicted iCloud files. A snapshot that lost most of the vault must fail the run before
    `forget` prunes the snapshots that still hold it."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "slim").mkdir()
    shutil.copy2(Path(__file__).resolve().parent.parent / "scripts" / "backup.sh",
                 repo / "scripts" / "backup.sh")
    shutil.copy2(Path(vaultpath.__file__), repo / "slim" / "vaultpath.py")
    vault = tmp_path / "vault"
    vault.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls"
    snaps = ('[{"time":"2026-09-21T12:30:00Z","summary":{"total_files_processed":996}},'
             '{"time":"2026-09-22T12:30:00Z","summary":{"total_files_processed":400}}]')
    restic = fake_bin / "restic"
    restic.write_text(
        f'#!/bin/sh\necho "$1" >> "{calls}"\n'
        f'[ "$1" = snapshots ] && echo \'{snaps}\'\nexit 0\n', encoding="utf-8")
    restic.chmod(0o755)
    (repo / ".env").write_text(
        f'export PATH="{fake_bin}:/usr/bin:/bin"\n'
        f'export SLIM_VAULT="{vault}"\n'
        'export RESTIC_REPOSITORY="test"\n'
        'export RESTIC_PASSWORD="test"\n', encoding="utf-8")

    out = subprocess.run(
        ["/bin/bash", str(repo / "scripts" / "backup.sh")],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True)

    assert out.returncode != 0
    assert "read 400 files; the previous one read 996" in out.stderr
    assert "forget" not in calls.read_text().split()
