"""Where the vault IS. Discovery replaced a hardcoded path, so every consumer of that path —
the CLI, the server's health report, the privacy guard, the backup — has to agree on it.
"""

import json
import subprocess
import sys
from pathlib import Path

from slim import vaultpath


def registry_at(home: Path, vaults: dict) -> None:
    """Write an Obsidian app registry under a fake home."""
    app_support = home / "Library" / "Application Support" / "obsidian"
    app_support.mkdir(parents=True, exist_ok=True)
    (app_support / "obsidian.json").write_text(json.dumps({"vaults": vaults}), encoding="utf-8")


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
