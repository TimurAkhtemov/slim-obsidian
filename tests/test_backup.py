"""The nightly backup script, driven against a fake restic."""

import shutil
import subprocess
from pathlib import Path

from slim import vaultpath


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
