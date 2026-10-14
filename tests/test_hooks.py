"""The privacy guard finds the vault the way SLIM does, and fails closed when it cannot."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from tests.test_vaultpath import registry_at

HOOK = Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "private-content-guard.py"


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
