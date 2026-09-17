"""Where the Obsidian vault is. ONE copy of that answer, for four callers.

`slim/config.py` imports it; `.claude/hooks/private-content-guard.py` and `scripts/backup.sh`
load this FILE directly, so it must stand alone:

  - stdlib only (the hook and the backup run on the system python, not the project venv),
  - no relative import (it is loaded by path, and run as `python3 slim/vaultpath.py`),
  - `from __future__ import annotations`, because the system python may be 3.9.

A second copy of this logic in the hook would be a copy that goes stale — and the copy that
went stale is exactly how the guard came to protect a folder that no longer existed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _vault_from_obsidian_registry() -> Path | None:
    """The open, or most recently opened, vault in Obsidian's own app registry."""
    candidates = [
        Path.home() / "Library" / "Application Support" / "obsidian" / "obsidian.json",
        Path.home() / ".config" / "obsidian" / "obsidian.json",
    ]
    appdata = os.environ.get("APPDATA")
    candidates.append(Path(appdata) / "obsidian" / "obsidian.json" if appdata
                      else Path.home() / "AppData" / "Roaming" / "obsidian" / "obsidian.json")

    for config_path in candidates:
        if not config_path.is_file():
            continue
        try:
            vaults = json.loads(config_path.read_text(encoding="utf-8")).get("vaults")
        except Exception:
            continue
        if not isinstance(vaults, dict):
            continue
        valid = []
        for v in vaults.values():
            # Per ENTRY: one malformed record must not discard the vaults already collected.
            try:
                if isinstance(v, dict) and v.get("path"):
                    p = Path(v["path"]).expanduser().resolve()
                    if p.is_dir():
                        valid.append((bool(v.get("open")), int(v.get("ts", 0) or 0), p))
            except Exception:
                continue
        if valid:
            valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return valid[0][2]
    return None


def _vault_from_icloud() -> Path | None:
    """A vault inside iCloud Drive's Obsidian container — the iOS app's sync location.

    ⚠ `.obsidian` is REQUIRED, not preferred. The container existing is not proof a vault is
    in it, and returning an arbitrary folder would pass every existence check downstream and
    then let `ingest`'s removal sweep mark every source in the real vault deleted.
    """
    icloud_dir = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
    if not icloud_dir.is_dir():
        return None
    try:
        vaults = [p.resolve() for p in icloud_dir.iterdir()
                  if p.is_dir() and (p / ".obsidian").is_dir()]
        if vaults:
            vaults.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return vaults[0]
    except Exception:
        pass
    return None


def discover_vault() -> Path:
    """The vault this machine is working with.

    1. `SLIM_VAULT` — the explicit override.
    2. Obsidian's app registry — where the vault actually is, including after a move.
    3. iCloud's Obsidian container — a vault synced to the iOS app.
    4. The historical default.
    """
    env_val = os.environ.get("SLIM_VAULT")
    if env_val:
        return Path(env_val).expanduser().resolve()
    return (_vault_from_obsidian_registry()
            or _vault_from_icloud()
            or Path.home() / "Documents" / "Obsidian Vault")


if __name__ == "__main__":
    # `scripts/backup.sh` and the README's install block ask here rather than repeating a path.
    sys.stdout.write(str(discover_vault()) + "\n")
