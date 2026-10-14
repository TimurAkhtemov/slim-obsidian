"""Which vault this index belongs to. Ingest's removal sweep deletes every source it cannot find
under the current vault, and discovery can change that path with nobody asking, so every command
that can write checks the pin first. `vaultpath.py` finds the vault; this module decides whether
the index may use it."""

import os
from pathlib import Path

from . import trace
from .config import VAULT


def pinned_vault() -> Path | None:
    """The vault this index was built from, or None before the first ingest."""
    from . import config
    pin = config.DATA_DIR / "vault_pin"
    if not pin.is_file():
        return None
    value = pin.read_text(encoding="utf-8").rstrip("\n")
    if not value:
        raise RuntimeError(f"vault pin is empty: {pin}")
    return Path(value)


def pin_vault(vault: Path) -> None:
    from . import config
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pin = config.DATA_DIR / "vault_pin"
    partial = pin.with_name(f".{pin.name}.{os.getpid()}.part")
    try:
        partial.write_text(str(vault) + "\n", encoding="utf-8")
        partial.replace(pin)
    finally:
        partial.unlink(missing_ok=True)


def require_vault_dir() -> None:
    if not VAULT.is_dir():
        trace.record("vault_identity", {
            "status": "refused", "reason": "vault_dir_missing", "vault": str(VAULT)})
        print(f"⚠ Vault directory not found at: {VAULT}")
        print("  Point SLIM at it:  export SLIM_VAULT=/path/to/vault")
        raise SystemExit(1)


def require_vault_identity(con, *, allow_move: bool = False) -> None:
    """Refuse to use derived state with a different or unidentified vault.

    An empty database is a new index and may claim the discovered vault. A populated database
    without a pin predates this invariant; only an explicit `--vault-moved` may adopt it. Pin
    before work begins: recorder and memo writes precede their indexing pass, and `ingest()`
    commits before embedding can fail.
    """
    pinned = pinned_vault()
    has_sources = bool(con.execute("SELECT EXISTS(SELECT 1 FROM sources)").fetchone()[0])

    if pinned is not None and pinned == VAULT:
        return

    if pinned is None and has_sources and not allow_move:
        trace.record("vault_identity", {
            "status": "refused", "reason": "unpinned_populated",
            "vault": str(VAULT), "pinned": None})
        print("⚠ This existing index has no vault pin, so SLIM cannot prove which vault built it.")
        print(f"  SLIM now finds the vault at: {VAULT}")
        print("  If this is that same vault, adopt it:  uv run slim ingest --vault-moved")
        print("  Otherwise name the right vault first: export SLIM_VAULT=/path/to/vault")
        raise SystemExit(1)

    if pinned is not None and pinned != VAULT and not allow_move:
        trace.record("vault_identity", {
            "status": "refused", "reason": "pinned_mismatch",
            "vault": str(VAULT), "pinned": str(pinned)})
        print(f"⚠ This index was built from: {pinned}")
        print(f"  SLIM now finds the vault at: {VAULT}")
        print("  Continuing could read or write against the wrong vault.")
        print("  If the vault MOVED, re-point the index:  uv run slim ingest --vault-moved")
        print("  If it did not, name the right one:       export SLIM_VAULT=/path/to/vault")
        raise SystemExit(1)

    if pinned != VAULT:
        pin_vault(VAULT)


def pin_status(con) -> str:
    """What `require_vault_identity` would decide, REPORTED rather than enforced: status reads
    and never pins, so it is the one command that can show a mismatch without acting on it."""
    try:
        pinned = pinned_vault()
    except RuntimeError as exc:
        return f"⚠ {exc}"
    if pinned == VAULT:
        return "matches the vault"
    has_sources = bool(con.execute("SELECT EXISTS(SELECT 1 FROM sources)").fetchone()[0])
    if pinned is None and not has_sources:
        return "none yet (the first ingest pins it)"
    if pinned is None:
        return ("⚠ none — this index cannot prove which vault built it; if it is this one, "
                "`slim ingest --vault-moved`")
    return (f"⚠ built from {pinned} — every writing command refuses; if the vault moved, "
            "`slim ingest --vault-moved`")
