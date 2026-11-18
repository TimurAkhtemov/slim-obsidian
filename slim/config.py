"""Paths and constants. Env overrides exist for tests, not for daily use."""

import os
from pathlib import Path

import yaml

from .vaultpath import discover_vault

REPO_ROOT = Path(__file__).resolve().parent.parent


# Discovered, not hardcoded — see `vaultpath.py`. ⚠ It is resolved ONCE, at import: a server
# that outlives a vault move keeps serving the old path, which is why `/api/health` reports it.
VAULT = discover_vault()
DATA_DIR = Path(os.environ.get("SLIM_DATA_DIR", str(Path.home() / "Library" / "Application Support" / "slim")))
DB_PATH = DATA_DIR / "slim.db"
TRACE_DIR = DATA_DIR / "traces"
PROJECTS_YAML = Path(os.environ.get("SLIM_PROJECTS", str(REPO_ROOT / "config" / "projects.yaml")))

# Ingestion exclusions: never indexed, never even read. Exclusion is by explicit NAME: a
# leading dot excludes nothing on its own.
EXCLUDED_DIRS = {".obsidian", ".trash"}
# `Profile/` is their self-description, excluded because it is not evidence — and since
# 2026-09-03 nothing reads it at all. `Skills/` holds the copilot's prompt templates
# (`copilot/skills.py`): instructions, not evidence.
EXCLUDED_TOP = {"Attachments", "Profile", "Skills"}

def _load() -> dict:
    # The file is the user's own vocabulary and is gitignored; a fresh clone has none until
    # `config/projects.example.yaml` is copied. Absent, every recording files into `Notes/_unfiled`.
    if not PROJECTS_YAML.exists():
        return {}
    return yaml.safe_load(PROJECTS_YAML.read_text()) or {}


def registry() -> dict:
    """The registered projects, as config. Closed set — nothing is discovered dynamically."""
    return _load().get("projects") or {}


# Classes of personal data masked in a recording's transcript (`capture/pii.py`). `phone` is off
# unless listed; an explicit empty list turns masking off.
PII_CLASSES = ("ssn", "card", "account", "password", "email", "address", "phone")
PII_DEFAULT = ("ssn", "card", "account", "password", "email", "address")


def _pii_mask(data: dict) -> tuple[str, ...]:
    """`pii.mask`, tidied. A name that is no class is an error at load: kept, it would mask
    nothing while every recording was stamped as checked."""
    mask = (data.get("pii") or {}).get("mask")
    if mask is None:
        return PII_DEFAULT
    if isinstance(mask, str):
        mask = [mask] if mask.strip() else []
    if not isinstance(mask, list):
        raise ValueError(f"pii.mask in {PROJECTS_YAML} must be a list of classes, not {mask!r}; "
                         f"allowed: {', '.join(PII_CLASSES)}")
    names = tuple(dict.fromkeys(str(m).strip().lower() for m in mask))
    unknown = [n for n in names if n not in PII_CLASSES]
    if unknown:
        raise ValueError(f"pii.mask in {PROJECTS_YAML}: unknown class {', '.join(map(repr, unknown))}; "
                         f"allowed: {', '.join(PII_CLASSES)}")
    return names


PII_MASK: tuple[str, ...] = _pii_mask(_load())


def _pii_skip(data: dict) -> tuple[str, ...]:
    """Vault folders `slim mask --all` leaves alone (`pii.skip`), as prefixes of whole path
    segments. Their names are the owner's, so they live in config, never in code. A note named
    on the command line is masked wherever it sits."""
    skip = (data.get("pii") or {}).get("skip") or ()
    if isinstance(skip, str):
        skip = [skip]
    return tuple(p for p in (str(s).strip().strip("/") for s in skip) if p)


PII_SKIP: tuple[str, ...] = _pii_skip(_load())

# The name the prompts address. Only interpretation reads it (labels, summaries);
# no path, id or filter ever does.
OWNER: str = str(_load().get("owner") or "the vault's owner")


def project_prefix(project_id: str) -> str:
    """The one vault folder that belongs to a project: `Notes/<id>`, recordings and the owner's
    writing alike. DERIVED from the id, never configured, so there is no second place for a
    path to drift out of sync."""
    return f"Notes/{project_id}"


# chunking targets (chars; ~4 chars/token)
CHUNK_TARGET = 2800
CHUNK_MAX = 4200
CHUNK_MIN = 400
