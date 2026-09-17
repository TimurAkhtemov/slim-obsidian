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

# Ingestion exclusions: never indexed, never even read. `.versions` holds superseded copies of
# imported transcripts — kept forever, but each is a near-duplicate of a live note, so indexing
# them would return the same meeting three times. Exclusion is by explicit NAME: a leading dot
# excludes nothing on its own.
EXCLUDED_DIRS = {".obsidian", ".trash", "_triage", ".versions"}
# `Profile/` is their self-description, excluded because it is not evidence — and since
# 2026-09-03 nothing reads it at all. `_Reflections/` is `slim reflect`'s own output: the brain
# must never retrieve its opinions back as evidence, and `reflect` reads that folder from disk
# rather than the DB to find its last window.
EXCLUDED_TOP = {"Attachments", "Profile", "_Reflections"}

def _load() -> dict:
    # The file is the user's own vocabulary and is gitignored; a fresh clone has none until
    # `config/projects.example.yaml` is copied. Absent, every recording files as `_unfiled`.
    if not PROJECTS_YAML.exists():
        return {}
    return yaml.safe_load(PROJECTS_YAML.read_text()) or {}


def registry() -> dict:
    """The registered projects, as config. Closed set — nothing is discovered dynamically."""
    return _load().get("projects") or {}


# The name the prompts address. Only interpretation reads it (labels, summaries, reflection);
# no path, id or filter ever does.
OWNER: str = str(_load().get("owner") or "the vault's owner")


def project_prefixes(project_id: str) -> tuple[str, ...]:
    """Every vault path prefix that belongs to one project: `Notes/<id>` (curated) and
    `Capture/<id>` (raw recordings, filed by the router).

    Both DERIVED from the id, never configured, so there is no second place for a path to
    drift out of sync. A caller that scopes by path must consider both subtrees.
    """
    return (f"Notes/{project_id}", f"Capture/{project_id}")


# chunking targets (chars; ~4 chars/token)
CHUNK_TARGET = 2800
CHUNK_MAX = 4200
CHUNK_MIN = 400
