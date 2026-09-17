"""Copy-only sweep of Apple's Voice Memos container into the journal inbox.

The owner records in the iOS Voice Memos app; iCloud syncs the recordings to this Mac's group
container. This module copies NEW recordings out of that container into the content-addressed
inbox flow (`inbox.process`), where they become Journal transcripts. Driven by launchd every
five minutes. The machinery here is only the copy: it calls no model and writes no prose.

The container is Apple's app data, so this reads only: nothing there is ever renamed, moved,
rewritten or deleted, and the title db is opened read-only AND immutable (`mode=ro&immutable=1`)
so it can never be locked against the Voice Memos app. That db's schema is unvalidated on this
machine, so `_titles` probes defensively and falls back to the UUID stem rather than blocking a
sweep — a missing title is cosmetic, a blocked sweep is a lost recording.

Safe to run every five minutes forever: idempotent by SHA-256 (a manifest in the brain dir, not
the vault), quiet when nothing is new, and loud and AUTHORED when it cannot read the container.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import inbox, trace
from . import transcribe as tx
from .config import DATA_DIR

# Apple's synced Voice Memos recordings folder. Overridable everywhere via `container=` so the
# tests can point at a fake dir — the real one is TCC-protected and cannot be touched from here.
CONTAINER = Path.home() / "Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings"

# The user-visible title + record metadata live in Voice Memos' own sqlite, a sibling of the
# recordings inside the same folder.
TITLE_DB = "CloudRecordings.db"

# Sweep idempotency manifest: digest -> what we copied. Rebuildable state, so it lives in the
# Mac-only brain dir (never the synced vault), and is written atomically like inbox's manifest.
MANIFEST = DATA_DIR / "audio" / "voicememos-swept.json"


@dataclass
class SweepResult:
    source: Path | None       # the file in Apple's container (None for a whole-sweep refusal)
    dest: Path | None         # where it landed in Inbox/Journal (None if refused/dry-run-noop)
    digest: str | None
    status: str               # copied | skipped | refused
    detail: str = ""


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    partial = MANIFEST.with_name(f".{MANIFEST.name}.part")
    try:
        partial.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        partial.replace(MANIFEST)
    finally:
        partial.unlink(missing_ok=True)


def _titles(container: Path) -> tuple[dict[str, str], str]:
    """Map container filename -> user-visible title from `CloudRecordings.db`, read-only.

    Returns `({}, reason)` on ANY failure — missing db, missing table/columns, undecodable
    values — because a failed title lookup must never block a sweep; the UUID stem is an
    acceptable name. The reason is recorded in the sweep trace.

    The real Apple schema is unvalidated on this machine — TCC blocks our shell from the
    container — so this probes through a few plausible column sets.
    """
    db_path = container / TITLE_DB
    if not db_path.exists():
        return {}, "db-missing"

    # Read-only AND immutable: we can never lock Apple's db or trigger WAL recovery on it.
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error as e:
        return {}, f"db-unopenable: {type(e).__name__}"

    # ZPATH names the recording; the title is a custom label or the (possibly encrypted) title.
    column_sets = (
        ("ZPATH", "ZCUSTOMLABEL", "ZENCRYPTEDTITLE"),
        ("ZPATH", "ZCUSTOMLABEL"),
        ("ZPATH", "ZENCRYPTEDTITLE"),
    )
    try:
        for cols in column_sets:
            try:
                rows = con.execute(
                    f"SELECT {', '.join(cols)} FROM ZCLOUDRECORDING"
                ).fetchall()
            except sqlite3.Error:
                continue
            titles: dict[str, str] = {}
            for row in rows:
                zpath = row[0]
                if not zpath:
                    continue
                title = next((c for c in row[1:] if isinstance(c, str) and c.strip()), "")
                if not title:
                    continue
                # Key by both the recorded filename and its stem: ZPATH may be a bare name or a
                # relative path, and the on-disk file may or may not carry the suffix.
                name = Path(str(zpath)).name
                titles[name] = title.strip()
                titles[Path(name).stem] = title.strip()
            if titles:
                return titles, "db"
        return {}, "db-schema"
    except sqlite3.Error as e:
        return {}, f"db-error: {type(e).__name__}"
    finally:
        con.close()


def _dest_path(journal_inbox: Path, source: Path, title: str, digest: str) -> Path:
    """Name in the inbox: the user-visible title (slugged) or the original stem, then
    `--<hash8>` before the suffix so two different recordings can never collide on a name."""
    base = inbox._slug(title) if title else source.stem
    return journal_inbox / f"{base}--{digest[:8]}{source.suffix.lower()}"


def _copy_out(source: Path, dest: Path, digest: str) -> None:
    """Copy container bytes into the inbox with inbox's partial-then-verify-then-replace
    pattern. Reads `source`; never writes to it. A partial copy is a dotfile that is unlinked
    unless it verifies, so an interrupted sweep can never leave a truncated recording behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f".{dest.name}.part")
    try:
        with source.open("rb") as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        if inbox._sha256(partial) != digest:
            raise RuntimeError(f"voice-memo copy failed verification: {source}")
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)


_PERMISSION_REFUSAL = (
    "Cannot read the Voice Memos recordings folder — macOS blocked access "
    "(\"Operation not permitted\"). Grant access to the Voice Memos folder in System Settings "
    "> Privacy & Security. Nothing was copied."
)


def _missing_refusal(container: Path) -> str:
    return (
        f"The Voice Memos recordings folder does not exist at {container} — iCloud sync for "
        "Voice Memos may not be set up yet. Nothing was copied."
    )


def sweep(container: Path = CONTAINER, dry_run: bool = False) -> list[SweepResult]:
    """Copy new recordings out of Apple's container into `Inbox/Journal`. Copy-only, idempotent
    by SHA-256, traced, and loud (never silent) when it cannot read the container."""
    # Distinguish "sync not set up" (missing dir) from "TCC not granted" (dir present, unreadable).
    try:
        present = container.exists()
    except OSError:
        present = True  # cannot even stat it -> treat as a permission problem, handled below
    if not present:
        trace.record("voicememo_sweep",
                     {"status": "refused", "reason": "missing-container",
                      "container": str(container)})
        return [SweepResult(None, None, None, "refused", _missing_refusal(container))]

    try:
        entries = sorted(p for p in container.iterdir() if not p.name.startswith("."))
    except (PermissionError, OSError) as e:
        # TCC / sandbox denial. Authored refusal naming the fix — never a bare traceback in a
        # launchd log. PermissionError covers EPERM/EACCES; the broad OSError is belt-and-braces.
        trace.record("voicememo_sweep",
                     {"status": "refused", "reason": "permission",
                      "container": str(container), "error": f"{type(e).__name__}: {e}"})
        return [SweepResult(None, None, None, "refused", _PERMISSION_REFUSAL)]

    journal_inbox = inbox.INBOX["journal"]
    titles, title_detail = _titles(container)

    manifest = _load_manifest()
    results: list[SweepResult] = []
    found = copied = skipped = 0

    for source in entries:
        if source.suffix.lower() not in tx.AUDIO_SUFFIXES:
            continue
        found += 1
        digest = inbox._sha256(source)
        if digest in manifest:
            skipped += 1
            results.append(SweepResult(source, Path(manifest[digest]["dest"]), digest,
                                       "skipped", "already swept (same audio)"))
            continue

        # A RECORDING's identity is its Voice Memos stem, not the bytes we were handed.
        # MEASURED 2026-07-20: the export is not byte-stable, so the same recording exported
        # twice looked new to content-addressed dedup and was transcribed again — four memos
        # had two notes each, and the five-minute timer would have duplicated every memo
        # forever. The stem carries Voice Memos' own recording id and survives re-export.
        # Content hashing stays the FIRST check; the stem is the second.
        if any(entry.get("source_stem") == source.stem for entry in manifest.values()):
            skipped += 1
            results.append(SweepResult(
                source, None, digest, "skipped",
                "already swept (same recording, re-exported)"))
            continue

        title = titles.get(source.name) or titles.get(source.stem) or ""
        dest = _dest_path(journal_inbox, source, title, digest)

        if dry_run:
            results.append(SweepResult(source, dest, digest, "copied", "(dry run)"))
            continue

        # Idempotency belt if the manifest was lost: a name embeds the content hash, so an
        # existing dest with matching bytes is the same recording — record it, don't re-copy.
        if dest.exists() and inbox._sha256(dest) == digest:
            results.append(SweepResult(source, dest, digest, "skipped",
                                       "already in the inbox (same audio)"))
        else:
            _copy_out(source, dest, digest)
            copied += 1
            results.append(SweepResult(source, dest, digest, "copied", title or source.stem))

        manifest[digest] = {
            "source": source.name,
            # The stable recording identity — see the re-export guard above.
            "source_stem": source.stem,
            "dest": str(dest),
            "title": title,
            "title_source": "db" if title else "filename",
            "swept_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    if not dry_run:
        _save_manifest(manifest)

    # ⚠ ONLY when something was actually copied (2026-09-03). The launchd timer fires every
    # five minutes, so an unconditional record wrote 288 identical "found 0" lines a day and
    # buried the records that say an event happened. A refusal still traces — being unable to
    # read Apple's container IS an event, and it is the one a launchd log must not lose.
    if copied:
        trace.record("voicememo_sweep", {
            "status": "ok",
            "container": str(container),
            "found": found,
            "copied": copied,
            "skipped": skipped,
            "title_source": "db" if titles else "filename",
            "title_detail": title_detail,
            "dry_run": dry_run,
        })
    return results
