"""slim CLI — the pipeline commands. `slim` with no arguments lists them."""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

from . import db, trace
from .index import embed as embed_mod, ingest as ingest_mod
from .vault import identity
from .config import DB_PATH, VAULT


def unembedded_count(con) -> int:
    """Fragments whose vector is missing or STALE — keyed on (fragment_id, content_hash), so a
    content change orphans the old row. `COUNT(*) FROM embeddings` cannot see that."""
    return con.execute(
        "SELECT count(*) FROM fragments f LEFT JOIN embeddings e"
        "  ON e.fragment_id = f.id AND e.content_hash = f.content_hash"
        " WHERE e.fragment_id IS NULL").fetchone()[0]


def cmd_ingest(args):
    identity.require_vault_dir()
    con = db.connect()
    # ⚠ The removal sweep marks every source it cannot find under the CURRENT vault deleted.
    # That was safe while the path only moved when someone changed the env by hand; discovery
    # can now change it with nobody asking (a second vault opened in Obsidian, an iCloud copy
    # found), and one such run would empty the index — and a rebuild mints new source ids,
    # which orphans every saved copilot chat. So: index one vault, or say the move was meant.
    identity.require_vault_identity(con, allow_move=getattr(args, "vault_moved", False))
    t0 = time.time()
    stats = ingest_mod.ingest(con, verbose=args.verbose)
    # Embedding is PART of indexing, never a second command someone must run: vectors are keyed on
    # (fragment_id, content_hash), so a content change orphans the old row, and a forgotten
    # embed once left 2,802 of 3,403 fragments unembedded for hours (2026-08-04) while the
    # lexical arm kept answers plausible.
    embed_stats = embed_mod.embed_missing(con, verbose=True)
    stats["unembedded"] = unembedded_count(con)
    dt = time.time() - t0
    print(f"ingest ({dt:.1f}s): {json.dumps(stats)}  embed: {json.dumps(embed_stats)}")
    trace.record("ingest", {"stats": stats, "embed": embed_stats, "duration_s": round(dt, 2)})
    if stats["unembedded"]:
        # Nonzero AFTER an embed pass means the embedder did not finish. Loud and non-zero,
        # so a launchd-driven run is marked failed instead of quietly degraded.
        print(f"  ⚠ {stats['unembedded']} fragment(s) still have no current vector — the "
              f"embedder did not finish; semantic retrieval is degraded. Re-run `slim ingest`.")
        raise SystemExit(1)


def cmd_chat(args):
    """Serve the Obsidian plugin's API. Loopback-only by construction — a non-loopback
    --host is refused before a socket exists."""
    from . import server
    if args.parent_pid is not None and args.parent_pid <= 0:
        raise SystemExit("--parent-pid must be positive")
    identity.require_vault_dir()
    identity_con = db.connect()
    try:
        identity.require_vault_identity(identity_con)
    finally:
        identity_con.close()
    port = args.port if args.port is not None else server.DEFAULT_PORT
    if args.reload:
        from . import devserver
        return devserver.supervise(args.host, port, parent_pid=args.parent_pid)
    try:
        server.serve(args.host, port, parent_pid=args.parent_pid)
    except server.ChatError as e:
        raise SystemExit(str(e))


def cmd_trace(args):
    for entry in trace.last(args.n):
        print(json.dumps(entry, indent=2, ensure_ascii=False))


def _staging_lines(vault: Path) -> list[str]:
    """What sits in the recorder's staging folder. Every `.webm` there is a recording that never
    finished filing; a `.webm.transcript.json` without its `.webm` is a cache that outlived it."""
    from .recorder import note as note_writer
    staging = vault / note_writer.STAGING_DIR
    files = sorted(staging.iterdir()) if staging.is_dir() else []
    names = {f.name for f in files}
    segments: dict[str, list[Path]] = {}
    for f in files:
        if f.suffix == ".webm":
            segments.setdefault(note_writer.recording_id_from_staged(f.name), []).append(f)
    orphans = sum(1 for f in files if f.name.endswith(".webm.transcript.json")
                  and f.name.removesuffix(".transcript.json") not in names)

    if not segments:
        lines = ["stranded:  none"]
    else:
        n = len(segments)
        lines = [f"stranded:  {n} recording{'s' if n > 1 else ''} in {note_writer.STAGING_DIR} "
                 "never filed  ⚠"]
        for rid, segs in segments.items():
            # The id starts with when recording began; a name without one falls back to disk.
            day = (f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}" if re.match(r"\d{8}T", rid) else
                   time.strftime("%Y-%m-%d", time.localtime(segs[0].stat().st_mtime)))
            more = f"  {len(segs)} segments" if len(segs) > 1 else ""
            lines.append(f"  {rid}  {day}{more}")
    if orphans:
        lines.append(f"orphaned:  {orphans} cached transcript{'s' if orphans > 1 else ''} "
                     "whose audio is gone")
    return lines


def _last_reflection(vault: Path) -> str:
    """The newest reflection's date, from its filename (`<date>--reflection[-N].md`)."""
    from . import reflect
    dates = [f.name[:10] for f in (vault / reflect.NAMESPACE).glob("*--reflection*.md")]
    return max(dates, default="never")


def cmd_status(args):
    con = db.connect()

    def one(sql):
        return con.execute(sql).fetchone()[0]

    vault_status = "" if VAULT.is_dir() else "  ⚠ directory not found"
    print(f"vault:     {VAULT}{vault_status}")
    print(f"pin:       {identity.pin_status(con)}")
    print(f"db:        {DB_PATH}")
    print(f"sources:   {one('SELECT COUNT(*) FROM sources WHERE deleted=0')} active, "
          f"{one('SELECT COUNT(*) FROM sources WHERE deleted=1')} deleted")
    print(f"fragments: {one('SELECT COUNT(*) FROM fragments')}")
    print(f"embedded:  {one('SELECT COUNT(*) FROM embeddings')}")
    unembedded = unembedded_count(con)
    print(f"unembedded: {unembedded}" + ("  ⚠ run `slim ingest`" if unembedded else ""))
    for row in con.execute(
            "SELECT substr(path, 1, instr(path, '/') - 1) AS area, COUNT(*) c "
            "FROM sources WHERE deleted = 0 GROUP BY area ORDER BY c DESC"):
        print(f"  {row['c']:5d}  {row['area']}")
    for line in _staging_lines(VAULT):
        print(line)
    print(f"reflected: {_last_reflection(VAULT)}")


def cmd_reflect(args):
    """REFLECT: read a window of journal entries with the resident model and write ONE
    reflection note into `_Reflections/`. Nightly. `--dry-run` previews the selection — window,
    entry counts, target path — with no model call and no write. It reads journals under
    `Journal/` only, with the LOCAL model."""
    from . import reflect as reflect_mod

    identity.require_vault_dir()
    con = db.connect()
    try:
        identity.require_vault_identity(con)
    except BaseException:
        con.close()
        raise
    result = reflect_mod.reflect(
        con, days=args.days, min_entries=args.min_entries, dry_run=args.dry_run)

    if args.dry_run:
        if result.skipped:
            print(f"quiet window ({result.reason}): {result.focus_count} journal(s) since "
                  f"{result.window_start} — nothing would be written.")
            return
        print(f"WOULD reflect on {result.focus_count} journal(s) "
              f"({result.window_start} → {result.window_end}), "
              f"{result.context_count} context ent(y/ies) for continuity.")
        print(f"  would write: {result.path}")
        print("Re-run without --dry-run to generate it (a model pass, ~1–2 min).")
        return

    if result.skipped:
        print(f"quiet window ({result.reason}): {result.focus_count} journal(s) since "
              f"{result.window_start} — nothing written.")
        return
    print(f"reflected on {result.focus_count} journal(s) "
          f"({result.window_start} → {result.window_end}) "
          f"with {result.context_count} context ent(y/ies) → {result.path}")
    print(f"Open it in Obsidian: {VAULT / result.path}")


def _mask_line(res: dict) -> str:
    """One note: classes and counts, never a value."""
    tail = []
    if res.get("skipped"):
        tail.append(f"({res['skipped']})")
    elif res["counts"]:
        tail.append(" ".join(f"{c}:{n}" for c, n in sorted(res["counts"].items())))
    else:
        tail.append("(unchanged)")
    if res.get("partial"):
        tail.append(f"(partial: {', '.join(res['partial'])})")
    return f"{res['note']}  " + " ".join(tail)


def cmd_mask(args):
    """Mask personal data in recordings already in the vault (`recorder/mask.py`). Prints class
    counts per note, never a value. Non-zero when a note failed or was written but not
    re-indexed: anything that writes a note must index it, or say plainly that it did not."""
    from . import config
    from .recorder import mask

    if not args.notes and not args.all:
        print("name the notes to mask, or pass --all")
        raise SystemExit(2)
    if not config.PII_MASK:
        print("masking is off: pii.mask is empty in projects.yaml")
        raise SystemExit(1)
    identity.require_vault_dir()
    if not args.dry_run:
        con = db.connect()
        try:
            identity.require_vault_identity(con)
        finally:
            con.close()

    rels = []
    for name in args.notes:
        path = Path(name)
        if path.is_absolute() and path.is_relative_to(VAULT):
            path = path.relative_to(VAULT)
        rels.append(path.as_posix())
    if args.all:
        rels += [r for r in mask.recordings(VAULT) if r not in rels]

    masked = unchanged = skipped = 0
    failed = changed = False
    for rel in rels:
        try:
            res = mask.mask_note(VAULT, rel, force=args.force, dry_run=args.dry_run)
        except Exception as exc:                   # noqa: BLE001 - one bad note never stops --all
            # The class only: a message can quote the file it choked on.
            trace.record("mask", {"note": rel, "status": "failed", "error": type(exc).__name__})
            print(f"{rel}  (error: {type(exc).__name__})")
            skipped += 1
            failed = True
            continue
        print(_mask_line(res))
        if res.get("skipped"):
            skipped += 1
        elif res["counts"]:
            masked += 1
        else:
            unchanged += 1
        changed = changed or res["changed"]
        if res.get("index_error"):
            print(f"  ⚠ not re-indexed: {res['index_error']}")
            failed = True

    if changed and not args.dry_run:
        con = db.connect()
        try:
            db.compact(con)
        except sqlite3.OperationalError:
            print("  ⚠ index not compacted (database in use): old text may remain in free pages "
                  "until the next `slim mask` run with the server stopped")
        finally:
            con.close()
    prefix = "dry run: " if args.dry_run else ""
    print(f"{prefix}{masked} notes masked, {unchanged} unchanged, {skipped} skipped")
    trace.record("mask", {"status": "done", "dry_run": args.dry_run, "masked": masked,
                          "unchanged": unchanged, "skipped": skipped, "failed": failed})
    if failed:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(prog="slim", description="Sparse Local Intelligence & Memory")
    sub = ap.add_subparsers(dest="cmd", required=False)

    p = sub.add_parser("ingest", help="scan the vault, index changes, and embed new fragments")
    p.add_argument("--vault-moved", action="store_true",
                   help="the vault is the same one, at a new path — re-point the index to it")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser(
        "chat", help="the Obsidian plugin's local API server: recorder + copilot (loopback only)")
    p.add_argument("--port", type=int, default=None,
                   help="port on 127.0.0.1 (default 7546)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address — must be loopback; anything else is refused")
    p.add_argument("--reload", action="store_true",
                   help="dev mode: restart the server whenever a slim/*.py file is saved "
                        "(cuts in-flight requests — not for use while recording)")
    p.add_argument("--parent-pid", type=int, default=None,
                   help="exit when this parent process exits (used by the Obsidian plugin)")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("trace", help="show recent traces")
    p.add_argument("-n", type=int, default=1)
    p.set_defaults(fn=cmd_trace)

    p = sub.add_parser("status", help="index statistics")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("reflect",
                       help="write ONE journal-reflection note into the vault (nightly)")
    p.add_argument("--days", type=int, default=None,
                   help="reflect on the last N days (default: since the last reflection)")
    p.add_argument("--min-entries", dest="min_entries", type=int, default=1,
                   help="skip (quiet night) if fewer than N new journals in the window (default: 1)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="preview the selection only — no model call, no write")
    p.set_defaults(fn=cmd_reflect)

    p = sub.add_parser("mask", help="mask personal data in recordings already in the vault")
    p.add_argument("notes", nargs="*", metavar="NOTE", help="vault-relative recording paths")
    p.add_argument("--all", action="store_true",
                   help="every recording, except folders listed under pii.skip")
    p.add_argument("--force", action="store_true",
                   help="redo notes already masked by the current version")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="report what would be masked; write nothing")
    p.set_defaults(fn=cmd_mask)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.fn(args)
