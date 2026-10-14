"""slim CLI — the pipeline commands. `slim` with no arguments lists them."""

import argparse
import json
import re
import time
from pathlib import Path

from . import db, embed as embed_mod, ingest as ingest_mod, trace, vaultidentity
from .config import DB_PATH, VAULT


def unembedded_count(con) -> int:
    """Fragments whose vector is missing or STALE — keyed on (fragment_id, content_hash), so a
    content change orphans the old row. `COUNT(*) FROM embeddings` cannot see that."""
    return con.execute(
        "SELECT count(*) FROM fragments f LEFT JOIN embeddings e"
        "  ON e.fragment_id = f.id AND e.content_hash = f.content_hash"
        " WHERE e.fragment_id IS NULL").fetchone()[0]


def cmd_ingest(args):
    vaultidentity.require_vault_dir()
    con = db.connect()
    # ⚠ The removal sweep marks every source it cannot find under the CURRENT vault deleted.
    # That was safe while the path only moved when someone changed the env by hand; discovery
    # can now change it with nobody asking (a second vault opened in Obsidian, an iCloud copy
    # found), and one such run would empty the index — and a rebuild mints new source ids,
    # which orphans every saved copilot chat. So: index one vault, or say the move was meant.
    vaultidentity.require_vault_identity(con, allow_move=getattr(args, "vault_moved", False))
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


def cmd_memos(args):
    """Sweep Apple's Voice Memos container into Inbox/Journal, transcribe, and (only if a note
    was written) index it. Driven by launchd every five minutes, or run by hand.

    Copy-only: nothing in Apple's container is ever renamed, moved, or deleted. Safe to run
    every five minutes — quiet when there is nothing new, loud (verbatim) when it is refused."""
    from types import SimpleNamespace

    from . import inbox as inbox_mod, voicememos

    vaultidentity.require_vault_dir()
    identity_con = db.connect()
    try:
        vaultidentity.require_vault_identity(identity_con)
    finally:
        identity_con.close()

    container = Path(args.container).expanduser() if args.container else voicememos.CONTAINER
    swept = voicememos.sweep(container=container, dry_run=args.dry_run)

    refusals = [s for s in swept if s.status == "refused"]
    copied = [s for s in swept if s.status == "copied"]
    for s in refusals:
        print(f"REFUSED: {s.detail}")
    if copied:
        for s in copied:
            print(f"  swept  {s.source.name}  ->  Inbox/Journal/{s.dest.name}"
                  f"{'   (dry run)' if args.dry_run else ''}")
    elif not refusals:
        print("no new voice memos to sweep.")

    processed = inbox_mod.process(dry_run=args.dry_run)
    for r in processed:
        rel = r.note.relative_to(VAULT) if r.note and VAULT in r.note.parents else r.note
        if r.status == "written" and r.transcript:
            t = r.transcript
            print(f"  {r.audio.name}  ->  {rel}   "
                  f"({t.audio_seconds/60:.1f} min, {t.words} words)")
        elif r.status == "written":
            print(f"  {r.audio.name}  ->  {rel}   {r.detail}")
        else:
            print(f"  {r.audio.name}  [{r.status}] {r.detail}")

    wrote = any(r.status == "written" for r in processed)
    if wrote and not args.dry_run:
        # Auto-index at the CLI level: a new transcript is retrievable without a second
        # command, through the exact `slim ingest` seam (which embeds at the end).
        cmd_ingest(SimpleNamespace(verbose=False))
    elif wrote:
        print("\n(dry run — not indexed)")


def cmd_chat(args):
    """Serve the Obsidian plugin's API. Loopback-only by construction — a non-loopback
    --host is refused before a socket exists."""
    from . import chat as chat_mod
    if args.parent_pid is not None and args.parent_pid <= 0:
        raise SystemExit("--parent-pid must be positive")
    vaultidentity.require_vault_dir()
    identity_con = db.connect()
    try:
        vaultidentity.require_vault_identity(identity_con)
    finally:
        identity_con.close()
    port = args.port if args.port is not None else chat_mod.DEFAULT_PORT
    if args.reload:
        from . import devserver
        return devserver.supervise(args.host, port, parent_pid=args.parent_pid)
    try:
        chat_mod.serve(args.host, port, parent_pid=args.parent_pid)
    except chat_mod.ChatError as e:
        raise SystemExit(str(e))


def cmd_trace(args):
    for entry in trace.last(args.n):
        print(json.dumps(entry, indent=2, ensure_ascii=False))


def _staging_lines(vault: Path) -> list[str]:
    """What sits in the recorder's staging folder. Every `.webm` there is a recording that never
    finished filing; a `.webm.transcript.json` without its `.webm` is a cache that outlived it."""
    from . import record
    staging = vault / record.STAGING_DIR
    files = sorted(staging.iterdir()) if staging.is_dir() else []
    names = {f.name for f in files}
    segments: dict[str, list[Path]] = {}
    for f in files:
        if f.suffix == ".webm":
            segments.setdefault(record.recording_id_from_staged(f.name), []).append(f)
    orphans = sum(1 for f in files if f.name.endswith(".webm.transcript.json")
                  and f.name.removesuffix(".transcript.json") not in names)

    if not segments:
        lines = ["stranded:  none"]
    else:
        n = len(segments)
        lines = [f"stranded:  {n} recording{'s' if n > 1 else ''} in {record.STAGING_DIR} "
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
    print(f"pin:       {vaultidentity.pin_status(con)}")
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
    entry counts, target path — with no model call and no write. The agent never reads journal
    content here; the LOCAL model does."""
    from . import reflect as reflect_mod

    vaultidentity.require_vault_dir()
    con = db.connect()
    try:
        vaultidentity.require_vault_identity(con)
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


def main():
    ap = argparse.ArgumentParser(prog="slim", description="Sparse Local Intelligence & Memory")
    sub = ap.add_subparsers(dest="cmd", required=False)

    p = sub.add_parser("ingest", help="scan the vault, index changes, and embed new fragments")
    p.add_argument("--vault-moved", action="store_true",
                   help="the vault is the same one, at a new path — re-point the index to it")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser(
        "memos", help="sweep synced Voice Memos into Inbox/Journal, transcribe, and index")
    p.add_argument("--container", default=None,
                   help="override the Voice Memos recordings folder (default: the synced container)")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be swept and transcribed; copy and write nothing")
    p.set_defaults(fn=cmd_memos)

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

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.fn(args)
