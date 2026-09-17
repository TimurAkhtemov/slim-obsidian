#!/usr/bin/env python3
"""Pull all Notion AI meeting notes (transcription blocks) into the vault as Markdown.

Notion AI meeting notes are `transcription` blocks embedded in pages, not pages
themselves. This script:

  1. enumerates every page visible to the integration (search API, paginated)
  2. scans each page's block tree (bounded depth) for `transcription` blocks
  3. renders each block's children (AI summary + full transcript) to Markdown
  4. writes Meetings/YYYY/YYYY-MM-DD--<slug>.md with provenance frontmatter

Idempotent: a manifest in the vault records page and block last_edited_time;
unchanged pages are skipped entirely, unchanged meetings are not re-fetched.

═══ THIS SCRIPT MUST NEVER DESTROY A TRANSCRIPT. READ BEFORE EDITING. ═══

For most of its life it did. `write_meeting` called `out_path.write_text(content)`
unconditionally: whenever Notion's `last_edited_time` moved, the local file was replaced
in place. The index records a HASH, not content, so the old bytes were gone — and raw
imported sources are never rewritten.

That is not theoretical. The last pull ran at the moment the vault became the ONLY copy
of the meeting archive that exists anywhere, so a bad render, a partial API response or a
Notion re-transcription during it would silently have destroyed the archive.

Three guarantees now, in this order:

  1. **Never overwrite in place.** If a file exists and its BODY differs, the old file is
     copied to `Meetings/.versions/` (content-addressed, append-only, never ingested)
     BEFORE the new one is written. Every version of every transcript survives forever.
  2. **Refuse a suspicious shrink.** If the new body is empty, or is less than
     SHRINK_FLOOR of the old one, the write is REFUSED and reported — that is what a
     partial fetch or a render bug looks like, and it is exactly the failure that would
     eat an archive. `--allow-shrink` overrides, deliberately and loudly.
  3. **Compare BODIES, not files.** `imported_at` changes on every run, so a byte
     comparison would call every file "changed" and archive the entire vault on any run
     with a cold manifest.

Usage:  uv run scripts/pull_notion_meetings.py [--vault PATH] [--dry-run] [--allow-shrink]
Env:    NOTION_TOKEN (read from .env in repo root if not set)
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from slim import voicetags

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
# The rate limiter said one thing and did another.
#
# This was 0.20, commented "~3 rps ceiling per Notion guidance". 1/0.20 is FIVE requests per
# second against a documented ceiling of THREE. The comment and the code disagreed and
# nothing ever checked — the exact failure this project keeps writing rules about.
#
# 0.34 s is ~2.9 rps, under the ceiling. Whether over-rating was CAUSING the short reads
# below is NOT established (see the note on SHRINK_FLOOR); it is corrected because it was
# simply wrong, not because it is a proven fix.
REQUEST_INTERVAL = 0.34  # ≈2.9 rps — UNDER Notion's documented 3 rps average
# container block types worth descending into when hunting for transcription blocks
CONTAINER_TYPES = {"toggle", "column_list", "column", "synced_block", "callout"}
SCAN_DEPTH = 2  # transcription blocks sit at or near page top level

# Every superseded version of every transcript, content-addressed, append-only. Lives INSIDE
# the vault so it inherits the vault's backup and sync; dot-prefixed so ingestion skips it
# (the same reason `.notion-manifest.json` sits where it does) — otherwise every revision
# would come back as a near-duplicate search hit and poison retrieval.
VERSIONS_DIR = ".versions"
# Where an imported transcript lands. 2026-08-03: the type segment is gone — `Capture/` holds
# every recording and SUBJECT is the only folder axis, so there is no `Meetings/` to write into.
# The subject subfolder is not chosen here: `_subject_dir` derives it from the frontmatter this
# importer itself writes, through the same single emitter the router uses, so an import lands
# right on first write.
CAPTURE_REL = ("Capture",)

# A new body smaller than this fraction of the old one is treated as a FETCH FAILURE, not as
# an edit. A meeting transcript does not lose half its content in the real world; a partial
# API response, a render bug, or a re-transcription that dropped the tail looks exactly like
# this. Refuse, report, and let a human look — the alternative is discovering it after the
# subscription is cancelled and the source is gone.
SHRINK_FLOOR = 0.75

_last_request = 0.0


def load_token(repo_root: Path) -> str:
    import os
    tok = os.environ.get("NOTION_TOKEN", "")
    if not tok:
        env = repo_root / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.strip().startswith("NOTION_TOKEN="):
                    tok = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        sys.exit("NOTION_TOKEN not found in environment or .env")
    return tok


def api(token: str, path: str, method: str = "GET", body: dict | None = None,
        retries: int = 5) -> dict:
    global _last_request
    for attempt in range(retries):
        wait = REQUEST_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            API + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={"Authorization": f"Bearer {token}",
                     "Notion-Version": NOTION_VERSION,
                     "Content-Type": "application/json"})
        try:
            _last_request = time.time()
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                retry_after = float(e.headers.get("Retry-After", 2 ** attempt))
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"Notion API {e.code} on {path}: {e.read().decode()[:200]}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"retries exhausted for {path}")


PAGE_SIZE = 100


def paginate(token: str, path: str, method: str = "GET", body: dict | None = None):
    """Yield results across cursor pages, re-asking when a page looks suspiciously full.

    A response that returns EXACTLY `page_size` results while claiming `has_more: false` is
    suspicious: a genuine final page is exactly full only when the child count happens to be
    a multiple of 100. When we see that shape we pause and ask once more before believing it.

    This is a CHEAP GUARD, not a diagnosed fix. Short renders inside a full scan are real and
    still unexplained; this check does not make them go away. An earlier version of this
    docstring asserted that Notion "lies about has_more under load" and gave a worked example
    from a specific interview — that example was measured against the WRONG BLOCK (an 8-char
    Notion ID prefix is not unique) and the whole claim is withdrawn. The protection that
    actually holds is SHRINK_FLOOR plus archive-before-write.
    """
    cursor = None
    while True:
        def fetch(cur):
            if method == "POST":
                b = dict(body or {})
                if cur:
                    b["start_cursor"] = cur
                return api(token, path, "POST", b)
            sep = "&" if "?" in path else "?"
            return api(token, path + (f"{sep}start_cursor={cur}" if cur else ""))

        resp = fetch(cursor)
        results = resp.get("results", [])

        # THE SUSPICIOUS SHAPE: a completely full page that claims to be the last one.
        if not resp.get("has_more") and len(results) == PAGE_SIZE:
            time.sleep(1.0)                       # let whatever throttled us settle
            again = fetch(cursor)
            if again.get("has_more"):
                resp = again                      # the first answer WAS a lie
                results = again.get("results", [])

        yield from results
        if not resp.get("has_more"):
            return
        cursor = resp.get("next_cursor")


def rich_text(arr: list) -> str:
    return "".join(t.get("plain_text", "") for t in arr or [])


def all_pages(token: str) -> list[dict]:
    return list(paginate(token, "/search", "POST",
                         {"page_size": 100,
                          "filter": {"value": "page", "property": "object"}}))


def page_title(page: dict) -> str:
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return rich_text(prop.get("title", []))
    return "(untitled)"


def find_transcriptions(token: str, block_id: str, depth: int = 0) -> list[dict]:
    """Scan children of block_id for transcription blocks, descending into containers."""
    found = []
    for child in paginate(token, f"/blocks/{block_id}/children?page_size=100"):
        if child.get("in_trash") or child.get("archived"):
            continue
        if child["type"] == "transcription":
            found.append(child)
        elif (depth < SCAN_DEPTH and child.get("has_children")
              and child["type"] in CONTAINER_TYPES):
            found.extend(find_transcriptions(token, child["id"], depth + 1))
    return found


# --- markdown rendering -----------------------------------------------------

def render_block(token: str, block: dict, indent: int = 0) -> list[str]:
    t = block["type"]
    payload = block.get(t, {})
    text = rich_text(payload.get("rich_text", []))
    pad = "    " * indent
    lines: list[str] = []

    if t in ("heading_1", "heading_2", "heading_3"):
        lines.append(f"{'#' * (int(t[-1]) + 1)} {text}")
    elif t == "paragraph":
        if text:
            lines.append(pad + text)
    elif t == "bulleted_list_item":
        lines.append(f"{pad}- {text}")
    elif t == "numbered_list_item":
        lines.append(f"{pad}1. {text}")
    elif t == "to_do":
        mark = "x" if payload.get("checked") else " "
        lines.append(f"{pad}- [{mark}] {text}")
    elif t == "quote":
        lines.append(f"{pad}> {text}")
    elif t == "code":
        lang = payload.get("language", "")
        lines.append(f"```{lang}\n{text}\n```")
    elif t == "divider":
        lines.append("---")
    elif t == "callout":
        lines.append(f"{pad}> {text}")
    elif t == "toggle":
        if text:
            lines.append(f"{pad}**{text}**")
    elif text:  # unknown type with text — keep the text, lose the styling
        lines.append(pad + text)

    if block.get("has_children"):
        child_indent = indent + 1 if t in ("bulleted_list_item", "numbered_list_item",
                                           "to_do", "toggle", "quote") else indent
        for child in paginate(token, f"/blocks/{block['id']}/children?page_size=100"):
            if not (child.get("in_trash") or child.get("archived")):
                lines.extend(render_block(token, child, child_indent))
    return lines


def looks_like_transcript(token: str, wrapper: dict) -> bool:
    """A wrapper whose children are mostly long plain paragraphs is the transcript."""
    if not wrapper.get("has_children"):
        return False
    kids = list(paginate(token, f"/blocks/{wrapper['id']}/children?page_size=100"))
    if len(kids) < 4:
        return False
    paras = [k for k in kids if k["type"] == "paragraph"]
    headings = [k for k in kids if k["type"].startswith("heading")]
    return not headings and len(paras) >= 0.8 * len(kids)


class UnstableRender(RuntimeError):
    """Two reads of the same immutable block disagreed. The fetch cannot be trusted."""


def _render_once(token: str, block: dict) -> str:
    wrappers = [c for c in paginate(token, f"/blocks/{block['id']}/children?page_size=100")
                if not (c.get("in_trash") or c.get("archived"))]
    parts: list[str] = []
    transcript_started = False
    for i, w in enumerate(wrappers):
        if not transcript_started and i > 0 and looks_like_transcript(token, w):
            parts.append("## Transcript")
            transcript_started = True
        parts.extend(render_block(token, w))
    return "\n\n".join(p for p in parts if p.strip())


def render_meeting(token: str, block: dict, reads: int = 2) -> str:
    """Render a meeting, and require two independent reads to agree before believing it.

    OBSERVED, AND NOT FULLY EXPLAINED. During a full 265-page scan, some meeting renders
    come back small enough to trip SHRINK_FLOOR; rendering the same block on its own returns
    a stable, complete result. Two reads inside the scan agreed with each other, so the
    quorum alone does not catch it, and neither a slower request rate nor a 20 s cooldown
    made it go away.

    An earlier version of this comment blamed Notion pagination and claimed a specific
    transcript had lost a quarter of its content. THAT CLAIM WAS WRONG and is withdrawn: it
    rested on matching an 8-character block-ID prefix that is NOT unique (Notion IDs are
    time-ordered — this vault has two distinct meetings both starting `3487c804`), so the
    "complete" render being compared against was a DIFFERENT MEETING.

    What is actually established:
      - something renders short inside a full scan, and SHRINK_FLOOR refuses to write it
      - no transcript has been overwritten or lost as a result
      - the cause is unknown

    So this stays as a cheap consistency check, and the real protection is SHRINK_FLOOR +
    archive-before-write: a short read cannot destroy a good file, whatever is causing it.
    """
    first = _render_once(token, block)
    for _ in range(reads - 1):
        again = _render_once(token, block)
        if again == first:
            return first
        # Keep the LONGER one only to describe the failure; we still refuse to use it.
        short, long = sorted([first, again], key=len)
        raise UnstableRender(
            f"two reads of block {block['id']} disagreed: {len(short):,} vs {len(long):,} "
            f"chars ({len(short) / max(len(long), 1):.0%}). Notion dropped a page of block "
            f"children on at least one read. NOT writing — a partial transcript is worse than "
            f"no transcript, because it looks complete.")
    return first


# --- output -----------------------------------------------------------------

def slugify(title: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:max_len].strip("-")
    return slug or "untitled"


class TranscriptShrank(RuntimeError):
    """A re-fetch came back materially smaller. That is a fetch failure, not an edit."""


def body_of(text: str) -> str:
    """Everything after the YAML frontmatter.

    Compare BODIES, never whole files: `imported_at` is stamped fresh on every run, so a
    byte comparison would report every file as changed and archive the entire vault the
    first time anyone runs this with a cold manifest.
    """
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:]
    return text


def archive_existing(vault: Path, out_path: Path, dry_run: bool) -> Path | None:
    """Preserve the CURRENT file before anything replaces it. Append-only, content-addressed.

    This is the whole fix. Everything else in this script is bookkeeping; this is the line
    that means a transcript can never be destroyed — not by a bad render, not by a partial
    fetch, and not by the final pull that runs the moment before the subscription dies.
    """
    if not out_path.exists():
        return None
    current = out_path.read_text()
    digest = hashlib.sha256(current.encode()).hexdigest()[:12]
    dest = vault.joinpath(*CAPTURE_REL) / VERSIONS_DIR / f"{out_path.stem}--{digest}.md"
    if dest.exists():
        return dest                       # this exact content is already preserved
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(current)
    return dest


def imported_by_block_id(vault: Path) -> dict[str, Path]:
    """`notion_block_id` -> where that meeting actually lives, scanning `Capture/` once.

    An imported meeting does not stay where this script wrote it: the owner drags it from
    `Capture/_unfiled/` to `Capture/<subject>/` by hand, and a
    path-based existence check then misses EVERY previously-imported meeting: no `unchanged`
    short-circuit, no `TranscriptShrank` guard, no `archive_existing` — the three protections
    that make re-running this safe. The next pull would write ~100 second copies, each with a
    `notion_block_id` identical to an existing note, and they would pile up in `_unfiled`.

    Identity is the block id, not the filename: a title edited in Notion changes the slug, so
    a name-keyed index would miss the file for the same reason a path-keyed one does.
    """
    out: dict[str, Path] = {}
    root = vault.joinpath(*CAPTURE_REL)
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.md")):
        if VERSIONS_DIR in path.parts:
            continue                      # archived revisions are not the live note
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                if fh.readline().strip() != "---":
                    continue
                for _ in range(64):
                    line = fh.readline()
                    if not line or line.strip() == "---":
                        break
                    if line.startswith("notion_block_id:"):
                        bid = line.split(":", 1)[1].strip()
                        if bid:
                            out.setdefault(bid, path)
                        break
        except OSError:
            continue
    return out


def write_meeting(vault: Path, block: dict, body: str, parent_title: str,
                  dry_run: bool, allow_shrink: bool = False,
                  existing: dict[str, Path] | None = None) -> tuple[Path, str]:
    """-> (path, one of 'new' | 'unchanged' | 'revised'). NEVER destroys existing content.

    `existing` maps `notion_block_id` -> current path (see `imported_by_block_id`). Pass it and
    a meeting is found wherever it has since been moved; omit it and only the write location
    is checked, which is correct for a fresh vault and wrong for a sorted one.
    """
    title = rich_text(block.get("transcription", {}).get("title", [])).strip() or "Untitled Meeting"
    created = block["created_time"]
    date = created[:10]
    short_id = block["id"].replace("-", "")[:8]
    # No year segment any more: the date is in `date:` and in `authored_at`, where it can be
    # ORDERED. Duplicating it in the path is what crossed with subject and type to make 11
    # folders for 116 notes. An import has no subject at write time, so it lands in the sentinel
    # and the owner drags it to its subject by hand.
    out_dir = vault.joinpath(*CAPTURE_REL) / voicetags.UNFILED
    out_path = out_dir / f"{date}--{slugify(title)}--{short_id}.md"

    # Where this meeting IS, which is not always where a new one would be written. A revision
    # is written back over the note in place — moving it to `_unfiled` would undo a filing
    # done by hand.
    landed = (existing or {}).get(block["id"])
    if landed is not None and landed.exists():
        out_path = landed

    fm = "\n".join([
        "---",
        f'title: "{title}"',
        f"date: {date}",
        "type: meeting-note",
        "origin: imported",
        "source: notion-transcription-block",
        f"notion_block_id: {block['id']}",
        f'notion_parent_page: "{parent_title}"',
        f"created_time: {created}",
        f"last_edited_time: {block['last_edited_time']}",
        f"imported_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "---",
    ])
    new_content = f"{fm}\n\n# {title}\n\n{body}\n"
    new_body = body_of(new_content)

    if out_path.exists():
        old_body = body_of(out_path.read_text())

        # Notion moved last_edited_time but the CONTENT is identical — a metadata touch.
        # Do not rewrite: it would churn imported_at and archive a version for nothing.
        if old_body == new_body:
            return out_path, "unchanged"

        # THE CATASTROPHE GUARD. A real transcript does not lose a quarter of its body. A
        # partial fetch, a render bug, or a re-transcription that dropped the tail does.
        if len(new_body.strip()) < SHRINK_FLOOR * len(old_body.strip()) and not allow_shrink:
            raise TranscriptShrank(
                f"{out_path.name}: body would shrink {len(old_body):,} -> {len(new_body):,} "
                f"chars ({len(new_body) / max(len(old_body), 1):.0%} of the original). "
                f"REFUSING to write. That is what a partial fetch looks like, and this file "
                f"may be the only copy that exists. Inspect it, then --allow-shrink if the "
                f"shrink is genuinely correct.")

        # Real revision: preserve the old bytes FIRST, then replace.
        archive_existing(vault, out_path, dry_run)
        if not dry_run:
            out_path.write_text(new_content)
        return out_path, "revised"

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_content)
    return out_path, "new"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=str(Path.home() / "Documents" / "Obsidian Vault"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-shrink", action="store_true",
                    help="permit a transcript to be replaced by a MATERIALLY SMALLER one. "
                         "Off by default: a shrink is what a partial fetch looks like, and "
                         "the file it would overwrite may be the only copy in existence. "
                         "Look at the diff in Capture/.versions/ before you use this.")
    ap.add_argument("--full", action="store_true",
                    help="IGNORE THE MANIFEST and rescan every page. The manifest skips a "
                         "page whose last_edited_time is unchanged -- but adding a "
                         "transcription block to a Notion page does NOT reliably bump that "
                         "timestamp, so new meetings inside an already-seen page are never "
                         "discovered. Measured: two meetings from 2025 were never pulled at "
                         "all, and the manifest reported the vault as current for months.")
    ap.add_argument("--verify", action="store_true",
                    help="THE PRE-CANCELLATION RITUAL (§17). Ignore the manifest, re-fetch "
                         "EVERY meeting, and report any that differ from the vault — writing "
                         "nothing. The manifest skips pages on trust; before you cancel the "
                         "subscription that holds the only other copy of this archive, you "
                         "want proof, not trust. Slow (~3 rps) and worth it exactly once.")
    args = ap.parse_args()
    if args.verify:
        args.dry_run = True          # verification NEVER writes. Not once, not ever.
        args.full = True             # and it never trusts the manifest -- that is the point.

    repo_root = Path(__file__).resolve().parent.parent
    vault = Path(args.vault)
    token = load_token(repo_root)
    manifest_path = vault.joinpath(*CAPTURE_REL) / ".notion-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "pages": {}, "meetings": {}}

    # Built ONCE, before any writing: previously-imported meetings have been filed across
    # `Capture/<subject>/`, and a per-write rglob over ~350 notes would be quadratic.
    existing = imported_by_block_id(vault)
    print(f"meetings already in the vault: {len(existing)}")

    pages = all_pages(token)
    print(f"pages visible: {len(pages)}")

    stats = {"pages_scanned": 0, "pages_skipped": 0, "meetings_found": 0,
             "new": 0, "revised": 0, "unchanged": 0, "refused_shrink": 0,
             "unstable_render": 0}

    for i, page in enumerate(pages, 1):
        pid, edited = page["id"], page["last_edited_time"]
        title = page_title(page)
        # --full ignores the manifest ENTIRELY. The manifest is the very thing being audited:
        # one that wrongly says "unchanged" is precisely how a meeting goes missing for a year.
        if not args.full and manifest["pages"].get(pid) == edited:
            stats["pages_skipped"] += 1
            continue
        stats["pages_scanned"] += 1
        if i % 25 == 0:
            print(f"  ...scanning page {i}/{len(pages)}")
        try:
            transcriptions = find_transcriptions(token, pid)
        except RuntimeError as e:
            print(f"  WARN: scan failed for '{title}': {e}", file=sys.stderr)
            continue

        page_ok = True
        for block in transcriptions:
            stats["meetings_found"] += 1
            bid = block["id"]
            if not args.full and \
                    manifest["meetings"].get(bid, {}).get("last_edited_time") == block["last_edited_time"]:
                stats["unchanged"] += 1
                continue
            try:
                # A shrink is a SYMPTOM of the throttling degradation, not a verdict. Quiet
                # reads of the same block are always complete, so on a suspicious result we
                # stop, let Notion recover, and ask again — rather than refusing outright and
                # leaving a truncated transcript unrepaired. Only if it shrinks every time do
                # we refuse, and then the existing file is left untouched.
                for attempt in range(3):
                    try:
                        body = render_meeting(token, block)
                        out, what = write_meeting(vault, block, body, title, args.dry_run,
                                                  args.allow_shrink, existing)
                        break
                    except TranscriptShrank:
                        if attempt == 2:
                            raise
                        cool = 10 * (attempt + 1)
                        print(f"  short read on '{title[:40]}' — cooling {cool}s and "
                              f"re-reading ({attempt + 1}/2)", flush=True)
                        time.sleep(cool)
                stats[what] += 1
                if what == "revised":
                    # Loud, because a transcript changing under you is a real event: the old
                    # bytes are preserved in .versions/, but somebody should know it happened.
                    print(f"  REVISED (old version archived) {out.relative_to(vault)}")
                elif what == "new":
                    print(f"  wrote {out.relative_to(vault)}")
                if not args.dry_run:
                    entry = {"last_edited_time": block["last_edited_time"],
                             "file": str(out.relative_to(vault))}
                    parent_page = block.get("parent", {}).get("page_id")
                    if parent_page:  # lets link reconciliation redirect container-page links
                        entry["parent_page_id"] = parent_page.replace("-", "")
                    manifest["meetings"][bid] = entry
            except TranscriptShrank as e:
                # NOT counted as done, and the manifest is NOT advanced — so the next run
                # retries it instead of recording a corrupt state as current.
                page_ok = False
                stats["refused_shrink"] += 1
                print(f"  REFUSED: {e}", file=sys.stderr)
            except UnstableRender as e:
                page_ok = False
                stats["unstable_render"] += 1
                print(f"  UNSTABLE: {e}", file=sys.stderr)
            except RuntimeError as e:
                page_ok = False
                print(f"  WARN: render failed for meeting in '{title}': {e}", file=sys.stderr)

        # only mark the page done if every meeting in it succeeded
        if page_ok and not args.dry_run:
            manifest["pages"][pid] = edited
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=1))

    print(json.dumps(stats, indent=2))

    if args.verify:
        drift = (stats["new"] + stats["revised"] + stats["refused_shrink"]
                 + stats["unstable_render"])
        if drift == 0:
            print(f"\n✓ VERIFIED: all {stats['meetings_found']} meetings in Notion match the "
                  f"vault byte-for-byte.\n  The vault is a faithful copy. It is safe to "
                  f"cancel WITHOUT re-running the pull.")
        else:
            print(f"\n✗ DRIFT: {drift} meeting(s) in Notion differ from the vault "
                  f"({stats['new']} missing, {stats['revised']} changed, "
                  f"{stats['refused_shrink']} refused).\n  Nothing was written — this was a "
                  f"verification. Re-run WITHOUT --verify to reconcile, and read every REVISED "
                  f"line before you cancel anything.")

    if stats["refused_shrink"]:
        print(f"\n⚠ {stats['refused_shrink']} transcript(s) came back MATERIALLY SMALLER and "
              f"were REFUSED.\n  The vault copies are intact. This is what a partial fetch "
              f"looks like — do not\n  reach for --allow-shrink until you have read the diff.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
