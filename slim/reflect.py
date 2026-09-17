"""REFLECT — nightly journal synthesis, written INTO the vault.

`slim reflect` reads a window of journal entries with the resident model and writes ONE note
into `_Reflections/` — throughlines, what shifted, threads back to earlier entries, with
Obsidian wikilinks to the sources. Driven by a launchd timer at 03:45; the CLI is the only
other caller. Reflection, not retrieval Q&A and not re-summary.

Three things are load-bearing:

  - The LOCAL model reads journal bodies; this module's CODE reads them only to feed it.
    Nothing here returns or logs body text — the trace records counts, dates, model and token
    stats, never a line of a journal or of the reflection derived from it.
  - SELECTION IS DETERMINISTIC. Which journals to reflect on is decided in SQL by
    `type='journal'` plus date, never by similarity: `authored_at` is a plain YYYY-MM-DD
    string, so the window is lexicographic date math. Retrieval is the wrong tool — it is
    subject-driven and needs a query term, and reflect wants EVERY journal in range.
  - `think=False`, explicit. If reflections ever read as invented, `think=True` with a widened
    `NUM_PREDICT` is the known lever.

The note is DERIVED and `_Reflections/` is in `config.EXCLUDED_TOP`, so the brain never
retrieves its own reflections as evidence. That exclusion is also what lets "since the last
reflection" be a disk-visible fact: `last_window_end` reads the folder, not the DB.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from . import llm, trace
from .chunk import parse_frontmatter
from .config import OWNER, VAULT

REFLECT_PROMPT_VERSION = "reflect-v2"

# Where reflection notes land. The leading underscore marks the folder as SLIM-generated and
# sorts it to the top in Obsidian. MUST match the entry in EXCLUDED_TOP.
NAMESPACE = "_Reflections"

# How much journal history BEFORE the focus window the model sees as read-only background. It
# reflects only on FOCUS but can see movement against CONTEXT, which is what turns a one-entry
# night from a paraphrase into "how this connects to your arc".
CONTEXT_LOOKBACK_DAYS = 14
CONTEXT_MAX_ENTRIES = 30          # bound the tokens on a first run over a long backfill

# First-ever run (no prior reflection to resume from): reflect on the last week.
DEFAULT_FIRST_WINDOW_DAYS = 7

# Prose the owner reads, so a real but bounded budget: no grammar here to loop, but the cap
# still stops a runaway.
NUM_PREDICT = 1400

# A wikilink: [[target]] or [[target|alias]]. Used to validate model-emitted links against the
# allow-list and to strip any the model invented.
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


SYSTEM = f"""You are reflecting on {OWNER}'s personal journal over a period of days. You are not \
summarizing. A summary restates what each entry said; a reflection surfaces what they might not see \
themselves — the threads that run across entries, what shifted, what keeps recurring, an intention \
stated and then dropped, a tension between two entries.

You are given FOCUS entries (this period — what you reflect on) and earlier CONTEXT entries \
(read-only background, so you can tell what changed). Reflect on the FOCUS entries; use CONTEXT \
only to notice movement — "you circled back to X," "the frustration in [earlier] eased here."

Write in second person, to {OWNER} — like a perceptive friend who has read it all, not a therapist \
or a report. Format it however reads best: headers, bullets, short sections, emphasis — whatever \
the material calls for.

If the FOCUS window is thin — one entry, or a quiet stretch — do not pad it and do not just \
restate it. Use it as a doorway: connect it to the recent CONTEXT (what it echoes, what it breaks \
from, a thread it revives), or offer one honest observation or open question about the longer arc. \
A quiet day is still worth a few sentences when they're about the shape of things.

Never invent a feeling, an event, or a throughline the entries don't support. When a thread \
connects to a specific entry, link to it using the wikilink given for that entry in the list \
below — copy it exactly, including the short label after the "|". Only use links from that list."""


@dataclass
class JournalEntry:
    """One journal source with its full body, loaded to feed the LOCAL model. Never logged."""
    source_id: str
    path: str            # vault-relative, e.g. "Journal/2026-07-15--...--hash.md"
    stem: str            # filename without .md — the Obsidian wikilink target
    authored_at: str     # YYYY-MM-DD
    title: str
    body: str


@dataclass
class ReflectionResult:
    """Outcome of a run. Metadata only — `text` is the written prose (returned to the CLI for the
    'wrote N' message), but it is NEVER placed in a trace or any durable operational record."""
    skipped: bool = False
    reason: str = ""
    dry_run: bool = False
    path: str | None = None          # vault-relative path written, if any
    window_start: str = "none"       # the lower bound used (a date, or "none" on the first run)
    window_end: str = ""             # max FOCUS authored_at; where the NEXT run resumes
    focus_count: int = 0
    context_count: int = 0
    sources: list[str] = field(default_factory=list)   # FOCUS stems
    model: str = ""
    text: str = ""


def _stem(path: str) -> str:
    return Path(path).stem


def _deslug(stem: str) -> str:
    """A journal stem is `YYYY-MM-DD--Title-Slug--hash`. Strip the date prefix and the hash suffix
    and turn the slug back into words — a readable fallback label when a note carries no title."""
    s = re.sub(r"^\d{4}-\d{2}-\d{2}--", "", stem)
    s = re.sub(r"--[0-9a-f]{6,}$", "", s)
    return s.replace("-", " ").strip()


def _display(entry: "JournalEntry") -> str:
    """The SHORT label a wikilink shows — the note's title, else its de-slugged stem. This is
    whole fix for 'the full stem clogs the prose': `[[stem|label]]` links to the note but renders
    only `label`. Kept link-safe (no `[`, `]`, `|`, newlines) and capped so a long title can't
    reintroduce the clutter it replaced."""
    t = (entry.title or "").strip()
    if not t or "/" in t or t == entry.stem:      # a path-like/absent title → de-slug the stem
        t = _deslug(entry.stem)
    t = t.replace("[", "").replace("]", "").replace("|", "-").replace("\n", " ").strip()
    if len(t) > 60:
        t = t[:59].rstrip() + "…"
    return t or entry.authored_at


def _link(entry: "JournalEntry") -> str:
    """An aliased Obsidian wikilink: links to the exact note, renders only the short label."""
    return f"[[{entry.stem}|{_display(entry)}]]"


def _load_entries(con, rows) -> list[JournalEntry]:
    """Attach each source's full body by concatenating its fragments in `seq` order.

    The body is read ONLY to hand to the local model. It is never returned past `reflect()`'s model
    call, never traced.
    """
    entries: list[JournalEntry] = []
    for r in rows:
        frags = con.execute(
            "SELECT text FROM fragments WHERE source_id = ? ORDER BY seq", (r["id"],)).fetchall()
        body = "\n\n".join(f["text"] for f in frags).strip()
        entries.append(JournalEntry(
            source_id=r["id"], path=r["path"], stem=_stem(r["path"]),
            authored_at=r["authored_at"], title=(r["title"] or _stem(r["path"])), body=body))
    return entries


def _journals(con, *, after: str | None = None, on_or_after: str | None = None,
              before: str | None = None) -> list[JournalEntry]:
    """Dated journals in a date range, oldest first. Bounds are YYYY-MM-DD strings compared
    lexicographically. `after` is exclusive, `on_or_after` inclusive."""
    where = ["type = 'journal'", "deleted = 0", "authored_at IS NOT NULL"]
    params: list[str] = []
    if after is not None:
        where.append("authored_at > ?")
        params.append(after)
    if on_or_after is not None:
        where.append("authored_at >= ?")
        params.append(on_or_after)
    if before is not None:
        where.append("authored_at < ?")
        params.append(before)
    rows = con.execute(
        f"SELECT id, path, title, authored_at FROM sources WHERE {' AND '.join(where)} "
        f"ORDER BY authored_at, path", params).fetchall()
    return _load_entries(con, rows)


def last_window_end(vault: Path = VAULT) -> str | None:
    """The `window_end` of the most recent reflection, read from `_Reflections/` on disk.

    `_Reflections/` is excluded from the index (config.EXCLUDED_TOP), so this is deliberately a
    filesystem read, not a DB query — the resume point is a visible, rebuildable fact in the vault.
    Returns None when no reflection exists yet.
    """
    folder = vault / NAMESPACE
    if not folder.is_dir():
        return None
    best: str | None = None
    for f in sorted(folder.glob("*.md")):
        try:
            fm, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        except OSError:
            continue
        we = fm.get("window_end")
        if isinstance(we, str) and we and (best is None or we > best):
            best = we
    return best


def _render_entries(entries: list[JournalEntry]) -> str:
    """One dated block per entry, each headed by its aliased wikilink. Body verbatim."""
    blocks = []
    for e in entries:
        blocks.append(f"### {e.authored_at} — {_link(e)}\n{e.body}")
    return "\n\n".join(blocks)


def _sanitize_links(text: str, labels: dict[str, str]) -> str:
    """Force every valid [[link]] to the canonical aliased form and strip any the model invented.

    Two jobs, both deterministic-truth-in-code: (1) a link whose target is a real entry is rewritten
    to `[[stem|short label]]` — so no matter what the model emitted (a bare stem, its own wording),
    the prose shows the compact label, never the long stem+hash. (2) a link the model made up (no
    matching entry) is stripped to plain text, so a reflection can never point at a note that does
    not exist. `labels` maps allowed stem → its display label.
    """
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        target, _, alias = inner.partition("|")
        target = target.strip()
        if target in labels:
            return f"[[{target}|{labels[target]}]]"
        return (alias or target).strip()
    return _WIKILINK_RE.sub(repl, text)


def _frontmatter(*, window_start: str, window_end: str, now: datetime,
                 sources: list[str], model: str) -> str:
    src = ", ".join(sources)
    lines = [
        "---",
        f'title: "Reflection — {window_start} to {window_end}"',
        "type: reflection",
        "generated_by: slim reflect",
        f"authored_at: {window_end}",
        f"reflected_at: {now.isoformat(timespec='seconds')}",
        f"window_start: {window_start}",
        f"window_end: {window_end}",
        f"model: {model}",
        f"sources: [{src}]",
        "---",
        "",
    ]
    return "\n".join(lines)


def _write_note(vault: Path, *, filename_date: str, frontmatter: str, body: str,
                footer: str) -> str:
    """Create-only atomic write into `_Reflections/` (temp file → os.link).

    Never overwrites: a same-date collision adds a numeric suffix rather than clobbering a note
    they may have read, and rather than erroring — a nightly job must not fail on a name clash.
    """
    folder = (vault / NAMESPACE)
    folder.mkdir(parents=True, exist_ok=True)
    text = frontmatter + body.strip() + "\n\n" + footer
    if not text.endswith("\n"):
        text += "\n"

    partial = folder / f".{uuid.uuid4().hex}.part"
    partial.write_text(text, encoding="utf-8")
    try:
        for n in range(1, 100):
            name = f"{filename_date}--reflection.md" if n == 1 \
                else f"{filename_date}--reflection-{n}.md"
            target = folder / name
            try:
                os.link(partial, target)
                return f"{NAMESPACE}/{name}"
            except FileExistsError:
                continue
        raise RuntimeError(f"could not find a free reflection filename for {filename_date}")
    finally:
        partial.unlink(missing_ok=True)


def reflect(con, *, vault: Path = VAULT, days: int | None = None, min_entries: int = 1,
            now: datetime | None = None, dry_run: bool = False) -> ReflectionResult:
    """Reflect on the journal window and (unless dry_run) write ONE note into `_Reflections/`.

    Window: `--days N` reflects on the last N days (inclusive); otherwise the window resumes from
    the last reflection's `window_end` (exclusive), or the last week on the first ever run. A window
    with fewer than `min_entries` new journals is a quiet night — nothing is written, and because
    the window only advances on a WRITE, a skipped stretch simply widens the next run.
    """
    now = now or datetime.now().astimezone()
    today = now.date()

    # --- resolve the FOCUS lower bound
    if days is not None:
        since = (today - timedelta(days=days)).isoformat()
        focus = _journals(con, on_or_after=since)
    else:
        lw = last_window_end(vault)
        if lw:
            since = lw
            focus = _journals(con, after=since)          # exclusive: don't re-reflect the boundary
        else:
            since = (today - timedelta(days=DEFAULT_FIRST_WINDOW_DAYS)).isoformat()
            focus = _journals(con, on_or_after=since)

    # --- G1: quiet nights stay quiet
    if len(focus) < min_entries:
        reason = "quiet" if not focus else "below-floor"
        result = ReflectionResult(
            skipped=True, reason=reason, dry_run=dry_run, window_start=since,
            focus_count=len(focus))
        if not dry_run:
            _trace(result)
        return result

    window_end = focus[-1].authored_at
    earliest = focus[0].authored_at

    # --- CONTEXT: read-only background, the entries just before the window (capped)
    context_from = (datetime.strptime(earliest, "%Y-%m-%d").date()
                    - timedelta(days=CONTEXT_LOOKBACK_DAYS)).isoformat()
    context = _journals(con, on_or_after=context_from, before=earliest)
    if len(context) > CONTEXT_MAX_ENTRIES:
        context = context[-CONTEXT_MAX_ENTRIES:]         # keep the most recent before the window

    sources = [e.stem for e in focus]
    result = ReflectionResult(
        dry_run=dry_run, window_start=since, window_end=window_end,
        focus_count=len(focus), context_count=len(context), sources=sources,
        model=llm.MODEL)

    if dry_run:
        result.path = f"{NAMESPACE}/{window_end}--reflection.md"   # what a real run WOULD write
        return result

    # --- one model call
    entries = focus + context
    labels = {e.stem: _display(e) for e in entries}
    allowed_list = "\n".join(_link(e) for e in entries) or "(none)"
    user = (f"FOCUS ENTRIES (reflect on these):\n\n{_render_entries(focus)}\n\n"
            f"CONTEXT ENTRIES (read-only background — do NOT summarize these):\n\n"
            f"{_render_entries(context) or '(none)'}\n\n"
            f"Wikilinks you may use (copy exactly, including the label after '|'):\n{allowed_list}")
    content, stats = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        model=llm.MODEL, num_ctx=llm.RESIDENT_CTX,
        temperature=llm.REPLY_TEMPERATURE, num_predict=NUM_PREDICT, think=False)

    body = _sanitize_links(content.strip(), labels)
    footer = "_Reflected on: " + ", ".join(_link(e) for e in focus) + "_\n"
    fm = _frontmatter(window_start=since, window_end=window_end, now=now,
                      sources=sources, model=stats.get("model", llm.MODEL))
    result.path = _write_note(vault, filename_date=window_end, frontmatter=fm, body=body,
                              footer=footer)
    result.model = stats.get("model", llm.MODEL)
    result.text = body
    _trace(result, stats=stats)
    return result


def _trace(result: ReflectionResult, *, stats: dict | None = None) -> None:
    """Operational record — METADATA ONLY. Never a journal body and never the reflection prose:
    neither may land in a durable log."""
    payload = {
        "prompt_version": REFLECT_PROMPT_VERSION,
        "skipped": result.skipped,
        "reason": result.reason,
        "window_start": result.window_start,
        "window_end": result.window_end,
        "focus_count": result.focus_count,
        "context_count": result.context_count,
        "n_sources": len(result.sources),
        "path": result.path,
        "model": result.model,
    }
    if stats:
        payload["output_tokens"] = stats.get("output_tokens")
        payload["duration_s"] = stats.get("duration_s")
        payload["ctx_saturated"] = stats.get("ctx_saturated")
    trace.record("reflect", payload)
