"""MASK — personal data in recordings ALREADY in the vault, at the owner's request (`slim mask`).
New recordings are masked at capture (`pipeline._mask_segments`); this is for the ones filed
before that, and for a rerun after the masker changes or its password pass failed.

What is masked: the transcript, and SLIM's summary (written from the raw transcript, so it can
repeat a value). Never `## Notes`: their typing is not read into the masker at all. Only the
masked spans change; every other byte of the note is sliced through (`note.replace_transcript`).

Which notes: `origin: recorded`, indexable, and not under a `pii.skip` prefix (`--all`); a note
named on the command line runs wherever it sits. Done means `pii_by` is the current version and
the password pass completed: a note left with only `audio` unfinished is done, since a text
rerun cannot fix audio.

Values are never returned, printed or traced: class counts and reasons only.
"""
import os
from pathlib import Path

from .. import config, trace
from ..capture import pii
from ..index.ingest import indexable
from ..vault.frontmatter import parse_frontmatter
from . import note as note_writer, state

# A note that keeps changing under the masker (an append while the password pass runs) is
# re-read and redone; past this many tries it is reported, never overwritten.
_ATTEMPTS = 3


def _under(rel: Path, prefixes) -> bool:
    """Whole path segments, case-folded like `indexable`: `Notes/a` skips `Notes/a/x.md`, not
    `Notes/ab/x.md`."""
    parts = tuple(p.casefold() for p in rel.parts)
    for prefix in prefixes:
        want = tuple(p.casefold() for p in Path(prefix).parts)
        if want and parts[:len(want)] == want:
            return True
    return False


def recordings(vault: Path) -> list[str]:
    """Vault-relative paths of the recordings `slim mask --all` selects, sorted.

    A skipped folder is pruned from the walk, so nothing under it is even opened."""
    vault = Path(vault)
    out = []
    for root, dirs, files in os.walk(vault):
        here = Path(root).relative_to(vault)
        dirs[:] = [d for d in dirs if not _under(here / d, config.PII_SKIP)]
        for name in files:
            rel = here / name
            if rel.suffix.casefold() != ".md" or not indexable(rel) or _under(rel, config.PII_SKIP):
                continue
            try:
                text = (vault / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue                 # not readable as a note, so not one we can mask
            if parse_frontmatter(text)[0].get("origin") == "recorded":
                out.append(rel.as_posix())
    return sorted(out)


def _read_raw(path: Path) -> str:
    """The note's exact characters: no newline translation, so a `\r` is seen, not hidden."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _restores(original: str, new: str, transcript: str, summary: str | None) -> bool:
    """INTEGRITY: put the original transcript, summary and frontmatter back into `new`, found
    by the same locators on the NEW text, and require the original byte for byte. Anything a
    splice moved outside those three places fails this."""
    try:
        back = new
        a, b = note_writer.transcript_span(back)
        back = back[:a] + transcript + back[b:]
        span = note_writer.summary_span(back)
        if (span is None) != (summary is None):
            return False
        if span is not None:
            back = back[:span[0]] + summary + back[span[1]:]
        _, old_skip = parse_frontmatter(original)
        _, new_skip = parse_frontmatter(back)
        old_lines = original.splitlines(keepends=True)
        back = "".join(old_lines[:old_skip] + back.splitlines(keepends=True)[new_skip:])
    except ValueError:
        return False
    return back == original


def _plan(text: str, force: bool) -> dict:
    """What masking `text` would write: {"text", "counts" (found now), "partial" (reasons)},
    or {"skipped": why}, or {"error": why}."""
    if "\r" in text:
        # Recordings SLIM writes are LF. Refused rather than converted: a second code path for
        # CR would be a second way to move bytes the owner did not ask to move.
        return {"error": "CR line endings"}
    fm, skip = parse_frontmatter(text)
    if not skip:
        return {"skipped": "no frontmatter"}
    if fm.get("origin") != "recorded":
        return {"skipped": "not a recording"}
    had_counts, had = note_writer.read_pii(text)
    if fm.get("pii_by") == pii.VERSION and "password" not in had and not force:
        return {"skipped": "already masked", "partial": had}
    try:
        transcript = note_writer.read_transcript(text)
    except ValueError as exc:
        if "no transcript" in str(exc):
            return {"skipped": "no transcript"}
        return {"error": str(exc)}
    span = note_writer.summary_span(text)
    summary = text[span[0]:span[1]] if span else None

    find = lambda window: pii.ask_model(window)          # noqa: E731 - read at call time
    counts: dict[str, int] = {}
    partial = False
    out = text
    found = []
    try:
        m = pii.mask(transcript, find_password=find)
        found.append(m)
        if m.text != transcript:
            out = note_writer.replace_transcript(out, m.text)
        if summary:                       # the exact span, unstripped: its whitespace is kept
            sm = pii.mask(summary, find_password=find)
            found.append(sm)
            if sm.text != summary:
                out = note_writer.replace_summary(out, sm.text)
    except ValueError:
        return {"error": "integrity"}
    for one in found:
        partial = partial or one.partial
        for cls, n in one.counts.items():
            counts[cls] = counts.get(cls, 0) + n

    reasons = set(had)
    if "password" in config.PII_MASK:     # the pass ran: its outcome replaces the old one
        reasons.discard("password")
        if partial:
            reasons.add("password")
    total = dict(had_counts)
    for cls, n in counts.items():
        total[cls] = total.get(cls, 0) + n
    out = note_writer.set_pii_frontmatter(out, total, reasons)
    if not _restores(text, out, transcript, summary):
        return {"error": "integrity"}
    return {"text": out, "counts": counts, "partial": sorted(reasons)}


def mask_note(vault: Path, rel: str, *, force: bool = False, dry_run: bool = False) -> dict:
    """Mask one recording in place, then re-index it.

    Returns {"note", "counts" (classes found this run), "partial" (unfinished reasons),
    "changed"}, plus "skipped" (left alone, nothing wrong), "error" (left alone and the run
    must fail: CR line endings, integrity, changed while masking) or "index_error" (written,
    not re-indexed). Never writes a value anywhere."""
    out: dict = {"note": rel, "counts": {}, "partial": [], "changed": False}
    if not config.PII_MASK:
        return {**out, "skipped": "masking is off"}
    if not indexable(Path(rel)):
        # A draft still being recorded, or a folder SLIM never indexes: a write here would race
        # the recorder or produce a note no reader can reach.
        return {**out, "skipped": "not indexed: draft or excluded folder"}
    src = state.vault_path(vault, rel)
    for _ in range(_ATTEMPTS):
        text = _read_raw(src)
        plan = _plan(text, force)
        if "error" in plan:
            trace.record("mask", {"note": rel, "status": "failed", "reason": plan["error"]})
        if "skipped" in plan or "error" in plan:
            return {**out, **plan}
        out.update(counts=plan["counts"], partial=plan["partial"])
        if dry_run or plan["text"] == text:
            return out
        # ⚠ The password pass can take minutes, and this lock is per PROCESS: a server
        # appending to the note is not held off by it. So the write happens only if the note
        # is still the text that was masked; otherwise it is masked again as it now stands.
        with state.note_lock(vault, rel):
            if not src.is_file() or _read_raw(src) != text:
                continue
            note_writer.write_note_text(src, plan["text"])
        out["changed"] = True
        indexed = state.index_now(vault, rel, sweep=False)
        if "error" in indexed:
            out["index_error"] = indexed["error"]
        return out
    trace.record("mask", {"note": rel, "status": "failed", "reason": "changed while masking"})
    return {**out, "error": "changed while masking"}
