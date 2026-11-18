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


def _plan(text: str, force: bool) -> dict:
    """What masking `text` would write: {"text", "counts" (found now), "partial" (reasons)},
    or {"skipped": why}."""
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
        return {"skipped": "no transcript" if "no transcript" in str(exc) else str(exc)}

    find = lambda window: pii.ask_model(window)          # noqa: E731 - read at call time
    counts: dict[str, int] = {}
    partial = False
    out = text
    m = pii.mask(transcript, find_password=find)
    if m.text != transcript:
        out = note_writer.replace_transcript(out, m.text)
    found = [m]
    summary = note_writer.read_review_sections(out)["summary"]
    if summary:
        s = pii.mask(summary, find_password=find)
        found.append(s)
        if s.text != summary:
            out = note_writer.replace_review_sections(out, summary_md=s.text)
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
    return {"text": out, "counts": counts, "partial": sorted(reasons)}


def mask_note(vault: Path, rel: str, *, force: bool = False, dry_run: bool = False) -> dict:
    """Mask one recording in place, then re-index it.

    Returns {"note", "counts" (classes found this run), "partial" (unfinished reasons),
    "changed"}, plus "skipped" (why it was left alone) or "index_error" (written, not
    re-indexed). Never writes a value anywhere."""
    out: dict = {"note": rel, "counts": {}, "partial": [], "changed": False}
    if not config.PII_MASK:
        return {**out, "skipped": "masking is off"}
    src = state.vault_path(vault, rel)
    for _ in range(_ATTEMPTS):
        text = src.read_text(encoding="utf-8")
        plan = _plan(text, force)
        if "skipped" in plan:
            return {**out, **plan}
        out.update(counts=plan["counts"], partial=plan["partial"])
        if dry_run or plan["text"] == text:
            return out
        # ⚠ The password pass can take minutes, and this lock is per PROCESS: a server
        # appending to the note is not held off by it. So the write happens only if the note
        # is still the text that was masked; otherwise it is masked again as it now stands.
        with state.note_lock(vault, rel):
            if state.reread_for_write(src, rel) != text:
                continue
            note_writer.write_note_text(src, plan["text"])
        out["changed"] = True
        if indexable(Path(rel)):
            indexed = state.index_now(vault, rel, sweep=False)
            if "error" in indexed:
                out["index_error"] = indexed["error"]
        return out
    trace.record("mask", {"note": rel, "status": "skipped", "reason": "changed while masking"})
    return {**out, "skipped": "changed while masking"}
