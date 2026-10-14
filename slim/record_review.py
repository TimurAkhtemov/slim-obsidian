"""The card's edits to a filed recording: their notes, a summary retry, apply (which can move
the note) and plain approval."""

import re
from pathlib import Path

from . import llm, record_pipeline, record_state, trace

def _strip_frontmatter_key(text: str, key: str) -> str:
    """Remove a single `key:` line from the frontmatter block only, leaving the body exact."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    out, in_fm = [lines[0]], True
    for line in lines[1:]:
        if in_fm and line.strip() == "---":
            in_fm = False
        elif in_fm and re.match(rf"^{re.escape(key)}:(\s|$)", line):
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def handle_record_notes(*, note: str, notes_md: str, vault: Path | None = None) -> dict:
    """Save THEIR notes, and nothing else.

    Deliberately NOT `handle_record_apply`: apply rewrites frontmatter, drops `filed_by` and
    can MOVE the file. Typing a note is not a filing decision, and an autosave that files is
    an autosave that surprises. The summary is preserved by `summary_md=None` rather than
    round-tripped through the client, which would one day hand back an empty one.
    """
    from . import record as record_mod
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    src = record_state._vault_path(vault, note)
    with record_state._note_lock(vault, note):
        text = src.read_text(encoding="utf-8")
        record_mod.write_note_text(
            src, record_mod.replace_review_sections(text, summary_md=None, notes_md=notes_md))
    record_state._index_now(vault, str(src.relative_to(vault)), sweep=False)
    trace.record("record", {"stage": "notes", "note": str(src)})
    return {"note": str(src.relative_to(vault))}


def handle_record_summary(*, note: str, instructions: str = "", job_id: str = "",
                          vault: Path | None = None) -> dict:
    """Retry the summary against the transcript the note already holds.

    The instructions field is their steer for this one retry, in their own words. Nothing tells
    the model what KIND of recording this is; the no-template-switch rule is intact.

    ⚠ A FAILED RETRY KEEPS THE SUMMARY IT HAS. The recorder's worst failure is reporting
    success while losing what it exists to keep, and blanking a good summary because the model
    died is that shape exactly. This raises and writes nothing.
    """
    from . import record as record_mod, summarize
    from .chunk import parse_frontmatter
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    src = record_state._vault_path(vault, note)
    text = src.read_text(encoding="utf-8")
    parts = record_mod.read_review_sections(text)
    if not parts["transcript"].strip():
        raise ValueError("this note has no transcript to summarize")

    record_state._record_progress_start(job_id)
    record_state._record_progress_update(job_id, stage="summarizing",
                                         label="Reasoning about the summary",
                                         detail="The local model is re-reading the transcript")
    activity = (lambda kind, chunk: record_state._record_progress_update(
        job_id, stage="summarizing", label="Reasoning about the summary",
        detail="Live local-model activity", kind=kind, text=chunk)) if job_id else None

    # Every knob matches the recording path, because this IS the recording path re-run: the
    # resident at its one context size, and thinking on (ratified 2026-08-26, lectures).
    clean_notes, note_images = record_pipeline._extract_note_images(parts["notes"], vault)
    s = summarize.summarize(parts["transcript"], model=llm.RECORD_MODEL,
                            num_ctx=llm.RECORD_CTX, think=True, on_activity=activity,
                            notes=clean_notes, instructions=instructions,
                            images=note_images or None)
    if not s.valid:
        record_state._record_progress_update(job_id, stage="failed", label="Retry failed",
                                             detail=s.error)
        raise ValueError(f"the summary could not be rewritten: {s.error}")

    fm, _ = parse_frontmatter(text)
    summary_md = summarize.render(
        s, open_questions=str(fm.get("type") or "") not in summarize.LECTURE_TYPES)
    # ⚠ `valid` means SCHEMA-valid, and `Summary()` — every field empty — passes it and renders
    # to "". Writing that removes the summary and its fence, which is the one thing a failed
    # retry promises not to do. An empty answer is a failed answer.
    if not summary_md.strip():
        record_state._record_progress_update(job_id, stage="failed", label="Retry failed",
                                             detail="the model returned an empty summary")
        raise ValueError("the summary could not be rewritten: the model returned an empty "
                         "summary, so the one you had is untouched")
    # The note as it stands NOW, not as it stood before the model started thinking. The lock
    # covers the read and the write, never the model call: holding it for the minutes a summary
    # takes would hang their notes autosave on every blur.
    with record_state._note_lock(vault, note):
        current = record_state._reread_for_write(src, note)
        current = record_mod.replace_review_sections(current, summary_md=summary_md)
        record_mod.write_note_text(src, current)
    record_state._index_now(vault, str(src.relative_to(vault)), sweep=False)
    record_state._record_progress_update(job_id, stage="done", label="Summary rewritten", detail="")
    trace.record("record", {"stage": "resummarize", "note": str(src),
                            "instructions": bool(instructions.strip()), "stats": s.stats})
    return {"note": str(src.relative_to(vault)), "summary": summary_md}


def handle_record_apply(*, note: str, dest_dir: str, type_tag: str, topics: list[str],
                        summary_md: str, notes_md: str | None = None,
                        title: str = "",
                        vault: Path | None = None) -> dict:
    """Apply the card's edits, and mark the edited fields AS THEIR.

    ⚠ Writing the new value is not enough. A model pass re-judges what it INFERRED but never
    overrides a declared value, so an edit that writes only the value leaves the field looking
    inferred and a later pass reverts it. Touching a field therefore also drops `filed_by`,
    whose ABSENCE is this vault's signal for "they declared this".
    """
    from . import record as record_mod
    from . import summarize
    from .chunk import set_frontmatter
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    src = record_state._vault_path(vault, note)
    # ⚠ Held across read, write AND move: apply is the one writer that can invalidate another
    # endpoint's path, and a bare acquire/release would leak the lock on the raise paths below.
    with record_state._note_lock(vault, note):
        text = src.read_text(encoding="utf-8")

        # ⚠ The summary was rendered against the type the CARD guessed, and this is where they
        # corrects that guess. Measured in the live plugin 2026-08-26: a lecture whose card said
        # otherwise kept its "Open questions" — the lecturer's rhetorical questions, sitting under
        # a heading that makes them read as the owner's own. Deterministic, and it touches only the
        # derived section; re-rendering properly would mean a second model call for one heading.
        if type_tag in summarize.LECTURE_TYPES:
            summary_md = summarize.drop_open_questions(summary_md)

        # `tags` moves with `type` or Obsidian's tag pane starts lying the moment they correct a
        # type on the card — the pane is the whole reason the field exists.
        updates: dict[str, object] = {"type": type_tag, "tags": [type_tag], "topics": topics,
                                      "subject_by": "declared", "tagged_by": "declared"}

        # A corrected title has to reach the FILENAME, not just the frontmatter. The filename is
        # what they read in the file tree and in Obsidian's quick switcher, so a note titled
        # correctly in its frontmatter and wrongly on disk is still wrongly named where it counts.
        # Date and content digest are preserved: the first is when it happened, the second is what
        # makes the name content-addressed, and neither is theirs to change by retitling.
        name = src.name
        if title.strip():
            from . import notefile
            updates["title"] = f'"{record_mod._clean_scalar(title)}"'
            parts = src.stem.split("--")
            if len(parts) >= 3:
                name = f"{parts[0]}--{notefile.slug(title)}--{parts[-1]}{src.suffix}"

        text = set_frontmatter(text, updates)
        text = record_mod.set_review_status(text, "complete")
        text = _strip_frontmatter_key(text, "filed_by")
        text = record_mod.replace_review_sections(
            text, summary_md=summary_md, notes_md=notes_md)

        dest = record_state._vault_path(vault, dest_dir) / name
        if dest != src and dest.exists():
            # Both movers grew this guard on 2026-08-04 after two notes could resolve to one
            # destination and one was silently lost. A third mover does not get to relearn it.
            raise ValueError(f"refusing to overwrite an existing note at {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        record_mod.write_note_text(dest, text)
        if dest != src:
            src.unlink()

    record_state._index_now(vault, str(dest.relative_to(vault)),
                            moved_from=str(src.relative_to(vault)) if dest != src else None)
    trace.record("record", {"stage": "apply", "note": str(dest), "moved": dest != src})
    return {"note": str(dest.relative_to(vault))}


def handle_record_review(*, note: str, vault: Path | None = None) -> dict:
    """Mark an unchanged recording reviewed without turning approval into a filing edit.

    Approving the card as it stands is agreeing with where SLIM put it, so `filed_by` goes,
    as it does on apply: its presence means "SLIM guessed and nobody has looked"."""
    from . import record as record_mod
    from .chunk import parse_frontmatter
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    src = record_state._vault_path(vault, note)
    with record_state._note_lock(vault, note):
        text = src.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        if fm.get("origin") != "recorded":
            raise ValueError("review status is only valid for recorded notes")
        text = record_mod.set_review_status(text, "complete")
        record_mod.write_note_text(src, _strip_frontmatter_key(text, "filed_by"))
    record_state._index_now(vault, str(src.relative_to(vault)), sweep=False)
    trace.record("record", {"stage": "review", "note": str(src)})
    return {"note": str(src.relative_to(vault))}
