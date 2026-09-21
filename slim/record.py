"""RECORD — turn one finished recording into ONE note on disk. Called by `chat.py`.

The sole writer of a recorded note's frontmatter and managed sections. A pure function of its
inputs — a path and strings in, a file out, and NO routing judgement, since the destination
arrives already decided from the card. Audio arrives as a PATH already inside the vault;
`archive_audio` moves it into the content-addressed archive and `write_recording` deletes the
staged copy once it is safely there.

Layout, and each part of it is load-bearing:

    frontmatter        queryable. Type, topics and provenance — the fields code reads.
    slim-meeting block the inline Obsidian object; ordinary Markdown can continue after it.
      <fenced summary> SLIM's opinion — blanked by chunk.strip_derived_blocks before ingestion.
      ## Notes         THEIR words, typed while recording. UNFENCED, therefore indexed.
      ## Transcript    the raw record. Never rewritten.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .chunk import SUMMARY_CLOSE, SUMMARY_OPEN, TRANSCRIPT_MARKER
from .config import VAULT

RECORD_PROMPT_VERSION = "record-v2"
MEETING_BLOCK_LANGUAGE = "slim-meeting"
# Where the plugin writes audio while a recording is in progress, and the ONLY directory whose
# files a server-side path is allowed to consume and unlink. Mirrors STAGING_DIR in main.js.
STAGING_DIR = "Attachments/_incoming"


@dataclass
class Recorded:
    note: Path
    audio: Path
    digest: str


@dataclass(frozen=True)
class MeetingBlock:
    before: str
    opener: str
    body: str
    closer: str
    after: str


def split_meeting_block(text: str) -> MeetingBlock | None:
    """Return the first durable inline meeting block without normalizing any surrounding text."""
    opener = re.search(
        rf"(?m)^(?P<fence>`{{3,}}){MEETING_BLOCK_LANGUAGE}[^\n]*\n", text)
    if opener is None:
        return None
    fence = opener.group("fence")
    closer = re.search(rf"(?m)^{re.escape(fence)}[ \t]*(?:\n|$)", text[opener.end():])
    if closer is None:
        raise ValueError("recorded note has an unterminated slim-meeting block")
    close_start = opener.end() + closer.start()
    close_end = opener.end() + closer.end()
    return MeetingBlock(
        before=text[:opener.start()],
        opener=text[opener.start():opener.end()],
        body=text[opener.end():close_start],
        closer=text[close_start:close_end],
        after=text[close_end:],
    )


def wrap_meeting_block(body: str) -> str:
    """Fence managed recording Markdown so Obsidian can replace it with one inline object.

    The fence is longer than every backtick run inside the recording, so a transcript that
    discusses Markdown cannot accidentally terminate its own meeting block.
    """
    longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
    fence = "`" * max(4, longest + 1)
    return f"{fence}{MEETING_BLOCK_LANGUAGE}\n{body.rstrip()}\n{fence}"


# ⚠ `(?:^|\n)`, NOT `\n`. A note whose summary is empty — exactly the note they presse Retry
# summary on — has a block that STARTS with `## Notes`, and a leading-newline pattern read it
# as having no notes, so every `notes_md=None` write dropped them. The plugin's parser has
# always used `(?:^|\n)`, so the UI showed their notes right up until the server dropped them
# (2026-08-26).
_NOTES_RE = (rf"(?:^|\n)## Notes[ \t]*\r?\n(?:\r?\n)?([\s\S]*?)"
             rf"(?=\r?\n{re.escape(TRANSCRIPT_MARKER)}[ \t]*(?:\r?\n|$))")


def _after_summary(body: str) -> str:
    """Everything past the derived summary's closing marker, or the whole body if there is
    none. One helper, because both readers must agree on where their source text begins."""
    at = body.find(SUMMARY_CLOSE)
    return body[at + len(SUMMARY_CLOSE):] if at >= 0 else body


def read_review_sections(text: str) -> dict:
    """The inverse of `replace_review_sections`: summary, notes and transcript as they stand.

    Lives beside the writer so the two move together; the plugin has its own parser for its
    tabs, and a second one on this side would be a second thing to get wrong.
    """
    block = split_meeting_block(text)
    scope = block.body if block is not None else text
    fm_end = 0
    if block is None and text.startswith("---\n"):
        fm_end = scope.find("\n---", 4)
        fm_end = fm_end + 4 if fm_end >= 0 else 0
    transcript_at = scope.rfind(f"\n{TRANSCRIPT_MARKER}")
    head = scope[fm_end:transcript_at] if transcript_at >= fm_end else scope[fm_end:]
    transcript = scope[transcript_at + len(TRANSCRIPT_MARKER) + 1:] if transcript_at >= fm_end else ""

    summary = re.search(
        rf"^<!-- slim:summary[^\n]*-->\n## Summary\n+([\s\S]*?)\n{re.escape(SUMMARY_CLOSE)}",
        head, re.M)
    # ⚠ AFTER the summary fence. The summary is hand-editable, so it can contain `## Notes`
    # itself; searching the whole block let that swallow the closing marker and their real notes,
    # which a later write would re-emit as SOURCE.
    notes = re.search(_NOTES_RE, _after_summary(scope[fm_end:]))
    return {
        "summary": summary.group(1).strip() if summary else "",
        "notes": notes.group(1).strip() if notes else "",
        "transcript": transcript.strip(),
    }


def replace_review_sections(text: str, *, summary_md: str | None,
                            notes_md: str | None = None) -> str:
    """Replace only the editable review sections of a recorded note.

    Frontmatter and the transcript are sliced from the original string and returned unchanged.
    The summary remains inside its derived fence; notes remain unfenced human-authored source.
    ``notes_md=None`` and ``summary_md=None`` each preserve that section exactly, so a caller
    never has to hand a section back just to leave it alone.
    """
    block = split_meeting_block(text)
    scope = block.body if block is not None else text
    if block is None and not text.startswith("---\n"):
        raise ValueError("recorded note has no frontmatter")
    fm_end = 0
    if block is None:
        fm_end = scope.find("\n---", 4)
        if fm_end < 0:
            raise ValueError("recorded note has unterminated frontmatter")
        fm_end += 4
    transcript_at = scope.rfind(f"\n{TRANSCRIPT_MARKER}")
    if transcript_at < fm_end:
        raise ValueError("recorded note has no transcript section")

    if notes_md is None:
        existing = re.search(_NOTES_RE, _after_summary(scope[fm_end:]))
        notes_md = existing.group(1) if existing else ""
    if summary_md is None:
        summary_md = read_review_sections(text)["summary"]
    # ⚠ STRIPPED, not refused. A hand-edited summary can contain the marker that tells
    # `chunk.strip_derived_blocks` where derived content ends, and two of them break the fence.
    # Refusing would block their approval on a paste they probably did not mean.
    for marker in (SUMMARY_CLOSE, SUMMARY_OPEN):
        summary_md = summary_md.replace(marker, "")

    opener = re.search(r"^<!-- slim:summary[^\n]*-->$", scope[fm_end:transcript_at], re.M)
    summary_open = opener.group(0) if opener else (
        f'{SUMMARY_OPEN} prompt="{RECORD_PROMPT_VERSION}" -->')

    body = ""
    if summary_md.strip():
        body += f"\n\n{summary_open}\n## Summary\n\n{summary_md.strip()}\n{SUMMARY_CLOSE}"
    if notes_md.strip():
        body += f"\n\n## Notes\n\n{notes_md}"
    # `scope[transcript_at:]` begins at the newline before ## Transcript. It is the raw source
    # boundary and is concatenated, not rendered or normalized.
    replaced = scope[:fm_end] + body + scope[transcript_at:]
    if block is None:
        return replaced
    # ⚠ RE-FENCED, not reopened. Edited notes or a summary can contain a backtick run as long
    # as the fence already there, which would close the block early and leave the transcript
    # outside the object. `wrap_meeting_block` sizes the fence to its contents.
    # ⚠ `block.after` is their ordinary Markdown and is concatenated untouched, byte for byte.
    return block.before + wrap_meeting_block(replaced.lstrip("\n")) + "\n" + block.after


# One owner for "which recording does this staged file belong to". The rule was written three
# times in two languages, they agreed only by luck, and missing one made EVERY recording fail
# on 2026-08-26 with "audio and draft recording ids do not match".
_SEGMENT_SUFFIX = re.compile(r"-\d{3}$")


def recording_id_from_staged(path: str) -> str:
    """`Attachments/_incoming/<id>-001.webm` -> `<id>`. A name with no segment number is
    returned unchanged: recordings made before segments existed still have to resolve."""
    return _SEGMENT_SUFFIX.sub("", Path(path).stem)


def append_audio_frontmatter(text: str, *, rels: list[str], digests: list[str],
                             seconds: float, asr_model: str = "",
                             speakers_by: str = "") -> str:
    """Add a resumed segment's audio to a recorded note's frontmatter.

    Here rather than in `chat.py`: these are `render_note`'s keys and the scalar-or-list rule
    that shapes them is this module's. A second writer is a second place to get it wrong.

    `asr_model` is written only when the note has none: a note first transcribed by one model
    keeps that provenance, and a resume that produced no model name never blanks it. `speakers_by`
    follows the same rule.
    """
    from .chunk import parse_frontmatter, set_frontmatter

    fm, _ = parse_frontmatter(text)

    def existing(key: str) -> list[str]:
        value = fm.get(key)
        if not value:
            return []
        return [str(v) for v in value] if isinstance(value, list) else [str(value)]

    try:
        total = float(fm.get("audio_seconds") or 0.0) + seconds
    except (TypeError, ValueError):
        total = seconds
    updates = {
        "audio": _scalar_or_list(existing("audio") + list(rels)),
        "audio_sha256": _scalar_or_list(existing("audio_sha256") + list(digests)),
        "audio_seconds": f"{total:.1f}",
    }
    if asr_model and not fm.get("asr_model"):
        updates["asr_model"] = asr_model
    if speakers_by and not fm.get("speakers_by"):
        updates["speakers_by"] = speakers_by
    return set_frontmatter(text, updates)


def write_note_text(path: Path, text: str) -> None:
    """The one way a recorded note is written after it exists.

    ⚠ `Path.write_text` TRUNCATES before it writes. A disk-full or a crash mid-write leaves the
    note SHORTER than it was — and what sits at the end of a recorded note is the raw
    transcript. Every derived edit (notes autosave, retry, append) goes through here so an
    interrupted write leaves the previous note intact rather than half a recording.
    """
    from . import inbox
    inbox._atomic_write_text(path, text)


def set_review_status(text: str, status: str) -> str:
    """Persist the recorder's approval state in the note itself.

    Workflow state in the source, not a queue: reopening the file reconstructs it. Older notes
    have no key and read as complete.
    """
    if status not in {"pending", "complete"}:
        raise ValueError(f"invalid recording review status: {status!r}")
    from .chunk import set_frontmatter
    return set_frontmatter(text, {"review_status": status})


def append_transcript(text: str, addition: str) -> str:
    """Add a resumed segment's words to the end of a note's transcript.

    ⚠ ADDS, NEVER REWRITES. Existing transcript is concatenated through untouched — no reflow,
    no normalization — because it is raw testimony. The block is re-fenced rather than reopened:
    a resumed segment can carry a longer backtick run than the fence already there.
    """
    if not addition.strip():
        return text
    block = split_meeting_block(text)
    scope = block.body if block is not None else text
    if f"\n{TRANSCRIPT_MARKER}" not in scope:
        raise ValueError("recorded note has no transcript section")
    grown = scope.rstrip() + "\n\n" + addition.strip() + "\n"
    if block is None:
        return grown
    return block.before + wrap_meeting_block(grown) + "\n" + block.after.lstrip("\n")


def archive_audio(src: Path) -> tuple[Path, str]:
    """Archive a staged file content-addressed, returning (path, sha256). Reuses inbox's
    hash-verified copier so there is exactly one implementation of 'the audio is safe'."""
    from . import inbox

    digest = inbox._sha256(src)
    archived = inbox._archive_audio(src, digest)
    return archived, digest


def _clean_scalar(value: str) -> str:
    """One frontmatter value, safe to write into a flow list.

    ⚠ Quoting does NOT solve this. SLIM's own reader splits a flow list on bare commas and
    ignores quotes entirely (`chunk.parse_frontmatter`: `val[1:-1].split(",")`), so a comma
    inside a value silently becomes TWO values — measured on a model-written field that
    returned "Mercer's condition, restated". `topics` is model-generated free text, so fixing
    the writer is the safe half; `parse_frontmatter` is what ingest reads every note with.

    Commas and newlines become spaces (a search phrase loses nothing); brackets are dropped so
    a value can never terminate the list early.
    """
    s = str(value).replace(",", " ").replace("[", "").replace("]", "")
    return " ".join(s.split())          # collapses newlines and runs of whitespace


def _yaml_list(values: list[str]) -> str:
    cleaned = [c for c in (_clean_scalar(v) for v in values) if c]
    return "[" + ", ".join(cleaned) + "]"


def _scalar_or_list(value: str | list[str]) -> str:
    """One value writes a scalar; several write a YAML flow list. Nothing in SLIM parses these
    keys, so the shape is chosen for the person reading the
    note, not for a parser."""
    if isinstance(value, str):
        return value
    return value[0] if len(value) == 1 else _yaml_list(list(value))


def render_note(*, title: str, when: datetime, type_tag: str, transcript: str,
                notes_md: str, summary_md: str,
                topics: list[str], audio_rel: str | list[str],
                filed_by_slim: bool, audio_sha256: str | list[str] = "", asr_model: str = "",
                speakers_by: str = "",
                audio_seconds: float | None = None,
                transcribed_at: datetime | None = None) -> str:
    """One recording's frontmatter.

    ⚠ THE KEY ORDER MATCHES `inbox._render_note`. Two lanes write recordings, and when they
    wrote different vocabularies in different orders two notes of the same kind did not look
    alike in the one place they read them. Nothing in SLIM parses key order; THEY do. Fields
    inbox derives from spoken routing stay absent here, because the recorder has none.
    """
    from . import inbox

    # The title arrives from the plugin. A newline in it would end the `title:` line and leave
    # the rest of the string sitting in the frontmatter block as a stray, unparseable line.
    fm = ["---",
          f'title: "{_clean_scalar(title).replace(chr(34), "")}"',
          f"date: {inbox.local_date(when)}",
          f"type: {type_tag}",
          # For OBSIDIAN's tag pane, graph and search — nothing in SLIM reads this. Without it
          # a recorded note is invisible where every memo note shows up, which is exactly the
          # difference between the two lanes that they noticed.
          f"tags: [{type_tag}]",
          f"topics: {_yaml_list(topics)}"]
    fm += ["origin: recorded", "review_status: pending"]
    if asr_model:
        # Omitted rather than faked when transcription failed: an `asr_model` on a note with
        # no transcript would say a model produced the silence. `asr_selected: false` records
        # that these words came from an unvalidated default, so a future re-transcription
        # knows which notes to revisit.
        fm += ["source: local-asr", f"asr_model: {asr_model}", "asr_selected: false"]
    if speakers_by:
        # Provenance like `tagged_by`: which layer decided who spoke. "channels" is the
        # capture itself — microphone against system audio — not a model's opinion.
        fm.append(f"speakers_by: {speakers_by}")
    # A RESUMED recording has several segments: a second MediaRecorder session writes its own
    # container header, so one file per segment is what keeps every recording playable. A single
    # segment still writes a plain scalar, which is what 300 existing notes carry.
    fm.append(f"audio: {_scalar_or_list(audio_rel)}")
    if audio_sha256:
        # Also in the audio's filename, which is what makes it content-addressed. Here it is
        # queryable without parsing a path.
        fm.append(f"audio_sha256: {_scalar_or_list(audio_sha256)}")
    if audio_seconds is not None:
        fm.append(f"audio_seconds: {audio_seconds:.1f}")
    fm.append(f"recorded_at: {when.isoformat(timespec='seconds')}")
    if transcribed_at is not None:
        fm.append(f"transcribed_at: {transcribed_at.isoformat(timespec='seconds')}")
    if filed_by_slim:
        # Provenance, not decoration: absence means THEY declared it.
        fm.append("filed_by: slim")
    fm.append("---")

    managed: list[str] = []
    if summary_md.strip():
        managed += [f"{SUMMARY_OPEN} prompt=\"{RECORD_PROMPT_VERSION}\" -->",
                    "## Summary", "", summary_md.strip(), SUMMARY_CLOSE, ""]
    if notes_md.strip():
        # The draft is THEIR source text. Keep it byte-for-byte inside the Notes section — in
        # particular trailing spaces (Markdown hard breaks) and image embeds are meaningful.
        managed += ["## Notes", "", notes_md]
    managed += [TRANSCRIPT_MARKER, "", transcript.strip(), ""]
    body = ["", wrap_meeting_block("\n".join(managed)), ""]
    return "\n".join(fm + body)


def write_recording(*, audio_src: Path | list[Path], title: str, when: datetime, type_tag: str,
                    transcript: str, notes_md: str, summary_md: str,
                    topics: list[str], dest_dir: str, filed_by_slim: bool = True,
                    asr_model: str = "", speakers_by: str = "",
                    audio_seconds: float | None = None,
                    transcribed_at: datetime | None = None,
                    draft_src: Path | None = None,
                    vault: Path = VAULT) -> Recorded:
    from . import inbox

    sources = [audio_src] if isinstance(audio_src, Path) else list(audio_src)
    archived_all, digests = [], []
    for src in sources:
        one, one_digest = archive_audio(src)
        archived_all.append(one)
        digests.append(one_digest)
    archived, digest = archived_all[0], digests[0]
    audio_rels = []
    for one in archived_all:
        try:
            audio_rels.append(str(one.relative_to(vault)))
        except ValueError:
            audio_rels.append(one.name)
    audio_rel = audio_rels if len(audio_rels) > 1 else audio_rels[0]

    text = render_note(title=title, when=when, type_tag=type_tag, transcript=transcript,
                       notes_md=notes_md, summary_md=summary_md,
                       topics=topics, audio_rel=audio_rel,
                       filed_by_slim=filed_by_slim,
                       audio_sha256=digests if len(digests) > 1 else digest,
                       asr_model=asr_model, speakers_by=speakers_by, audio_seconds=audio_seconds,
                       transcribed_at=transcribed_at)

    name = f"{inbox.local_date(when)}--{inbox._slug(title)}--{digest[:8]}.md"
    note = vault / dest_dir / name
    if note.exists() and (draft_src is None or note.resolve() != draft_src.resolve()):
        raise ValueError(f"refusing to overwrite an existing note at {note}")
    inbox._atomic_write_text(note, text)

    # A completed note is the commit point. Until it exists, both human sources stay where the
    # retry path can name them. Once it does, remove the draft before the staged audio so a late
    # cleanup failure leaves MORE recoverable state, never less.
    draft_cleaned = True
    if draft_src is not None and draft_src.exists() and draft_src.resolve() != note.resolve():
        try:
            draft_src.unlink()
        except OSError:
            # The note is already committed. Keep the matching staged audio as well so the
            # startup recovery notice can still identify this pair; cleanup is never allowed
            # to turn a safely filed recording into a failed request.
            draft_cleaned = False
    if draft_cleaned:
        for src, one in zip(sources, archived_all):
            if src.exists() and src.resolve() != one.resolve():
                try:
                    src.unlink()
                except OSError:
                    # The content-addressed archive and final note are both safe. A leftover
                    # staged copy is visible to the startup check and can be removed by hand.
                    pass

    return Recorded(note=note, audio=archived, digest=digest)
