"""RECORD — turn one finished recording into ONE note on disk. Called by `recorder.pipeline`
and `recorder.review`.

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

from ..index.chunk import SUMMARY_CLOSE, SUMMARY_OPEN, TRANSCRIPT_MARKER
from ..config import VAULT

RECORD_PROMPT_VERSION = "record-v2"
MEETING_BLOCK_LANGUAGE = "slim-meeting"
# Where the plugin writes audio while a recording is in progress, and the ONLY directory whose
# files a server-side path is allowed to consume and unlink. Mirrors STAGING_DIR in
# obsidian/src/recorder/meeting-note.js.
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


_TRANSCRIPT_LINE = re.compile(rf"(?m)^{re.escape(TRANSCRIPT_MARKER)}[ \t]*\r?$")


# The summary's TEXT is group 1: after `## Summary` and its blank lines, up to the line before
# the closing marker. One pattern for the reader and the splice.
_SUMMARY_RE = re.compile(
    rf"^<!-- slim:summary[^\n]*-->\n## Summary\n+([\s\S]*?)\n{re.escape(SUMMARY_CLOSE)}", re.M)


def _transcript_at(scope: str, fm_end: int = 0) -> int:
    """THE transcript locator: where the last line that is exactly `## Transcript` begins in
    `scope`, at or after `fm_end`, or -1.

    One for every reader and writer here. The two-and-a-half copies there were took any
    `\n## Transcript` prefix, so an owner heading `## Transcript notes` below the real one moved
    the transcript, and a block that OPENS with the heading (a note filed with no summary and
    no notes) read as having no transcript at all."""
    found = -1
    for m in _TRANSCRIPT_LINE.finditer(scope, fm_end):   # `^` holds at fm_end only after a newline
        found = m.start()
    return found


def _before_transcript(scope: str, at: int) -> int:
    """The boundary the section writers splice at: the newline before the heading, or the
    heading itself when it opens the scope."""
    return at - 1 if at > 0 else at


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
    at = _transcript_at(scope, fm_end)
    head = scope[fm_end:_before_transcript(scope, at)] if at >= 0 else scope[fm_end:]
    transcript = scope[at + len(TRANSCRIPT_MARKER):] if at >= 0 else ""

    summary = _SUMMARY_RE.search(head)
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
    at = _transcript_at(scope, fm_end)
    if at < 0:
        raise ValueError("recorded note has no transcript section")
    transcript_at = _before_transcript(scope, at)

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
    # boundary and is concatenated, not rendered or normalized. A block that opens with the
    # heading has no newline before it, so one is supplied.
    tail = scope[transcript_at:] if at > 0 else "\n" + scope[at:]
    replaced = scope[:fm_end] + body + tail
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

    Here rather than in `recorder.pipeline`: these are `render_note`'s keys and the scalar-or-list rule
    that shapes them is this module's. A second writer is a second place to get it wrong.

    `asr_model` is written only when the note has none: a note first transcribed by one model
    keeps that provenance, and a resume that produced no model name never blanks it. `speakers_by`
    follows the same rule.
    """
    from ..vault.frontmatter import parse_frontmatter, set_frontmatter

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


_PII_COUNT = re.compile(r"([a-z]+):([0-9]+)", re.ASCII)


def read_pii(text: str) -> tuple[dict[str, int], list[str]]:
    """A note's masking as its frontmatter states it: (class counts, unfinished reasons).

    TOTAL: the frontmatter is theirs to edit, and this runs inside the append's note lock after
    the audio is archived, where an exception orphans the archived copies. An item that is not
    `<known class>:<ASCII digits>` is ignored (`"²".isdigit()` is True and `int("²")` raises).
    A legacy `pii_partial: true` was the password pass."""
    from ..capture import pii
    from ..vault.frontmatter import parse_frontmatter

    fm, _ = parse_frontmatter(text)
    listed = fm.get("pii_masked") or []
    counts: dict[str, int] = {}
    for item in listed if isinstance(listed, list) else [listed]:
        m = _PII_COUNT.fullmatch(str(item).strip())
        if m and m.group(1) in pii.PLACEHOLDER:
            counts[m.group(1)] = counts.get(m.group(1), 0) + int(m.group(2))
    was = fm.get("pii_partial")
    had = set(was) if isinstance(was, list) else {"password"} if was == "true" else set()
    return counts, sorted(had & set(pii.PARTIAL_REASONS))


def merge_pii_frontmatter(text: str, counts: dict[str, int], reasons) -> str:
    """Add a resumed segment's masking to the note's: counts add, and `pii_partial` is the
    union of the note's reasons and the new ones (a legacy `pii_partial: true` was the password
    pass, so it reads as `password`).

    ⚠ `pii_by` is never ADDED here. It says the whole transcript was checked, and a note filed
    before masking existed was not: only the new words were, and `slim mask` finds the note by
    the missing key. Such a note gets its counts and reasons, and nothing at all when the new
    words held nothing. A note that has it is brought to the current version."""
    from ..capture import pii
    from ..vault.frontmatter import parse_frontmatter, set_frontmatter

    fm, _ = parse_frontmatter(text)
    checked = bool(fm.get("pii_by"))
    if not checked and not counts and not reasons:
        return text
    total, had = read_pii(text)
    for cls, n in counts.items():
        total[cls] = total.get(cls, 0) + n
    unfinished = sorted((set(had) | set(reasons)) & set(pii.PARTIAL_REASONS))
    updates: dict[str, object] = {"pii_masked": [f"{c}:{n}" for c, n in sorted(total.items())]}
    if checked:
        updates["pii_by"] = pii.VERSION
    if unfinished:
        updates["pii_partial"] = unfinished
    return set_frontmatter(text, updates)


def set_pii_frontmatter(text: str, counts: dict[str, int], reasons) -> str:
    """The whole note was checked (`slim mask`): counts, the masker's version, and `pii_partial`
    exactly as `reasons` say — REMOVED when none are left, which `set_frontmatter` cannot do.

    Unlike `merge_pii_frontmatter` this adds `pii_by`, because here the whole transcript was
    read. Counts arrive already totalled; nothing is added to what the note says."""
    from ..capture import pii
    from ..vault.frontmatter import set_frontmatter

    updates = pii.frontmatter(counts, reasons)
    text = set_frontmatter(text, updates)
    return text if "pii_partial" in updates else _drop_frontmatter_key(text, "pii_partial")


def _drop_frontmatter_key(text: str, key: str) -> str:
    """`text` without `key` in its frontmatter, block-list lines included; every other byte kept."""
    from ..vault.frontmatter import parse_frontmatter

    _, skip = parse_frontmatter(text)
    lines = text.splitlines(keepends=True)
    head, body = lines[:skip], lines[skip:]
    kept, i = [], 0
    while i < len(head):
        line = head[i]
        i += 1
        if 0 < i - 1 < len(head) - 1 and re.match(rf"^{re.escape(key)}:(\s|$)", line):
            if not line[len(key) + 1:].strip():        # a block list may follow
                while i < len(head) - 1 and (head[i].strip().startswith("- ")
                                             or (head[i][:1].isspace() and head[i].strip())):
                    i += 1
            continue
        kept.append(line)
    return "".join(kept + body)


def _managed_scope(text: str) -> tuple[str, int, int]:
    """(scope, its offset in `text`, where its sections may begin): the meeting block's body,
    or for a note filed before the block existed, the text after its frontmatter."""
    block = split_meeting_block(text)
    if block is not None:
        return block.body, len(block.before) + len(block.opener), 0
    if not text.startswith("---\n"):
        raise ValueError("recorded note has no frontmatter")
    fm_end = text.find("\n---", 4)
    if fm_end < 0:
        raise ValueError("recorded note has unterminated frontmatter")
    return text, 0, fm_end + 4


def transcript_span(text: str) -> tuple[int, int]:
    """Where the transcript's words sit in `text`: just past `## Transcript` to the end of the
    meeting block's body, or of the file for a note filed before the block existed. Found
    through the one locator `read_review_sections` uses."""
    scope, offset, fm_end = _managed_scope(text)
    at = _transcript_at(scope, fm_end)
    if at < 0:
        raise ValueError("recorded note has no transcript section")
    return offset + at + len(TRANSCRIPT_MARKER), offset + len(scope)


def summary_spans(text: str) -> list[tuple[int, int]]:
    """Where the TEXT of every well-formed derived summary block sits in `text` (inside its
    fence, after `## Summary`), in order. The first is what `read_review_sections` returns."""
    scope, offset, fm_end = _managed_scope(text)
    at = _transcript_at(scope, fm_end)
    end = _before_transcript(scope, at) if at >= 0 else len(scope)
    return [(offset + m.start(1), offset + m.end(1)) for m in _SUMMARY_RE.finditer(scope, fm_end, end)]


def summary_openers(text: str) -> int:
    """How many summary openers the note holds anywhere. More than `summary_spans` found means a
    block the splice cannot see (an unclosed fence, one below the transcript): its text would be
    left as it is."""
    return text.count(SUMMARY_OPEN)


def read_transcript(text: str) -> str:
    """The transcript exactly as it sits in the note, from just past `## Transcript` on:
    leading newlines and trailing whitespace included, so a caller that changes a few spans
    and hands it to `replace_transcript` moves no other byte."""
    start, end = transcript_span(text)
    return text[start:end]


def read_summaries(text: str) -> list[str]:
    """Each summary block's text exactly as it sits in the note, unstripped."""
    return [text[a:b] for a, b in summary_spans(text)]


def _longest_backticks(s: str) -> int:
    return max((len(r) for r in re.findall(r"`+", s)), default=0)


def _splice(text: str, span: tuple[int, int], new: str) -> str:
    """`new` in place of `text[span]`, nothing else moved. ⚠ NEVER re-fenced: a longer backtick
    run than the old text had could close the meeting block early, so it is refused instead."""
    a, b = span
    if _longest_backticks(new) > _longest_backticks(text[a:b]):
        raise ValueError("the new text has a longer backtick run than the old")
    return text[:a] + new + text[b:]


def replace_transcript(text: str, transcript: str) -> str:
    """Swap the transcript (as `read_transcript` returns it) and nothing else.

    ⚠ ONLY FOR A CHANGE THE OWNER ASKED FOR (`slim mask`): SLIM never rewrites a transcript on
    its own. A pure splice: every byte outside it — frontmatter, summary, their notes, the
    fence, `block.after` — is sliced from the original."""
    return _splice(text, transcript_span(text), transcript)


def replace_summaries(text: str, summaries: list[str]) -> str:
    """Swap each summary block's text (as `read_summaries` returns them) and nothing else.

    Not `replace_review_sections`, which REBUILDS the head from summary and notes: on a note
    that is not in the canonical shape that drops the owner's text around them."""
    spans = summary_spans(text)
    if len(spans) != len(summaries):
        raise ValueError(f"the note has {len(spans)} summary blocks, not {len(summaries)}")
    for span, summary in reversed(list(zip(spans, summaries))):   # later spans first: offsets hold
        if SUMMARY_CLOSE in summary or SUMMARY_OPEN in summary:
            raise ValueError("a summary cannot carry the summary fence's markers")
        text = _splice(text, span, summary)
    return text


def write_note_text(path: Path, text: str) -> None:
    """The one way a recorded note is written after it exists.

    ⚠ `Path.write_text` TRUNCATES before it writes. A disk-full or a crash mid-write leaves the
    note SHORTER than it was — and what sits at the end of a recorded note is the raw
    transcript. Every derived edit (notes autosave, retry, append) goes through here so an
    interrupted write leaves the previous note intact rather than half a recording.
    """
    from ..vault import notefile
    notefile.atomic_write_text(path, text)


def set_review_status(text: str, status: str) -> str:
    """Persist the recorder's approval state in the note itself.

    Workflow state in the source, not a queue: reopening the file reconstructs it. Older notes
    have no key and read as complete.
    """
    if status not in {"pending", "complete"}:
        raise ValueError(f"invalid recording review status: {status!r}")
    from ..vault.frontmatter import set_frontmatter
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
    fm_end = 0
    if block is None and text.startswith("---\n"):
        fm_end = scope.find("\n---", 4)
        fm_end = fm_end + 4 if fm_end >= 0 else 0
    if _transcript_at(scope, fm_end) < 0:
        raise ValueError("recorded note has no transcript section")
    grown = scope.rstrip() + "\n\n" + addition.strip() + "\n"
    if block is None:
        return grown
    return block.before + wrap_meeting_block(grown) + "\n" + block.after.lstrip("\n")


def archive_audio(src: Path) -> tuple[Path, str]:
    """Archive a staged file content-addressed, returning (path, sha256). Reuses `audio`'s
    hash-verified copier so there is exactly one implementation of 'the audio is safe'."""
    from ..capture import audio

    digest = audio.sha256(src)
    archived = audio.archive_audio(src, digest)
    return archived, digest


def _clean_scalar(value: str) -> str:
    """One frontmatter value, safe to write into a flow list.

    ⚠ Quoting does NOT solve this. SLIM's own reader splits a flow list on bare commas and
    ignores quotes entirely (`frontmatter.parse_frontmatter`: `val[1:-1].split(",")`), so a comma
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
                transcribed_at: datetime | None = None,
                pii: dict[str, object] | None = None) -> str:
    """One recording's frontmatter.

    ⚠ THE KEY ORDER MATCHES inbox's `_render_note`. Two lanes write recordings, and when they
    wrote different vocabularies in different orders two notes of the same kind did not look
    alike in the one place they read them. Nothing in SLIM parses key order; THEY do. Fields
    inbox derives from spoken routing stay absent here, because the recorder has none.
    """
    from ..vault import notefile

    # The title arrives from the plugin. A newline in it would end the `title:` line and leave
    # the rest of the string sitting in the frontmatter block as a stray, unparseable line.
    fm = ["---",
          f'title: "{_clean_scalar(title).replace(chr(34), "")}"',
          f"date: {notefile.local_date(when)}",
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
    for key, value in (pii or {}).items():
        # `pii.frontmatter`: class counts and the masker's version, never a masked value.
        fm.append(f"{key}: {_yaml_list(value) if isinstance(value, list) else value}")
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
                    pii: dict[str, object] | None = None,
                    draft_src: Path | None = None,
                    vault: Path = VAULT) -> Recorded:
    from ..vault import notefile

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
                       transcribed_at=transcribed_at, pii=pii)

    name = f"{notefile.local_date(when)}--{notefile.slug(title)}--{digest[:8]}.md"
    note = vault / dest_dir / name
    if note.exists() and (draft_src is None or note.resolve() != draft_src.resolve()):
        raise ValueError(f"refusing to overwrite an existing note at {note}")
    notefile.atomic_write_text(note, text)

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
