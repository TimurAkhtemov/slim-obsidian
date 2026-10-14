"""One recording, from staged audio to a filed note: the draft as the notes source,
segment-cached transcription so a resumed recording never pays twice, the append a resume
makes, and the read-only transcript of a paused one."""

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config, copilot_images, llm, record_state, trace

_DRAFT_SUFFIX = ".slim-draft.md"


def _draft_source(*, vault: Path, draft_rel: str, audio_rel: str | list[str],
                  declared_type: str) -> tuple[Path, str]:
    from . import record

    draft = record_state._vault_path(vault, draft_rel)
    if not draft.is_file():
        raise FileNotFoundError(f"no draft at {draft_rel!r}")
    if not draft.name.endswith(_DRAFT_SUFFIX):
        raise ValueError(f"draft must end with {_DRAFT_SUFFIX}")

    rel = draft.relative_to(vault)
    parent = tuple(part.casefold() for part in rel.parts[:-1])
    expected = ("journal",) if declared_type == "journal" else ("capture", "_unfiled")
    if parent != expected:
        wanted = "Journal" if declared_type == "journal" else "Capture/_unfiled"
        raise ValueError(f"a {declared_type or 'non-journal'} draft must be directly under {wanted}")

    base = draft.name[:-len(_DRAFT_SUFFIX)]
    if "--" not in base:
        raise ValueError("draft filename has no recording id")
    recording_id = base.rsplit("--", 1)[1]
    # ⚠ A SEGMENT BELONGS TO THE SAME RECORDING. Staging is `<id>-001.webm`, so the stem
    # carries a segment number the draft's id never has, and comparing them raw made EVERY
    # recording fail with "ids do not match". EVERY segment, not just the first: a request
    # naming two recordings would merge both transcripts and delete both staged files.
    rels = [audio_rel] if isinstance(audio_rel, str) else list(audio_rel)
    if any(record.recording_id_from_staged(rel) != recording_id for rel in rels):
        raise ValueError("audio and draft recording ids do not match")
    source = draft.read_text(encoding="utf-8")
    block = record.split_meeting_block(source)
    if block is not None and not block.before.strip() and not block.body.strip():
        # The empty block is the inline recorder's durable anchor. Notes typed in Obsidian live
        # after it and remain the only human-authored input sent to the filing pipeline.
        source = block.after[1:] if block.after.startswith("\n") else block.after
    return draft, source


_MAX_NOTE_IMAGES = 15


def _extract_note_images(notes_md: str, vault: Path) -> tuple[str, list[str]]:
    """The images embedded in their notes, for the summary (`copilot_images.note_images`).

    Returns (cleaned_notes, base64_images) where cleaned_notes has the embeds stripped (a file
    name in the text is noise to the summarizer) and base64_images is ready for Ollama's message
    ``images`` field.
    """
    embeds = copilot_images.NOTE_EMBED_RE.findall(notes_md)
    if not embeds:
        return notes_md, []
    found, skipped = copilot_images.note_images(notes_md, vault, limit=_MAX_NOTE_IMAGES)
    images = [data for _name, data in found]
    trace.record("record_images", {
        "embeds_found": len(embeds),
        "images_extracted": len(images),
        "skipped": skipped,
        "total_bytes": sum(len(base64.b64decode(data)) for data in images),
    })
    cleaned = copilot_images.NOTE_EMBED_RE.sub("", notes_md)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned, images


def _segment_cache(staged: Path) -> Path:
    """Where one segment's finished transcript waits for the rest of the recording.

    ⚠ THIS IS WHAT MAKES CANCEL AFFORDABLE: without a cache, resuming charges them a second time
    for the forty minutes they cancelled to escape. Deleted with the staged file.
    """
    return staged.with_suffix(staged.suffix + ".transcript.json")


def _join_segments(gots: list[dict], labelled: bool = False) -> str:
    """One transcript from its segments, in order. No seam marker: where they paused is not
    information they want back inside a lecture transcript, and a marker there would be read by
    chunking and retrieval as content.

    ⚠ Speakers are decided per segment, and a one-sided segment comes back unlabelled — right
    for a lecture, wrong beside a labelled call, where it reads as the last speaker still
    talking. So once anything is labelled (a segment here, or the note it is resumed onto),
    a one-sided segment carries its one label."""
    from . import speakers

    labelled = labelled or any(got.get("speakers_by") for got in gots)
    parts = []
    for got in gots:
        text = str(got.get("text") or "").strip()
        solo = got.get("solo_speaker") or ""
        if text and labelled and solo and not got.get("speakers_by"):
            text = speakers.render([(solo, text)])
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _transcribe_segment(staged: Path, transcribe_mod) -> tuple[dict, bool]:
    """One segment's words, from the cache when the cache is still about THIS audio.

    ⚠ BOUND TO THE BYTES, not to the path. Staged names are reused across sessions, and a cache
    trusted by name alone hands back the previous recording's words with no transcription at
    all — the quietest way imaginable to file the wrong transcript.
    """
    from . import audio

    cache = _segment_cache(staged)
    digest = ""
    try:
        digest = audio.sha256(staged)
    except OSError:
        pass
    if cache.is_file():
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if digest and got.get("audio_sha256") == digest:
                return got, True
        except (OSError, json.JSONDecodeError):
            pass                      # a corrupt cache is not a reason to lose the segment
    audio.normalize_container(staged)
    t = transcribe_mod.transcribe(staged)
    got = {"text": t.text, "model": t.model, "audio_seconds": t.audio_seconds,
           "words": t.words, "wall_seconds": t.wall_seconds, "rtf": t.rtf,
           "speakers_by": t.speakers_by, "solo_speaker": t.solo_speaker}
    # After normalize_container, which rewrites the file — hash what was actually read.
    try:
        got["audio_sha256"] = audio.sha256(staged)
    except OSError:
        pass
    try:
        cache.write_text(json.dumps(got), encoding="utf-8")
    except OSError:
        pass                          # the cache is an optimisation, never a requirement
    return got, False


def handle_record(*, audio_rel: str | list[str], notes_md: str | None = None,
                  draft_rel: str | None = None, declared_type: str, title: str,
                  when: datetime | None = None, vault: Path | None = None,
                  on_progress=None, job_id: str = "") -> dict:
    """One finished recording -> one FILED note, plus the card that refines it.

    ⚠ **The note is written before this returns, and filing never depends on a model.** The
    card adjusts a note that is already on disk; it is not a gate. A SUMMARY or TITLE failure
    degrades the note's content rather than costing the recording.

    ⚠ **A TRANSCRIPT FAILURE IS DIFFERENT AND RAISES** (2026-08-23): the transcript IS the
    artifact, and filing a note without one showed a SUCCESS card for a recording that had not
    been transcribed. Because `write_recording` never runs, the audio stays staged in
    `_incoming`, where the plugin's error screen names it and a retry is one button.
    """
    from . import enrich, record, suggest, summarize, transcribe
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    when = when or datetime.now(timezone.utc)
    # A RESUMED recording arrives as several segments sharing one recording id, in the order they
    # recorded them. One segment is the ordinary case and stays a bare string on the wire.
    audio_rels = [audio_rel] if isinstance(audio_rel, str) else list(audio_rel)
    if not audio_rels:
        raise ValueError("audio (a vault-relative path) is required")
    segments = [record_state._vault_path(vault, rel) for rel in audio_rels]
    # ⚠ THEIR title, captured before any fallback overwrites it. A title they typed is TESTIMONY,
    # so `build_card` gets it. A title the MODEL just wrote must never be fed back in: it is
    # derived from the same transcript the card reads, so one bad guess would reinforce the
    # next.
    declared_title = title.strip()
    # Fail before the expensive part. Without this a bad path still costs a transcription
    # attempt and a model call before dying on the missing file at write time.
    for rel, seg in zip(audio_rels, segments):
        if not seg.is_file():
            raise FileNotFoundError(f"no audio at {rel!r}")

    def progress(stage: str, label: str, detail: str = "", **extra) -> None:
        if on_progress is not None:
            on_progress(stage=stage, label=label, detail=detail, **extra)

    # ⚠ CANCEL IS CHECKED BEFORE EVERY EXPENSIVE STEP AND WRITES NOTHING WHEN IT FIRES. The
    # draft note and the staged audio are the only copies of what they said, so a cancelled pass
    # leaves the recording exactly as it found it — they resume into the same draft.
    record_state._check_record_cancel(job_id)

    draft: Path | None = None
    if draft_rel is not None:
        if notes_md is not None:
            raise ValueError("provide exactly one notes source: draft or notes_md")
        draft, notes_md = _draft_source(vault=vault, draft_rel=draft_rel,
                                        audio_rel=audio_rels, declared_type=declared_type)
    elif notes_md is None:
        raise ValueError("provide exactly one notes source: draft or notes_md")

    # ⚠ Before anything reads it. A browser recorder streams its container and never writes a
    # duration, which cost a whole 37-minute meeting its transcript on the first real run —
    # and left Obsidian's own player unable to seek the file. A 0.12 s lossless remux fixes
    # both, and doing it HERE means every reader downstream gets a sane file. `_transcribe_segment`
    # does it per segment, right before reading each one.
    record_state._check_record_cancel(job_id)
    progress("preparing", "Preparing audio", "Making the recording seekable and readable")

    # Keep the Transcript, not just its text: `asr_model` and `audio_seconds` are deterministic
    # facts about how these words were produced, and the note records them the same way the
    # memo lane does. They stay absent when transcription fails rather than being invented.
    asr_model, audio_seconds, transcribed_at, speakers_by = "", None, None, ""
    gots, words, seconds, wall = [], 0, 0.0, 0.0
    try:
        for i, seg in enumerate(segments, start=1):
            record_state._check_record_cancel(job_id)
            of = f" ({i} of {len(segments)})" if len(segments) > 1 else ""
            progress("transcribing", f"Transcribing{of}",
                     "Converting the local audio into exact source text")
            got, cached = _transcribe_segment(seg, transcribe)
            gots.append(got)
            words += got.get("words") or 0
            seconds += got.get("audio_seconds") or 0.0
            wall += 0.0 if cached else (got.get("wall_seconds") or 0.0)
            asr_model = got.get("model") or asr_model
            speakers_by = got.get("speakers_by") or speakers_by
        text = _join_segments(gots)
        audio_seconds = seconds or None
        transcribed_at = datetime.now(timezone.utc)
        progress("transcribed", "Transcript ready", f"Captured {words} words",
                 metrics={"transcript_words": words, "audio_seconds": seconds,
                          "transcription_seconds": wall, "segments": len(segments),
                          "transcription_realtime_factor": round(seconds / wall, 2) if wall else 0})
    except record_state.RecordCancelled:
        raise
    except Exception as exc:                       # noqa: BLE001 - re-raised, see below
        trace.record("record", {"stage": "transcribe", "status": "failed", "error": str(exc)})
        raise ValueError(f"transcription failed: {exc}") from exc

    if not text.strip():
        # Silence is the same outcome by a different route — a muted mic, the wrong input — and
        # it produces exactly the same worthless note. Saying so is the point.
        trace.record("record", {"stage": "transcribe", "status": "empty"})
        raise record_state.NoSpeech("no words were captured — nothing was filed, and your audio "
                                    "and notes are exactly where they were")

    summary = None
    if text.strip():
        try:
            progress("summarizing", "Reasoning about the summary",
                     "The local model is identifying substance, decisions and open questions")
            # The one stage a cancel can interrupt MID-CALL: llm.chat_stream reads the
            # response line by line, so raising from here stops a summary that is already
            # generating instead of waiting out its remaining minutes.
            def activity_fn(kind, chunk):
                record_state._check_record_cancel(job_id)
                progress("summarizing", "Reasoning about the summary",
                         "Live local-model activity", kind=kind, text=chunk)
            activity = activity_fn if on_progress is not None else None
            record_state._check_record_cancel(job_id)
            # ⚠ RECORD_MODEL is the resident (llm.py says why). Their notes ride along as
            # context — what they noticed, how they spell things — never as a filter.
            clean_notes, note_images = _extract_note_images(notes_md or "", vault)
            s = summarize.summarize(text, model=llm.RECORD_MODEL,
                                    num_ctx=llm.RECORD_CTX, think=True,
                                    on_activity=activity, notes=clean_notes,
                                    images=note_images or None)
            # Rendered below, once the card has decided the type: a lecture carries no
            # "Open questions" (summarize.LECTURE_TYPES). ⚠ render(), NOT s.overview — the
            # overview is 2-4 sentences and the SUBSTANCE is in the other fields. Writing only
            # the overview once reduced a 37-minute meeting to two sentences, which reads as a
            # lazy model when the caller was throwing its work away.
            summary = s
            progress("summarized", "Summary drafted", "Structured output is ready",
                     metrics={"summary_model": s.stats.get("model"),
                              "summary_seconds": s.stats.get("duration_s"),
                              "summary_output_tokens": s.stats.get("output_tokens"),
                              "summary_decode_tok_s": s.stats.get("decode_tok_s")})
        except record_state.RecordCancelled:
            # ⚠ NOT swallowed by the summary's catch-all. A cancel is their instruction, not a
            # degraded summary, and continuing on to file the note would ignore it.
            raise
        except Exception as exc:                   # noqa: BLE001 - a summary, never a recording
            trace.record("record", {"stage": "summarize", "status": "failed",
                                    "error": str(exc)})

    if not title.strip() and text.strip():
        # A timestamp fallback names nothing, and the filename embeds it, so the vault fills
        # with rows that all look the same. `enrich` is the titling implementation this
        # project already has, so there is one answer to "what is a good title", not two.
        try:
            progress("titling", "Writing the title", "Naming the note from the transcript")
            # ⚠ RECORD_CTX — it moves with the summarize call above, and the alias exists so
            # the recorder's calls cannot drift apart (llm.py says why). A runner is ALREADY
            # LOADED at that size, so a different one buys nothing and costs a second runner:
            # one untitled recording once asked for 32768 -> 8192 -> 32768, two reloads while
            # the owner waited.
            labels, _ = enrich.enrich(text, model=llm.RECORD_MODEL,
                                      num_ctx=llm.RECORD_CTX, registry=config.registry())
            title = (labels.get("title") or "").strip()
        except Exception as exc:                   # noqa: BLE001 - a title, never a recording
            trace.record("record", {"stage": "title", "status": "failed", "error": str(exc)})
    if not title.strip():
        title = f"Recording {when.strftime('%Y-%m-%d %H%M')}"

    # No empty-transcript branch: it was unreachable once an empty transcript became an error,
    # and its job — keeping a failed journal inside the privacy floor — belongs to `build_card`,
    # which returns `JOURNAL_FOLDER` for a declared journal before it consults the model at all.
    record_state._check_record_cancel(job_id)
    progress("filing", "Choosing filing details", "Comparing the recording with the vault's existing structure")
    card = suggest.build_card(text, declared_type=declared_type, title=declared_title,
                              notes=notes_md, vault=vault, model=llm.RECORD_MODEL)
    summary_md = summarize.render(
        summary, open_questions=card.type_tag not in summarize.LECTURE_TYPES
    ) if summary is not None else ""

    # The last chance: past this line the note is on disk and cancelling would mean deleting
    # a written recording, which this never does.
    record_state._check_record_cancel(job_id)
    progress("saving", "Saving the note", "Archiving audio, writing Markdown and refreshing the index")
    r = record.write_recording(
        audio_src=segments, title=title, when=when, type_tag=card.type_tag, transcript=text,
        notes_md=notes_md, summary_md=summary_md,
        topics=card.topics, dest_dir=card.dest_dir, filed_by_slim=True, asr_model=asr_model,
        speakers_by=speakers_by,
        audio_seconds=audio_seconds, transcribed_at=transcribed_at, draft_src=draft,
        vault=vault)

    # The note is committed, so the per-segment transcript caches have nothing left to save.
    for seg in segments:
        cache = _segment_cache(seg)
        if cache.exists():
            try:
                cache.unlink()
            except OSError:
                pass                  # visible in _incoming, harmless, removable by hand

    record_state._index_now(vault, str(r.note.relative_to(vault)))
    trace.record("record", {"note": str(r.note), "dest": card.dest_dir,
                            "type": card.type_tag, "has_summary": bool(summary_md),
                            "speakers_by": speakers_by})
    progress("complete", "AI review ready", "The note is filed and ready for approval")
    return {
        "note": str(r.note.relative_to(vault)),
        "audio": str(r.audio.relative_to(vault)),
        "title": title,
        "summary": summary_md,
        "card": {"dest_dir": card.dest_dir, "confident_depth": card.confident_depth,
                 "type_tag": card.type_tag,
                 "topics": card.topics, **_card_lists(vault)},
    }


def _card_lists(vault: Path) -> dict:
    """What the card browses and badges: every folder a note may go to, every tag in use.
    Also served alone, for a pending card reopened after a reload (`/api/record/folders`)."""
    from . import suggest
    return {"tag_counts": suggest.tag_counts(vault),
            "folders": [{"path": f.path, "count": f.count} for f in suggest.folder_tree(vault)]}


def _record_args(data: dict) -> dict:
    """Normalize the two-version HTTP contract without creating two sources of truth."""
    has_draft = "draft" in data
    has_notes = "notes_md" in data
    if has_draft == has_notes:
        raise ValueError("provide exactly one notes source: draft or notes_md")
    # One segment stays a bare string on the wire; a resumed recording sends the list, in the
    # order they recorded it. Both shapes are accepted so an older plugin keeps working.
    raw_audio = data.get("audio")
    if isinstance(raw_audio, list):
        audio_rel = [str(a).strip() for a in raw_audio if str(a).strip()]
    else:
        audio_rel = str(raw_audio or "").strip()
    if not audio_rel:
        raise ValueError("audio (a vault-relative path) is required")
    draft_rel = str(data.get("draft") or "").strip() if has_draft else None
    if has_draft and not draft_rel:
        raise ValueError("draft (a vault-relative path) is required")
    return {
        "audio_rel": audio_rel,
        "draft_rel": draft_rel,
        "notes_md": None if has_draft else str(data.get("notes_md") or ""),
        "declared_type": str(data.get("type") or "").strip(),
        "title": str(data.get("title") or "").strip(),
    }


def handle_record_append(*, note: str, audio_rel: str | list[str], job_id: str = "",
                         on_progress=None, vault: Path | None = None) -> dict:
    """Resume a recording whose note already exists: transcribe the new segments and APPEND.

    Not a second recording and not a rewrite: the existing transcript passes through byte for
    byte, the new words go on the end, and the new audio joins `audio:` and `audio_sha256:`.

    ⚠ THE SUMMARY IS LEFT ALONE (their call, 2026-08-26) until they press Retry summary: nothing
    costing a minute of model time runs unasked, and nothing overwrites a summary they may have
    edited by hand. ⚠ Silence is refused exactly as the recording path refuses it.
    """
    from . import record as record_mod, transcribe
    from .frontmatter import parse_frontmatter
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    src = record_state._vault_path(vault, note)
    text = src.read_text(encoding="utf-8")
    rels = [audio_rel] if isinstance(audio_rel, str) else list(audio_rel)
    segments = [record_state._vault_path(vault, rel) for rel in rels]
    staging = record_state._vault_path(vault, record_mod.STAGING_DIR)
    for rel, seg in zip(rels, segments):
        if not seg.is_file():
            raise FileNotFoundError(f"no audio at {rel!r}")
        # ⚠ STAGED ONLY. This function unlinks its inputs once they are archived, and
        # `archive_audio` returns the SAME path when handed a file that is already the
        # content-addressed archive — so a note could end up naming a recording this request
        # had just deleted. Nothing outside _incoming is ever an input here.
        if seg.parent.resolve() != staging.resolve():
            raise ValueError(f"{rel!r} is not staged audio — append only takes files in "
                             f"{record_mod.STAGING_DIR}")

    def progress(stage: str, label: str, detail: str = "", **extra) -> None:
        if on_progress is not None:
            on_progress(stage=stage, label=label, detail=detail, **extra)

    gots, seconds, asr_model, speakers_by = [], 0.0, "", ""
    for i, seg in enumerate(segments, start=1):
        record_state._check_record_cancel(job_id)
        of = f" ({i} of {len(segments)})" if len(segments) > 1 else ""
        progress("transcribing", f"Transcribing{of}", "Adding the new words to this note")
        got, _cached = _transcribe_segment(seg, transcribe)
        gots.append(got)
        seconds += got.get("audio_seconds") or 0.0
        asr_model = got.get("model") or asr_model
        speakers_by = got.get("speakers_by") or speakers_by
    fm, _ = parse_frontmatter(text)
    addition = _join_segments(gots, labelled=bool(fm.get("speakers_by")))
    if not addition:
        raise record_state.NoSpeech("no words were captured — nothing was added, and your audio "
                                    "is exactly where it was")

    # ⚠ PAST THE LOOP, AND BEFORE ARCHIVING. A resume normally has ONE segment, so a check
    # only at the top of the loop ran once and never again — they were told "nothing is written"
    # and the note grew anyway. `archive_audio` writes durable files, so a cancel landing after
    # it left orphan copies. This is the last boundary; nothing durable happens before it.
    record_state._check_record_cancel(job_id)
    with record_state._note_lock(vault, note):
        text = record_state._reread_for_write(src, note)

        archived_rels, digests = [], []
        for seg in segments:
            one, one_digest = record_mod.archive_audio(seg)
            try:
                archived_rels.append(str(one.relative_to(vault)))
            except ValueError:
                archived_rels.append(one.name)
            digests.append(one_digest)

        # ⚠ record.py is the sole writer of a recorded note's frontmatter (CLAUDE.md), and
        # that includes `asr_model`: it decides whether an existing model name is kept.
        text = record_mod.append_audio_frontmatter(text, rels=archived_rels, digests=digests,
                                                   seconds=seconds, asr_model=asr_model,
                                                   speakers_by=speakers_by)
        record_mod.write_note_text(src, record_mod.append_transcript(text, addition))

    archived_paths = {record_state._vault_path(vault, rel).resolve() for rel in archived_rels}
    for seg in segments:
        cache = _segment_cache(seg)
        # Belt as well as braces: never unlink a file that IS the archive it was copied to.
        drop = [cache] + ([seg] if seg.resolve() not in archived_paths else [])
        for path in drop:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
    record_state._index_now(vault, str(src.relative_to(vault)), sweep=False)
    trace.record("record", {"stage": "append", "note": str(src), "segments": len(segments),
                            "words": len(addition.split())})
    return {"note": str(src.relative_to(vault)), "words": len(addition.split()),
            "audio_seconds": seconds}


def handle_record_transcript(*, audio_rel: str | list[str],
                             vault: Path | None = None) -> dict:
    """What has been transcribed of a paused recording, from the caches alone.

    ⚠ READS, NEVER TRANSCRIBES. This serves a screen meant to be instant. A segment with no
    cache yet is COUNTED, not invented.
    """
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    rels = [audio_rel] if isinstance(audio_rel, str) else list(audio_rel)
    gots, pending = [], 0
    for rel in rels:
        cache = _segment_cache(record_state._vault_path(vault, rel))
        if not cache.is_file():
            pending += 1
            continue
        try:
            gots.append(json.loads(cache.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pending += 1
    text = _join_segments(gots)
    return {"text": text, "pending": pending, "segments": len(rels),
            "words": len(text.split())}
