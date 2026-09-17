"""Recording -> transcript -> vault note: the library behind the recorder and the memo sweep.

`process()` transcribes every audio file the Voice Memos sweep dropped in `Inbox/Journal` into
a note in `Journal/` — the phone lane writes JOURNALS, and only journals (2026-09-03). A spoken
opening can still carry one out of the floor (`voicetags.route`); nothing else can. `record.py`
reuses the helpers below — hashing, archiving, slugs, the note renderer — so both lanes write
one vocabulary in one key order.

Hard rules honored here: raw audio is never destroyed (content-addressed under the synced
vault, idempotent by SHA-256); a note is never overwritten, only refused; hash, duration,
dates and ids are deterministic, never inferred; and no summary is written on this path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import enrich, trace, transcribe as tx, voicetags
from .config import DATA_DIR, VAULT

INBOX = {"journal": VAULT / "Inbox" / "Journal"}
AUDIO_ARCHIVE = VAULT / "Attachments" / "Recordings"
TMP_AUDIO_DIR = DATA_DIR / "audio" / "tmp"
MANIFEST = DATA_DIR / "audio" / "manifest.json"


@dataclass
class Processed:
    audio: Path
    note: Path | None
    status: str           # written | skipped (already done) | refused (note exists)
    detail: str = ""
    transcript: tx.Transcript | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _archive_audio(source: Path, digest: str) -> Path:
    """Copy raw source bytes into the synced vault before the Inbox copy may be removed.

    The temporary file and hash verification make a partial copy distinguishable from a
    preserved source. An existing content-addressed path is trusted only after re-hashing it.
    """
    archived = AUDIO_ARCHIVE / f"{digest}{source.suffix.lower()}"
    archived.parent.mkdir(parents=True, exist_ok=True)
    if archived.exists():
        if _sha256(archived) != digest:
            raise RuntimeError(f"audio archive hash mismatch: {archived}")
        return archived

    partial = archived.with_name(f".{archived.name}.part")
    try:
        with source.open("rb") as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        if _sha256(partial) != digest:
            raise RuntimeError(f"audio archive copy failed verification: {source}")
        partial.replace(archived)
    finally:
        partial.unlink(missing_ok=True)
    return archived


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    try:
        partial.write_text(text, encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"[\s_]+", "-", s)[:60] or "recording"


def recorded_at(path: Path) -> datetime:
    """When was this actually recorded? Prefer the container's creation tag (Voice Memos and
    most recorders set it); fall back to the file's mtime. Deterministic, and never guessed by
    a model."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return datetime.fromisoformat(out.replace("Z", "+00:00")).astimezone()
    except (subprocess.CalledProcessError, ValueError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _default_dir() -> str:
    """Where a memo lands when nothing decided a type: the floor. The phone lane records
    journals, so an undecidable drop stays in the most protected place it could have been."""
    return voicetags.folder_for(voicetags.JOURNAL, None)


def local_date(when: datetime) -> str:
    """The LOCAL calendar date of an instant — for filenames and the `date:` field.

    ⚠ NEVER `when.strftime("%Y-%m-%d")` DIRECTLY. `when` is `datetime.now(timezone.utc)`, so a
    bare strftime yields the UTC date and every recording made after 20:00 local is stamped
    TOMORROW — measured 2026-08-27 on three evening lectures that carried the next day's date.
    It also MISSORTS: the date is the filename prefix, so an evening lecture sorts ahead of the
    following morning's. `recorded_at` stays the UTC instant, which is a point in time and
    correct as stored. This is the CALENDAR DAY, which only exists in a timezone."""
    return when.astimezone().strftime("%Y-%m-%d")


def _note_path(when: datetime, title: str, digest: str, folder: str | None = None) -> Path:
    date = local_date(when)
    name = f"{date}--{_slug(title)}--{digest[:8]}.md"
    return VAULT / (folder or _default_dir()) / name


def _render_note(title: str, when: datetime, t: tx.Transcript, digest: str,
                 labels: dict | None = None, routing: dict | None = None) -> str:
    labels = labels or {"type_tag": "", "topics": []}
    routing = routing or {}
    type_from_routing = routing.get("kind") or ""
    note_type = voicetags.TYPE_FRONTMATTER.get(type_from_routing, "journal")
    fm = [
        "---",
        f'title: "{title}"',
        f"date: {local_date(when)}",
        f"type: {note_type}",
    ]
    # LEVEL 1 — the type tag. `detect_type` is the whole classifier, so `type:` and `tags:` are
    # the same word by construction. `voicetags.route` folds a spoken type and a model-inferred
    # one into the same field and records which in `routed_by`; with neither there is no tag to
    # write, so the note says so by omission.
    if type_from_routing:
        type_tags, tagged_by = [type_from_routing], routing.get("routed_by") or "spoken"
    else:
        type_tags, tagged_by = [], "none"
    if type_tags:
        fm.append(f"tags: [{', '.join(type_tags)}]")
    fm.append(f"tagged_by: {tagged_by}")
    # LEVEL 2 — topic tags. These exist for OBSIDIAN, not for retrieval: its tag pane, graph
    # and backlinks read frontmatter. Gated on groundedness — every tag's words appear in the
    # transcript — because an invented subject written into their notes is the one failure an
    # autonomous write cannot have. The SPOKEN subject leads the list when they named one:
    # deterministic first, model second.
    topics = list(labels["topics"])
    subject = routing.get("subject")
    if subject and subject not in topics:
        topics.insert(0, subject)
    if topics:
        fm.append(f"topics: [{', '.join(topics)}]")
    if routing.get("folder"):
        # How this note got where it is, so a misfiled note is always traceable to a word they
        # said rather than to a guess. "default" means they said nothing and nothing moved.
        fm.append(f"routed_by: {routing.get('routed_by') or 'default'}")
    if routing.get("subject_by"):
        # The SUBJECT's provenance, a separate axis from the type's. `spoken` and `topic` are
        # literal matches decided in code; `description` is the one model judgement, so it is
        # the only value worth auditing, and grep finds it.
        fm.append(f"subject_by: {routing['subject_by']}")
    fm += [
        "origin: recorded",
        "source: local-asr",
        f"asr_model: {t.model}",
        # `asr_selected: false` records that these words came from an unvalidated default, so
        # a future re-transcription knows which notes to revisit.
        "asr_selected: false",
        f"audio_sha256: {digest}",
        f"audio_seconds: {t.audio_seconds:.1f}",
        f"recorded_at: {when.isoformat(timespec='seconds')}",
        f"transcribed_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "---",
    ]
    # The `## Transcript` H2 is the same synthetic marker `pull_notion_meetings.py` injects,
    # and `summarize.py` splits on it. One convention, whatever the source.
    return "\n".join(fm) + f"\n\n# {title}\n\n## Transcript\n\n{t.text}\n"


def process(dry_run: bool = False) -> list[Processed]:
    manifest = _load_manifest()
    results: list[Processed] = []

    folder = INBOX["journal"]
    if folder.exists():
        for audio in sorted(folder.iterdir()):
            if audio.suffix.lower() not in tx.AUDIO_SUFFIXES or audio.name.startswith("."):
                continue

            digest = _sha256(audio)
            if digest in manifest:
                results.append(Processed(audio, Path(manifest[digest]["note"]),
                                         "skipped", "already transcribed (same audio)"))
                continue

            when = recorded_at(audio)
            # The sweep names its copies `<stem>--<hash8>` and the note path adds the same
            # digest again, so a trailing hash is stripped from the TITLE or the filename and
            # the H1 both read "…--f70f13c6--f70f13c6" forever.
            fallback_title = re.sub(r"--[0-9a-f]{8}$", "", audio.stem).replace("_", " ").strip()

            if dry_run:
                results.append(Processed(audio, _note_path(when, fallback_title, digest),
                                         "written", "(dry run)"))
                continue

            # 16 kHz mono is what every ASR wants; normalize once so the runtime never
            # has to guess how to resample.
            wav = TMP_AUDIO_DIR / f"{digest}.wav"
            tx.to_wav16k(audio, wav)
            try:
                t = tx.transcribe(wav)
            finally:
                wav.unlink(missing_ok=True)

            # Enrichment: a model-written title and tags, written without confirmation (their
            # call, 2026-07-20 — a timestamp filename they will never rename is worse). A
            # failure returns the deterministic fallback and never blocks the transcript.
            labels, enrich_stats = enrich.enrich(t.text, fallback_title=fallback_title,
                                                  model=enrich.MODEL)
            title = labels["title"] or fallback_title

            # Spoken routing (2026-08-01): what they SAID this is decides where it lands, and
            # what they said it is ABOUT becomes the subject. Silent when they said nothing, so an
            # undetectable recording keeps the folder it would have had. Enrichment ran just
            # above, so its guess is the FALLBACK; spoken still wins inside `route`.
            r = voicetags.route(
                t.text, default_dir=_default_dir(),
                inferred_type=labels["type_tag"], inferred_topics=labels["topics"],
                inferred_project=labels.get("project"))
            routing = r._asdict()
            note = _note_path(when, title, digest, folder=r.folder)
            if note.exists():
                results.append(Processed(audio, note, "refused",
                                         "a note already exists at that path"))
                continue

            # Archive and verify the irreplaceable source before committing derived output or
            # removing the Inbox copy. The archive stays in the synced vault.
            archived = _archive_audio(audio, digest)
            _atomic_write_text(note, _render_note(title, when, t, digest, labels, routing))
            trace.record("enrich", {
                "note": str(note), "audio_sha256": digest,
                "type_tag": labels["type_tag"], "topics": labels["topics"],
                "routed_to": r.folder, "spoken_type": r.kind,
                "subject": r.subject, "routed_by": r.routed_by,
                "subject_by": r.subject_by,
                "title_from_model": bool(labels["title"]),
                "prompt_version": enrich.PROMPT_VERSION,
                "stats": enrich_stats,
            })

            manifest[digest] = {
                "note": str(note),
                "audio": str(archived),
                "asr_model": t.model,
                "audio_seconds": round(t.audio_seconds, 1),
                "transcribed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _atomic_write_text(MANIFEST, json.dumps(manifest, indent=2))
            # The source is already archived and hash-verified in the vault, so the staged
            # copy has nothing left to protect.
            audio.unlink()

            results.append(Processed(audio, note, "written", transcript=t))

    return results
