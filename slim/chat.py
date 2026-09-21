"""The Obsidian plugin's local API server — `slim chat`, 127.0.0.1:7546.

Eighteen routes, called by `obsidian/main.js` (and `GET /api/health` also by the stale-server
hook): the health guard, ten `/api/record*` routes for the recorder and seven `/api/copilot*`
routes for the open-note sidebar.

Loopback-only and Host/Origin-gated because this server moves and overwrites vault files. Two
properties are load-bearing — do not "simplify" either away. `make_server` refuses any bind
address that is not loopback (`""` binds every interface and is refused too). And every
request whose `Host` is not localhost is refused before routing, because DNS rebinding is the
one way a remote page reaches a loopback server. Mutating routes additionally refuse a foreign
`Origin` and any non-JSON body (`_refuse_cross_site_write`): a blind cross-site POST cannot
read the response, but a write does not need reading.

The recording pipeline (`handle_record` and friends) is the bulk of the file: one bounded FIFO
for the heavy model jobs, a job-local progress registry the plugin polls, cooperative cancel,
and segment-cached transcription so a resumed recording never pays twice.
"""

import base64
import binascii
import ipaddress
import json
import os
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import (config, copilot as copilot_mod, copilot_images, db,
               llm, threads as threads_mod,
               trace)

DEFAULT_PORT = 7546          # "slim" on a phone keypad
MAX_QUESTION_CHARS = 4000
MAX_BODY_BYTES = 64_000
# A thread save carries the WHOLE conversation, so it legitimately dwarfs the default body
# cap. Capped at the store's own limit plus envelope headroom: a 64 kB HTTP cap silently
# 400'd saves the store would accept, the turn never reached disk, and the whole chat
# vanished on the next reload.
MAX_THREAD_BODY_BYTES = threads_mod.MAX_THREAD_BYTES + 100_000
MAX_COPILOT_STREAM_BODY_BYTES = (
    MAX_THREAD_BODY_BYTES
    + copilot_images.MAX_IMAGES_PER_TURN * ((copilot_images.MAX_IMAGE_BYTES + 2) // 3 * 4 + 4096)
)
# The recorder body carries notes they TYPED during a recording, plus a path — never the audio,
# which the plugin writes into the vault itself precisely because a recording dwarfs any
# sane HTTP cap. An hour of typed notes still fits comfortably here.
MAX_RECORD_BODY_BYTES = 1_000_000

# Recorder progress is local, short-lived diagnostic state. The recording request remains the
# source of truth; this registry only lets the Obsidian object observe a long synchronous job.
_RECORD_PROGRESS_LOCK = threading.Lock()
_RECORD_PROGRESS: dict[str, dict] = {}
_RECORD_PROGRESS_LIMIT = 32


class RecordQueueFull(RuntimeError):
    """The bounded local recording pipeline has no admission slot left."""


class RecordingPipelineQueue:
    """A bounded FIFO around the heavyweight local recording pipeline.

    Request threads stay alive independently, but Parakeet, Ollama, filing and indexing run one
    recording at a time. The deque includes the active job, so ``limit`` bounds admitted work.
    """

    def __init__(self, *, limit: int = _RECORD_PROGRESS_LIMIT):
        if limit < 1:
            raise ValueError("recording pipeline limit must be positive")
        self.limit = limit
        self._condition = threading.Condition()
        self._jobs: deque[object] = deque()

    @property
    def size(self) -> int:
        with self._condition:
            return len(self._jobs)

    @contextmanager
    def slot(self, job_id: str, *, on_queued=None):
        token = object()
        with self._condition:
            if len(self._jobs) >= self.limit:
                raise RecordQueueFull(
                    f"local recording pipeline already has {self.limit} admitted jobs")
            position = len(self._jobs)
            self._jobs.append(token)
        try:
            if position and on_queued is not None:
                on_queued(position)
            with self._condition:
                while self._jobs[0] is not token:
                    self._condition.wait()
            yield
        finally:
            with self._condition:
                try:
                    self._jobs.remove(token)
                except ValueError:
                    pass
                self._condition.notify_all()


_RECORD_PIPELINE = RecordingPipelineQueue()


@contextmanager
def _record_pipeline_slot(job_id: str):
    """Admit one heavy recorder job and publish a job-local queued state while it waits."""
    def queued(position: int) -> None:
        noun = "recording" if position == 1 else "recordings"
        _record_progress_update(
            job_id,
            stage="queued",
            label=f"Queued behind {position} {noun}",
            detail="Waiting for the local recording pipeline",
        )

    with _RECORD_PIPELINE.slot(job_id, on_queued=queued):
        yield


def _record_progress_start(job_id: str) -> None:
    if not job_id:
        return
    with _RECORD_PROGRESS_LOCK:
        if len(_RECORD_PROGRESS) >= _RECORD_PROGRESS_LIMIT:
            oldest = min(_RECORD_PROGRESS, key=lambda key: _RECORD_PROGRESS[key]["updated_at"])
            _RECORD_PROGRESS.pop(oldest, None)
        _RECORD_PROGRESS[job_id] = {
            "job_id": job_id, "stage": "queued", "label": "Queued",
            "detail": "Waiting for the local pipeline", "steps": [],
            "metrics": {}, "done": False,
            "error": "", "updated_at": time.time(),
        }


def _record_progress_update(job_id: str, *, stage: str, label: str, detail: str = "",
                            kind: str = "", text: str = "", metrics: dict | None = None,
                            done: bool = False, error: str = "") -> None:
    if not job_id:
        return
    with _RECORD_PROGRESS_LOCK:
        state = _RECORD_PROGRESS.get(job_id)
        if state is None:
            return
        if stage != state["stage"]:
            if state["steps"]:
                state["steps"][-1]["status"] = "done"
            state["steps"].append({"stage": stage, "label": label, "status": "active"})
        state.update({"stage": stage, "label": label, "detail": detail,
                      "done": done, "error": error, "updated_at": time.time()})
        if kind in ("thinking", "draft") and text:
            state["metrics"]["model_activity_events"] = (
                state["metrics"].get("model_activity_events", 0) + 1)
        if metrics:
            state["metrics"].update(metrics)
        if done and state["steps"]:
            state["steps"][-1]["status"] = "error" if error else "done"


class NoSpeech(Exception):
    """The recording had no words in it. NOT an error, and the distinction is theirs (2026-08-26).

    Nothing is written either way, but silence must not reach them as a red failure screen
    asking whether to delete their audio. The caller tells this apart from a broken ASR.
    """


class RecordCancelled(Exception):
    """They asked for this pass to stop. Not an error: nothing is wrong and nothing is lost."""


def request_record_cancel(job_id: str) -> bool:
    """Flag a running pass. Returns False if there is no such job to flag.

    ⚠ COOPERATIVE, and it has to be. The summary is a STREAMING call, so the flag raises out of
    it mid-generation; every other stage is a blocking call, so the flag lands at the next stage
    boundary. Nothing here kills a thread.
    """
    if not job_id:
        return False
    with _RECORD_PROGRESS_LOCK:
        state = _RECORD_PROGRESS.get(job_id)
        if state is None:
            return False
        state["cancelled"] = True
        state.update({"stage": "cancelling", "label": "Cancelling",
                      "detail": "Stopping at the end of the current step",
                      "updated_at": time.time()})
        return True


def _record_cancelled(job_id: str) -> bool:
    if not job_id:
        return False
    with _RECORD_PROGRESS_LOCK:
        state = _RECORD_PROGRESS.get(job_id)
        return bool(state and state.get("cancelled"))


def _check_record_cancel(job_id: str) -> None:
    if _record_cancelled(job_id):
        raise RecordCancelled("cancelled")


def record_progress(job_id: str) -> dict | None:
    with _RECORD_PROGRESS_LOCK:
        state = _RECORD_PROGRESS.get(job_id)
        if state is None:
            return None
        return {**state, "steps": [dict(step) for step in state["steps"]],
                "metrics": dict(state["metrics"])}


# The copilot's history window: what the plugin sends back with each question, cleaned by
# `clean_history` before it reaches the model. Bounded so a long sidebar thread cannot crowd
# the evidence out of the resident's context.
MAX_HISTORY_TURNS = 12
MAX_HISTORY_TURN_CHARS = 1000


class ChatError(Exception):
    """A refusal with an authored message (bad bind address). Reaches the user verbatim."""


def validate_bind_host(host: str) -> None:
    """Refuse any bind that is not loopback, before a socket exists."""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ChatError(
        f"refusing to bind {host!r} — the chat surface can display Journal/ content and must "
        f"only ever listen on loopback (127.0.0.1 or localhost)")


def _host_header_allowed(value: str | None) -> bool:
    """True only for a localhost/loopback Host header (the DNS-rebinding gate)."""
    if not value:
        return False
    host = value.strip()
    if host.startswith("["):                      # bracketed IPv6, e.g. [::1]:7546
        end = host.find("]")
        if end < 0:
            return False
        name = host[1:end]
    elif host.count(":") == 1:                    # name:port
        name = host.rsplit(":", 1)[0]
    else:
        name = host
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False



def _origin_allowed(value: str) -> bool:
    """True only for a local page origin (the cross-site-WRITE gate, v1.3). Browsers attach
    Origin to every cross-site POST; a present non-local value is another site's page
    firing at this server, and `null` (sandboxed/file: contexts) is refused with it."""
    parsed = urlparse(value)
    return parsed.scheme == "http" and _host_header_allowed(parsed.netloc)


def clean_history(raw) -> list[dict]:
    """Sanitize the page-supplied transcript: keep only well-formed recent turns, bounded.
    Bad history is DROPPED, never a 400 — a broken transcript must not brick the chat."""
    if not isinstance(raw, list):
        return []
    turns = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role, text = item.get("role"), item.get("text")
        if role not in ("you", "slim") or not isinstance(text, str) or not text.strip():
            continue
        turn = {"role": role, "text": text.strip()[:MAX_HISTORY_TURN_CHARS]}
        if role == "you" and item.get("attachments"):
            try:
                turn["attachments"] = copilot_images.clean_refs(item.get("attachments"))
            except copilot_images.ImageError:
                pass                         # malformed history never bricks a new question
        turns.append(turn)
    return turns[-MAX_HISTORY_TURNS:]


# --------------------------------------------------------------------------
# the recorder
# --------------------------------------------------------------------------

def _vault_path(vault: Path, rel: str) -> Path:
    """Resolve a caller-supplied vault-relative path, refusing anything that escapes.

    `rel` arrives over HTTP: without this a caller could name `../../etc/passwd`. Loopback-only
    is not an argument against checking — the browser is the untrusted party, not the network.

    ⚠ Validates with `resolve()` but RETURNS THE UNRESOLVED JOIN, so every path here stays in
    one coordinate system. Returning the resolved path breaks `relative_to(vault)` wherever the
    vault sits under a symlink: on macOS `/var` is a symlink to `/private/var`, which made
    `handle_record_apply` move a note correctly and then report a 400. Tests hide it — pytest's
    `tmp_path` is already resolved, so the two spellings coincide there.
    """
    target = vault / rel
    if not target.resolve().is_relative_to(vault.resolve()):
        raise ValueError(f"path escapes the vault: {rel!r}")
    return target


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


def _index_now(vault: Path, note_rel: str, *, sweep: bool = True,
               moved_from: str | None = None) -> dict:
    """Make THIS note reachable by everything that reads the INDEX rather than the vault.

    Filing makes a note findable in Obsidian immediately; it does NOT reach the two readers of
    SQLite, the copilot and `reflect`. ⚠ The second fails silently: `reflect._journals()`
    selects `type = 'journal'` FROM THE DATABASE, so a journal they record would be skipped by
    the nightly synthesis with no error anywhere.

    One note in the request, the vault in the background (`_schedule_sweep`) — a full pass here
    once embedded 1,287 fragments for a ~50-fragment note inside the pause after the summary.
    ⚠ `sweep=False` for edits to a note that ALREADY EXISTS: on every blur of the notes field
    that pass is a full hash-and-embed of the whole vault, and typing in a filed note changes
    no other note.

    ⚠ Called after record AND after apply, so a note is indexed twice with both its content and
    its path changed in between — the exact sequence the hash rename rule cannot follow. The
    note is NOT lineage-free by then: the copilot can be opened on the draft, and its threads
    are keyed by source id (a filed note was re-minted and its chat orphaned, 2026-09-22).
    Apply therefore passes `moved_from`, and ingest keeps the id.

    Never raises: a note filed but not indexed is recoverable by the sweep. One that failed to
    file is not.
    """
    from . import db, embed as embed_mod, ingest as ingest_mod

    try:
        con = db.connect()
        try:
            source = ingest_mod.ingest_note(con, vault, note_rel, moved_from=moved_from)
            embedded = embed_mod.embed_source(con, source["id"])
        finally:
            con.close()
        out = {"note": note_rel, "source_id": source["id"], "embed": embedded}
        trace.record("record", {"stage": "index", "status": "ok", **out})
    except Exception as exc:                       # noqa: BLE001 - see the docstring
        trace.record("record", {"stage": "index", "status": "failed", "error": str(exc)})
        out = {"error": str(exc)}
    if sweep:
        _schedule_sweep(vault)
    return out


# One sweep at a time: record and apply arrive seconds apart, and two full passes writing the
# same tables would contend for SQLite's write lock (5 s default busy timeout) for nothing.
_SWEEP_LOCK = threading.Lock()


def _serialized(fn, *args):
    with _SWEEP_LOCK:
        return fn(*args)


def _sweep_vault(vault: Path) -> dict:
    """The full incremental pass — hash every note, index what changed, embed what has no
    vector — off the request path. WAL lets the sidebar keep reading while it runs; a sidebar
    WRITE (ingest_note on the note they just opened) waits on the lock like any other writer."""
    from . import db, embed as embed_mod, ingest as ingest_mod

    def pass_once(vault):
        try:
            con = db.connect()
            try:
                stats = ingest_mod.ingest(con, vault=vault)
                embedded = embed_mod.embed_missing(con)
            finally:
                con.close()
            out = {"ingest": stats, "embed": embedded}
            trace.record("record", {"stage": "sweep", "status": "ok", **out})
            return out
        except Exception as exc:                   # noqa: BLE001 - background; never raises
            trace.record("record", {"stage": "sweep", "status": "failed", "error": str(exc)})
            return {"error": str(exc)}

    return _serialized(pass_once, vault)


def _schedule_sweep(vault: Path) -> threading.Thread:
    # Daemon: a sweep cut off by shutdown is redone by the next one — ingest commits at the
    # end of a pass and embed per batch, so nothing half-written survives.
    thread = threading.Thread(target=lambda: _sweep_vault(vault), name="slim-index-sweep",
                              daemon=True)
    thread.start()
    return thread


_DRAFT_SUFFIX = ".slim-draft.md"


def _draft_source(*, vault: Path, draft_rel: str, audio_rel: str | list[str],
                  declared_type: str) -> tuple[Path, str]:
    from . import record

    draft = _vault_path(vault, draft_rel)
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


_NOTE_IMAGE_RE = re.compile(
    r'!\[\[(Attachments/Recorder/[^\]\n]+\.(?:png|jpe?g|webp))\]\]', re.I)
# The plugin emits bare ![[path]] embeds, never ![[path|size]]; if that changes, widen this.
_MAX_NOTE_IMAGES = 15
_MAX_IMAGE_FILE_BYTES = 10_000_000
_IMAGE_MAX_EDGE = 1600


def _extract_note_images(notes_md: str, vault: Path) -> tuple[str, list[str]]:
    """Parse ![[Attachments/Recorder/...]] embeds, load and resize the images.

    Returns (cleaned_notes, base64_images) where cleaned_notes has the embed lines stripped
    and base64_images is ready for Ollama's message ``images`` field.
    """
    matches = _NOTE_IMAGE_RE.findall(notes_md)
    if not matches:
        return notes_md, []

    from io import BytesIO
    from PIL import Image

    images: list[str] = []
    skipped: list[dict] = []
    total_bytes = 0

    for rel in matches[:_MAX_NOTE_IMAGES]:
        try:
            path = _vault_path(vault, rel)
            if not path.is_file():
                skipped.append({"path": rel, "reason": "missing"})
                continue
            raw_size = path.stat().st_size
            if raw_size > _MAX_IMAGE_FILE_BYTES:
                skipped.append({"path": rel, "reason": f"too large ({raw_size} bytes)"})
                continue
            img = Image.open(path)
            img.load()
            has_alpha = img.mode in ("RGBA", "LA", "PA")
            w, h = img.size
            if max(w, h) > _IMAGE_MAX_EDGE:
                scale = _IMAGE_MAX_EDGE / max(w, h)
                img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            if hasattr(img, "_getexif") and img._getexif():
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            buf = BytesIO()
            if has_alpha:
                img.save(buf, format="PNG", optimize=True)
            else:
                img = img.convert("RGB")
                img.save(buf, format="JPEG", quality=85)
            data = base64.b64encode(buf.getvalue()).decode("ascii")
            images.append(data)
            total_bytes += len(buf.getvalue())
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": rel, "reason": str(exc)})

    if matches:
        trace.record("record_images", {
            "embeds_found": len(matches),
            "images_extracted": len(images),
            "skipped": skipped,
            "total_bytes": total_bytes,
        })

    cleaned = _NOTE_IMAGE_RE.sub("", notes_md)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned, images


def _segment_cache(staged: Path) -> Path:
    """Where one segment's finished transcript waits for the rest of the recording.

    ⚠ THIS IS WHAT MAKES CANCEL AFFORDABLE: without a cache, resuming charges them a second time
    for the forty minutes they cancelled to escape. Deleted with the staged file.
    """
    return staged.with_suffix(staged.suffix + ".transcript.json")


def _transcribe_segment(staged: Path, transcribe_mod) -> tuple[dict, bool]:
    """One segment's words, from the cache when the cache is still about THIS audio.

    ⚠ BOUND TO THE BYTES, not to the path. Staged names are reused across sessions, and a cache
    trusted by name alone hands back the previous recording's words with no transcription at
    all — the quietest way imaginable to file the wrong transcript.
    """
    from . import inbox

    cache = _segment_cache(staged)
    digest = ""
    try:
        digest = inbox._sha256(staged)
    except OSError:
        pass
    if cache.is_file():
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if digest and got.get("audio_sha256") == digest:
                return got, True
        except (OSError, json.JSONDecodeError):
            pass                      # a corrupt cache is not a reason to lose the segment
    transcribe_mod.normalize_container(staged)
    t = transcribe_mod.transcribe(staged)
    got = {"text": t.text, "model": t.model, "audio_seconds": t.audio_seconds,
           "words": t.words, "wall_seconds": t.wall_seconds, "rtf": t.rtf,
           "speakers_by": t.speakers_by}
    # After normalize_container, which rewrites the file — hash what was actually read.
    try:
        got["audio_sha256"] = inbox._sha256(staged)
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
    segments = [_vault_path(vault, rel) for rel in audio_rels]
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
    _check_record_cancel(job_id)

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
    _check_record_cancel(job_id)
    progress("preparing", "Preparing audio", "Making the recording seekable and readable")

    # Keep the Transcript, not just its text: `asr_model` and `audio_seconds` are deterministic
    # facts about how these words were produced, and the note records them the same way the
    # memo lane does. They stay absent when transcription fails rather than being invented.
    asr_model, audio_seconds, transcribed_at, speakers_by = "", None, None, ""
    parts, words, seconds, wall = [], 0, 0.0, 0.0
    try:
        for i, seg in enumerate(segments, start=1):
            _check_record_cancel(job_id)
            of = f" ({i} of {len(segments)})" if len(segments) > 1 else ""
            progress("transcribing", f"Transcribing{of}",
                     "Converting the local audio into exact source text")
            got, cached = _transcribe_segment(seg, transcribe)
            parts.append(got["text"])
            words += got.get("words") or 0
            seconds += got.get("audio_seconds") or 0.0
            wall += 0.0 if cached else (got.get("wall_seconds") or 0.0)
            asr_model = got.get("model") or asr_model
            speakers_by = got.get("speakers_by") or speakers_by
        # No seam marker: where they paused is not information they want back inside a lecture
        # transcript, and a marker there would be read by chunking and retrieval as content.
        text = "\n\n".join(part.strip() for part in parts if part.strip())
        audio_seconds = seconds or None
        transcribed_at = datetime.now(timezone.utc)
        progress("transcribed", "Transcript ready", f"Captured {words} words",
                 metrics={"transcript_words": words, "audio_seconds": seconds,
                          "transcription_seconds": wall, "segments": len(segments),
                          "transcription_realtime_factor": round(seconds / wall, 2) if wall else 0})
    except RecordCancelled:
        raise
    except Exception as exc:                       # noqa: BLE001 - re-raised, see below
        trace.record("record", {"stage": "transcribe", "status": "failed", "error": str(exc)})
        raise ValueError(f"transcription failed: {exc}") from exc

    if not text.strip():
        # Silence is the same outcome by a different route — a muted mic, the wrong input — and
        # it produces exactly the same worthless note. Saying so is the point.
        trace.record("record", {"stage": "transcribe", "status": "empty"})
        raise NoSpeech("no words were captured — nothing was filed, and your audio and notes "
                       "are exactly where they were")

    summary = None
    if text.strip():
        try:
            progress("summarizing", "Reasoning about the summary",
                     "The local model is identifying substance, decisions and open questions")
            # The one stage a cancel can interrupt MID-CALL: llm.chat_stream reads the
            # response line by line, so raising from here stops a summary that is already
            # generating instead of waiting out its remaining minutes.
            def activity_fn(kind, chunk):
                _check_record_cancel(job_id)
                progress("summarizing", "Reasoning about the summary",
                         "Live local-model activity", kind=kind, text=chunk)
            activity = activity_fn if on_progress is not None else None
            _check_record_cancel(job_id)
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
        except RecordCancelled:
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
    _check_record_cancel(job_id)
    progress("filing", "Choosing filing details", "Comparing the recording with the vault's existing structure")
    card = suggest.build_card(text, declared_type=declared_type, title=declared_title,
                              notes=notes_md, vault=vault, model=llm.RECORD_MODEL)
    summary_md = summarize.render(
        summary, open_questions=card.type_tag not in summarize.LECTURE_TYPES
    ) if summary is not None else ""

    # The last chance: past this line the note is on disk and cancelling would mean deleting
    # a written recording, which this never does.
    _check_record_cancel(job_id)
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

    _index_now(vault, str(r.note.relative_to(vault)))
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
                 "topics": card.topics,
                 "tag_counts": suggest.tag_counts(vault),
                 "folders": [{"path": f.path, "count": f.count}
                             for f in suggest.folder_tree(vault)]},
    }


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
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    src = _vault_path(vault, note)
    text = src.read_text(encoding="utf-8")
    rels = [audio_rel] if isinstance(audio_rel, str) else list(audio_rel)
    segments = [_vault_path(vault, rel) for rel in rels]
    staging = _vault_path(vault, record_mod.STAGING_DIR)
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

    parts, seconds, asr_model, speakers_by = [], 0.0, "", ""
    for i, seg in enumerate(segments, start=1):
        _check_record_cancel(job_id)
        of = f" ({i} of {len(segments)})" if len(segments) > 1 else ""
        progress("transcribing", f"Transcribing{of}", "Adding the new words to this note")
        got, _cached = _transcribe_segment(seg, transcribe)
        parts.append(got["text"])
        seconds += got.get("audio_seconds") or 0.0
        asr_model = got.get("model") or asr_model
        speakers_by = got.get("speakers_by") or speakers_by
    addition = "\n\n".join(part.strip() for part in parts if part.strip())
    if not addition:
        raise NoSpeech("no words were captured — nothing was added, and your audio is exactly "
                       "where it was")

    # ⚠ PAST THE LOOP, AND BEFORE ARCHIVING. A resume normally has ONE segment, so a check
    # only at the top of the loop ran once and never again — they were told "nothing is written"
    # and the note grew anyway. `archive_audio` writes durable files, so a cancel landing after
    # it left orphan copies. This is the last boundary; nothing durable happens before it.
    _check_record_cancel(job_id)
    with _note_lock(vault, note):
        text = _reread_for_write(src, note)

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

    archived_paths = {_vault_path(vault, rel).resolve() for rel in archived_rels}
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
    _index_now(vault, str(src.relative_to(vault)), sweep=False)
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
    parts, pending = [], 0
    for rel in rels:
        cache = _segment_cache(_vault_path(vault, rel))
        if not cache.is_file():
            pending += 1
            continue
        try:
            parts.append(str(json.loads(cache.read_text(encoding="utf-8")).get("text") or ""))
        except (OSError, json.JSONDecodeError):
            pending += 1
    text = "\n\n".join(part.strip() for part in parts if part.strip())
    return {"text": text, "pending": pending, "segments": len(rels),
            "words": len(text.split())}


# ⚠ ONE WRITER AT A TIME, PER NOTE. `ThreadingHTTPServer` runs handlers in parallel and five
# endpoints write the same file. Re-reading before the write does NOT close two writers
# interleaving between one another's read and write, which is how a notes save silently
# reverted a finished append; the lock is held across read AND write, so the pair is atomic.
# Keyed on the resolved path, so `apply` moving a note releases the old key naturally.
_NOTE_LOCKS_GUARD = threading.Lock()
_NOTE_LOCKS: dict[str, threading.RLock] = {}


def _note_lock(vault: Path, note: str) -> threading.RLock:
    key = str(_vault_path(vault, note).resolve())
    with _NOTE_LOCKS_GUARD:
        lock = _NOTE_LOCKS.get(key)
        if lock is None:
            if len(_NOTE_LOCKS) > 256:               # a recorder touches a handful of notes
                _NOTE_LOCKS.clear()
            lock = _NOTE_LOCKS[key] = threading.RLock()
        return lock


def _reread_for_write(src: Path, note: str) -> str:
    """The note as it stands RIGHT NOW, for a job that has been thinking for minutes.

    ⚠ A summary retry takes minutes and the notes editor autosaves on blur the whole time, so
    writing back the text read at entry reverts whatever they typed in between. And `write_text`
    CREATES: if `apply` moved the note, writing the old path would put a second, stale copy of
    the recording into the vault. Both are one re-read away.
    """
    if not src.is_file():
        raise FileNotFoundError(
            f"{note} is no longer there — it was moved or renamed while this ran, so nothing "
            f"was written")
    return src.read_text(encoding="utf-8")


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
    src = _vault_path(vault, note)
    with _note_lock(vault, note):
        text = src.read_text(encoding="utf-8")
        record_mod.write_note_text(
            src, record_mod.replace_review_sections(text, summary_md=None, notes_md=notes_md))
    _index_now(vault, str(src.relative_to(vault)), sweep=False)
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
    src = _vault_path(vault, note)
    text = src.read_text(encoding="utf-8")
    parts = record_mod.read_review_sections(text)
    if not parts["transcript"].strip():
        raise ValueError("this note has no transcript to summarize")

    _record_progress_start(job_id)
    _record_progress_update(job_id, stage="summarizing", label="Reasoning about the summary",
                            detail="The local model is re-reading the transcript")
    activity = (lambda kind, chunk: _record_progress_update(
        job_id, stage="summarizing", label="Reasoning about the summary",
        detail="Live local-model activity", kind=kind, text=chunk)) if job_id else None

    # Every knob matches the recording path, because this IS the recording path re-run: the
    # resident at its one context size, and thinking on (ratified 2026-08-26, lectures).
    clean_notes, note_images = _extract_note_images(parts["notes"], vault)
    s = summarize.summarize(parts["transcript"], model=llm.RECORD_MODEL,
                            num_ctx=llm.RECORD_CTX, think=True, on_activity=activity,
                            notes=clean_notes, instructions=instructions,
                            images=note_images or None)
    if not s.valid:
        _record_progress_update(job_id, stage="failed", label="Retry failed", detail=s.error)
        raise ValueError(f"the summary could not be rewritten: {s.error}")

    fm, _ = parse_frontmatter(text)
    summary_md = summarize.render(
        s, open_questions=str(fm.get("type") or "") not in summarize.LECTURE_TYPES)
    # ⚠ `valid` means SCHEMA-valid, and `Summary()` — every field empty — passes it and renders
    # to "". Writing that removes the summary and its fence, which is the one thing a failed
    # retry promises not to do. An empty answer is a failed answer.
    if not summary_md.strip():
        _record_progress_update(job_id, stage="failed", label="Retry failed",
                                detail="the model returned an empty summary")
        raise ValueError("the summary could not be rewritten: the model returned an empty "
                         "summary, so the one you had is untouched")
    # The note as it stands NOW, not as it stood before the model started thinking. The lock
    # covers the read and the write, never the model call: holding it for the minutes a summary
    # takes would hang their notes autosave on every blur.
    with _note_lock(vault, note):
        current = _reread_for_write(src, note)
        current = record_mod.replace_review_sections(current, summary_md=summary_md)
        record_mod.write_note_text(src, current)
    _index_now(vault, str(src.relative_to(vault)), sweep=False)
    _record_progress_update(job_id, stage="done", label="Summary rewritten", detail="")
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
    src = _vault_path(vault, note)
    # ⚠ Held across read, write AND move: apply is the one writer that can invalidate another
    # endpoint's path, and a bare acquire/release would leak the lock on the raise paths below.
    with _note_lock(vault, note):
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
            from . import inbox
            updates["title"] = f'"{record_mod._clean_scalar(title)}"'
            parts = src.stem.split("--")
            if len(parts) >= 3:
                name = f"{parts[0]}--{inbox._slug(title)}--{parts[-1]}{src.suffix}"

        text = set_frontmatter(text, updates)
        text = record_mod.set_review_status(text, "complete")
        text = _strip_frontmatter_key(text, "filed_by")
        text = record_mod.replace_review_sections(
            text, summary_md=summary_md, notes_md=notes_md)

        dest = _vault_path(vault, dest_dir) / name
        if dest != src and dest.exists():
            # Both movers grew this guard on 2026-08-04 after two notes could resolve to one
            # destination and one was silently lost. A third mover does not get to relearn it.
            raise ValueError(f"refusing to overwrite an existing note at {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        record_mod.write_note_text(dest, text)
        if dest != src:
            src.unlink()

    _index_now(vault, str(dest.relative_to(vault)),
               moved_from=str(src.relative_to(vault)) if dest != src else None)
    trace.record("record", {"stage": "apply", "note": str(dest), "moved": dest != src})
    return {"note": str(dest.relative_to(vault))}


def handle_record_review(*, note: str, vault: Path | None = None) -> dict:
    """Mark an unchanged recording reviewed without turning approval into a filing edit."""
    from . import record as record_mod
    from .chunk import parse_frontmatter
    from .config import VAULT as _VAULT

    vault = vault or _VAULT
    src = _vault_path(vault, note)
    with _note_lock(vault, note):
        text = src.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        if fm.get("origin") != "recorded":
            raise ValueError("review status is only valid for recorded notes")
        record_mod.write_note_text(src, record_mod.set_review_status(text, "complete"))
    _index_now(vault, str(src.relative_to(vault)), sweep=False)
    trace.record("record", {"stage": "review", "note": str(src)})
    return {"note": str(src.relative_to(vault))}


def _live_source_paths(source_ids: list[str]) -> dict[str, str]:
    """Where each note IS, not where it was when its thread was saved. A thread file records
    the path at save time; the note moves (a drag, the card filing it) and the plugin opens
    a thread's note by path, so both thread readers answer from the sources table."""
    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return {}
    con = db.connect()
    try:
        marks = ",".join("?" * len(ids))
        return {row["id"]: row["path"] for row in con.execute(
            f"SELECT id, path FROM sources WHERE deleted=0 AND id IN ({marks})", ids)}
    finally:
        con.close()


class _Handler(BaseHTTPRequestHandler):
    server_version = "slim-chat"

    # ---- plumbing ---------------------------------------------------------------

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Belt to the Host check's braces: never let another origin read these responses.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
            "object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", body)

    def _refuse_foreign_host(self) -> bool:
        if _host_header_allowed(self.headers.get("Host")):
            return False
        self._send_json(403, {
            "error": "refused: this local surface only answers requests addressed to "
                     "localhost (DNS-rebinding guard)"})
        return True

    def log_message(self, format, *args):  # noqa: A002 — stdlib signature
        pass  # quiet; every query already writes a replayable trace

    # ---- routes -----------------------------------------------------------------

    def do_GET(self):
        if self._refuse_foreign_host():
            return
        url = urlparse(self.path)
        # `stale` cannot see every bad server: one whose uv parent was adopted by PID 1
        # answered stale:false while ffprobe got EPERM on an ordinary file (2026-08-31).
        # When in doubt, kill the port.
        if url.path == "/api/health":
            self._send_json(200, code_state(getattr(self.server, "managed_parent_pid", None)))
            return
        if url.path == "/api/record/progress":
            job_id = (parse_qs(url.query).get("job_id") or [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
                self._send_json(400, {"error": "a valid job_id is required"})
                return
            state = record_progress(job_id)
            if state is None:
                self._send_json(404, {"error": "no such recording job"})
            else:
                self._send_json(200, state)
            return
        if url.path == "/api/copilot/threads":
            listed = threads_mod.list_copilot_threads()
            live = _live_source_paths([item.source_id for item in listed])
            self._send_json(200, {"threads": [
                {"id": item.id, "title": item.title, "updated_at": item.updated_at,
                 "turns": item.turns, "source_id": item.source_id,
                 "source_path": live.get(item.source_id, item.source_path),
                 "reasoning_mode": item.reasoning_mode}
                for item in listed]})
            return
        if url.path == "/api/copilot/image":
            self._handle_copilot_image(parse_qs(url.query))
            return
        if url.path == "/api/copilot/thread":
            thread_id = (parse_qs(url.query).get("id") or [""])[0]
            try:
                data = threads_mod.load(thread_id)
            except threads_mod.ThreadError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            if data is None or not data.get("source_id"):
                self._send_json(404, {"error": "no such copilot thread"})
            else:
                live = _live_source_paths([data["source_id"]])
                data["source_path"] = live.get(data["source_id"], data.get("source_path"))
                self._send_json(200, data)
            return
        self._send_json(404, {"error": f"no such path {url.path!r}"})

    def do_POST(self):
        if self._refuse_foreign_host():
            return
        path = urlparse(self.path).path
        if path == "/api/copilot/context":
            self._handle_copilot_context()
        elif path == "/api/copilot/stream":
            self._handle_copilot_stream()
        elif path == "/api/copilot/thread/save":
            self._handle_copilot_thread_save()
        elif path == "/api/copilot/thread/delete":
            self._handle_copilot_thread_delete()
        elif path == "/api/record":
            self._handle_record()
        elif path == "/api/record/live":
            self._handle_record_live()
        elif path == "/api/record/apply":
            self._handle_record_apply()
        elif path == "/api/record/review":
            self._handle_record_review()
        elif path == "/api/record/notes":
            self._handle_record_notes()
        elif path == "/api/record/summary":
            self._handle_record_summary()
        elif path == "/api/record/cancel":
            self._handle_record_cancel()
        elif path == "/api/record/transcript":
            self._handle_record_transcript()
        elif path == "/api/record/append":
            self._handle_record_append()
        else:
            self._send_json(404, {"error": f"no such path {self.path!r}"})

    # Read-only, loopback, Host-gated like every GET: the sidebar shows a sent image as a
    # thumbnail from the same store the model reads, instead of a name chip.
    def _handle_copilot_image(self, query: dict) -> None:
        thread_id = (query.get("thread") or [""])[0]
        image_id = (query.get("id") or [""])[0]
        try:
            found = copilot_images.image_path(thread_id, image_id)
        except threads_mod.ThreadError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        if found is None:
            self._send_json(404, {"error": "no such image"})
            return
        path, mime = found
        try:
            data = path.read_bytes()
        except OSError:                     # deleted between the stat and the read
            self._send_json(404, {"error": "no such image"})
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=86400")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _handle_copilot_context(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body()
        if data is None:
            return
        path = str(data.get("path") or "").strip()
        if not path:
            self._send_json(400, {"error": "path is required"})
            return
        # related=false is the plugin's pre-question refresh: sync and embed the note, skip
        # the nearby-notes search it already has.
        related = bool(data.get("related", True))
        con = db.connect()
        try:
            payload = copilot_mod.context_payload(con, config.VAULT, path, related=related)
        except copilot_mod.CopilotError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        finally:
            con.close()
        self._send_json(200, payload)

    def _handle_copilot_thread_save(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_THREAD_BODY_BYTES)
        if data is None:
            return
        source_id = str(data.get("source_id") or "").strip()
        con = db.connect()
        try:
            source = con.execute(
                "SELECT path FROM sources WHERE id=? AND deleted=0", (source_id,)
            ).fetchone()
        finally:
            con.close()
        if source is None:
            self._send_json(400, {"error": "no indexed source for that source id"})
            return
        try:
            summary = threads_mod.save_copilot(
                data.get("id"), str(data.get("title") or ""), data.get("turns"),
                source_id=source_id, source_path=source["path"],
                reasoning_mode=data.get("reasoning_mode"))
        except threads_mod.ThreadError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, {
            "status": "saved", "id": summary.id, "updated_at": summary.updated_at,
            "turns": summary.turns, "source_id": summary.source_id,
            "source_path": summary.source_path, "reasoning_mode": summary.reasoning_mode})

    def _handle_copilot_stream(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_COPILOT_STREAM_BODY_BYTES)
        if data is None:
            return
        question = str(data.get("question") or "").strip()
        source_id = str(data.get("source_id") or "").strip()
        if not question:
            self._send_json(400, {"error": "question is required"})
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._send_json(400, {"error": f"question exceeds {MAX_QUESTION_CHARS} chars"})
            return
        if not source_id:
            self._send_json(400, {"error": "source id is required"})
            return
        history = clean_history(data.get("history"))
        # Quick unless they asked for Deep. There is no third setting to resolve (2026-09-03).
        mode = str(data.get("reasoning_mode") or "quick").strip().lower()
        thread_id = data.get("thread_id")
        image_lock = None
        try:
            image_lock = copilot_images.request_lock(thread_id)
            image_lock.acquire()
            attachments, created = copilot_images.save_images(thread_id, data.get("images"))
            attachments, images = copilot_images.load_refs(thread_id, attachments)
            history = copilot_images.hydrate_history(thread_id, history)
        except (copilot_images.ImageError, threads_mod.ThreadError) as exc:
            if image_lock is not None:
                image_lock.release()
            self._send_json(400, {"error": str(exc)})
            return
        except OSError as exc:
            if image_lock is not None:
                image_lock.release()
            self._send_json(500, {"error": f"image attachment storage failed ({type(exc).__name__})"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()

        def emit(event, payload):
            body = json.dumps(payload, ensure_ascii=False)
            self.wfile.write(f"event: {event}\ndata: {body}\n\n".encode("utf-8"))
            self.wfile.flush()

        con = None
        completed = False
        try:
            con = db.connect()
            turn = copilot_mod.run_turn(
                con, source_id, question, history, mode, images=images,
                on_stage=lambda stage: emit("stage", {"stage": stage}),
                on_delta=lambda text: emit("delta", {"text": text}))
            turn["attachments"] = attachments
            emit("turn", turn)
            completed = True
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                emit("error", {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            if con is not None:
                con.close()
            if not completed:
                copilot_images.discard_created(created)
            image_lock.release()

    def _read_json_body(self, max_bytes: int = MAX_BODY_BYTES) -> dict | None:
        """Bounded JSON object body, or None after a 400 has already been sent.

        `max_bytes` is the small default guard; the thread save path passes
        MAX_THREAD_BODY_BYTES, since a resumable conversation legitimately exceeds 64 kB.
        The store enforces its own cap and returns a page-safe message beyond it."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not 0 < length <= max_bytes:
            self._send_json(400, {"error": f"expected a JSON body under {max_bytes // 1000} kB"})
            return None
        try:
            data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "body is not valid JSON"})
            return None
        if not isinstance(data, dict):
            self._send_json(400, {"error": "body must be a JSON object"})
            return None
        return data

    def _handle_record(self):
        """One finished recording. The body names an audio file the plugin already wrote into
        the vault — it never carries the audio itself."""
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        job_id = str(data.get("job_id") or "").strip()
        if job_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
            self._send_json(400, {"error": "job_id is invalid"})
            return
        _record_progress_start(job_id)
        update = (lambda **event: _record_progress_update(job_id, **event)) if job_id else None
        try:
            # Blank title stays blank. `handle_record` owns the transcript-derived fallback,
            # and only after the model has had its turn.
            with _record_pipeline_slot(job_id):
                result = handle_record(**_record_args(data), on_progress=update, job_id=job_id)
        except RecordQueueFull as e:
            if job_id:
                _record_progress_update(job_id, stage="failed", label="Processing queue is full",
                                        detail=str(e), done=True, error=str(e))
            self._send_json(503, {"error": str(e)})
            return
        except NoSpeech as e:
            # 422 and `silent: true`: the plugin shows a calm line and the ordinary way
            # forward, never the red "delete or keep your files?" screen.
            if job_id:
                _record_progress_update(job_id, stage="silent", label="No words captured",
                                        detail=str(e), done=True)
            self._send_json(422, {"silent": True, "error": str(e)})
            return
        except RecordCancelled:
            # 409, and a body that says nothing was lost: the plugin returns to the paused
            # state with the draft and the audio exactly where they were.
            if job_id:
                _record_progress_update(job_id, stage="cancelled", label="Cancelled",
                                        detail="Your recording and notes are untouched",
                                        done=True)
            self._send_json(409, {"cancelled": True,
                                  "error": "cancelled — nothing was written or deleted"})
            return
        except (ValueError, FileNotFoundError) as e:
            if job_id:
                _record_progress_update(job_id, stage="failed", label="Processing stopped",
                                        detail=str(e), done=True, error=str(e))
            self._send_json(400, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001 - surface the local failure and finish progress
            if job_id:
                _record_progress_update(job_id, stage="failed", label="Processing stopped",
                                        detail=str(e), done=True, error=str(e))
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        if job_id:
            _record_progress_update(job_id, stage="complete", label="AI review ready",
                                    detail="The note is filed and ready for approval", done=True)
        self._send_json(200, result)

    def _handle_record_live(self):
        """Local incremental captions. Audio arrives as bounded 16 kHz PCM, never leaves loopback."""
        if self._refuse_cross_site_write():
            return
        from . import transcribe

        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        session_id = str(data.get("session_id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", session_id):
            self._send_json(400, {"error": "a valid session_id is required"})
            return
        action = str(data.get("action") or "").strip()
        try:
            if action == "start":
                transcribe.live_start(session_id)
                text = ""
            elif action == "chunk":
                encoded = str(data.get("pcm16") or "")
                if not encoded or len(encoded) > 800_000:
                    raise ValueError("live transcript chunk is empty or too large")
                try:
                    pcm = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("live transcript chunk is not valid base64") from exc
                text = transcribe.live_add(session_id, pcm)
            elif action == "stop":
                text = transcribe.live_stop(session_id)
            else:
                raise ValueError("action must be start, chunk, or stop")
        except (ValueError, KeyError) as exc:
            self._send_json(400, {"error": str(exc).strip("'")})
            return
        except Exception as exc:  # captions are optional; report without touching the recording
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, {"transcript": text, "provisional": True})

    def _handle_record_apply(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        note = str(data.get("note") or "").strip()
        if not note:
            self._send_json(400, {"error": "note is required"})
            return
        try:
            result = handle_record_apply(
                note=note,
                dest_dir=str(data.get("dest_dir") or "").strip(),
                type_tag=str(data.get("type_tag") or "").strip(),
                topics=[str(t) for t in (data.get("topics") or [])],
                summary_md=str(data.get("summary_md") or ""),
                notes_md=(str(data.get("notes_md") or "") if "notes_md" in data else None),
                title=str(data.get("title") or ""))
        except (ValueError, FileNotFoundError) as e:
            self._send_json(400, {"error": str(e)})
            return
        self._send_json(200, result)

    def _handle_record_review(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        note = str(data.get("note") or "").strip()
        if not note:
            self._send_json(400, {"error": "note is required"})
            return
        try:
            result = handle_record_review(note=note)
        except (ValueError, FileNotFoundError) as e:
            self._send_json(400, {"error": str(e)})
            return
        self._send_json(200, result)

    def _handle_record_cancel(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body()
        if data is None:
            return
        job_id = str(data.get("job_id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id or ""):
            self._send_json(400, {"error": "a valid job_id is required"})
            return
        self._send_json(200, {"cancelling": request_record_cancel(job_id)})

    def _handle_record_append(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        note = str(data.get("note") or "").strip()
        raw = data.get("audio")
        rels = [str(a).strip() for a in raw] if isinstance(raw, list) else [str(raw or "").strip()]
        rels = [r for r in rels if r]
        if not note or not rels:
            self._send_json(400, {"error": "note and audio are required"})
            return
        job_id = str(data.get("job_id") or "").strip()
        if job_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
            self._send_json(400, {"error": "job_id is invalid"})
            return
        _record_progress_start(job_id)
        update = (lambda **event: _record_progress_update(job_id, **event)) if job_id else None
        try:
            with _record_pipeline_slot(job_id):
                result = handle_record_append(note=note, audio_rel=rels, job_id=job_id,
                                              on_progress=update)
        except RecordQueueFull as e:
            if job_id:
                _record_progress_update(job_id, stage="failed", label="Processing queue is full",
                                        detail=str(e), done=True, error=str(e))
            self._send_json(503, {"error": str(e)})
            return
        except NoSpeech as e:
            if job_id:
                _record_progress_update(job_id, stage="silent", label="No words captured",
                                        detail=str(e), done=True)
            self._send_json(422, {"silent": True, "error": str(e)})
            return
        except RecordCancelled:
            self._send_json(409, {"cancelled": True,
                                  "error": "cancelled — nothing was written or deleted"})
            return
        except (ValueError, FileNotFoundError) as e:
            if job_id:
                _record_progress_update(job_id, stage="failed", label="Processing stopped",
                                        detail=str(e), done=True, error=str(e))
            self._send_json(400, {"error": str(e)})
            return
        if job_id:
            _record_progress_update(job_id, stage="complete", label="Added to the note",
                                    detail=f"{result['words']} more words", done=True)
        self._send_json(200, result)

    def _handle_record_transcript(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        raw = data.get("audio")
        rels = [str(a).strip() for a in raw] if isinstance(raw, list) else [str(raw or "").strip()]
        rels = [r for r in rels if r]
        if not rels:
            self._send_json(400, {"error": "audio (a vault-relative path) is required"})
            return
        try:
            self._send_json(200, handle_record_transcript(audio_rel=rels))
        except (ValueError, FileNotFoundError) as e:
            self._send_json(400, {"error": str(e)})

    def _handle_record_notes(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        note = str(data.get("note") or "").strip()
        if not note:
            self._send_json(400, {"error": "note is required"})
            return
        try:
            result = handle_record_notes(note=note, notes_md=str(data.get("notes_md") or ""))
        except (ValueError, FileNotFoundError) as e:
            self._send_json(400, {"error": str(e)})
            return
        self._send_json(200, result)

    def _handle_record_summary(self):
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_RECORD_BODY_BYTES)
        if data is None:
            return
        note = str(data.get("note") or "").strip()
        if not note:
            self._send_json(400, {"error": "note is required"})
            return
        job_id = str(data.get("job_id") or "").strip()
        if job_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
            self._send_json(400, {"error": "job_id is invalid"})
            return
        _record_progress_start(job_id)
        try:
            with _record_pipeline_slot(job_id):
                result = handle_record_summary(
                    note=note, instructions=str(data.get("instructions") or ""), job_id=job_id)
        except RecordQueueFull as e:
            if job_id:
                _record_progress_update(job_id, stage="failed", label="Processing queue is full",
                                        detail=str(e), done=True, error=str(e))
            self._send_json(503, {"error": str(e)})
            return
        except (ValueError, FileNotFoundError) as e:
            self._send_json(400, {"error": str(e)})
            return
        self._send_json(200, result)

    def _refuse_cross_site_write(self) -> bool:
        """The write-endpoint gate, shared by every mutating route. The Host check alone
        cannot protect a mutation: a hostile page can fire a BLIND cross-site POST whose Host
        header is legitimately ours. Two properties close that — the body must be
        application/json (which forces a CORS preflight this server never grants), and a
        present Origin header must itself be local."""
        origin = self.headers.get("Origin")
        if origin is not None and not _origin_allowed(origin):
            self._send_json(403, {"error": "refused: cross-origin writes are not accepted"})
            return True
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(415, {
                "error": "refused: write endpoints require an application/json body"})
            return True
        return False

    def _handle_copilot_thread_delete(self):
        """Remove ONE copilot thread, behind the same write gate as every mutating route. An
        absent thread is a clean 200 removed=False; a bad id is an authored 400."""
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body()
        if data is None:
            return
        try:
            removed = threads_mod.delete_thread(data.get("id"))
        except threads_mod.ThreadError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, {"removed": removed})


# When THIS process started importing. `uv run` reads Python from disk at spawn time and never
# hot-reloads, so a server that outlives an edit serves old code with no outward sign.
SERVER_STARTED = time.time()


def _process_alive(pid: int) -> bool:
    """Whether ``pid`` still exists. EPERM means it exists but is owned by another user."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def code_state(parent_pid: int | None = None) -> dict:
    """Is the code this process is RUNNING older than the code on disk?

    ⚠ THIS EXISTS BECAUSE "the plugin owns the server's lifecycle" IS NOT TRUE. Obsidian does
    not reliably call `onunload` on a whole-app quit — measured 2026-08-23, a server started the
    previous afternoon survived a full Cmd+Q. The plugin then REUSES whatever is on the port
    (deliberately, so a terminal `--reload` server is not killed), so a stale server is silently
    PREFERRED over a correct one. A timestamp comparison, not a hash: mtime is what a save
    changes.
    """
    newest, newest_file = 0.0, ""
    for path in Path(__file__).resolve().parent.rglob("*.py"):
        try:
            m = path.stat().st_mtime
        except OSError:                  # a file vanishing mid-walk is not an error here
            continue
        if m > newest:
            newest, newest_file = m, path.name
    return {
        "ok": True,
        "pid": os.getpid(),
        "started_at": datetime.fromtimestamp(SERVER_STARTED, timezone.utc).isoformat(
            timespec="seconds"),
        "newest_source": datetime.fromtimestamp(newest, timezone.utc).isoformat(
            timespec="seconds") if newest else "",
        "newest_file": newest_file,
        "managed_parent_pid": parent_pid,
        # The vault this process resolved AT IMPORT. The plugin sends vault-relative paths and
        # reuses whatever is on the port, so a server that discovered a different vault (or was
        # started before a move) must be visible from outside — the same job as `stale`.
        "vault": str(config.VAULT),
        "managed_parent_alive": _process_alive(parent_pid) if parent_pid is not None else None,
        # Reporting it is the whole job. Refusing to serve would turn a warning into an outage.
        "stale": bool(newest and newest > SERVER_STARTED),
    }


def make_server(host: str, port: int, *, parent_pid: int | None = None) -> ThreadingHTTPServer:
    validate_bind_host(host)
    if parent_pid is not None and parent_pid <= 0:
        raise ChatError("parent pid must be positive")
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    server.managed_parent_pid = parent_pid
    return server


def _stop_with_parent(server: ThreadingHTTPServer, parent_pid: int,
                      interval: float = 1.0) -> threading.Thread:
    """Stop ``server`` after its Obsidian parent exits.

    The first wait is intentional: ``shutdown`` must run after ``serve_forever`` starts.
    A daemon thread is sufficient because this server is disposable derived state.
    """
    def watch():
        while True:
            time.sleep(interval)
            if not _process_alive(parent_pid):
                server.shutdown()
                return

    worker = threading.Thread(target=watch, name="slim-parent-watch", daemon=True)
    worker.start()
    return worker


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          *, parent_pid: int | None = None) -> None:
    if parent_pid is not None and not _process_alive(parent_pid):
        raise ChatError(f"parent process {parent_pid} is not running")
    server = make_server(host, port, parent_pid=parent_pid)
    bound_port = server.server_address[1]
    print(f"SLIM chat — http://{host}:{bound_port}/   (Ctrl-C to stop)")
    if parent_pid is not None:
        _stop_with_parent(server, parent_pid)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
