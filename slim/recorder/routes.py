"""The `/api/record*` routes: thin HTTP adapters over the recorder, which lives in
`record_pipeline` (a recording, a resume), `record_review` (the card's edits) and
`record_state` (the job queue and the progress the plugin polls).

A POST adapter is `fn(h, data)`: `chat._Handler` has already refused a foreign Host and a
cross-site write, and read the JSON body under the cap `POST` names for the route. `h` is
the handler, for its `_send_json`.
"""

import base64
import binascii
import re

from .. import config
from . import pipeline, review as record_review, state as record_state

# The recorder body carries notes they TYPED during a recording, plus a path — never the audio,
# which the plugin writes into the vault itself precisely because a recording dwarfs any
# sane HTTP cap. An hour of typed notes still fits comfortably here.
MAX_RECORD_BODY_BYTES = 1_000_000


def progress(h, query: dict) -> None:
    job_id = (query.get("job_id") or [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
        h._send_json(400, {"error": "a valid job_id is required"})
        return
    state = record_state.record_progress(job_id)
    if state is None:
        h._send_json(404, {"error": "no such recording job"})
    else:
        h._send_json(200, state)


def folders(h, query: dict) -> None:
    h._send_json(200, pipeline._card_lists(config.VAULT))


def record(h, data: dict) -> None:
    """One finished recording. The body names an audio file the plugin already wrote into
    the vault — it never carries the audio itself."""
    job_id = str(data.get("job_id") or "").strip()
    if job_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
        h._send_json(400, {"error": "job_id is invalid"})
        return
    record_state._record_progress_start(job_id)
    update = ((lambda **event: record_state._record_progress_update(job_id, **event))
              if job_id else None)
    try:
        # Blank title stays blank. `handle_record` owns the transcript-derived fallback,
        # and only after the model has had its turn.
        with record_state._record_pipeline_slot(job_id):
            result = pipeline.handle_record(**pipeline._record_args(data),
                                                   on_progress=update, job_id=job_id)
    except record_state.RecordQueueFull as e:
        if job_id:
            record_state._record_progress_update(
                job_id, stage="failed", label="Processing queue is full",
                detail=str(e), done=True, error=str(e))
        h._send_json(503, {"error": str(e)})
        return
    except record_state.NoSpeech as e:
        # 422 and `silent: true`: the plugin shows a calm line and the ordinary way
        # forward, never the red "delete or keep your files?" screen.
        if job_id:
            record_state._record_progress_update(job_id, stage="silent",
                                                 label="No words captured",
                                                 detail=str(e), done=True)
        h._send_json(422, {"silent": True, "error": str(e)})
        return
    except record_state.RecordCancelled:
        # 409, and a body that says nothing was lost: the plugin returns to the paused
        # state with the draft and the audio exactly where they were.
        if job_id:
            record_state._record_progress_update(job_id, stage="cancelled", label="Cancelled",
                                                 detail="Your recording and notes are untouched",
                                                 done=True)
        h._send_json(409, {"cancelled": True,
                           "error": "cancelled — nothing was written or deleted"})
        return
    except (ValueError, FileNotFoundError) as e:
        if job_id:
            record_state._record_progress_update(job_id, stage="failed",
                                                 label="Processing stopped",
                                                 detail=str(e), done=True, error=str(e))
        h._send_json(400, {"error": str(e)})
        return
    except Exception as e:  # noqa: BLE001 - surface the local failure and finish progress
        if job_id:
            record_state._record_progress_update(job_id, stage="failed",
                                                 label="Processing stopped",
                                                 detail=str(e), done=True, error=str(e))
        h._send_json(500, {"error": f"{type(e).__name__}: {e}"})
        return
    if job_id:
        record_state._record_progress_update(job_id, stage="complete", label="AI review ready",
                                             detail="The note is filed and ready for approval",
                                             done=True)
    h._send_json(200, result)


def live(h, data: dict) -> None:
    """Local incremental captions. Audio arrives as bounded 16 kHz PCM, never leaves loopback."""
    from ..capture import transcribe

    session_id = str(data.get("session_id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", session_id):
        h._send_json(400, {"error": "a valid session_id is required"})
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
        h._send_json(400, {"error": str(exc).strip("'")})
        return
    except Exception as exc:  # captions are optional; report without touching the recording
        h._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
        return
    h._send_json(200, {"transcript": text, "provisional": True})


def apply(h, data: dict) -> None:
    note = str(data.get("note") or "").strip()
    if not note:
        h._send_json(400, {"error": "note is required"})
        return
    try:
        result = record_review.handle_record_apply(
            note=note,
            dest_dir=str(data.get("dest_dir") or "").strip(),
            type_tag=str(data.get("type_tag") or "").strip(),
            topics=[str(t) for t in (data.get("topics") or [])],
            summary_md=str(data.get("summary_md") or ""),
            notes_md=(str(data.get("notes_md") or "") if "notes_md" in data else None),
            title=str(data.get("title") or ""))
    except (ValueError, FileNotFoundError) as e:
        h._send_json(400, {"error": str(e)})
        return
    h._send_json(200, result)


def review(h, data: dict) -> None:
    note = str(data.get("note") or "").strip()
    if not note:
        h._send_json(400, {"error": "note is required"})
        return
    try:
        result = record_review.handle_record_review(note=note)
    except (ValueError, FileNotFoundError) as e:
        h._send_json(400, {"error": str(e)})
        return
    h._send_json(200, result)


def cancel(h, data: dict) -> None:
    job_id = str(data.get("job_id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id or ""):
        h._send_json(400, {"error": "a valid job_id is required"})
        return
    h._send_json(200, {"cancelling": record_state.request_record_cancel(job_id)})


def append(h, data: dict) -> None:
    note = str(data.get("note") or "").strip()
    raw = data.get("audio")
    rels = [str(a).strip() for a in raw] if isinstance(raw, list) else [str(raw or "").strip()]
    rels = [r for r in rels if r]
    if not note or not rels:
        h._send_json(400, {"error": "note and audio are required"})
        return
    job_id = str(data.get("job_id") or "").strip()
    if job_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
        h._send_json(400, {"error": "job_id is invalid"})
        return
    record_state._record_progress_start(job_id)
    update = ((lambda **event: record_state._record_progress_update(job_id, **event))
              if job_id else None)
    try:
        with record_state._record_pipeline_slot(job_id):
            result = pipeline.handle_record_append(note=note, audio_rel=rels,
                                                          job_id=job_id, on_progress=update)
    except record_state.RecordQueueFull as e:
        if job_id:
            record_state._record_progress_update(
                job_id, stage="failed", label="Processing queue is full",
                detail=str(e), done=True, error=str(e))
        h._send_json(503, {"error": str(e)})
        return
    except record_state.NoSpeech as e:
        if job_id:
            record_state._record_progress_update(job_id, stage="silent",
                                                 label="No words captured",
                                                 detail=str(e), done=True)
        h._send_json(422, {"silent": True, "error": str(e)})
        return
    except record_state.RecordCancelled:
        h._send_json(409, {"cancelled": True,
                           "error": "cancelled — nothing was written or deleted"})
        return
    except (ValueError, FileNotFoundError) as e:
        if job_id:
            record_state._record_progress_update(job_id, stage="failed",
                                                 label="Processing stopped",
                                                 detail=str(e), done=True, error=str(e))
        h._send_json(400, {"error": str(e)})
        return
    if job_id:
        record_state._record_progress_update(job_id, stage="complete",
                                             label="Added to the note",
                                             detail=f"{result['words']} more words", done=True)
    h._send_json(200, result)


def transcript(h, data: dict) -> None:
    raw = data.get("audio")
    rels = [str(a).strip() for a in raw] if isinstance(raw, list) else [str(raw or "").strip()]
    rels = [r for r in rels if r]
    if not rels:
        h._send_json(400, {"error": "audio (a vault-relative path) is required"})
        return
    try:
        h._send_json(200, pipeline.handle_record_transcript(audio_rel=rels))
    except (ValueError, FileNotFoundError) as e:
        h._send_json(400, {"error": str(e)})


def notes(h, data: dict) -> None:
    note = str(data.get("note") or "").strip()
    if not note:
        h._send_json(400, {"error": "note is required"})
        return
    try:
        result = record_review.handle_record_notes(note=note,
                                                   notes_md=str(data.get("notes_md") or ""))
    except (ValueError, FileNotFoundError) as e:
        h._send_json(400, {"error": str(e)})
        return
    h._send_json(200, result)


def summary(h, data: dict) -> None:
    note = str(data.get("note") or "").strip()
    if not note:
        h._send_json(400, {"error": "note is required"})
        return
    job_id = str(data.get("job_id") or "").strip()
    if job_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job_id):
        h._send_json(400, {"error": "job_id is invalid"})
        return
    record_state._record_progress_start(job_id)
    try:
        with record_state._record_pipeline_slot(job_id):
            result = record_review.handle_record_summary(
                note=note, instructions=str(data.get("instructions") or ""), job_id=job_id)
    except record_state.RecordQueueFull as e:
        if job_id:
            record_state._record_progress_update(
                job_id, stage="failed", label="Processing queue is full",
                detail=str(e), done=True, error=str(e))
        h._send_json(503, {"error": str(e)})
        return
    except (ValueError, FileNotFoundError) as e:
        h._send_json(400, {"error": str(e)})
        return
    h._send_json(200, result)


GET = {
    "/api/record/progress": progress,
    "/api/record/folders": folders,
}

# Each route's adapter and its body cap. None is the server's 64 kB default: a cancel carries a
# job id and nothing else.
POST = {
    "/api/record": (record, MAX_RECORD_BODY_BYTES),
    "/api/record/live": (live, MAX_RECORD_BODY_BYTES),
    "/api/record/apply": (apply, MAX_RECORD_BODY_BYTES),
    "/api/record/review": (review, MAX_RECORD_BODY_BYTES),
    "/api/record/notes": (notes, MAX_RECORD_BODY_BYTES),
    "/api/record/summary": (summary, MAX_RECORD_BODY_BYTES),
    "/api/record/cancel": (cancel, None),
    "/api/record/transcript": (transcript, MAX_RECORD_BODY_BYTES),
    "/api/record/append": (append, MAX_RECORD_BODY_BYTES),
}
