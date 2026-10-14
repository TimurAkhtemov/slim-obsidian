"""The recorder's shared state: the one bounded job queue, the progress and cancel registry
the plugin polls, one writer per note, and the index-now / sweep-later pair that makes a filed
note reachable."""

import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from . import trace

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
            if any(stats.get(key) for key in ("new", "edited", "renamed", "removed")) \
                    or embedded.get("embedded"):
                trace.record("record", {"stage": "sweep", "status": "ok", **out})   # quiet ones write nothing
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
