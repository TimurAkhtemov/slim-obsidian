"""The Obsidian plugin's local API server — `slim chat`, 127.0.0.1:7546.

Twenty-two routes, called by `obsidian/main.js` (and `GET /api/health` also by the stale-server
hook): the health guard, eleven `/api/record*` routes for the recorder and ten `/api/copilot*`
routes for the open-note sidebar. The copilot never writes a note: an edit streams back as
changes, and the plugin applies each one through the editor.

Loopback-only and Host/Origin-gated because this server moves and overwrites vault files. Two
properties are load-bearing — do not "simplify" either away. `make_server` refuses any bind
address that is not loopback (`""` binds every interface and is refused too). And every
request whose `Host` is not localhost is refused before routing, because DNS rebinding is the
one way a remote page reaches a loopback server. Mutating routes additionally refuse a foreign
`Origin` and any non-JSON body (`_refuse_cross_site_write`): a blind cross-site POST cannot
read the response, but a write does not need reading.

The `/api/record*` routes are adapters in `recorder.routes`, past the same gates. The recorder
itself is `recorder.pipeline` (segment-cached transcription, so a resumed recording never pays
twice), `recorder.review` (the card's edits) and `recorder.state` (one bounded FIFO for the heavy
model jobs, a job-local progress registry the plugin polls, cooperative cancel).
"""

import ipaddress
import json
import os
import select
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, db
from .recorder import routes
from .copilot import (blocks, context, edits as edits_mod, images as copilot_images,
                      skills as skills_mod, threads as threads_mod, turn as copilot_mod)

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
RESTORE_REFUSED = ("This recording changed since (a new transcript or card edit); restoring would remove "
                   "that. Use ↺ on the changes instead.")


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
    """Sanitize the page-supplied transcript: keep only well-formed turns, each whole; what fits
    the model's window is `copilot.turn.fit_window`'s call. Bad history is DROPPED, never a 400 — a
    broken transcript must not brick the chat."""
    if not isinstance(raw, list):
        return []
    turns = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role, text = item.get("role"), item.get("text")
        if role not in ("you", "slim") or not isinstance(text, str) or not text.strip():
            continue
        turn = {"role": role, "text": text.strip()}
        if role == "you" and item.get("attachments"):
            try:
                turn["attachments"] = copilot_images.clean_refs(item.get("attachments"))
            except copilot_images.ImageError:
                pass                         # malformed history never bricks a new question
        turns.append(turn)
    return turns


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


def _ended(connection: socket.socket) -> bool:
    """The client has closed its side of the connection: it reads as ended, or as reset."""
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        return bool(readable) and connection.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


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
        if url.path in routes.GET:
            routes.GET[url.path](self, parse_qs(url.query))
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
        if url.path == "/api/copilot/skills":
            # Read on every call: a skill saved in `Skills/` a moment ago is listed at once.
            self._send_json(200, {"skills": [
                skill.listing() for skill in skills_mod.load(config.VAULT).values()]})
            return
        if url.path == "/api/copilot/image":
            self._handle_copilot_image(parse_qs(url.query))
            return
        if url.path == "/api/copilot/before":
            # The note as it was before an editing run, for See before and Restore.
            query = parse_qs(url.query)
            try:
                text = threads_mod.load_before((query.get("thread") or [""])[0],
                                               (query.get("ref") or [""])[0])
            except threads_mod.ThreadError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            if text is None:
                self._send_json(404, {"error": "no such copy"})
            else:
                self._send_json(200, {"text": text})
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
        elif path == "/api/copilot/restore":
            self._handle_copilot_restore()
        elif path == "/api/copilot/thread/save":
            self._handle_copilot_thread_save()
        elif path == "/api/copilot/thread/delete":
            self._handle_copilot_thread_delete()
        elif path in routes.POST:
            handle, max_bytes = routes.POST[path]
            if self._refuse_cross_site_write():
                return
            data = self._read_json_body(max_bytes or MAX_BODY_BYTES)
            if data is not None:
                handle(self, data)
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
            payload = context.context_payload(con, config.VAULT, path, related=related)
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

    def _handle_copilot_restore(self):
        """The note as it was before an editing run, to put back — unless the recording's own parts
        changed since (a resume added a transcript, the card was edited): restoring would undo
        that, so it is refused with what to do instead."""
        if self._refuse_cross_site_write():
            return
        data = self._read_json_body(MAX_THREAD_BODY_BYTES)
        if data is None:
            return
        doc = data.get("doc")
        if not isinstance(doc, str):
            self._send_json(400, {"error": "doc must be text"})
            return
        try:
            before = threads_mod.load_before(data.get("thread_id"), data.get("ref"))
        except threads_mod.ThreadError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        if before is None:
            self._send_json(404, {"error": "no such copy"})
        elif blocks.same_recording(before, doc):
            self._send_json(200, {"text": before})
        else:
            self._send_json(200, {"refused": RESTORE_REFUSED})

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
        # The selection is the input of a selection skill; it writes nothing here. Where it
        # starts, in the editor's UTF-16 units: the same words can be in the note twice.
        selection = data.get("selection")
        selection = selection[:context.MAX_SELECTION_CHARS] if isinstance(selection, str) else ""
        selection_from = data.get("selection_from")
        if type(selection_from) is not int or selection_from < 0:
            selection_from = None
        # The editor's text and its hash: an editing run works on exactly what the owner sees. A
        # note too big to edit is still asked about; run_turn refuses only an edit of it.
        doc, doc_hash = data.get("doc"), data.get("doc_hash")
        if doc is not None and not (isinstance(doc, str) and isinstance(doc_hash, str)):
            self._send_json(400, {"error": "doc and doc_hash must be text"})
            return
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

        # Stop is the plugin closing the socket. During an editing run a failed write marks the
        # client gone, and so does its socket reading as ended — a write into a closed socket can
        # still succeed for a moment, and the plugin sends nothing after its request. The run
        # then ends before its next model call, its trace written and the lock freed; what
        # landed stays.
        gone, before = [], {}

        def quietly(write):
            if gone:
                return
            try:
                write()
            except (BrokenPipeError, ConnectionResetError):
                gone.append(True)

        def on_edit(event, payload):
            if event == "start":
                text = payload["before_text"]
                before.update(before_ref=threads_mod.save_before(thread_id, text),
                              hash_before=edits_mod.digest(text))
                # The plugin keeps a run's turn however it ends, and a stopped or failed run
                # never gets the `turn` naming its images: the start names them, and they stay.
                event, payload = "edit", {**before, "attachments": attachments}
            quietly(lambda: emit(event, payload))

        def should_stop():
            quietly(lambda: (self.wfile.write(b": alive\n\n"), self.wfile.flush()))
            if not gone and _ended(self.connection):
                gone.append(True)
            return bool(gone)

        con = None
        completed = False
        try:
            con = db.connect()
            turn = copilot_mod.run_turn(
                con, source_id, question, history, mode, vault=config.VAULT, images=images,
                selection=selection, selection_from=selection_from, doc=doc, doc_hash=doc_hash,
                on_stage=lambda stage: emit("stage", {"stage": stage}),
                on_delta=lambda text: emit("delta", {"text": text}),
                on_edit=on_edit, should_stop=should_stop)
            turn["attachments"] = attachments
            if turn.get("edit_run"):
                turn["edit_run"]["before_ref"] = before.get("before_ref")
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
            if not (completed or before):       # a run that started keeps them (see on_edit)
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
