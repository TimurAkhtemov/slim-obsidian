"""HTTP contract for the Obsidian open-note copilot."""

import base64
import json
import threading
import urllib.error
import urllib.request

import pytest

from slim import chat, copilot, copilot_images, db


@pytest.fixture
def server_url():
    server = chat.make_server("127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def post(url, path, body):
    request = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read().decode()


def seed_source(source_id="source-1", path="Notes/school/lesson.md"):
    con = db.connect()
    con.execute(
        "INSERT INTO sources (id, path, current_hash) VALUES (?, ?, 'hash')",
        (source_id, path))
    con.commit()
    con.close()


def test_context_endpoint_returns_the_active_source(server_url, monkeypatch):
    monkeypatch.setattr(copilot, "context_payload", lambda con, vault, path, related=True: {
        "source": {"id": "source-1", "path": path},
        "related": []})
    status, _headers, body = post(
        server_url, "/api/copilot/context", {"path": "Notes/school/lesson.md"})
    assert status == 200
    assert json.loads(body)["source"]["id"] == "source-1"


def test_copilot_thread_endpoints_require_and_return_source_metadata(server_url):
    seed_source()
    status, _headers, body = post(server_url, "/api/copilot/thread/save", {
        "id": "chat-1", "title": "Lesson", "turns": [], "source_id": "source-1",
        "source_path": "Journal/spoofed.md", "reasoning_mode": "deep"})
    assert status == 200
    with urllib.request.urlopen(server_url + "/api/copilot/threads") as response:
        listed = json.load(response)["threads"]
    assert listed[0]["source_id"] == "source-1"
    assert listed[0]["source_path"] == "Notes/school/lesson.md"
    assert listed[0]["reasoning_mode"] == "deep"

    status, _headers, body = post(server_url, "/api/copilot/thread/save", {
        "id": "bad", "title": "No anchor", "turns": []})
    assert status == 400 and "source id" in json.loads(body)["error"]

    status, _headers, body = post(server_url, "/api/copilot/thread/save", {
        "id": "missing", "title": "Missing", "turns": [], "source_id": "not-indexed",
        "source_path": "Notes/missing.md", "reasoning_mode": "quick"})
    assert status == 400 and "no indexed source" in json.loads(body)["error"]


def test_stream_endpoint_emits_stage_delta_and_turn(server_url, monkeypatch):
    def fake_run(con, source_id, question, history, mode, **kwargs):
        kwargs["on_stage"]("Thinking deeply")
        kwargs["on_delta"]("Grounded")
        return {"answer": "Grounded", "mode": "deep", "citations": []}

    monkeypatch.setattr(copilot, "run_turn", fake_run)
    status, headers, body = post(server_url, "/api/copilot/stream", {
        "thread_id": "chat-1", "source_id": "source-1", "question": "Compare",
        "history": [], "reasoning_mode": "deep"})
    assert status == 200
    assert headers.get_content_type() == "text/event-stream"
    assert "event: stage\ndata: {\"stage\": \"Thinking deeply\"}" in body
    assert "event: delta\ndata: {\"text\": \"Grounded\"}" in body
    assert "event: turn\ndata:" in body


def test_stream_endpoint_persists_images_and_passes_current_and_history_bytes(server_url, monkeypatch):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    encoded = base64.b64encode(png).decode()
    old_refs, _created = copilot_images.save_images("chat-1", [
        {"name": "old.png", "mime": "image/png", "data": encoded}])
    seen = {}

    def fake_run(_con, _source_id, _question, history, _mode, **kwargs):
        seen["history"] = history
        seen["images"] = kwargs["images"]
        return {"answer": "Equation", "mode": "quick", "citations": []}

    monkeypatch.setattr(copilot, "run_turn", fake_run)
    status, _headers, body = post(server_url, "/api/copilot/stream", {
        "thread_id": "chat-1", "source_id": "source-1", "question": "Explain",
        "history": [{"role": "you", "text": "Earlier", "attachments": old_refs}],
        "images": [{"name": "new.png", "mime": "image/png", "data": encoded}],
        "reasoning_mode": "quick"})
    assert status == 200
    assert base64.b64decode(seen["images"][0]) == png
    assert base64.b64decode(seen["history"][0]["images"][0]) == png
    turn = json.loads(body.split("event: turn\ndata: ", 1)[1].split("\n\n", 1)[0])
    assert turn["attachments"][0]["name"] == "new.png"
    assert "data" not in turn["attachments"][0]


def test_image_endpoint_serves_a_stored_image_and_refuses_the_rest(server_url):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    refs, _created = copilot_images.save_images("chat-1", [
        {"name": "eq.png", "mime": "image/png", "data": base64.b64encode(png).decode()}])
    with urllib.request.urlopen(f"{server_url}/api/copilot/image?thread=chat-1&id={refs[0]['id']}") as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/png"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.read() == png
    for query in (f"thread=chat-9&id={refs[0]['id']}", "thread=chat-1&id=" + "0" * 64, "thread=chat-1&id=x"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{server_url}/api/copilot/image?{query}")
        assert caught.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"{server_url}/api/copilot/image?thread=..%2Fx&id={refs[0]['id']}")
    assert caught.value.code == 400


def test_stream_endpoint_emits_error_event_not_a_dead_socket(server_url, monkeypatch):
    monkeypatch.setattr(
        copilot, "run_turn",
        lambda *args, **kwargs: (_ for _ in ()).throw(copilot.CopilotError("local-only")))
    status, _headers, body = post(server_url, "/api/copilot/stream", {
        "thread_id": "chat-1", "source_id": "source-1", "question": "Question",
        "history": [], "reasoning_mode": "quick"})
    assert status == 200
    assert "event: error\ndata: {\"error\": \"local-only\"}" in body


def test_context_endpoint_can_skip_the_nearby_notes_search(server_url, monkeypatch):
    # The plugin re-syncs the open note before every question; that call needs the fresh
    # source row, not a second hybrid search for nearby notes.
    seen = {}

    def fake_context(con, vault, path, *, related=True):
        seen["related"] = related
        return {"source": {"id": "source-1", "path": path}, "related": []}

    monkeypatch.setattr(copilot, "context_payload", fake_context)
    status, _headers, _body = post(
        server_url, "/api/copilot/context", {"path": "Notes/school/lesson.md", "related": False})
    assert status == 200
    assert seen["related"] is False


# --- per-thread delete (the sidebar's table stake; only obsidian/main.js calls it) -----------

def _save(url, thread_id):
    status, _headers, body = post(url, "/api/copilot/thread/save", {
        "id": thread_id, "title": thread_id, "turns": [], "source_id": "source-1",
        "source_path": "Notes/school/lesson.md", "reasoning_mode": "quick"})
    assert status == 200, body


def test_copilot_thread_delete_round_trip(server_url):
    """Save two, delete one, and only the survivor lists. A second delete of the same id is
    a clean removed=False, never an error — delete is idempotent."""
    seed_source()
    _save(server_url, "keep")
    _save(server_url, "drop")

    status, _headers, body = post(server_url, "/api/copilot/thread/delete", {"id": "drop"})
    assert status == 200 and json.loads(body)["removed"] is True

    with urllib.request.urlopen(server_url + "/api/copilot/threads") as response:
        ids = [t["id"] for t in json.load(response)["threads"]]
    assert ids == ["keep"], "the deleted thread is gone; the other survives"

    status, _headers, body = post(server_url, "/api/copilot/thread/delete", {"id": "drop"})
    assert status == 200 and json.loads(body)["removed"] is False, "absent delete is not an error"


@pytest.mark.parametrize("evil", ["../evil", "a/b", "..", ".hidden"])
def test_copilot_thread_delete_invalid_id_is_authored_400(server_url, evil):
    """A traversal-shaped id is refused with authored JSON, never a traceback to the plugin."""
    status, _headers, body = post(server_url, "/api/copilot/thread/delete", {"id": evil})
    assert status == 400, evil
    assert "error" in json.loads(body)


# --- the shared write gate covers every mutating copilot route -----------------------------

COPILOT_WRITES = ("/api/copilot/context", "/api/copilot/stream",
                  "/api/copilot/thread/save", "/api/copilot/thread/delete")


@pytest.mark.parametrize("endpoint", COPILOT_WRITES)
def test_copilot_write_endpoints_refuse_cross_site_writes(server_url, endpoint):
    """The Host check alone cannot protect a mutation: a hostile page can fire a BLIND
    cross-site POST at 127.0.0.1. A foreign Origin is refused outright, and a no-preflight
    content type is refused with no Origin at all."""
    foreign = urllib.request.Request(
        server_url + endpoint, data=json.dumps({"path": "x", "id": "x"}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Origin": "https://evil.example.com"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(foreign)
    assert excinfo.value.code == 403, endpoint

    form = urllib.request.Request(
        server_url + endpoint, data=b"path=x", method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(form)
    assert excinfo.value.code == 415, endpoint


def test_thread_listing_and_load_report_the_source_current_path(server_url):
    """A thread remembers the path its note had when it was saved. The note moves — a drag,
    or the card filing it — and the plugin opens a thread by path, so both readers must
    answer with where the note is NOW."""
    seed_source()
    status, _headers, _body = post(server_url, "/api/copilot/thread/save", {
        "id": "chat-1", "title": "Lesson", "turns": [], "source_id": "source-1",
        "source_path": "Notes/school/lesson.md", "reasoning_mode": "quick"})
    assert status == 200
    con = db.connect()
    con.execute("UPDATE sources SET path='Notes/school/cs-201/lesson.md' WHERE id='source-1'")
    con.commit()
    con.close()
    with urllib.request.urlopen(server_url + "/api/copilot/threads") as response:
        listed = json.load(response)["threads"]
    assert listed[0]["source_path"] == "Notes/school/cs-201/lesson.md"
    with urllib.request.urlopen(server_url + "/api/copilot/thread?id=chat-1") as response:
        assert json.load(response)["source_path"] == "Notes/school/cs-201/lesson.md"
