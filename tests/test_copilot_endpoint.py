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

COPILOT_WRITES = ("/api/copilot/context", "/api/copilot/stream", "/api/copilot/restore",
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


def test_stream_endpoint_passes_a_bounded_selection(server_url, monkeypatch):
    seen = {}

    def fake_run(con, source_id, question, history, mode, **kwargs):
        seen.update(selection=kwargs["selection"], edit="edit" in kwargs)
        return {"answer": "ok", "mode": "quick", "citations": [], "edit_run": None}

    monkeypatch.setattr(copilot, "run_turn", fake_run)
    post(server_url, "/api/copilot/stream", {
        "thread_id": "chat-1", "source_id": "source-1", "question": "tidy", "history": [],
        "edit": True, "selection": "x" * (copilot.MAX_SELECTION_CHARS + 50)})
    assert seen == {"selection": "x" * copilot.MAX_SELECTION_CHARS, "edit": False}   # no Edit mode left
    post(server_url, "/api/copilot/stream", {
        "thread_id": "chat-1", "source_id": "source-1", "question": "tidy", "history": [],
        "selection": 5})
    assert seen == {"selection": "", "edit": False}


def test_skills_endpoint_lists_builtins_and_the_vaults_own(server_url, monkeypatch, tmp_path):
    (tmp_path / "Skills").mkdir()
    (tmp_path / "Skills/standup.md").write_text("---\ndescription: Mine\n---\nDo it.\n")
    (tmp_path / "Skills/broken.md").write_text("---\noutput: somewhere\n---\nx\n")
    monkeypatch.setattr(chat.config, "VAULT", tmp_path)
    with urllib.request.urlopen(server_url + "/api/copilot/skills") as response:
        listed = {item["name"]: item for item in json.load(response)["skills"]}
    assert listed["standup"]["description"] == "Mine" and listed["standup"]["origin"] == "vault"
    assert listed["broken"]["error"].startswith("output must be")
    assert listed["consolidate-my-notes"]["output"] == "new-note"


def test_folders_endpoint_gives_a_reopened_card_its_tree_and_tags(server_url, monkeypatch, tmp_path):
    # 2026-09-29: a pending card reopened after a reload showed an empty "browse the vault"
    # and every topic as "new" — only a fresh recording's response carried either list.
    (tmp_path / "Capture/job-search").mkdir(parents=True)
    (tmp_path / "Capture/job-search/a.md").write_text("---\ntopics: [job-search]\n---\nx\n")
    (tmp_path / "Notes/empty").mkdir(parents=True)
    monkeypatch.setattr(chat.config, "VAULT", tmp_path)
    with urllib.request.urlopen(server_url + "/api/record/folders") as response:
        body = json.load(response)
    paths = {f["path"]: f["count"] for f in body["folders"]}
    assert paths["Capture/job-search"] == 1 and paths["Notes/empty"] == 0
    assert body["tag_counts"].get("job-search") == 1


# --- an editing turn over the stream ---------------------------------------------------------

from slim import config, edits, llm, threads  # noqa: E402

NOTE = "# Title\n\nIntro para.\n\n## A\n\nText a.\n"


def events_of(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line and not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def a_note(tmp_path, monkeypatch):
    """A vault holding one indexed note, and a scripted model for turns that may edit."""
    vault = tmp_path / "vault"
    (vault / "Notes").mkdir(parents=True)
    (vault / "Notes/n.md").write_text(NOTE, encoding="utf-8")
    monkeypatch.setattr(config, "VAULT", vault)
    monkeypatch.setattr(copilot.search_mod, "hybrid", lambda *a, **k: [])
    monkeypatch.setattr(copilot.llm, "chat_json", lambda *a, **k: ({"notes": []}, {}))
    con = db.connect()
    source = copilot.sync_note(con, vault, "Notes/n.md", embed=False)
    con.close()

    def script(*replies, intent=True):
        def chat_tools(messages, tools=None, **opts):
            if opts.get("fmt") is not None:    # the intent gate
                return {"content": json.dumps({"change_note": intent}), "tool_calls": [], "raw_tool_calls": []}, {}
            content, calls = replies[script.calls]
            script.calls += 1
            return ({"content": content, "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                     "raw_tool_calls": []}, {"duration_s": 1, "output_tokens": 5})
        def chat_stream(messages, **opts):     # today's Ask path
            content, _calls = replies[script.calls]
            script.calls += 1
            opts["on_delta"](content)
            return content, {"duration_s": 1}
        script.calls = 0
        monkeypatch.setattr(copilot.llm, "chat_tools", chat_tools)
        monkeypatch.setattr(copilot.llm, "chat_stream", chat_stream)
    return source, script


def stream(url, source, question, **extra):
    return post(url, "/api/copilot/stream", {
        "thread_id": "chat-1", "source_id": source["id"], "question": question, "history": [],
        "reasoning_mode": "quick", "doc": NOTE, "doc_hash": edits.digest(NOTE), **extra})


def test_the_stream_emits_the_edit_events_in_order(server_url, a_note):
    source, script = a_note
    script(("", [("plan", {"intro": "Rename A and add a line.", "steps": ["Rename A", "Add a line"]})]),
           ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
           ("", [("insert", {"after": "b4", "markdown": "More."})]))
    status, _headers, body = stream(server_url, source, "rename A and add a line")
    assert status == 200
    events = events_of(body)
    assert [e for e, _ in events] == ["stage", "edit", "plan", "step", "change", "step", "step", "change",
                                      "step", "turn"]
    edit = events[1][1]
    assert set(edit) == {"before_ref", "hash_before"} and edit["hash_before"] == edits.digest(NOTE)
    assert threads.load_before("chat-1", edit["before_ref"]) == NOTE
    assert events[2][1] == {"intro": "Rename A and add a line.", "steps": ["Rename A", "Add a line"],
                            "outline": None}
    assert events[4][1]["id"] == "c1" and events[7][1]["id"] == "c2"
    run = events[-1][1]["edit_run"]
    assert run["before_ref"] == edit["before_ref"] and run["hash_before"] == edit["hash_before"]
    text = NOTE
    for change in run["changes"]:
        text = edits.apply_splices(text, change["splices"])
    assert run["hash_after"] == edits.digest(text) == edits.digest(
        "# Title\n\nIntro para.\n\n## A renamed\n\nText a.\n\nMore.\n")
    assert ": alive" in body                       # the run asked whether the client was still there
    assert (config.VAULT / "Notes/n.md").read_text() == NOTE         # the server never writes the note


def test_a_text_answer_streams_as_today_with_no_edit_events(server_url, a_note):
    source, script = a_note
    script(("It is a note.", []), intent=False)
    _status, _headers, body = stream(server_url, source, "what is this?")
    events = events_of(body)
    assert [e for e, _ in events] == ["stage", "delta", "stage", "turn"]
    assert events[1][1] == {"text": "It is a note."} and events[2][1] == {"stage": "Noting sources"}
    assert events[-1][1]["edit_run"] is None


def test_a_doc_hash_that_does_not_match_reaches_the_plugin_as_an_error(server_url, a_note):
    source, script = a_note
    script(("never", []))
    _status, _headers, body = stream(server_url, source, "q", doc_hash=edits.digest("other"))
    assert events_of(body) == [("error", {"error": "the note changed while sending; try again"})]
    assert script.calls == 0


def test_a_doc_too_large_or_not_text_is_refused(server_url, a_note, monkeypatch):
    source, script = a_note
    monkeypatch.setattr(edits, "MAX_EDIT_CHARS", 10)
    status, _headers, body = stream(server_url, source, "q")
    assert status == 400 and "too large" in json.loads(body)["error"]
    monkeypatch.setattr(edits, "MAX_EDIT_CHARS", 400_000)
    status, _headers, body = stream(server_url, source, "q", doc=5)
    assert status == 400 and "doc" in json.loads(body)["error"]


def test_before_returns_the_saved_text_and_refuses_a_bad_ref(server_url):
    ref = threads.save_before("chat-1", NOTE)
    with urllib.request.urlopen(f"{server_url}/api/copilot/before?thread=chat-1&ref={ref}") as response:
        assert response.status == 200 and json.load(response) == {"text": NOTE}
    for query, code in ((f"thread=chat-1&ref={'0' * 32}", 404), ("thread=chat-1&ref=../x", 400),
                        (f"thread=..%2Fx&ref={ref}", 400), ("thread=chat-1", 400)):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{server_url}/api/copilot/before?{query}")
        assert caught.value.code == code, query


def test_a_client_that_hangs_up_stops_the_run_after_the_current_call(server_url, a_note, monkeypatch):
    """Stop is a closed socket. The run ends before its next model call, what landed stays in
    the trace, and the lock is free. The client here ends its side while its kernel still takes
    the server's writes: a write alone succeeds then, so the server must also see the end."""
    import socket
    from urllib.parse import urlparse

    source, script = a_note
    inside, hung_up = threading.Event(), threading.Event()
    real = []
    script(("", [("plan", {"intro": "x", "steps": ["Rename A", "Add a line", "Add another"]})]),
           ("", [("replace", {"target": "b3", "markdown": "## A renamed"})]),
           ("", [("insert", {"after": "b4", "markdown": "More."})]),
           ("", [("insert", {"after": "b4", "markdown": "Even more."})]))
    scripted = copilot.llm.chat_tools

    def chat_tools(messages, tools=None, **opts):
        if opts.get("fmt") is not None:        # the intent gate
            return scripted(messages, tools, **opts)
        if len(real) == 2:                     # the third call waits until the client has gone
            inside.set()
            hung_up.wait(5)
        real.append(1)
        return scripted(messages, tools, **opts)
    monkeypatch.setattr(copilot.llm, "chat_tools", chat_tools)
    records = []
    monkeypatch.setattr(copilot.trace, "record", lambda kind, payload: records.append(payload))

    png = base64.b64encode(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")).decode()
    body = json.dumps({"thread_id": "chat-1", "source_id": source["id"], "question": "go", "history": [],
                       "reasoning_mode": "quick", "doc": NOTE, "doc_hash": edits.digest(NOTE),
                       "images": [{"name": "eq.png", "mime": "image/png", "data": png}]}).encode()
    host, port = urlparse(server_url).hostname, urlparse(server_url).port
    client = socket.create_connection((host, port))
    client.sendall(b"POST /api/copilot/stream HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                   b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode()
                   + b"\r\n\r\n" + body)
    seen = b""
    while b"event: change" not in seen:
        seen += client.recv(65536)
    assert inside.wait(5)                      # the server is in its third call: Stop
    client.shutdown(socket.SHUT_WR)
    hung_up.set()
    for _ in range(100):
        if records:
            break
        threading.Event().wait(0.05)
    (record,) = records
    assert record["status"] == "edited" and record["edit_run"]["stopped"] == "stopped"
    assert record["edit_run"]["model_calls"] == 3 and record["edit_run"]["changes"] == 2
    assert not llm._CALL_LOCK.locked()
    handler = copilot_images.request_lock("chat-1")    # held by the handler until it finishes
    assert handler.acquire(timeout=5)
    handler.release()
    # The changes stay in the editor and the plugin keeps the turn, so its image stays too.
    assert len(list((threads.THREADS_DIR / "chat-1.images").glob("*.png"))) == 1
    client.close()



# --- Restore: the note as it was, unless a recording's own parts changed since -------------------

MEETING = ("---\ntitle: Call\n---\n\n```slim-meeting\n## Transcript\n\nMe: hello\n```\n\n"
           "My notes, long and rambling.\n")
REFUSED = ("This recording changed since (a new transcript or card edit); restoring would remove that. "
           "Use ↺ on the changes instead.")


def restore(url, ref, doc, thread="chat-1"):
    status, _headers, body = post(url, "/api/copilot/restore", {"thread_id": thread, "ref": ref, "doc": doc})
    return status, json.loads(body)


def test_restore_gives_back_the_copy_when_the_recordings_parts_are_as_they_were(server_url):
    ref = threads.save_before("chat-1", MEETING)
    now = MEETING.replace("My notes, long and rambling.", "My notes.")
    assert restore(server_url, ref, now) == (200, {"text": MEETING})


def test_restore_is_refused_when_the_recording_changed_since(server_url):
    ref = threads.save_before("chat-1", MEETING)
    resumed = MEETING.replace("Me: hello\n", "Me: hello\nThem: and a second segment\n")
    assert restore(server_url, ref, resumed) == (200, {"refused": REFUSED})
    carded = MEETING.replace("title: Call", "title: Call with Ana")
    assert restore(server_url, ref, carded) == (200, {"refused": REFUSED})


def test_a_plain_note_is_always_restorable(server_url):
    ref = threads.save_before("chat-1", NOTE)
    assert restore(server_url, ref, "anything at all\n") == (200, {"text": NOTE})


def test_restore_validates_its_ids_and_its_doc(server_url):
    ref = threads.save_before("chat-1", NOTE)
    assert restore(server_url, "0" * 32, NOTE)[0] == 404
    assert restore(server_url, "../x", NOTE)[0] == 400
    assert restore(server_url, ref, NOTE, thread="../x")[0] == 400
    assert restore(server_url, ref, 5)[0] == 400
