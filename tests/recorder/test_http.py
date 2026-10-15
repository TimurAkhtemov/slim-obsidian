"""The /api/record* routes over HTTP: status codes, body caps and the progress a job ends in.

The handlers have their own suites; these pin what the plugin sees on the wire. The slow parts
are faked exactly as in those suites, so no model is ever called.
"""
import http.client
import json
import threading
import urllib.error
import urllib.request

import pytest

from slim import server, config
from slim.recorder import state
from slim.capture import suggest, summarize, transcribe
from tests.recorder.fakes import _archive_in_tmp, _draft, _existing, _fakes, _staged  # noqa: F401


@pytest.fixture
def server_url(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT", tmp_path)
    httpd = server.make_server("127.0.0.1", 0)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


def post(url, path, body):
    request = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def progress(url, job_id):
    try:
        with urllib.request.urlopen(f"{url}/api/record/progress?job_id={job_id}") as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _silent(monkeypatch):
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="   ", model="fake", audio_seconds=30.0, wall_seconds=1.0))


def _full_queue(monkeypatch):
    monkeypatch.setattr(state._RECORD_PIPELINE, "limit", 0)


# --- POST /api/record -------------------------------------------------------------------

def test_record_files_the_note_and_ends_complete(server_url, tmp_path, monkeypatch):
    _fakes(monkeypatch)
    status, body = post(server_url, "/api/record", {
        "audio": _staged(tmp_path), "notes_md": "", "type": "lecture", "title": "T",
        "job_id": "job-ok"})
    assert status == 200
    assert (tmp_path / body["note"]).is_file()
    assert body["card"]["dest_dir"] == "Notes/school/CS201 ML"
    status, job = progress(server_url, "job-ok")
    assert status == 200
    assert (job["stage"], job["label"], job["done"], job["error"]) == \
        ("complete", "AI review ready", True, "")
    assert job["steps"][-1]["status"] == "done"


def test_record_silence_is_422_and_ends_silent(server_url, tmp_path, monkeypatch):
    _fakes(monkeypatch)
    _silent(monkeypatch)
    status, body = post(server_url, "/api/record", {
        "audio": _staged(tmp_path), "notes_md": "", "type": "lecture", "title": "T",
        "job_id": "job-silent"})
    assert status == 422 and body["silent"] is True and "no words" in body["error"]
    _, job = progress(server_url, "job-silent")
    assert (job["stage"], job["done"], job["error"]) == ("silent", True, "")


def test_record_cancel_is_409_and_ends_cancelled(server_url, tmp_path, monkeypatch):
    _fakes(monkeypatch)

    def cancel_midway(*a, **k):
        assert post(server_url, "/api/record/cancel", {"job_id": "job-c"}) == \
            (200, {"cancelling": True})
        return summarize.Summary(overview="never used")
    monkeypatch.setattr(summarize, "summarize", cancel_midway)
    audio, draft = _staged(tmp_path), _draft(tmp_path)
    status, body = post(server_url, "/api/record", {
        "audio": audio, "draft": draft, "type": "lecture", "title": "T", "job_id": "job-c"})
    assert status == 409 and body["cancelled"] is True
    assert (tmp_path / audio).is_file() and (tmp_path / draft).is_file()
    _, job = progress(server_url, "job-c")
    assert (job["stage"], job["label"], job["done"]) == ("cancelled", "Cancelled", True)


def test_record_missing_audio_is_400_and_ends_failed(server_url, tmp_path, monkeypatch):
    _fakes(monkeypatch)
    status, body = post(server_url, "/api/record", {
        "audio": "Attachments/_incoming/gone.webm", "notes_md": "", "type": "lecture",
        "job_id": "job-400"})
    assert status == 400 and "no audio" in body["error"]
    _, job = progress(server_url, "job-400")
    assert (job["stage"], job["label"], job["done"]) == \
        ("failed", "Processing stopped", True)
    assert "no audio" in job["error"]


def test_record_contract_errors_are_400(server_url, tmp_path):
    assert post(server_url, "/api/record", {"audio": "a.webm"})[0] == 400
    status, body = post(server_url, "/api/record",
                        {"audio": "a.webm", "notes_md": "", "job_id": "bad id!"})
    assert status == 400 and body["error"] == "job_id is invalid"


def test_record_unexpected_failure_is_500_and_ends_failed(server_url, tmp_path, monkeypatch):
    _fakes(monkeypatch)

    def broken_card(*a, **k):
        raise RuntimeError("card exploded")
    monkeypatch.setattr(suggest, "build_card", broken_card)
    status, body = post(server_url, "/api/record", {
        "audio": _staged(tmp_path), "notes_md": "", "type": "lecture", "job_id": "job-500"})
    assert status == 500 and body["error"] == "RuntimeError: card exploded"
    _, job = progress(server_url, "job-500")
    assert (job["stage"], job["done"], job["error"]) == ("failed", True, "card exploded")
    assert job["steps"][-1]["status"] == "error"


def test_record_full_queue_is_503_and_ends_failed(server_url, tmp_path, monkeypatch):
    _fakes(monkeypatch)
    _full_queue(monkeypatch)
    status, body = post(server_url, "/api/record", {
        "audio": _staged(tmp_path), "notes_md": "", "type": "lecture", "job_id": "job-503"})
    assert status == 503 and "admitted jobs" in body["error"]
    _, job = progress(server_url, "job-503")
    assert (job["stage"], job["label"], job["done"]) == \
        ("failed", "Processing queue is full", True)


# --- GET /api/record/progress -----------------------------------------------------------

def test_progress_refuses_a_bad_id_and_names_an_unknown_one(server_url):
    assert progress(server_url, "bad%20id")[0] == 400
    assert progress(server_url, "never-started") == (404, {"error": "no such recording job"})


# --- POST /api/record/append ------------------------------------------------------------

def test_append_adds_words_and_ends_complete(server_url, tmp_path, monkeypatch):
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(transcribe, "transcribe", lambda p, model=None: transcribe.Transcript(
        text="and more", model="fake", audio_seconds=60.0, wall_seconds=1.0))
    note = str(r.note.relative_to(tmp_path))
    status, body = post(server_url, "/api/record/append", {
        "note": note, "audio": [_staged(tmp_path, "more-001.webm")], "job_id": "app-ok"})
    assert status == 200 and body["words"] == 2 and body["note"] == note
    _, job = progress(server_url, "app-ok")
    assert (job["stage"], job["label"], job["detail"], job["done"]) == \
        ("complete", "Added to the note", "2 more words", True)


def test_append_silence_is_422_and_ends_silent(server_url, tmp_path, monkeypatch):
    r = _existing(tmp_path, monkeypatch)
    _silent(monkeypatch)
    status, body = post(server_url, "/api/record/append", {
        "note": str(r.note.relative_to(tmp_path)),
        "audio": _staged(tmp_path, "more-001.webm"), "job_id": "app-silent"})
    assert status == 422 and body["silent"] is True
    _, job = progress(server_url, "app-silent")
    assert (job["stage"], job["done"]) == ("silent", True)


def test_append_cancel_is_409_and_leaves_progress_cancelling(server_url, tmp_path, monkeypatch):
    r = _existing(tmp_path, monkeypatch)

    def cancel_then_transcribe(p, model=None):
        post(server_url, "/api/record/cancel", {"job_id": "app-c"})
        return transcribe.Transcript(text="words", model="fake", audio_seconds=1.0,
                                     wall_seconds=1.0)
    monkeypatch.setattr(transcribe, "transcribe", cancel_then_transcribe)
    status, body = post(server_url, "/api/record/append", {
        "note": str(r.note.relative_to(tmp_path)),
        "audio": [_staged(tmp_path, "more-001.webm")], "job_id": "app-c"})
    assert status == 409 and body["cancelled"] is True
    _, job = progress(server_url, "app-c")
    assert (job["stage"], job["done"]) == ("cancelling", False)


def test_append_errors_are_400_and_a_full_queue_503(server_url, tmp_path, monkeypatch):
    assert post(server_url, "/api/record/append", {"note": "x.md"}) == \
        (400, {"error": "note and audio are required"})
    assert post(server_url, "/api/record/append",
                {"note": "x.md", "audio": "a.webm", "job_id": "bad id"}) == \
        (400, {"error": "job_id is invalid"})
    status, body = post(server_url, "/api/record/append", {
        "note": "Capture/_unfiled/missing.md", "audio": "Attachments/_incoming/a.webm",
        "job_id": "app-400"})
    assert status == 400
    _, job = progress(server_url, "app-400")
    assert (job["stage"], job["done"]) == ("failed", True)

    _full_queue(monkeypatch)
    status, _ = post(server_url, "/api/record/append", {
        "note": "x.md", "audio": "a.webm", "job_id": "app-503"})
    assert status == 503
    _, job = progress(server_url, "app-503")
    assert (job["stage"], job["label"]) == ("failed", "Processing queue is full")


# --- POST /api/record/summary -----------------------------------------------------------

def test_summary_rewrites_and_ends_done(server_url, tmp_path, monkeypatch):
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(summarize, "summarize",
                        lambda *a, **k: summarize.Summary(overview="A fresh summary."))
    status, body = post(server_url, "/api/record/summary", {
        "note": str(r.note.relative_to(tmp_path)), "job_id": "sum-ok"})
    assert status == 200 and "A fresh summary." in body["summary"]
    _, job = progress(server_url, "sum-ok")
    assert (job["stage"], job["label"], job["done"]) == ("done", "Summary rewritten", False)


def test_summary_failure_is_400_and_ends_failed(server_url, tmp_path, monkeypatch):
    r = _existing(tmp_path, monkeypatch)
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: summarize.Summary())
    status, body = post(server_url, "/api/record/summary", {
        "note": str(r.note.relative_to(tmp_path)), "job_id": "sum-bad"})
    assert status == 400 and "untouched" in body["error"]
    _, job = progress(server_url, "sum-bad")
    assert (job["stage"], job["label"]) == ("failed", "Retry failed")


def test_summary_errors_are_400_and_a_full_queue_503(server_url, monkeypatch):
    assert post(server_url, "/api/record/summary", {}) == (400, {"error": "note is required"})
    assert post(server_url, "/api/record/summary", {"note": "x.md", "job_id": "bad id"}) == \
        (400, {"error": "job_id is invalid"})
    _full_queue(monkeypatch)
    status, _ = post(server_url, "/api/record/summary", {"note": "x.md", "job_id": "sum-503"})
    assert status == 503
    _, job = progress(server_url, "sum-503")
    assert (job["stage"], job["label"], job["done"]) == \
        ("failed", "Processing queue is full", True)


# --- the small routes, and which body cap each one keeps -----------------------------------

def test_the_small_record_routes_answer(server_url, tmp_path, monkeypatch):
    r = _existing(tmp_path, monkeypatch)
    note = str(r.note.relative_to(tmp_path))
    assert post(server_url, "/api/record/notes", {"note": note, "notes_md": "- typed"}) == \
        (200, {"note": note})
    assert post(server_url, "/api/record/review", {"note": note}) == (200, {"note": note})
    assert post(server_url, "/api/record/transcript",
                {"audio": "Attachments/_incoming/none.webm"})[1]["pending"] == 1
    assert post(server_url, "/api/record/cancel", {"job_id": "nobody"}) == \
        (200, {"cancelling": False})
    status, body = post(server_url, "/api/record/apply", {
        "note": note, "dest_dir": "Capture/school", "type_tag": "lecture", "topics": [],
        "summary_md": "s"})
    assert status == 200 and body["note"].startswith("Capture/school/")
    for path in ("/api/record/notes", "/api/record/review", "/api/record/apply"):
        assert post(server_url, path, {}) == (400, {"error": "note is required"})
    assert post(server_url, "/api/record/transcript", {})[0] == 400
    assert post(server_url, "/api/record/cancel", {})[0] == 400


def _claimed_length(url, path, length):
    """A request that only CLAIMS a body this long: the cap is checked before anything is read."""
    port = int(url.rsplit(":", 1)[1])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(length))
        conn.endheaders(b"{}")
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_cancel_keeps_the_small_body_cap_and_the_others_the_large_one(server_url):
    assert _claimed_length(server_url, "/api/record/cancel", 100_000) == \
        (400, {"error": "expected a JSON body under 64 kB"})
    for path in ("/api/record", "/api/record/append", "/api/record/summary", "/api/record/notes",
                 "/api/record/apply", "/api/record/review", "/api/record/transcript",
                 "/api/record/live"):
        assert _claimed_length(server_url, path, 1_000_001) == \
            (400, {"error": "expected a JSON body under 1000 kB"}), path
