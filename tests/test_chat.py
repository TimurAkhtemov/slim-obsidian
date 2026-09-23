"""The local API server behind the Obsidian plugin. What must never silently break:

1. The bind refusal — this server can serve Journal content to the plugin, so a non-loopback
   bind is refused BEFORE a socket exists, and a spoofed Host header (DNS rebinding, the one
   way a remote page reaches a loopback server) is refused before routing.
2. The stale-server guard — `GET /api/health` reports whether the running process predates
   the newest edit under `slim/`, and it never refuses to serve.

The route contracts themselves live beside their callers: `test_record_endpoint.py` for
`/api/record*` and `test_copilot_endpoint.py` for `/api/copilot*`.
"""

from types import SimpleNamespace

import pytest

from slim import chat, cli


# --- loopback-only, by construction ---------------------------------------------------------

@pytest.mark.parametrize("host", ["0.0.0.0", "", "192.168.1.23", "10.0.0.5", "example.com"])
def test_non_loopback_bind_is_refused_before_a_socket_exists(host):
    with pytest.raises(chat.ChatError):
        chat.make_server(host, 0)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_binds_are_accepted(host):
    server = chat.make_server(host, 0)
    server.server_close()


def test_validate_accepts_the_whole_loopback_range():
    # macOS only has the 127.0.0.1 alias configured by default, so don't BIND to .5 —
    # but the validator must not be narrower than "loopback".
    chat.validate_bind_host("127.0.0.5")


@pytest.mark.parametrize("value,ok", [
    ("127.0.0.1:7546", True),
    ("localhost:7546", True),
    ("localhost", True),
    ("[::1]:7546", True),
    ("evil.example.com", False),          # DNS rebinding: attacker domain -> 127.0.0.1
    ("evil.example.com:7546", False),
    ("192.168.1.10:7546", False),
    (None, False),
    ("", False),
])
def test_host_header_gate(value, ok):
    assert chat._host_header_allowed(value) is ok


# --- the stale-server guard (2026-08-23) -----------------------------------------------------
# "The plugin owns the server's lifecycle" is not true: a server started the previous afternoon
# survived a full Cmd+Q and relaunch, the plugin REUSED it, and two recordings failed against
# code that had already been fixed on disk.

def test_a_server_older_than_the_code_reports_itself_stale(monkeypatch):
    monkeypatch.setattr(chat, "SERVER_STARTED", 0.0)     # started at the epoch
    state = chat.code_state()
    assert state["stale"] is True
    assert state["newest_file"].endswith(".py")


def test_a_server_started_after_the_last_edit_is_not_stale(monkeypatch):
    import time as _time
    monkeypatch.setattr(chat, "SERVER_STARTED", _time.time() + 3600)
    assert chat.code_state()["stale"] is False


def test_health_never_refuses_to_serve(monkeypatch):
    """⚠ Reporting it is the whole job. Refusing on stale would turn a warning into an outage
    mid-recording — the failure this is meant to prevent, not cause."""
    monkeypatch.setattr(chat, "SERVER_STARTED", 0.0)
    assert chat.code_state()["ok"] is True


def test_managed_health_reports_the_obsidian_parent(monkeypatch):
    monkeypatch.setattr(chat, "_process_alive", lambda pid: pid == 4321)
    state = chat.code_state(parent_pid=4321)
    assert state["managed_parent_pid"] == 4321
    assert state["managed_parent_alive"] is True


def test_terminal_health_has_no_managed_parent():
    state = chat.code_state()
    assert state["managed_parent_pid"] is None
    assert state["managed_parent_alive"] is None


def test_parent_watch_stops_the_server_when_obisidian_exits(monkeypatch):
    stopped = threading.Event()
    server = type("Server", (), {"shutdown": stopped.set})()
    monkeypatch.setattr(chat, "_process_alive", lambda _pid: False)

    worker = chat._stop_with_parent(server, 4321, interval=0.001)

    worker.join(timeout=1)
    assert stopped.is_set()


def test_a_missing_managed_parent_refuses_to_bind(monkeypatch):
    monkeypatch.setattr(chat, "_process_alive", lambda _pid: False)
    with pytest.raises(chat.ChatError, match="parent process 4321 is not running"):
        chat.serve("127.0.0.1", 0, parent_pid=4321)


@pytest.mark.parametrize("parent_pid", [0, -1])
def test_a_managed_parent_pid_must_be_positive(parent_pid):
    with pytest.raises(chat.ChatError, match="parent pid must be positive"):
        chat.make_server("127.0.0.1", 0, parent_pid=parent_pid)


def test_chat_cli_passes_the_managed_parent_to_the_server(monkeypatch):
    called = {}
    monkeypatch.setattr(cli, "_require_vault_dir", lambda: None)
    monkeypatch.setattr(cli.db, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "require_vault_identity", lambda _con: None)
    monkeypatch.setattr(chat, "serve", lambda host, port, *, parent_pid: called.update(
        host=host, port=port, parent_pid=parent_pid))

    cli.cmd_chat(SimpleNamespace(host="127.0.0.1", port=7654, reload=False,
                                 parent_pid=4321))

    assert called == {"host": "127.0.0.1", "port": 7654, "parent_pid": 4321}


@pytest.mark.parametrize("parent_pid", [0, -1])
def test_chat_cli_rejects_a_nonpositive_parent(parent_pid):
    with pytest.raises(SystemExit, match="--parent-pid must be positive"):
        cli.cmd_chat(SimpleNamespace(host="127.0.0.1", port=7546, reload=False,
                                     parent_pid=parent_pid))


# --- the two gates, END TO END ---------------------------------------------------------------
# `_host_header_allowed` and `_refuse_cross_site_write` are predicates. What must not silently
# break is that `do_GET`/`do_POST` actually CALL them before routing — measured 2026-08-27: with
# the Host wiring patched out, the predicate tests above stayed green while a spoofed request
# was served. So these drive a real loopback server, not the predicates.

import json
import threading
import urllib.error
import urllib.request

RECORD_WRITE_ROUTES = [
    "/api/record", "/api/record/live", "/api/record/apply", "/api/record/review",
    "/api/record/notes", "/api/record/summary", "/api/record/cancel",
    "/api/record/transcript", "/api/record/append",
]


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


def _status(url, path, *, method="GET", body=None, headers=None) -> int:
    request = urllib.request.Request(url + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_a_spoofed_host_header_is_refused_before_routing(server_url):
    """DNS rebinding: the attacker's domain resolves to 127.0.0.1, so their page's requests
    arrive with THEIR Host. Refused on GET and POST alike, before any handler runs."""
    assert _status(server_url, "/api/health") == 200
    assert _status(server_url, "/api/health", headers={"Host": "evil.example.com"}) == 403
    body = json.dumps({"note": "Capture/_unfiled/x.md", "notes_md": "x"}).encode()
    assert _status(server_url, "/api/record/notes", method="POST", body=body, headers={
        "Host": "evil.example.com", "Content-Type": "application/json"}) == 403


def test_every_record_write_route_runs_the_cross_site_gate(server_url):
    """A blind cross-site POST cannot read the response, but a write does not need reading —
    and until 2026-08-27 only the four copilot routes were gated while all nine record routes
    (which overwrite notes and move files) were not. Every one is refused on a foreign Origin
    (403) and on a no-preflight content type (415), before the body is read."""
    body = json.dumps({"session_id": "s", "action": "nope"}).encode()
    for path in RECORD_WRITE_ROUTES:
        assert _status(server_url, path, method="POST", body=body, headers={
            "Content-Type": "application/json", "Origin": "https://evil.example.com"}) == 403, path
        assert _status(server_url, path, method="POST", body=body, headers={
            "Content-Type": "text/plain"}) == 415, path
    # The plugin's own shape — JSON, no Origin (Electron's requestUrl / Node http) — gets past
    # the gate and into the handler: a bad live action is the handler's 400, not the gate's.
    assert _status(server_url, "/api/record/live", method="POST", body=body, headers={
        "Content-Type": "application/json"}) == 400


# --- _extract_note_images -----------------------------------------------------------------

def _make_test_image(path, width=100, height=80, mode="RGB"):
    """Create a minimal real image file for testing."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, (width, height), color="red")
    img.save(str(path))


def test_extract_parses_recorder_embeds(tmp_path):
    from slim.chat import _extract_note_images
    img = tmp_path / "Attachments" / "Recorder" / "abc123" / "paste-001.png"
    _make_test_image(img)
    notes = "some notes\n![[Attachments/Recorder/abc123/paste-001.png]]\nmore notes"
    cleaned, images = _extract_note_images(notes, tmp_path)
    assert len(images) == 1
    assert "![[" not in cleaned
    assert "some notes" in cleaned
    assert "more notes" in cleaned


def test_extract_sends_an_embed_from_any_folder(tmp_path):
    from slim.chat import _extract_note_images
    img = tmp_path / "Attachments" / "other.png"
    _make_test_image(img)
    notes = "text\n![[Attachments/other.png]]\nmore"
    cleaned, images = _extract_note_images(notes, tmp_path)
    assert len(images) == 1
    assert "![[" not in cleaned


def test_extract_finds_a_pasted_screenshot_by_name_the_way_obsidian_does(tmp_path):
    """⚠ The real shape (2026-09-22): Obsidian's own paste writes a BARE name, often with a size,
    into `./Attachements` beside wherever the note was at the time — and filing then moves the
    note away from it. Matching only `Attachments/Recorder/…` sent no image to any summary,
    ever: 43 recorded notes had screenshots and every trace said `n_images: 0`."""
    from slim.chat import _extract_note_images
    _make_test_image(tmp_path / "Capture/_unfiled/Attachements/Screenshot 2026-09-17 at 7.59.13 AM.png")
    _make_test_image(tmp_path / "Capture/_unfiled/Attachements/Screenshot 2026-09-17 at 7.03.34 AM.png")
    notes = ("## Attention\n![[Screenshot 2026-09-17 at 7.59.13 AM.png|444]]\nsoftmax over scores\n"
             "![[Screenshot 2026-09-17 at 7.03.34 AM.png]]")
    cleaned, images = _extract_note_images(notes, tmp_path)
    assert len(images) == 2
    assert "Screenshot" not in cleaned and "softmax over scores" in cleaned


def test_extract_sends_an_image_embedded_twice_once(tmp_path):
    from slim.chat import _extract_note_images
    _make_test_image(tmp_path / "Attachements/shot.png")
    _, images = _extract_note_images("![[shot.png]]\n![[shot.png|300]]", tmp_path)
    assert len(images) == 1


def test_extract_never_reads_outside_the_vault(tmp_path):
    from slim.chat import _extract_note_images
    vault = tmp_path / "vault"
    vault.mkdir()
    _make_test_image(tmp_path / "secret.png")
    _, images = _extract_note_images("![[../secret.png]]", vault)
    assert images == []


def test_extract_does_not_search_obsidians_trash(tmp_path):
    from slim.chat import _extract_note_images
    _make_test_image(tmp_path / ".trash/gone.png")
    _, images = _extract_note_images("![[gone.png]]", tmp_path)
    assert images == []


def test_extract_skips_missing_files(tmp_path, monkeypatch):
    from slim.chat import _extract_note_images
    from slim import trace
    traced = []
    monkeypatch.setattr(trace, "record", lambda kind, data: traced.append((kind, data)))
    notes = "![[Attachments/Recorder/abc/paste-001.png]]"
    cleaned, images = _extract_note_images(notes, tmp_path)
    assert len(images) == 0
    assert any(d.get("skipped") for _, d in traced)


def test_extract_caps_at_15_images(tmp_path):
    from slim.chat import _extract_note_images
    lines = []
    for i in range(20):
        img = tmp_path / "Attachments" / "Recorder" / "r1" / f"paste-{i:03d}.png"
        _make_test_image(img)
        lines.append(f"![[Attachments/Recorder/r1/paste-{i:03d}.png]]")
    notes = "\n".join(lines)
    _, images = _extract_note_images(notes, tmp_path)
    assert len(images) == 15


def test_extract_resizes_large_images(tmp_path):
    from slim.chat import _extract_note_images
    import base64
    from io import BytesIO
    from PIL import Image
    img = tmp_path / "Attachments" / "Recorder" / "r1" / "paste-001.png"
    _make_test_image(img, width=3000, height=2000)
    notes = "![[Attachments/Recorder/r1/paste-001.png]]"
    _, images = _extract_note_images(notes, tmp_path)
    assert len(images) == 1
    decoded = base64.b64decode(images[0])
    result = Image.open(BytesIO(decoded))
    assert max(result.size) <= 1600


def test_extract_keeps_alpha_as_png(tmp_path):
    from slim.chat import _extract_note_images
    import base64
    from io import BytesIO
    from PIL import Image
    img = tmp_path / "Attachments" / "Recorder" / "r1" / "paste-001.png"
    _make_test_image(img, mode="RGBA")
    notes = "![[Attachments/Recorder/r1/paste-001.png]]"
    _, images = _extract_note_images(notes, tmp_path)
    decoded = base64.b64decode(images[0])
    result = Image.open(BytesIO(decoded))
    assert result.format == "PNG"
