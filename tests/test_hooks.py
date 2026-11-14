"""The session-start hooks."""

import json
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "stale-server-check.sh"


def test_the_stale_hook_names_a_file_in_a_subfolder():
    """`newest_file` is a path under `slim/` ("recorder/pipeline.py"); the hook's sed must keep
    the whole of it, slash included."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    body = json.dumps({"ok": True, "stale": True, "started_at": "2026-10-14T09:00:00+00:00",
                       "newest_file": "recorder/pipeline.py"}).encode()

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        hook = HOOK
        out = subprocess.run(["/bin/bash", str(hook)], capture_output=True, text=True,
                             env={"SLIM_CHAT_PORT": str(httpd.server_address[1]),
                                  "PATH": "/usr/bin:/bin:/usr/sbin"})
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert out.returncode == 0
    assert "and recorder/pipeline.py has been saved since" in out.stdout
