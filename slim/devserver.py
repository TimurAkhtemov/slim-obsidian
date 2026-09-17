"""Dev-mode reloader for `slim chat`: watch slim/*.py, respawn the server on a save.

The whole mechanism: a parent process that stats the package tree and restarts a child. Stat
polling rather than watchfiles, which keeps the install to four production deps.

It watches .py ONLY — a `config/*.yaml` edit will not reload — and it does not drain in-flight
requests, which is why the Obsidian toggle defaults to off.

The child is spawned WITHOUT start_new_session so it stays in the parent's process group: the
plugin kills the server with `process.kill(-pid, SIGTERM)` and that must reach every layer. It
runs `sys.executable -m slim` rather than `uv run slim` for the same reason — a `uv` wrapper
would swallow the signal.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
POLL_INTERVAL = 0.4

# A child that dies this fast may have lost a race for the socket rather than being broken.
# ONE retry, not a ladder: a syntax error also crashes inside two seconds and is
# indistinguishable from here, so every extra attempt is another full traceback in the log
# for an error that retrying cannot fix.
FAST_CRASH_SEC = 2.0
RETRY_DELAY = 0.3


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def snapshot(root: Path) -> dict[Path, int]:
    """Modification times of every .py under root, keyed by path.

    Scoped to the package, never the repo: `.venv/` and other large trees live under
    the repo root, and walking them costs far more per tick than this package-only scan.
    """
    seen: dict[Path, int] = {}
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            seen[path] = path.stat().st_mtime_ns
        except OSError:
            continue          # deleted between the walk and the stat; next poll catches it
    return seen


def changed(before: dict[Path, int], after: dict[Path, int]) -> list[Path]:
    """Paths that were added, removed, or written since the previous snapshot."""
    return sorted(set(before) ^ set(after)
                  | {p for p in set(before) & set(after) if before[p] != after[p]})


def _spawn(host: str, port: int, parent_pid: int | None = None) -> subprocess.Popen:
    """Start the server. It is never passed --reload, so it cannot supervise itself."""
    # -u: the child's stdout is a pipe or a redirected file as often as a terminal, and
    # block buffering there swallows the startup banner until the process happens to die.
    argv = [sys.executable, "-u", "-m", "slim", "chat", "--host", host, "--port", str(port)]
    if parent_pid is not None:
        argv.extend(["--parent-pid", str(parent_pid)])
    return subprocess.Popen(argv)


def _stop(child: subprocess.Popen) -> None:
    """Terminate and WAIT. Waiting is what frees the socket before the next bind."""
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def supervise(host: str = "127.0.0.1", port: int = 7546,
              interval: float = POLL_INTERVAL, parent_pid: int | None = None) -> None:
    if parent_pid is not None and parent_pid <= 0:
        raise ValueError("parent pid must be positive")
    child = _spawn(host, port, parent_pid)
    started = time.monotonic()
    retried = False
    parked = False
    state = snapshot(PACKAGE_ROOT)
    # flush on every print: these lines interleave with the child's stderr, and a buffered
    # "fix it and save" arriving after the traceback it explains is worse than no message.
    say = lambda msg: print(msg, flush=True)
    say(f"reload: watching {PACKAGE_ROOT}/**/*.py — save to restart the server")

    # SIGTERM only. Python's default SIGTERM exits without running `finally`, so the
    # child would outlive us and hold the port. SIGINT needs no handler: it reaches the
    # whole foreground group anyway, and KeyboardInterrupt already unwinds through
    # `finally` below — installing one here would race the child's own clean exit.
    def _terminated(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminated)

    try:
        while True:
            time.sleep(interval)
            if parent_pid is not None and not _process_alive(parent_pid):
                say("reload: Obsidian exited — stopping")
                return
            fresh = snapshot(PACKAGE_ROOT)
            hits = changed(state, fresh)
            state = fresh

            if hits:
                names = ", ".join(str(p.relative_to(PACKAGE_ROOT)) for p in hits)
                say(f"reload: {names}")
                _stop(child)
                child = _spawn(host, port, parent_pid)
                started, retried, parked = time.monotonic(), False, False
                continue

            if child.poll() is None:
                continue

            if time.monotonic() - started < FAST_CRASH_SEC and not retried:
                say(f"reload: server exited at startup (code {child.returncode}) — retrying")
                time.sleep(RETRY_DELAY)
                child = _spawn(host, port, parent_pid)
                started, retried = time.monotonic(), True
            elif not parked:
                # A real error in the source. Park rather than restart-loop: the next
                # save is the retry, which is exactly the edit-fix-save cycle.
                parked = True
                say(f"reload: server is down (code {child.returncode}) — "
                    f"fix the error above and save to restart")
    except KeyboardInterrupt:
        say("\nstopped")
    finally:
        _stop(child)
