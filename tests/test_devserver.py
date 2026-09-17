"""The dev-mode reloader for `slim chat`. What must never silently break:

1. The watch is scoped to the package. `.venv/` and other large trees live under the repo
   root, so a root-scoped walk wastes most of every polling tick. `snapshot` is handed one
   root and looks at .py only.
2. A save is DETECTED — a rewrite, a new file, and a deletion all have to register, because
   the whole value of the feature is that no manual kill is needed.
3. The child is never passed --reload, which is what makes self-supervision impossible by
   construction rather than by an environment-variable guard.
"""

import sys

import pytest

from slim import devserver


def _touch(path, text="x = 1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- what the watcher sees --------------------------------------------------------------

def test_snapshot_covers_python_files_at_any_depth(tmp_path):
    _touch(tmp_path / "chat.py")
    _touch(tmp_path / "nested" / "deep" / "llm.py")

    seen = devserver.snapshot(tmp_path)

    assert set(seen) == {tmp_path / "chat.py", tmp_path / "nested" / "deep" / "llm.py"}


def test_snapshot_ignores_pycache_and_non_python_files(tmp_path):
    _touch(tmp_path / "chat.py")
    _touch(tmp_path / "__pycache__" / "chat.cpython-313.pyc", "junk")
    _touch(tmp_path / "notes.md", "# not code")
    _touch(tmp_path / "projects.yaml", "a: 1")

    assert set(devserver.snapshot(tmp_path)) == {tmp_path / "chat.py"}


def test_snapshot_of_a_missing_root_is_empty_not_an_error(tmp_path):
    assert devserver.snapshot(tmp_path / "gone") == {}


# --- what counts as a change ------------------------------------------------------------

def test_an_unchanged_tree_reports_nothing(tmp_path):
    _touch(tmp_path / "chat.py")
    before = devserver.snapshot(tmp_path)

    assert devserver.changed(before, devserver.snapshot(tmp_path)) == []


def test_a_rewrite_is_detected(tmp_path):
    path = _touch(tmp_path / "chat.py")
    before = devserver.snapshot(tmp_path)
    # Written, not slept for: mtime_ns resolution is finer than any edit cadence, but the
    # test must not depend on that, so the timestamp is moved explicitly.
    path.write_text("x = 2\n")
    after = dict(devserver.snapshot(tmp_path))
    after[path] = before[path] + 1

    assert devserver.changed(before, after) == [path]


def test_a_new_file_is_detected(tmp_path):
    _touch(tmp_path / "chat.py")
    before = devserver.snapshot(tmp_path)
    _touch(tmp_path / "devserver.py")

    assert devserver.changed(before, devserver.snapshot(tmp_path)) == [
        tmp_path / "devserver.py"]


def test_a_deletion_is_detected(tmp_path):
    _touch(tmp_path / "chat.py")
    doomed = _touch(tmp_path / "gone.py")
    before = devserver.snapshot(tmp_path)
    doomed.unlink()

    assert devserver.changed(before, devserver.snapshot(tmp_path)) == [doomed]


# --- the child cannot supervise itself --------------------------------------------------

def test_the_child_is_launched_without_reload(monkeypatch):
    captured = {}
    monkeypatch.setattr(devserver.subprocess, "Popen", lambda argv: captured.setdefault(
        "argv", argv))

    devserver._spawn("127.0.0.1", 7546)

    assert "--reload" not in captured["argv"]
    assert captured["argv"][:5] == [sys.executable, "-u", "-m", "slim", "chat"]
    assert captured["argv"][5:] == ["--host", "127.0.0.1", "--port", "7546"]


def test_a_plugin_managed_child_receives_the_obsidian_pid(monkeypatch):
    captured = {}
    monkeypatch.setattr(devserver.subprocess, "Popen", lambda argv: captured.setdefault(
        "argv", argv))

    devserver._spawn("127.0.0.1", 7546, parent_pid=4321)

    assert captured["argv"][-2:] == ["--parent-pid", "4321"]


def test_reload_supervisor_stops_its_child_when_obsidian_exits(monkeypatch):
    child = object()
    stopped = []
    monkeypatch.setattr(devserver, "_spawn", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(devserver, "_stop", stopped.append)
    monkeypatch.setattr(devserver, "_process_alive", lambda _pid: False)
    monkeypatch.setattr(devserver.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(devserver, "snapshot", lambda _root: {})

    devserver.supervise(parent_pid=4321)

    assert stopped == [child]


@pytest.mark.parametrize("parent_pid", [0, -1])
def test_reload_supervisor_rejects_a_nonpositive_parent(parent_pid):
    with pytest.raises(ValueError, match="parent pid must be positive"):
        devserver.supervise(parent_pid=parent_pid)


def test_the_watch_root_is_the_package_not_the_repo():
    # The repo root holds .venv/ and other large trees; walking it per tick is much slower.
    assert devserver.PACKAGE_ROOT.name == "slim"
    assert (devserver.PACKAGE_ROOT / "chat.py").exists()
    assert not (devserver.PACKAGE_ROOT / ".venv").exists()
