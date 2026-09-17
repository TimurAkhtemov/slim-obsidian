"""The local thread store (chat-fluidity F5, north-star P1). What must never silently break:

- Per-thread delete is the store's one destructive operation, and it is traced — deleting
  conversational history is exactly what the trace log exists to remember happened.
- A bad id is refused before the filesystem is touched; a missing file is a valid outcome
  (idempotent delete), not an error.

The conftest fixtures already isolate THREADS_DIR and the trace directory at per-test tmp
paths, so these tests exercise the real store without reaching production state.
"""
import pytest

from slim import threads, trace


def test_delete_removes_the_file_and_reports_true():
    threads.save_copilot("keep-me", "A thread", [{"role": "you", "text": "hi"}],
                         source_id="src1", source_path="Notes/x.md",
                         reasoning_mode="quick")
    assert (threads.THREADS_DIR / "keep-me.json").exists()

    assert threads.delete_thread("keep-me") is True
    assert not (threads.THREADS_DIR / "keep-me.json").exists()


def test_delete_missing_thread_returns_false_not_an_error():
    assert threads.delete_thread("never-existed") is False


@pytest.mark.parametrize("evil", ["../evil", "a/b", "..", ".hidden", "x" * 65, "", None])
def test_delete_refuses_a_bad_id_before_touching_the_filesystem(evil):
    with pytest.raises(threads.ThreadError):
        threads.delete_thread(evil)


def test_delete_writes_a_trace_record_present_and_absent():
    threads.save_copilot("gone", "t", [], source_id="src1",
                         source_path="Notes/x.md", reasoning_mode="quick")

    threads.delete_thread("gone")
    entry = trace.last(1)[0]
    assert entry["kind"] == "thread_delete"
    assert entry["status"] == "deleted"
    assert entry["id"] == "gone"

    threads.delete_thread("gone")            # already absent — still traced, status flips
    entry = trace.last(1)[0]
    assert entry["kind"] == "thread_delete"
    assert entry["status"] == "absent"
    assert entry["id"] == "gone"
