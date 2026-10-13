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


# --- the note as it was before an editing run ------------------------------------------------

def test_a_copy_from_before_round_trips_byte_for_byte_and_its_ref_is_32_hex():
    text = "# Note\r\n\nkeeps é and a trailing space \n"
    ref = threads.save_before("chat-1", text)
    assert len(ref) == 32 and all(c in "0123456789abcdef" for c in ref)
    assert threads.load_before("chat-1", ref) == text
    assert threads.save_before("chat-1", text) != ref                     # one copy per run
    assert threads.load_before("chat-1", "0" * 32) is None
    assert (threads.THREADS_DIR / "chat-1.before" / f"{ref}.md").is_file()


@pytest.mark.parametrize("ref", ["../x", "0" * 31, "0" * 33, "G" * 32, "", None, "0" * 31 + "/"])
def test_a_bad_ref_is_refused_before_touching_the_filesystem(ref):
    with pytest.raises(threads.ThreadError, match="32"):
        threads.load_before("chat-1", ref)


@pytest.mark.parametrize("evil", ["../evil", "a/b", "..", ".hidden", None])
def test_a_copy_needs_a_valid_thread_id(evil):
    with pytest.raises(threads.ThreadError):
        threads.save_before(evil, "x")
    with pytest.raises(threads.ThreadError):
        threads.load_before(evil, "0" * 32)


def test_deleting_a_thread_deletes_its_copies_from_before():
    ref = threads.save_before("chat-1", "before")
    threads.save_copilot("chat-1", "t", [], source_id="src1", source_path="Notes/x.md",
                         reasoning_mode="quick")
    assert threads.delete_thread("chat-1") is True
    assert not (threads.THREADS_DIR / "chat-1.before").exists()
    assert threads.load_before("chat-1", ref) is None
    threads.save_before("orphan", "before")                              # copies without a thread
    assert threads.delete_thread("orphan") is True
