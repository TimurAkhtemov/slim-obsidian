"""The one bug in this repo that can DESTROY DATA. Every test here tries to destroy a
transcript and asserts that it couldn't.

`write_meeting` used to call `out_path.write_text(content)` unconditionally. §17's cutover
sequence re-runs this pull IMMEDIATELY BEFORE the Notion subscription is cancelled — so a
bad render or a partial API response during that final run would have overwritten the
archive at the exact moment the vault became the only copy of it in existence.

The index stores a HASH, not content. There was no way back.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pull_notion_meetings import (CAPTURE_REL, SHRINK_FLOOR,  # noqa: E402
                                          TranscriptShrank,
                                          body_of, write_meeting)

BLOCK = {
    "id": "2f87c804-1111-2222-3333-444444444444",
    "created_time": "2026-01-30T10:00:00.000Z",
    "last_edited_time": "2026-01-30T10:00:00.000Z",
    "transcription": {"title": [{"plain_text": "Power BI Overview"}]},
}
BODY = "## Transcript\n\n" + ("the quick brown fox jumped over the lazy dog. " * 200)


def _write(vault, body, block=None, **kw):
    return write_meeting(vault, block or BLOCK, body, "Parent", dry_run=False, **kw)


def test_first_write_creates(tmp_path):
    path, what = _write(tmp_path, BODY)
    assert what == "new" and path.exists()
    assert BODY in path.read_text()


def test_identical_content_is_not_rewritten(tmp_path):
    """Notion touches last_edited_time without changing content all the time. Rewriting then
    would churn imported_at and archive a version for nothing."""
    path, _ = _write(tmp_path, BODY)
    before = path.read_text()
    touched = dict(BLOCK, last_edited_time="2026-07-14T09:00:00.000Z")
    path2, what = _write(tmp_path, BODY, touched)
    assert what == "unchanged"
    assert path2.read_text() == before, "not one byte may move"
    assert not (tmp_path.joinpath(*CAPTURE_REL) / ".versions").exists(), "nothing to archive"


def test_a_real_revision_preserves_the_old_bytes(tmp_path):
    """THE FIX. A changed transcript is archived BEFORE it is replaced. Every version of
    every transcript survives forever."""
    path, _ = _write(tmp_path, BODY)
    original = path.read_text()

    longer = BODY + "\nand then somebody said one more thing.\n"
    path, what = _write(tmp_path, longer)
    assert what == "revised"
    assert "one more thing" in path.read_text(), "the new version is live"

    archived = list((tmp_path.joinpath(*CAPTURE_REL) / ".versions").glob("*.md"))
    assert len(archived) == 1
    assert archived[0].read_text() == original, "the OLD bytes survive, exactly"


def test_a_shrinking_transcript_is_REFUSED(tmp_path):
    """THE CATASTROPHE. A 9,000-char transcript comes back as a stub because the API
    returned partial blocks. Before the fix this silently replaced the only copy."""
    path, _ = _write(tmp_path, BODY)
    good = path.read_text()

    with pytest.raises(TranscriptShrank, match="REFUSING to write"):
        _write(tmp_path, "## Transcript\n\n(unintelligible)\n")

    assert path.read_text() == good, "the good transcript MUST be untouched on refusal"


def test_an_empty_body_is_refused(tmp_path):
    path, _ = _write(tmp_path, BODY)
    good = path.read_text()
    with pytest.raises(TranscriptShrank):
        _write(tmp_path, "")
    assert path.read_text() == good


def test_shrink_floor_boundary(tmp_path):
    """A small trim is a legitimate edit; a big one is a fetch failure. The line is drawn,
    and it is drawn generously toward keeping data."""
    _write(tmp_path, BODY)
    n = len(body_of(f"---\nx: 1\n---\n\n# T\n\n{BODY}\n").strip())

    just_over = "x" * int(n * (SHRINK_FLOOR + 0.05))
    _, what = _write(tmp_path, just_over)
    assert what == "revised", "a modest trim is allowed through"

    with pytest.raises(TranscriptShrank):
        _write(tmp_path, "x" * int(n * (SHRINK_FLOOR - 0.30)))


def test_allow_shrink_is_an_explicit_override(tmp_path):
    """The escape hatch exists — but it must be asked for, and the old bytes are STILL
    archived. There is no path through this function that loses content."""
    path, _ = _write(tmp_path, BODY)
    original = path.read_text()

    path, what = _write(tmp_path, "## Transcript\n\nshort.\n", allow_shrink=True)
    assert what == "revised"
    archived = list((tmp_path.joinpath(*CAPTURE_REL) / ".versions").glob("*.md"))
    assert archived and archived[0].read_text() == original, \
        "even a deliberate shrink preserves the original"


def test_archive_is_content_addressed_so_reruns_do_not_pile_up(tmp_path):
    """Re-running the pull must not accumulate identical copies of the same old version."""
    _write(tmp_path, BODY)
    _write(tmp_path, BODY + "\nrevision one.\n")
    _write(tmp_path, BODY)                      # flip-flop back
    _write(tmp_path, BODY + "\nrevision one.\n")
    archived = list((tmp_path.joinpath(*CAPTURE_REL) / ".versions").glob("*.md"))
    assert len(archived) == 2, "two distinct contents -> two archived versions, not four"


def test_versions_dir_is_excluded_from_ingestion():
    """Archived transcripts are near-duplicates of live ones. Indexed, they would return the
    same meeting several times and let retrieval quote a STALE version as current.

    Exclusion is by NAME — a leading dot excludes nothing on its own, which is exactly the
    trap this asserts against."""
    from slim.config import EXCLUDED_DIRS
    assert ".versions" in EXCLUDED_DIRS


def test_body_of_ignores_frontmatter():
    """imported_at is stamped fresh every run. Comparing whole FILES would mark every
    transcript as changed and archive the entire vault on the first cold-manifest run."""
    a = "---\nimported_at: 2026-07-13T00:00:00\n---\n\n# T\n\nsame body\n"
    b = "---\nimported_at: 2026-07-14T23:59:59\n---\n\n# T\n\nsame body\n"
    assert body_of(a) == body_of(b)
    assert a != b


# --- the pagination bug: a dropped page looks EXACTLY like a short meeting ------------

def test_two_disagreeing_reads_are_refused(monkeypatch):
    """THE BUG THAT MUTILATED THE ARCHIVE.

    Notion paginates block children at 100/page and, under load, returns `has_more: false`
    on a page that is not the last. `paginate` believes it, the render comes back short, and
    NOTHING RAISES — a dropped page is indistinguishable from a short meeting.

    Measured: one onsite interview (207 children, the only meeting of 109 needing
    more than one page) was stored as 39,694 chars — two pages of three. Its true length is
    52,729. A quarter of a job interview had been missing since the day it was imported, and
    every ingest and every M4 number built on it used the truncated text.

    An immutable block must render identically twice. If it doesn't, we do not write."""
    from scripts import pull_notion_meetings as m
    calls = {"n": 0}

    def flaky(token, block):
        calls["n"] += 1
        return "full transcript, all three pages" if calls["n"] % 2 else "page one only"

    monkeypatch.setattr(m, "_render_once", flaky)
    with pytest.raises(m.UnstableRender, match="dropped a page"):
        m.render_meeting("tok", BLOCK)


def test_two_agreeing_reads_are_accepted(monkeypatch):
    from scripts import pull_notion_meetings as m
    monkeypatch.setattr(m, "_render_once", lambda t, b: "stable content")
    assert m.render_meeting("tok", BLOCK) == "stable content"


def test_an_unstable_render_never_reaches_the_disk(tmp_path, monkeypatch):
    """The guarantee that matters: an unreliable fetch cannot produce a file. Not a short
    one, not a partial one, not one that 'looks complete'."""
    from scripts import pull_notion_meetings as m
    path, _ = _write(tmp_path, BODY)
    good = path.read_text()

    seq = iter(["a" * 5000, "b" * 9000])
    monkeypatch.setattr(m, "_render_once", lambda t, b: next(seq))
    with pytest.raises(m.UnstableRender):
        m.render_meeting("tok", BLOCK)
    assert path.read_text() == good, "the existing transcript is untouched"


# --- Notion lies about has_more under load --------------------------------------------

def test_a_full_page_claiming_to_be_last_is_re_asked(monkeypatch):
    """THE ROOT CAUSE. Under a 265-page scan, Notion returned 100 results with
    `has_more: false` on a block that actually had 207 children — and returned the SAME lie
    on a second read, defeating the two-read quorum.

    A genuine final page is exactly full only when the child count is a multiple of 100.
    That shape is therefore suspicious, and we re-ask instead of believing it."""
    from scripts import pull_notion_meetings as m
    pages = [
        {"results": [{"i": i} for i in range(100)], "has_more": False},            # THE LIE
        {"results": [{"i": i} for i in range(100)], "has_more": True,
         "next_cursor": "c1"},                                                     # re-ask
        {"results": [{"i": i} for i in range(100, 207)], "has_more": False},       # the rest
    ]
    calls = []
    monkeypatch.setattr(m, "api", lambda t, p, *a, **k: (calls.append(p), pages.pop(0))[1])
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    got = list(m.paginate("tok", "/blocks/x/children?page_size=100"))
    assert len(got) == 207, "must recover all 207 children, not stop at the lie"


def test_an_honest_short_final_page_is_not_re_asked(monkeypatch):
    """The check must not cost a request on every normal block — only on the suspicious
    shape. A final page that is not exactly full is believed immediately."""
    from scripts import pull_notion_meetings as m
    pages = [{"results": [{"i": i} for i in range(7)], "has_more": False}]
    calls = []
    monkeypatch.setattr(m, "api", lambda t, p, *a, **k: (calls.append(p), pages.pop(0))[1])
    got = list(m.paginate("tok", "/blocks/x/children?page_size=100"))
    assert len(got) == 7 and len(calls) == 1, "no extra request for an obviously-final page"


def test_a_genuinely_full_last_page_still_terminates(monkeypatch):
    """A block with exactly 100 children IS a full final page. The re-ask confirms it and we
    stop — the guard must not loop forever on a legitimate multiple of the page size."""
    from scripts import pull_notion_meetings as m
    pages = [
        {"results": [{"i": i} for i in range(100)], "has_more": False},
        {"results": [{"i": i} for i in range(100)], "has_more": False},   # re-ask agrees
    ]
    monkeypatch.setattr(m, "api", lambda t, p, *a, **k: pages.pop(0))
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    assert len(list(m.paginate("tok", "/blocks/x/children?page_size=100"))) == 100
