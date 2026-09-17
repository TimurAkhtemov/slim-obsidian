"""Tests for deterministic spoken routing (P2).

Pure functions, regex only — never a model. The spoken TYPE vocabulary itself is pinned in
`tests/test_voice_routing.py`; these cover determinism and the path invariants around it.
"""
import pytest

from slim import voicetags


# --- determinism ----------------------------------------------------------------------------

def test_equal_length_subject_names_resolve_the_same_in_every_process():
    """`sorted` is stable over a `set`, whose iteration order is hash-randomized per process.
    Length alone therefore left equal-length aliases tie-breaking differently in every
    interpreter — `interview` (job-search) and `workplace` (work) are both 9 chars — so two
    sweeps, being different processes, could file the same opening two ways."""
    reg = {"work": {"name": "Work", "aliases": ["workplace"]},
           "job-search": {"name": "Job Search", "aliases": ["interview"]}}
    order = voicetags.subject_names(reg)
    assert order == sorted(order, key=lambda p: (-len(p[0]), p[0], p[1]))
    # Same registry, shuffled insertion order — the resolution must not move.
    reshuffled = {"job-search": reg["job-search"], "work": reg["work"]}
    assert voicetags.subject_names(reshuffled) == order


def test_a_project_id_cannot_escape_the_two_segment_ceiling():
    """`registered_project` proves an id is IN the registry; it does not prove the id is a
    single path segment, and `folder_for` interpolates it directly. A per-course id like
    `school/cs201` is natural to add and yields three owned segments, with the ceiling test
    still green because it hardcodes a single-segment id. `..` resolves out of `Capture/` entirely."""
    assert voicetags.folder_for(voicetags.LECTURE, "school") == "Capture/school"
    for bad in ("school/cs201", "..", "../Journal", "./x", ""):
        if bad == "":
            assert voicetags.folder_for(voicetags.LECTURE, bad) == "Capture/_unfiled"
            continue
        with pytest.raises(ValueError, match="single path segment"):
            voicetags.folder_for(voicetags.LECTURE, bad)


def test_the_journal_tie_break_is_documented_as_unreachable():
    """Behaviour is deliberately UNCHANGED — this pins the claim the comment now makes, so it
    fails if the vocabulary ever grows a pattern that can share JOURNAL's offset. At that point
    the tie-break stops being dead code and the comment has to be rewritten, not the code."""
    journal_pattern = next(p for p, k in voicetags._TYPE_VOCAB if k == voicetags.JOURNAL)
    others = [p for p, k in voicetags._TYPE_VOCAB if k != voicetags.JOURNAL]
    for text in ("journal", "journaling", "journalling"):
        m = journal_pattern.search(text)
        assert m is not None
        for other in others:
            hit = other.search(text)
            assert hit is None or hit.start() != m.start(), (
                f"{other.pattern} can now tie with journal on {text!r} — the tie-break is live "
                f"again and voicetags.detect_type's comment is stale")


# --- THE INVARIANT (lived in tests/test_refile.py until refile was cut, 2026-09-01) ---------

def test_no_route_ever_has_more_than_two_path_segments():
    """Every type × every registered subject, plus no subject and an unknown one: exactly
    `Capture/<subject>` or the `Journal` floor. Nothing may add a third owned segment. Lose
    this test and expect reorg #4."""
    from slim import config
    subjects = [*config.registry(), None, "made-up"]
    for kind in [*voicetags.TYPE_FRONTMATTER, ""]:
        for subject in subjects:
            folder = voicetags.folder_for(kind, subject)
            parts = folder.split("/")
            assert parts[0] in (voicetags.NAMESPACE, voicetags.JOURNAL_FOLDER), folder
            assert len(parts) <= 2, f"{folder} has {len(parts)} segments"


def test_the_live_config_project_ids_are_all_single_path_segments():
    """A project id is not just a key — `folder_for` interpolates it into `Capture/<id>`, so
    it IS a path segment. The registry the owner actually ships is checked here so that a bad id
    is caught when it is added, before it reaches the filesystem."""
    import re

    from slim import config
    segment = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
    for pid in config.registry():
        assert segment.fullmatch(pid), f"{pid!r} would break Capture/<id>"
