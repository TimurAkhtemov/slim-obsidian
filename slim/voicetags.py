"""Deterministic spoken routing: what a recording IS (type) and what it is ABOUT (subject).

The owner opens a memo by saying what it is — "this is a journal", "meeting with Jordan about the
roster", "lecture note for school". That preamble is parsed in CODE from a small explicit
vocabulary, never handed to a model. `route` is called by the memo sweep (`voicememos`) and
by the recorder's filing card (`suggest`).

The rules that make it safe to run unattended: only the opening ~120 chars are examined, no
match routes nowhere new, the functions are pure, and spoken always beats a model's guess.
`routed_by` and `subject_by` record which layer decided.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from . import config

# Only the opening words carry the spoken tag. 120 chars is a few seconds of speech: long
# enough for a full declaration, short enough that a tag word used later in the memo's own
# content cannot retro-label the whole note.
OPENING_CHARS = 120

# --- TYPE: what kind of recording is this -----------------------------------------------
# ONE spoken vocabulary, and it is the whole classifier: explicit, ordered, small on purpose.
# EARLIEST match wins (see `detect_type`) — a recording lands in exactly one folder, so the
# type is single-valued, and `type:` and `tags:` are the same word by construction.
# A lecture is anything they TOOK IN — a course, a talk, a podcast, an article. `learning` was a
# separate type until 2026-09-22; nobody, model or owner, drew that line the same way twice.
JOURNAL, MEETING, LECTURE, IDEA, NOTE = (
    "journal", "meeting", "lecture", "idea", "note")

# The closed set of routable types. `enrich.TYPE_TAGS` IS this list — a type they SAID and a
# type the model inferred have to be the same kind of thing, or the inferred one cannot be
# routed on.
ROUTING_TYPES = (JOURNAL, MEETING, LECTURE, IDEA, NOTE)

_TYPE_VOCAB: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bjournal(ing|ling)?\b"), JOURNAL),
    (re.compile(r"\bmeeting\b|\bstand-?up\b|\bone[- ]on[- ]one\b|\b1:1\b|\bsync with\b"
                r"|\bcall with\b|\bon a call\b|\bdebrief\b|\binterview with\b"), MEETING),
    (re.compile(r"\blecture\b|\bclass note|\bcourse note|\bconference talk\b"
                r"|\bwatching a talk\b|\bstudy note"), LECTURE),
    (re.compile(r"\bproject idea\b|\b(an? )?idea for\b|\bbrainstorm"), IDEA),
    (re.compile(r"\bpodcast\b|\blistening to\b|\barticle\b|\bwatched a video\b"
                r"|\btakeaways? from\b"), LECTURE),
    (re.compile(r"\bnote to self\b|\bquick note\b|\bwork (thing|note)\b|\babout work\b"), NOTE),
]

# --- WHERE IT LANDS: `Capture/<subject>/<note>.md`, or the flat floor ----------------------
# SUBJECT is the ONLY axis, and that is not a taste call. A path is a single ordered sequence,
# so encoding N independent facts in it forces a Cartesian product: the previous scheme crossed
# type × subject × year and produced 11 folders for 116 recordings, with `school` in 6 of them
# (measured 2026-08-03). Type, year and course live in frontmatter, where they are queryable.
#
# TWO SEGMENTS IS THE CEILING, enforced rather than intended:
# tests/test_voicetags.py::test_no_route_ever_has_more_than_two_path_segments.
NAMESPACE = "Capture"

# The floor is its own top-level, a SIBLING of Capture/ rather than a hole punched inside it,
# so the protection is structural rather than a rule that has to out-rank another rule.
JOURNAL_FOLDER = "Journal"

# An unknown subject gets the sentinel, never a bare `Capture/note.md`. This is what makes the
# invariant EXACTLY two owned segments instead of "at most two".
# The leading underscore sorts it out of the way in Obsidian.
UNFILED = "_unfiled"

# How each type is spelled in a note's `type:` frontmatter. `meeting` keeps the historical
# `meeting-note` spelling so the 108 imported Notion transcripts and a freshly captured one are
# the same type to every query that already filters on it.
TYPE_FRONTMATTER = {
    JOURNAL: "journal",
    MEETING: "meeting-note",
    LECTURE: "lecture",
    IDEA: "idea",
    NOTE: "note",
}

# Inferred routing is live by the owner's call (2026-08-01): a misfile costs one drag. A
# journal-typed recording never routes out.


def detect_type(text: str) -> str | None:
    """The single spoken TYPE in the opening, or None when they did not say one.

    EARLIEST MATCH WINS, not first-in-list. List order gave `journal` — first in `_TYPE_VOCAB`,
    and a bare noun — veto power over every other type. Measured 2026-08-04, all of these typed
    `journal`:

        "Meeting with Sam about the journal tab redesign."     -> meeting
        "Lecture note on journaling as a research method."       -> lecture
        "Quick note: the journal view is broken."                -> note
        "I have an idea for the personal journal feature."       -> idea

    A journal-flavoured opening can still type as something else — one drag, accepted 2026-08-01.
    """
    opening = (text or "")[:OPENING_CHARS].lower()
    best: tuple[int, str] | None = None
    # Strict `<`: on an equal offset the earlier pattern in `_TYPE_VOCAB` keeps the win, so
    # list order is the tie-break and `journal` comes first.
    for pattern, kind in _TYPE_VOCAB:
        m = pattern.search(opening)
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), kind)
    return best[1] if best else None


def subject_names(registry: dict) -> list[tuple[str, str]]:
    """(spoken_name, project_id) pairs from the registry, longest name first.

    Longest-first stops a two-word alias being shadowed by a one-word one. The registry is
    the ONLY source of subjects, so a memo can never invent a project by naming a noun.
    """
    pairs: list[tuple[str, str]] = []
    for pid, entry in (registry or {}).items():
        entry = entry if isinstance(entry, dict) else {}
        names = [pid, *(entry.get("aliases") or []), entry.get("name") or ""]
        for name in names:
            name = str(name).strip().lower()
            if name:
                pairs.append((name, pid))
    # The secondary key is not cosmetic. `set` iteration order is hash-randomized per process
    # and `sorted` is stable, so length alone left equal-length names resolving DIFFERENTLY in
    # every interpreter: `interview` (job-search) and `analytics` (work) are both 9 chars, and
    # one opening returned job-search under PYTHONHASHSEED 0/2/5/6/7 and work under 1/3/4.
    # Every 5-minute sweep is a new process, so the same opening could land in two subject
    # folders on two days.
    return sorted(set(pairs), key=lambda p: (-len(p[0]), p[0], p[1]))


def detect_subject(text: str, registry: dict | None = None) -> str | None:
    """The registered project id named in the opening, or None.

    Closed-set: registry ids, aliases and display names only, never a model's guess.
    """
    registry = config.registry() if registry is None else registry
    opening = (text or "")[:OPENING_CHARS].lower()
    for name, pid in subject_names(registry):
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", opening):
            return pid
    return None


def subject_from_topics(topics, registry: dict | None = None) -> str | None:
    """A registered project id among the model's topic slugs, or None.

    The SUBJECT is never asked of the model directly: it proposes open-vocabulary topics and
    CODE decides whether any of them names a registered project, so the set stays closed.
    """
    registry = config.registry() if registry is None else registry
    known = {name: pid for name, pid in subject_names(registry)}
    for topic in (topics or []):
        pid = known.get(str(topic).strip().lower())
        if pid:
            return pid
    return None


class Route(NamedTuple):
    """One routing verdict. `subject_by` names WHICH layer decided, so a misfile is
    explainable."""

    folder: str
    subject: str | None
    kind: str
    routed_by: str
    subject_by: str


def route(text: str, *, default_dir: str,
          registry: dict | None = None,
          inferred_type: str | None = None,
          inferred_topics=None,
          inferred_project: str | None = None) -> Route:
    """Decide where one recording lands.

    `routed_by` is "spoken" (they said it), "inferred" (the model did), or "default" (neither, so
    nothing moved). `subject_by` is the SUBJECT's own provenance, a separate axis: `spoken` (they
    named the project), `topic` (a registered name appeared in the model's topics — still a
    literal match decided in code), `description` (the model matched the recording against the
    registered descriptions, the one layer that is a judgement), or `none`.

    **Spoken beats inferred, always**, and the layers are tried cheapest-first. Inferred
    routing is live by the owner's call (2026-08-01): a misfile costs one drag.

    `default_dir` is where the caller would have put it anyway, and is what an undecidable
    recording keeps. Nothing here writes; the caller owns the file.
    """
    registry = config.registry() if registry is None else registry
    kind = detect_type(text)
    subject = detect_subject(text, registry)
    routed_by = "spoken"
    subject_by = "spoken" if subject else "none"

    if kind is None and inferred_type in ROUTING_TYPES:
        kind, routed_by = inferred_type, "inferred"
    if subject is None:
        subject = subject_from_topics(inferred_topics, registry)
        if subject:
            subject_by = "topic"
    if subject is None:
        # Last resort, and the only judgement in the chain: the model read the recording
        # against the registered descriptions. Validated against the registry here rather
        # than trusted, so an invented id can never create a project (or a folder).
        subject = registered_project(inferred_project, registry)
        if subject:
            subject_by = "description"

    if kind is None:
        return Route(default_dir, subject, "", "default", subject_by)

    return Route(folder_for(kind, subject), subject, kind, routed_by, subject_by)


def registered_project(candidate, registry: dict | None = None) -> str | None:
    """`candidate` if it is a registered project id, else None.

    The gate on anything a model proposes as a subject. `enrich` already constrains its
    output to a grammar enum over these ids; this check is the redundant half on purpose,
    because a grammar that silently degrades would be the only thing between a hallucinated
    id and a new folder.
    """
    if not candidate:
        return None
    registry = config.registry() if registry is None else registry
    pid = str(candidate).strip().lower()
    return pid if pid in (registry or {}) else None


_SUBJECT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _one_segment(subject: str | None) -> str | None:
    """`subject` if it is a single safe path segment, else raise. None passes through."""
    if subject is None or subject == "":
        return None
    if not _SUBJECT_SEGMENT_RE.fullmatch(str(subject)):
        raise ValueError(
            f"project id {subject!r} is not a single path segment; it would break the "
            "two-segment ceiling under Capture/")
    return str(subject)


def folder_for(kind: str, subject: str | None) -> str:
    """The vault-relative folder for one (type, subject). Pure; creates nothing.

    Exactly two segments, always — `Capture/<subject>` or the flat floor. Type does not appear
    in the path; see the NAMESPACE note above.

    The subject is SHAPE-CHECKED here, not merely membership-checked upstream: a registry key
    of `school/cs201` would yield three owned segments, and `..` would resolve to the vault
    root, moving a recording out of `Capture/` forever.
    """
    if kind == JOURNAL:
        return JOURNAL_FOLDER          # a privacy floor, not a taxonomy slot
    return f"{NAMESPACE}/{_one_segment(subject) or UNFILED}"
