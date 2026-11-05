"""The labelling vocabulary the recorder shares: the closed set of types, the filing folders,
and the gate on a model-proposed subject.
"""
from __future__ import annotations

from .. import config

# --- TYPE: what kind of recording is this -----------------------------------------------
# Single-valued, and `type:` and `tags:` are the same word by construction.
# A lecture is anything they TOOK IN — a course, a talk, a podcast, an article. `learning` was a
# separate type until 2026-09-22; nobody, model or owner, drew that line the same way twice.
JOURNAL, MEETING, LECTURE, IDEA, NOTE = (
    "journal", "meeting", "lecture", "idea", "note")

# The closed set of types. `enrich.TYPE_TAGS` IS this list, so the model can only infer a
# type the code knows.
ROUTING_TYPES = (JOURNAL, MEETING, LECTURE, IDEA, NOTE)

# --- WHERE IT LANDS: `Capture/<subject>/<note>.md`, or the flat floor ----------------------
# SUBJECT is the ONLY axis, and that is not a taste call. A path is a single ordered sequence,
# so encoding N independent facts in it forces a Cartesian product: the previous scheme crossed
# type × subject × year and produced 11 folders for 116 recordings, with `school` in 6 of them
# (measured 2026-08-03). Type, year and course live in frontmatter, where they are queryable.
NAMESPACE = "Capture"

# The floor is its own top-level, a SIBLING of Capture/ rather than a hole punched inside it,
# so the protection is structural rather than a rule that has to out-rank another rule.
JOURNAL_FOLDER = "Journal"

# An unknown subject gets the sentinel, never a bare `Capture/note.md`. This is what makes the
# invariant EXACTLY two owned segments instead of "at most two".
# The leading underscore sorts it out of the way in Obsidian.
UNFILED = "_unfiled"


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
