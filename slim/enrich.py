"""Model-written title, type, topics and project for one transcript.

Writes frontmatter only: the transcript is raw testimony and stays byte-identical.
Provenance is always recorded — `tagged_by: spoken` when the deterministic opening
vocabulary matched, `llm` when this module classified. The type is a CLOSED enum,
`voicetags.ROUTING_TYPES`, so a type the model inferred can be routed on; topics are open
vocabulary, but code decides their form (slugged, bounded, deduped). Called by the
recorder's filing card and by the memo sweep.
"""

import re

from . import config, llm, voicetags
from .config import OWNER

PROMPT_VERSION = "enrich-v4"   # v4: + `project`, chosen against the registry's descriptions

# ON since 2026-07-20, by the owner, after reading real proposals for their own memos. Tests run
# with it OFF (conftest leaves the default) — enrichment must never put a live model call on
# a test's critical path.
ENABLED = True

# `None` means the resident default. A small model was measured against it on their own memos
# and labelled accurately 40% of the time to the resident's 50%, inventing ~14% of its tags.
# Nobody waits on a label, so the bigger model's latency buys that accuracy for free.
MODEL = None

# LITERALLY the vocabulary `voicetags` detects from spoken openings. A model-assigned type
# and a spoken one must be interchangeable, or the inferred one cannot be routed on.
TYPE_TAGS = list(voicetags.ROUTING_TYPES)

MAX_TOPICS = 5
MAX_TOPIC_CHARS = 30
MAX_TITLE_CHARS = 70
# Enough transcript to judge subject matter without paying to prefill a 40-minute memo.
# Head AND tail: a recording often names its subject at the start and its conclusion at the
# end, and the middle is the least informative part of a rambling memo.
HEAD_CHARS = 3000
TAIL_CHARS = 1000
NUM_PREDICT = 200          # a title and a few tags; the cap is the repetition guard

# The value the model returns when no registered project fits. THE ENUM MUST CARRY IT: a
# required enum field with no escape hatch means the JSON grammar FORCES a project onto every
# recording, including ones about nothing in the registry — grammar-guaranteed false positives
# rather than model error, and no amount of prompt wording can talk it out of a constrained
# decode. This is the null.
NO_PROJECT = "none"


def _schema(registry: dict | None = None) -> dict:
    """The output schema, with `project` constrained to the registered ids plus `none`.

    A function rather than a constant because the enum is derived from the registry: register
    a project and the grammar admits it on the next call, with no second list to update.
    """
    ids = sorted((registry if registry is not None else config.registry()) or {})
    return {
        "type": "object",
        "properties": {
            # No maxLength anywhere near Ollama's ~1500 grammar-compile ceiling, and every
            # free-text field is short by nature — `num_predict` is the real bound.
            "title": {"type": "string", "maxLength": MAX_TITLE_CHARS},
            "type_tag": {"type": "string", "enum": TYPE_TAGS},
            "topics": {
                "type": "array",
                "maxItems": MAX_TOPICS,
                "items": {"type": "string", "maxLength": MAX_TOPIC_CHARS},
            },
            "project": {"type": "string", "enum": [*ids, NO_PROJECT]},
        },
        "required": ["title", "type_tag", "topics", "project"],
    }


def _project_block(registry: dict | None = None) -> str:
    """The `project` instruction, listing each registered project and its description.

    Kept out of the static SYSTEM string because the candidates are config: the descriptions
    the owner already wrote in `config/projects.yaml` are the matching criteria, so there is one
    place to edit and no prompt copy to drift from it. Costs ~54 words at six projects.
    """
    registry = (registry if registry is not None else config.registry()) or {}
    lines = []
    for pid, entry in sorted(registry.items()):
        entry = entry if isinstance(entry, dict) else {}
        name = entry.get("name") or pid
        desc = " ".join(str(entry.get("description") or "").split())
        lines.append(f"  {pid} — {name}" + (f": {desc}" if desc else ""))
    return (
        f"\nproject — which of {OWNER}'s registered projects this recording belongs to, judged\n"
        "against the descriptions below. This is the LAST resort for filing: if they named the\n"
        f"project out loud, code already caught it. Answer `{NO_PROJECT}` whenever no project\n"
        "clearly fits — an honest `none` files the note as unsorted, which they can fix in one\n"
        "drag, while a wrong project buries it somewhere they will not think to look.\n"
        + "\n".join(lines)
    )

SYSTEM = f"""You label {OWNER}'s own voice recordings so they can find them later. They record
and forget — they will never do this by hand, so your labels are the only ones these notes
will ever have. Be concrete and be literal.

title — 4 to 7 words naming what this recording is ABOUT. A person reading it in a file
list should know whether to open it.
  GOOD: "Northwind round three debrief and next steps"
  GOOD: "Idea for tagging voice memos by spoken keyword"
  BAD:  "Voice memo" / "Personal thoughts" / "Recording about work"  (says nothing)
  BAD:  "{OWNER} discusses their feelings about the interview process"   (narrates, not names)
Use the words they actually used. Never invent a detail to make a title sound better, and
never carry over an example from these instructions.

type_tag — the FORM of the recording, not its subject. THIS DECIDES WHERE THE NOTE IS FILED,
so choose the form, never the topic. Decide in this order and stop at the first that fits:
  1. meeting   more than one voice: a call, an interview, a conversation. If people are
               talking to each other it is a `meeting` NO MATTER what it is about — a work
               call is a meeting, a recruiter call is a meeting.
  2. lecture   they are listening to someone teach: a course lecture, a conference talk, a
               tutorial. One voice explaining material, and it is not their own.
  3. idea      they propose or work out something to build or do, alone.
  4. journal   they reflect — how something went, how they feel, thinking out loud about their
               own life. When a recording is BOTH reflection and idea, prefer journal.
  5. learning  they are capturing something they CONSUMED — a podcast, an article, a video, a
               talk — and their takeaways from it. Distinct from `lecture`, which is a course.
  6. note      a short factual note to themself that is none of the above.
The subject belongs in `topics`, never here. "Acme survey meeting" is type `meeting`
with topics [acme, survey] — putting the subject here throws away what KIND of thing
it is and files it in the wrong place.

topics — up to 5 SPECIFIC subjects: the companies, people, projects, courses, and concrete
subjects the recording actually names. Lowercase words, no punctuation, no "#".
  GOOD: northwind, job search, school, slim, power bi, sam
  BAD:  thoughts, ideas, personal, misc, life, stuff  (these find nothing later)
Only what the recording genuinely covers. An empty list is correct for a recording with no
identifiable subject — never pad it to look thorough."""


def _slug(text: str) -> str:
    """One topic tag's canonical form. CODE decides shape; the model proposes meaning.

    Without this the same subject arrives four ways across four recordings and the tag
    stops gathering anything, which is the entire reason to have tags.
    """
    text = re.sub(r"[^a-z0-9]+", "-", (text or "").lower().strip())
    return re.sub(r"-{2,}", "-", text).strip("-")[:MAX_TOPIC_CHARS]


def normalize(result: dict, fallback_title: str, registry: dict | None = None) -> dict:
    """Bound and canonicalize the model's output. Never raises — a bad label must not cost
    a transcript, so anything unusable degrades to the deterministic fallback."""
    title = " ".join((result.get("title") or "").split())[:MAX_TITLE_CHARS].strip(' "')
    # A title that is empty, or that echoes a prompt example verbatim, is worse than the
    # timestamp it replaces: it would be confidently wrong on a file they cannot re-listen to
    # at a glance.
    if len(title) < 3:
        title = fallback_title

    type_tag = (result.get("type_tag") or "").strip().lower()
    if type_tag not in TYPE_TAGS:
        type_tag = ""

    topics, seen = [], set()
    raw = result.get("topics")
    for item in (raw if isinstance(raw, list) else []):
        slug = _slug(str(item))
        # Two-character slugs are noise ("ai", "hr" survive; "a", "" do not).
        if len(slug) < 2 or slug in seen or slug in TYPE_TAGS:
            continue
        seen.add(slug)
        topics.append(slug)
        if len(topics) >= MAX_TOPICS:
            break

    # Validated against the registry even though the grammar already constrained it. The enum
    # makes the closed set structural; this check is what survives a grammar that silently
    # degrades, and it is the only thing between a hallucinated id and a new folder.
    project = voicetags.registered_project(result.get("project"), registry) or ""

    return {"title": title, "type_tag": type_tag, "topics": topics, "project": project}


def _unlabelled(fallback_title: str) -> dict:
    """The deterministic fallback. Same keys as `normalize` so no caller has to know which
    path produced its labels."""
    return {"title": fallback_title, "type_tag": "", "topics": [], "project": ""}


def _excerpt(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return f"{text[:HEAD_CHARS]}\n\n[…]\n\n{text[-TAIL_CHARS:]}"


def enrich(transcript: str, fallback_title: str = "", model: str | None = None,
           num_ctx: int | None = None, registry: dict | None = None) -> tuple[dict, dict]:
    """Label one transcript. Returns (labels, stats); labels are always usable.

    `num_ctx` follows the ONE-SIZE-PER-MODEL rule: Ollama spawns a separate runner per
    (model, num_ctx), so a caller passing a bespoke size here would load a second copy of
    the weights and evict whatever was resident. Callers pass the size that model already
    uses everywhere else, or nothing at all.
    """
    if not ENABLED:
        return (_unlabelled(fallback_title), {"skipped": "enrichment disabled"})
    excerpt = _excerpt(transcript)
    if not excerpt:
        return (_unlabelled(fallback_title), {"skipped": "empty transcript"})
    registry = config.registry() if registry is None else registry
    try:
        result, stats = llm.chat_json(
            [{"role": "system", "content": SYSTEM + _project_block(registry)},
             {"role": "user", "content": f"RECORDING TRANSCRIPT:\n\n{excerpt}"}],
            _schema(registry), model=model, num_predict=NUM_PREDICT,
            # EXPLICIT, never inherited: an unset flag on a thinking model returns
            # content="". See llm.chat's docstring.
            think=False,
            **({"num_ctx": num_ctx} if num_ctx else {}))
    except llm.LLMError as e:
        # A labeling failure must never cost a transcript. The note still lands, with its
        # deterministic name and no model tags — visibly unlabelled rather than silently wrong.
        return (_unlabelled(fallback_title), {"error": str(e)})
    labels = normalize(result, fallback_title, registry)
    stats["prompt_version"] = PROMPT_VERSION
    return labels, stats
