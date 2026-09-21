"""Transcript -> structured summary: overview, key points, decisions, actions, questions.

One JSON-grammar call to the resident model, plus `render` for the markdown. Called by
`chat.py` for the recorder's summary, its retry and its append. Writes nothing: the
caller decides where the markdown goes.

One schema for every kind of recording — nothing tells the model "this is a lecture",
so an empty `action_items` is the schema working, not a degenerate case.
"""

from dataclasses import dataclass, field

from . import llm, trace
from .config import OWNER

# The version tracks the SHAPE as well as the wording — a v3 summary is a flat list of key
# points and a v4 summary groups them — so summaries written under different versions are
# not comparable and must not be scored as if they were.
PROMPT_VERSION = "summarize-v6"

# Ollama compiles a JSON schema into a GBNF grammar, and a bounded string becomes an N-way
# character repetition rule: maxLength 1500 compiles, 2000 dies with "failed to parse grammar".
#
# EVERY free-text field is bounded, INCLUDING `overview`. Greedy decoding into a JSON grammar
# loops forever on an unbounded string: one model degenerated into "(unintelligible)
# (unintelligible)…" for 19,203 characters, never closing the JSON, on 48% of transcripts.
# num_predict caps the damage but the JSON still comes back truncated and unparseable — only
# the GRAMMAR can forbid the loop, and that is what maxLength does. The bound never binds on
# legitimate output: measured on 98 real transcripts, overviews run 279–719 chars.
SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string", "maxLength": 1200},
        # GROUPED, not flat: [{"topic": str, "points": [str]}].
        # ⚠ The extra nesting does NOT relax the bound. Both strings in here carry a
        # maxLength for the same reason `overview` does; the repetition trap does not care
        # how deep the field is.
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "maxLength": 80},
                    "points": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 300},
                    },
                },
                "required": ["topic", "points"],
            },
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "maxLength": 300},
                    "owner": {"type": "string", "maxLength": 80},
                },
                "required": ["task"],
            },
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
        },
    },
    "required": ["overview", "key_points", "decisions", "action_items", "open_questions"],
}

# The loop guard, second line of defence behind the maxLengths above. If the model runs
# away it hits this cap, the JSON comes back truncated, the parse fails, and summarize()
# returns a visible failure instead of a hang. It is a loop guard, not a length target —
# it must sit clear of legitimate output, or a truncated parse looks exactly like a model
# that cannot follow a schema. Grouped output pays for topic labels and LaTeX for
# backslashes, which is why it is 2400 and not the 1600 the flat list needed.
NUM_PREDICT = 2400

# Ollama bills a reasoning model's chain-of-thought against num_predict while returning it
# in a SEPARATE field from content — so with think=True and the cap left at NUM_PREDICT the
# model can spend the whole budget thinking and return content="" (measured on this exact
# trap: 6,964 chars of thinking, all 1600 tokens spent, empty content). A thinking pass must
# budget for the reasoning AND the JSON both.
THINKING_PREDICT_MULTIPLIER = 4

SYSTEM = f"""You summarize meeting transcripts for {OWNER}'s personal memory system. The
transcript is the ONLY source of truth. You have no knowledge of {OWNER}'s life outside it.

THE RESTRAINT RULE — it governs `action_items` and `decisions`, AND ONLY THOSE TWO:

Only record an action item if someone ACTUALLY COMMITTED to doing something. Only record a
decision if one was ACTUALLY MADE. If nobody committed to anything, `action_items` is an
empty list. If nothing was decided, `decisions` is an empty list.

⚠ RESTRAINT DOES NOT APPLY TO `key_points`. It exists to stop you inventing obligations that
do not exist. Leaving out what was actually discussed is a DIFFERENT failure, and applying
restraint there does real damage: a 37-minute engineering meeting about dashboard latency,
browser automation and file-sync paths was once summarized in two sentences with an empty
`key_points`, and everything technical in it was lost. Recording what was said is not
inventing anything. Say what happened.

AN EMPTY `action_items` OR `decisions` IS A CORRECT AND EXPECTED ANSWER. It is not a failure,
and it is not you doing a bad job. Many of these recordings are lectures, conference talks,
training sessions, interviews, or social calls. Those have NO action items and NO decisions,
by their nature. A lecture on support vector machines does not have action items. A catch-up
call about someone's weekend does not have action items.

That same lecture is FULL of key points, and you must write them. The two lists above are
about what people COMMITTED to; `key_points` is about what was SAID. Emptiness is correct for
the first pair and almost never correct for the second.

Do NOT invent an action item to fill the section. Do NOT convert a topic that was merely
DISCUSSED into a decision that was MADE. Do NOT turn "we should probably look at that
sometime" into a commitment. Do NOT list "continue the discussion" or "follow up as needed"
as action items — those are filler, and filler is worse than an empty list, because it
teaches the reader that the list means nothing.

Inventing an action item nobody agreed to is the single worst thing you can do here. It puts
a false obligation into {OWNER}'s memory, and they will act on it.

WHAT TO WRITE:
- overview: 2-4 sentences. What was this, who was involved, what was it about?
- key_points: THE SUBSTANCE, and usually the most valuable part of the note. Capture the
  technical content: systems and tools named, architectural options weighed, workflows
  described, data sources, file paths, performance problems, constraints, numbers, and the
  tradeoffs argued over. Include a point even when nothing was concluded — "considered X
  because Y, but Z was slow" is exactly what they will want back later. A working session with
  no decisions still has a dozen key points. An empty `key_points` on a substantive
  conversation is a bug, not restraint. Write each as a specific sentence, not a topic label:
  "Tableau extracts refresh hourly, which is too slow for the headcount view" beats
  "discussed Tableau".

  GROUP the points. Each group is a `topic` and the points that belong to it. Use the topics
  the recording itself moved through, in the order it moved through them — a lecture has the
  sections the lecturer taught, a meeting has the things the people talked about. You are
  ORGANISING what was said, not interpreting it: every point still comes from the transcript,
  and grouping never licenses a conclusion nobody reached.

  Three to seven groups for an hour. Fewer for something short — one group is correct for a
  five-minute recording about one thing. DO NOT INVENT A TOPIC to hold a single stray point;
  put it in the group it is closest to. A topic is a short label, not a sentence.
- decisions: only things that were genuinely settled.
- action_items: only genuine commitments. `owner` only if a specific person was named as
  responsible; omit it otherwise rather than guessing.
- open_questions: things explicitly left unresolved.

FORMULAS. Some of these recordings are technical lectures, and a formula is often the point.
Write it between dollar signs — $x$ inside a sentence, $$x$$ when it stands on its own — and
Obsidian will render it.

⚠ NEVER USE A BACKSLASH. No \\frac, no \\text, no \\mid, no \\sum, no \\alpha. Write plain
symbols instead: $P(positive | w_1, ..., w_n)$, $P(w_n | w_1 ... w_(n-1))$, $2/||w||$,
$w^T x + b = 0$, $f(x) -> y$. These render correctly and they survive.

When the speaker states a formula IN WORDS, prefer the notation to the paraphrase. "The
probability that the review is positive given word one through word n" is a stated formula:
write $P(positive | w_1, ..., w_n)$, not a sentence about it.

Two limits, and they matter more than the notation does:
- Only a formula the speaker ACTUALLY STATED. You may transcribe "the margin is two over the
  norm of w" as $2/||w||$. You may NEVER DERIVE, complete, correct or simplify a step the
  speaker skipped. A derivation you filled in yourself is invention wearing notation, and it
  is indistinguishable from the real thing once it is in their notes.
- Only a formula a key point DEPENDS ON. A lecture puts a dozen formulas on its slides in
  passing; they are not all load-bearing. If the point stands without the formula, leave the
  formula out.

SCREENSHOTS. When images are attached, they are screenshots pasted during the recording:
equations from slides, diagrams, whiteboard photos, or other visual material. Reference their
content in your summary where it adds substance — a formula shown on a slide is the same as
one stated aloud and deserves the same treatment. Do not describe the screenshot itself ("a
screenshot shows…") — describe what it CONTAINS, as if the speaker had said it.

The transcript is machine-generated from audio. It has no reliable punctuation, and it
contains misheard words — especially names of people and companies. Work with what is
there; do not speculate about what a garbled word might have been.

Some transcripts mark who spoke: a paragraph beginning **Me:** is {OWNER}, and one beginning
**Them:** is whoever was on the other end of the call — possibly several people, not told
apart. Those marks come from the recording itself and are exact. A transcript without them
has no speaker information at all.

THEIR NOTES, when present after the transcript, are what {OWNER} typed while listening: they
tell you what they noticed and how they spell names and terms. Use them for that. They are NOT
a filter: brief the whole recording at your own discretion, do not favour passages because
they match their notes, and do not treat a question in their notes as an open question."""


# THE JSON/LaTeX TRAP. Measured 2026-08-26: the model wrote `$P(\text{positive})$`, Ollama's
# grammar admitted it because `\t` IS a valid JSON escape, and json decoded it to a TAB, so the
# note received `$P(<tab>ext{positive})$` with no exception and no counter. Five LaTeX commands
# collide with JSON's escapes: \b \f \n \r \t (\beta, \frac, \nabla, \rho, \text). The prompt
# steers the model off backslashes, but a prompt is a request and this is the guarantee: a raw
# control character in a summary field is a command that lost its backslash, so putting the
# backslash back reverses the decode. \n is NOT repaired — a real newline is ordinary there.
_MANGLED = {"\t": "\\t", "\f": "\\f", "\b": "\\b", "\r": "\\r"}


def _unmangle_latex(value: str) -> str:
    # ⚠ ONLY WHERE MATH LIVES. A raw tab is a mangled `\t` when it sits inside a formula and an
    # ordinary tab everywhere else — a summary that lines something up with tabs should keep
    # them, not grow visible backslashes. `$` is the only place this project puts LaTeX, so it
    # is the only place the repair applies.
    if "$" not in value:
        return value
    for control, command in _MANGLED.items():
        value = value.replace(control, command)
    return value


def _repair(value):
    """_unmangle_latex over whatever shape the schema produced. The model can put notation in
    any free-text field, so the repair cannot be attached to one of them by hand."""
    if isinstance(value, str):
        return _unmangle_latex(value)
    if isinstance(value, list):
        return [_repair(v) for v in value]
    if isinstance(value, dict):
        return {k: _repair(v) for k, v in value.items()}
    return value


@dataclass
class Summary:
    """A meeting summary, plus whether the call produced a usable one."""
    overview: str = ""
    # [{"topic": str, "points": [str]}] — NOT a flat list of strings.
    key_points: list[dict] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    # Validity is a first-class property, not an exception: a failed call comes back as a
    # Summary the card can show, so the recorder can offer Retry instead of dying.
    valid: bool = True
    error: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def n_key_points(self) -> int:
        """POINTS, not groups. The traces record this number, and `len(key_points)` counts
        the groups — comparable-looking and wrong."""
        return sum(len(g.get("points") or []) for g in self.key_points)


# Bounded like suggest.build_card bounds them: notes are typed live and can run long, and
# the transcript still has to fit.
NOTES_CHARS = 4000

# Their retry steer. Short on purpose: it is a one-line field in the review card, not a second
# system prompt, and a long one would quietly outweigh the rules above it.
INSTRUCTIONS_CHARS = 500


def summarize(transcript: str, model: str | None = None,
              num_ctx: int | None = None, think: bool = False,
              on_activity=None, notes: str = "", instructions: str = "",
              images: list[str] | None = None) -> Summary:
    """Transcript -> structured summary. Transcripts only — never feed this a summary.

    `num_ctx` follows the ONE-SIZE-PER-MODEL rule: Ollama spawns a separate runner per
    (model, num_ctx), so a caller MUST pass the ctx that `model` already uses everywhere
    else. Default (None → RESIDENT_CTX) is the resident's one size.

    `think` DEFAULTS TO FALSE. The recorder's summary and retry pass True unconditionally.
    Thinking halves invention at an ~8x latency tax (2026-08-25, their call), and nobody
    waits on a summary. Do not flip this default anywhere a person is waiting."""
    user = f"Summarize this meeting transcript.\n\n{transcript}"
    if notes.strip():
        # After the transcript, so the source stays primary; see SYSTEM on what notes are for.
        user += f"\n\nHIS NOTES WHILE LISTENING (context, not a filter):\n{notes.strip()[:NOTES_CHARS]}"
    if instructions.strip():
        # LAST, so it is the most recent thing the model reads, and fenced by the one rule it
        # may not unlock. A steer about emphasis is exactly what they should be able to give;
        # "invent three action items" is not, because the action-item list is only worth
        # having while it is trustworthy.
        user += (f"\n\nTHEIR INSTRUCTIONS FOR THIS SUMMARY (how to organise and what to "
                 f"emphasise):\n{instructions.strip()[:INSTRUCTIONS_CHARS]}\n\n"
                 f"Follow them, except where they conflict with THE RESTRAINT RULE — never "
                 f"record a decision that was not made or an action item nobody committed to, "
                 f"whatever they ask for.")
    msg = {"role": "user", "content": user}
    if images:
        msg["images"] = images
    messages = [{"role": "system", "content": SYSTEM}, msg]
    # The cap must widen for thinking or think=True comes back with empty content — see
    # THINKING_PREDICT_MULTIPLIER above.
    predict = NUM_PREDICT * (THINKING_PREDICT_MULTIPLIER if think else 1)
    try:
        call = llm.chat_json_stream if on_activity is not None else llm.chat_json
        activity = ({
            "on_thinking": lambda chunk: on_activity("thinking", chunk),
            "on_delta": lambda chunk: on_activity("draft", chunk),
        } if on_activity is not None else {})
        result, stats = call(
            messages, SCHEMA, model=model,
            # The longest transcript in the corpus is ~17.5k tokens, so the resident's
            # window holds every one.
            num_ctx=num_ctx or llm.RESIDENT_CTX,
            num_predict=predict,
            # Always explicit: whether a model thinks BY DEFAULT is a per-model property,
            # and an unset `think` on qwen3.6 returns content="". See llm.chat.
            think=think,
            **activity,
        )
    except llm.LLMError as e:
        # A JSON-validity failure is RETURNED, not raised: the card offers Retry summary.
        return Summary(valid=False, error=str(e), stats={"model": model or llm.MODEL})

    stats["prompt_version"] = PROMPT_VERSION
    result = _repair(result)
    s = Summary(
        overview=result.get("overview", ""),
        key_points=result.get("key_points") or [],
        decisions=result.get("decisions") or [],
        action_items=[a for a in (result.get("action_items") or []) if a.get("task")],
        open_questions=result.get("open_questions") or [],
        stats=stats,
    )

    # Ollama truncates a prompt to num_ctx WITHOUT erroring. If that happened the model
    # summarized a transcript with its ending cut off, and the summary must not pass as good.
    if stats.get("ctx_saturated"):
        s.valid = False
        s.error = (f"ctx_saturated: prompt hit num_ctx={stats.get('num_ctx')} — the "
                   f"transcript was TRUNCATED and this summary is not scorable")

    trace.record("summarize", {
        "model": stats.get("model"),
        "prompt_version": PROMPT_VERSION,
        "transcript_words": len(transcript.split()),
        "n_images": len(images or []),
        "valid": s.valid,
        "error": s.error,
        "n_key_points": s.n_key_points,
        "n_decisions": len(s.decisions),
        "n_action_items": len(s.action_items),
        "n_open_questions": len(s.open_questions),
        "stats": stats,
    })
    return s


# Placeholders a model reaches for when the prompt says to omit the field. Printing any
# of them is worse than an empty owner, which already means "nobody was named".
_NO_OWNER = frozenset({"", "unnamed", "unknown", "n/a", "na", "none", "someone",
                       "unspecified", "tbd", "participant", "speaker"})


# Recordings whose "open questions" can only be the SPEAKER's: the summarizer reads the
# transcript, so for a lecture the field holds rhetorical questions they pose and answers and
# examples on a slide — which, under an "Open questions" heading, read as the owner's confusions
# (2026-08-26, the NLP lecture). A meeting's unresolved question is real information.
LECTURE_TYPES = frozenset({"lecture", "learning"})


OPEN_QUESTIONS_HEADING = "## Open questions"


def drop_open_questions(summary_md: str) -> str:
    """Remove the Open questions section from a summary that is already written.

    `render` decides this at GENERATION time from the model's suggested type. When they
    corrects that type on the review card the summary is already on disk, so the correction
    needs a way into the markdown that costs no model call. Deterministic, and it only ever
    removes a DERIVED section — the transcript and their notes are elsewhere.
    """
    lines = summary_md.split("\n")
    out, skipping = [], False
    for line in lines:
        if line.strip() == OPEN_QUESTIONS_HEADING:
            skipping = True
            continue
        # They can edit the summary by hand, so the section is not reliably last: any following
        # H2 ends it.
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).strip()


def render(s: Summary, *, open_questions: bool = True) -> str:
    """Summary -> the markdown that goes in the note's summary section.

    Empty sections are OMITTED, not printed with a placeholder: "Action Items: none" and no
    section at all say the same thing, but the first one nags.
    `open_questions=False` drops that section (LECTURE_TYPES)."""
    if not s.valid:
        return f"_summarization failed: {s.error}_"

    out = []
    if s.overview:
        out.append(s.overview)
    if s.key_points:
        # A group with no points prints nothing: an empty heading is worse than no heading,
        # and a `topic` the model left blank must not cost the points underneath it.
        blocks = []
        for group in s.key_points:
            points = [p for p in (group.get("points") or []) if str(p).strip()]
            if not points:
                continue
            topic = (group.get("topic") or "").strip()
            blocks.append((f"### {topic}\n" if topic else "")
                          + "\n".join(f"- {p}" for p in points))
        if blocks:
            out.append("## Key points\n\n" + "\n\n".join(blocks))
    if s.decisions:
        out.append("## Decisions\n" + "\n".join(f"- {d}" for d in s.decisions))
    if s.action_items:
        lines = []
        for a in s.action_items:
            owner = a.get("owner", "").strip()
            # The prompt says to omit `owner` when nobody was named; a model that fills it with
            # "unnamed" has technically complied and produced "— **unnamed**", which is the
            # filler the Restraint Rule exists to prevent. Absence already says "nobody named".
            if owner.lower().strip(".") in _NO_OWNER:
                owner = ""
            lines.append(f"- [ ] {a['task']}" + (f" — **{owner}**" if owner else ""))
        out.append("## Action items\n" + "\n".join(lines))
    if s.open_questions and open_questions:
        out.append("## Open questions\n" + "\n".join(f"- {q}" for q in s.open_questions))
    return "\n\n".join(out)
