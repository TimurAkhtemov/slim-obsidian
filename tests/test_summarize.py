"""Summarization tests — the DETERMINISTIC half. No model is called.

Same split as test_ask.py: the model's judgment (is this an action item?) is a matter of
measurement, not assertion. What belongs here is the machinery that must never silently
break — and every test below encodes a failure that was actually measured on the real
corpus, not one imagined:

- A truncated prompt must invalidate the summary, not quietly produce a worse one.
- Restraint is counted from `decisions` and `action_items` — and NOT from key_points,
  because a lecture legitimately has plenty of those.
"""

import re

from slim import summarize as summarize_mod
from slim.summarize import Summary, render


_OK = {"overview": "o", "key_points": [], "decisions": [],
       "action_items": [], "open_questions": []}


def test_ctx_saturation_invalidates_the_summary(monkeypatch):
    """Ollama truncates a prompt to num_ctx WITHOUT erroring — no exception, no warning,
    just a summary of a half-fed transcript. Scored naively it would look like a normal
    (if worse) summary, and its coverage and fabrication rates would be measuring the
    truncation rather than the model. It must come back INVALID."""
    monkeypatch.setattr(summarize_mod.llm, "chat_json",
                        lambda *a, **k: (_OK, {"ctx_saturated": True, "num_ctx": 32768}))
    s = summarize_mod.summarize("...")
    assert not s.valid
    assert "TRUNCATED" in s.error


def test_llm_failure_is_counted_not_raised(monkeypatch):
    """§23 gates on >=95% JSON validity, so a failure has to be COUNTABLE. A run that dies
    on one bad meeting tells you nothing; one that records which meeting failed is data."""
    def boom(*a, **k):
        raise summarize_mod.llm.LLMError("model returned non-JSON under a schema")
    monkeypatch.setattr(summarize_mod.llm, "chat_json", boom)
    s = summarize_mod.summarize("...")
    assert not s.valid and "non-JSON" in s.error


def test_every_free_text_field_is_bounded():
    """THE BUG THAT COST A MODEL ITS PLACE IN THE BAKE-OFF.

    `overview` was left unbounded, copying synthesize.SCHEMA's `answer`. Under a greedy JSON
    grammar that is a repetition trap (llm.py documents it), and gemma4:26b fell in: it
    emitted "(unintelligible) (unintelligible) …" for 19,203 chars, never closed the JSON,
    and failed 48% of transcripts. It looked like a model defect. It was ours.

    num_predict caps the damage but CANNOT prevent it — the JSON still comes back truncated.
    Only the grammar can forbid the loop. So every free-text field carries a maxLength, and
    this test exists so nobody quietly removes one again."""
    for name, field in summarize_mod.SCHEMA["properties"].items():
        target = field.get("items", field)
        if target.get("type") == "string":
            assert "maxLength" in target, f"{name} is an unbounded string — repetition trap"
            assert target["maxLength"] <= 1500, (
                f"{name}: Ollama cannot compile maxLength > ~1500 into GBNF")
    # and the bound must not bind on real output: qwen's overviews measured 279–719 chars.
    assert summarize_mod.SCHEMA["properties"]["overview"]["maxLength"] >= 1000


def test_one_context_size_is_used(monkeypatch):
    """A second num_ctx makes Ollama spawn a second runner and reload 19 GB of weights
    mid-run. Every call in the codebase must use RESIDENT_CTX."""
    seen = {}
    monkeypatch.setattr(summarize_mod.llm, "chat_json",
                        lambda *a, **k: (seen.update(k), (_OK, {}))[1])
    summarize_mod.summarize("...")
    assert seen["num_ctx"] == summarize_mod.llm.RESIDENT_CTX == 131072
    assert seen["num_predict"] == summarize_mod.NUM_PREDICT, "unbounded decode = infinite loop"


def test_invalid_summary_renders_as_a_failure_not_as_silence():
    """An empty render and a failed render look identical to a reader. They must not."""
    out = render(Summary(valid=False, error="ctx_saturated"))
    assert "failed" in out and "ctx_saturated" in out


def test_render_omits_empty_sections():
    """§3.6, quiet by default: a section reading 'Action Items: none' and a section that
    isn't there say the same thing, but the first one nags."""
    out = render(Summary(overview="A lecture.", key_points=_groups(("Topic", ["p"]))))
    assert "Action items" not in out and "Decisions" not in out
    assert "## Key points" in out


def test_render_marks_owner_only_when_named():
    out = render(Summary(action_items=[
        {"task": "Send the deck", "owner": "Dean"},
        {"task": "Track time"},
    ]))
    assert "- [ ] Send the deck — **Dean**" in out
    assert "- [ ] Track time\n" in out + "\n"


def test_thinking_is_disabled_explicitly(monkeypatch):
    """THE SECOND HARNESS BUG THAT LOOKED LIKE A MODEL FAILURE.

    Ollama returns a reasoning model's chain-of-thought in message.thinking — a separate
    field from message.content — but those tokens still bill against num_predict. So a
    thinking model spends the whole output budget reasoning and returns content="", and
    the caller sees "this model cannot produce valid JSON."

    Measured: qwen3.6:35b emitted 6,964 chars of thinking, burned all 1600 tokens, and
    returned EMPTY content on every transcript. With think=False: 375 tokens, 8.5s, valid.

    Worse, it is NON-UNIFORM — whether a model thinks by default is a per-model property.
    Leaving it unset hands each arm of a bake-off a different output budget."""
    seen = {}
    monkeypatch.setattr(summarize_mod.llm, "chat_json",
                        lambda *a, **k: (seen.update(k), (_OK, {}))[1])
    summarize_mod.summarize("...")
    assert seen["think"] is False, "thinking must be set explicitly, never left to the model"


# --- Task 4: retrieval index + thinking recorder pass --------------------------------

def test_thinking_raises_the_output_cap(monkeypatch):
    """⚠ Ollama bills thinking tokens against num_predict while returning them separately.
    Leave the cap alone and content comes back EMPTY — indistinguishable from a model that
    cannot produce valid JSON. This test is the guard on a documented rabbit hole."""
    from slim import llm
    seen = {}
    def fake(*a, **k):
        seen.update(k)
        return ('{"overview": "x", "key_points": [], "decisions": [],'
                ' "action_items": [], "open_questions": []}', {})
    monkeypatch.setattr(llm, "chat", fake)
    summarize_mod.summarize("t", think=False)
    baseline = seen["num_predict"]
    summarize_mod.summarize("t", think=True)
    assert seen["think"] is True
    assert seen["num_predict"] > baseline


def test_streaming_surfaces_real_reasoning_and_draft_chunks(monkeypatch):
    """The recorder's progress view must show Ollama output, not a decorative timer."""
    events = []

    def fake_stream(*args, on_thinking, on_delta, **kwargs):
        on_thinking("checking decisions")
        on_delta('{"overview":"')
        on_delta('A real draft"}')
        return dict(_OK), {"output_tokens": 42, "decode_tok_s": 17}

    monkeypatch.setattr(summarize_mod.llm, "chat_json_stream", fake_stream)
    out = summarize_mod.summarize(
        "transcript", think=True,
        on_activity=lambda kind, chunk: events.append((kind, chunk)),
    )

    assert out.valid is True
    assert events == [
        ("thinking", "checking decisions"),
        ("draft", '{"overview":"'),
        ("draft", 'A real draft"}'),
    ]
    assert out.stats["output_tokens"] == 42


def test_an_extra_field_in_the_response_is_ignored_rather_than_fatal(monkeypatch):
    """Callers must survive a response carrying a key the schema no longer has — `index_terms`
    was deleted in summarize-v5, and a stale or hand-fed payload must not crash the recorder."""
    monkeypatch.setattr(summarize_mod.llm, "chat_json",
                        lambda *a, **k: (dict(_OK, index_terms=["kernel trick"]), {}))
    s = summarize_mod.summarize("...")
    assert s.valid
    assert not hasattr(s, "index_terms")


def test_a_placeholder_owner_is_dropped_rather_than_printed():
    """The prompt says omit `owner` when nobody was named. A model that instead writes
    "unnamed" has complied on paper and produced "— **unnamed**" in the note, which is exactly
    the filler the Restraint Rule exists to prevent. Seen on a real meeting, 2026-08-05."""
    from slim import summarize
    s = summarize.Summary(action_items=[{"task": "Send the tables", "owner": "unnamed"},
                                        {"task": "Review it", "owner": "Unknown"},
                                        {"task": "Ship it", "owner": "Sarah"}])
    out = summarize.render(s)
    assert "unnamed" not in out.lower()
    assert "- [ ] Send the tables\n" in out + "\n"
    assert "— **Sarah**" in out          # a real name still survives


def test_key_points_are_rendered_when_nothing_was_decided():
    """A working session with no decisions still has substance. An empty `key_points` beside a
    populated transcript was the defect that lost a 37-minute engineering meeting."""
    from slim import summarize
    s = summarize.Summary(overview="A review.",
                          key_points=_groups(("Refresh cadence",
                                              ["Tableau refreshes hourly, too slow for headcount"])),
                          decisions=[], action_items=[])
    out = summarize.render(s)
    assert "## Key points" in out
    assert "## Decisions" not in out     # restraint still holds where it belongs


# --- 2026-08-25: their notes are context for the summary, never a filter on it -------------

def test_their_notes_are_context_after_the_transcript_not_a_second_source(monkeypatch):
    """They asked for both halves: the summarizer should know what they noted down, and it must
    still brief the WHOLE recording rather than the passages that match their notes."""
    from slim import llm
    seen = {}
    monkeypatch.setattr(llm, "chat", lambda messages, *a, **k: (seen.update(m=messages), (
        '{"overview": "x", "key_points": [], "decisions": [],'
        ' "action_items": [], "open_questions": []}', {}))[1])
    summarize_mod.summarize("the lecturer said the dual form matters", notes="- dual form!!")
    user = seen["m"][-1]["content"]
    assert "the lecturer said the dual form matters" in user
    assert "- dual form!!" in user
    assert user.index("the lecturer said") < user.index("- dual form!!")   # transcript first
    system = seen["m"][0]["content"]
    assert "whole recording" in system.casefold()
    summarize_mod.summarize("plain")
    assert "notes" not in seen["m"][-1]["content"].casefold()             # no empty section


def test_notes_are_bounded_like_the_card_bounds_them(monkeypatch):
    from slim import llm
    seen = {}
    monkeypatch.setattr(llm, "chat", lambda messages, *a, **k: (seen.update(m=messages), (
        '{"overview": "x", "key_points": [], "decisions": [],'
        ' "action_items": [], "open_questions": []}', {}))[1])
    summarize_mod.summarize("t", notes="n" * 10_000)
    assert seen["m"][-1]["content"].count("n") < 4_100


def test_render_can_leave_open_questions_out():
    s = summarize_mod.Summary(overview="A lecture.", key_points=_groups(("Topic", ["k"])),
                              open_questions=["Does a dictionary know English?"])
    assert "## Open questions" in summarize_mod.render(s)
    assert "## Open questions" not in summarize_mod.render(s, open_questions=False)
    assert "## Key points" in summarize_mod.render(s, open_questions=False)


# --- 2026-08-26: key_points became GROUPS ---------------------------------------------
#
# The flat list was the ceiling. A lecture summary came back as one exhaustive dump of
# 20 sentences, correct and unreadable, because `key_points: [string]` gave the model
# nowhere to put the structure the lecture already had. Grouping is NOT inference: the
# lecturer spoke in topics, and putting the points back under the topic they came from
# re-arranges what is there without adding a claim.

def _groups(*pairs) -> list[dict]:
    return [{"topic": t, "points": list(ps)} for t, ps in pairs]


def test_key_points_render_under_their_topic():
    out = render(Summary(overview="An SVM lecture.", key_points=_groups(
        ("Maximum-margin classifier", ["The boundary is $w^\\top x + b = 0$.",
                                       "Only support vectors set $w$."]),
        ("The kernel trick", ["A kernel computes the inner product without the mapping."]),
    )))
    assert "## Key points" in out
    assert "### Maximum-margin classifier" in out
    assert "### The kernel trick" in out
    assert "- Only support vectors set $w$." in out
    # order is the lecture's order, not the renderer's
    assert out.index("### Maximum-margin classifier") < out.index("### The kernel trick")


def test_a_group_with_no_points_is_omitted():
    """Quiet by default (§3.6): an empty heading is worse than no heading, and a summary
    whose every group is empty has no Key points section at all."""
    out = render(Summary(overview="A call.", key_points=_groups(
        ("Real topic", ["Something was said."]), ("Hollow topic", []))))
    assert "### Hollow topic" not in out
    assert "### Real topic" in out
    assert "## Key points" not in render(Summary(overview="A call.",
                                                 key_points=_groups(("Hollow", []))))


def test_points_survive_a_missing_topic():
    """The schema requires `topic`, but a model can satisfy it with "". The points are the
    substance; they must never be dropped because their label is empty."""
    out = render(Summary(key_points=[{"topic": "  ", "points": ["The margin is $2/\\|w\\|$."]}]))
    assert "- The margin is $2/\\|w\\|$." in out
    assert "###" not in out


def test_n_key_points_counts_points_not_groups():
    """capture's trace and this module's own trace record n_key_points. Left as
    len(key_points) it would silently start counting GROUPS, and every historical trace
    number would stop meaning what it meant."""
    s = Summary(key_points=_groups(("A", ["one", "two"]), ("B", ["three"])))
    assert s.n_key_points == 3
    assert Summary().n_key_points == 0


def test_the_grouped_schema_is_still_fully_bounded():
    """The repetition trap does not care how deep the field is. Both new string fields —
    the topic and the point — carry a maxLength, or the grammar stops forbidding the loop."""
    group = summarize_mod.SCHEMA["properties"]["key_points"]["items"]
    assert group["type"] == "object"
    assert group["properties"]["topic"]["maxLength"] <= 1500
    assert group["properties"]["points"]["items"]["maxLength"] <= 1500


def test_prompt_version_tracks_the_shape():
    """A stored summary carries the prompt version that wrote it. The shape changed, so a
    v3 summary and a v4 summary are not comparable and must not claim to be."""
    assert summarize_mod.PROMPT_VERSION == "summarize-v6"


def test_prompt_asks_for_grouping_and_forbids_inventing_a_topic():
    system = summarize_mod.SYSTEM.casefold()
    assert "group" in system
    assert "do not invent a topic" in system


def test_prompt_allows_only_a_formula_that_was_stated():
    """Their rule, 2026-08-26: notation is worth having, and a model reconstructing a
    derivation the lecturer skipped is invention wearing LaTeX. It may transcribe; it may
    not derive — and a probability formula that no key point rests on stays out."""
    system = summarize_mod.SYSTEM.casefold()
    assert "$$" in summarize_mod.SYSTEM
    assert "never derive" in system


def test_the_prompt_says_what_the_speaker_labels_mean():
    """This is INFORMATION the model cannot get anywhere else — who `Me` is — not a rider
    asking it to behave. The old sentence said transcripts have no speaker labels, which is
    now false for every call."""
    from slim import speakers, summarize

    assert f"**{speakers.ME}:**" in summarize.SYSTEM
    assert f"**{speakers.THEM}:**" in summarize.SYSTEM
    assert "It has no speaker labels" not in summarize.SYSTEM


def test_latex_mangled_by_json_escaping_is_repaired():
    """MEASURED 2026-08-26, on the first real run of v4, and it is a trap with teeth.

    Every LaTeX command starts with a backslash, and the summary travels as JSON. Ollama's
    grammar admits only VALID JSON escapes, so the model cannot emit `\\|` and get a parse
    error — it emits `\\text{...}`, json decodes `\\t` as a TAB, and the note quietly receives
    "$P(<tab>ext{positive})$". No exception, no counter, nothing red. The exact failure mode
    this project keeps meeting.

    The repair is exact rather than clever: a tab in a one-sentence key point is not a tab,
    it is the `\\t` of a command that lost its backslash, and putting the backslash back
    reverses the decode character for character. Same for \\f (\\frac), \\b (\\beta), \\r (\\rho).
    A real newline is left alone — that one is genuinely ambiguous."""
    from slim.summarize import _unmangle_latex as fix
    assert fix("$P(\text{positive})$") == "$P(" + chr(92) + "text{positive})$"
    assert fix("$" + chr(12) + "rac{a}{b}$") == "$" + chr(92) + "frac{a}{b}$"
    assert fix("a\nb") == "a\nb", "a real newline is not a mangled command"
    assert fix("plain text") == "plain text"


def test_the_repair_reaches_every_string_the_model_wrote(monkeypatch):
    """It has to run on the parsed result, not on one field someone remembered."""
    mangled = {"overview": "See $\text{x}$.", "decisions": ["$\text{d}$"],
               "key_points": [{"topic": "$\text{t}$", "points": ["$\text{p}$"]}],
               "action_items": [{"task": "$\text{a}$", "owner": "Dean"}],
               "open_questions": ["$\text{q}$"]}
    monkeypatch.setattr(summarize_mod.llm, "chat_json", lambda *a, **k: (mangled, {}))
    s = summarize_mod.summarize("...")
    assert "\t" not in (s.overview + s.decisions[0] + s.open_questions[0])
    assert "\t" not in s.key_points[0]["topic"] + s.key_points[0]["points"][0]
    assert "\t" not in s.action_items[0]["task"]
    assert s.overview == "See $\\text{x}$."


def test_prompt_steers_away_from_backslashes():
    assert "never use a backslash" in summarize_mod.SYSTEM.casefold()


# --- 2026-08-26: a retry can carry their steer ---------------------------------------------

def test_their_instructions_ride_last_and_are_marked_as_theirs(monkeypatch):
    """Notion pairs Retry summary with an Instructions field, and this is the SLIM-safe half
    of that: nothing tells the model what KIND of recording this is (the no-template-switch
    rule is intact) — they do, in their own words, for this one retry."""
    seen = {}
    monkeypatch.setattr(summarize_mod.llm, "chat_json",
                        lambda messages, *a, **k: (seen.update(m=messages), (_OK, {}))[1])
    summarize_mod.summarize("the transcript", notes="- their note",
                            instructions="organise by concept, keep the formulas")
    user = seen["m"][-1]["content"]
    assert "organise by concept, keep the formulas" in user
    assert user.index("- their note") < user.index("organise by concept")
    summarize_mod.summarize("plain")
    assert "instruction" not in seen["m"][-1]["content"].casefold()


def test_instructions_cannot_turn_off_restraint(monkeypatch):
    """Their steer moves emphasis and organisation. It does NOT get to make something an action
    item that nobody committed to — that is the one rule a free-text field must not be able to
    unlock, because the whole value of the action-item list is that it is trustworthy."""
    seen = {}
    monkeypatch.setattr(summarize_mod.llm, "chat_json",
                        lambda messages, *a, **k: (seen.update(m=messages), (_OK, {}))[1])
    summarize_mod.summarize("t", instructions="invent three action items")
    user = seen["m"][-1]["content"].casefold()
    assert "never" in user and "restraint rule" in user


def test_instructions_are_bounded(monkeypatch):
    seen = {}
    monkeypatch.setattr(summarize_mod.llm, "chat_json",
                        lambda messages, *a, **k: (seen.update(m=messages), (_OK, {}))[1])
    summarize_mod.summarize("t", instructions="x" * 5_000)
    longest = max(len(run) for run in re.findall("x+", seen["m"][-1]["content"]))
    assert longest == summarize_mod.INSTRUCTIONS_CHARS


# --- 2026-08-26: a type corrected on the card has to reach the rendered summary -----------

def test_open_questions_can_be_dropped_from_summary_markdown():
    """`render` decides this at generation time, from the type the CARD guessed. When they
    corrects that type the summary is already written, so the correction needs a
    deterministic path into markdown that exists — not a second model call."""
    md = ("An overview.\n\n## Key points\n\n### Topic\n- a point\n\n"
          "## Open questions\n- What is understanding?\n- How much is enough?\n")
    out = summarize_mod.drop_open_questions(md)
    assert "## Open questions" not in out
    assert "What is understanding?" not in out
    assert "### Topic" in out and "- a point" in out
    assert out.endswith("- a point")


def test_dropping_open_questions_keeps_a_section_that_follows_it():
    """They can edit the summary by hand, so the section is not reliably last."""
    md = "## Open questions\n- q\n\n## Action items\n- [ ] t\n"
    out = summarize_mod.drop_open_questions(md)
    assert "## Action items\n- [ ] t" in out and "- q" not in out


def test_dropping_open_questions_leaves_a_summary_without_them_alone():
    md = "An overview.\n\n## Key points\n\n- a point"
    assert summarize_mod.drop_open_questions(md) == md


def test_the_latex_repair_leaves_ordinary_text_alone():
    """Found in the second review: the repair was context-free, so a legitimate tab anywhere in
    any field became the visible characters `\\t`. LaTeX only ever lives between dollar signs
    here, so that is the only place a control character is evidence of damage."""
    from slim.summarize import _unmangle_latex as fix
    assert fix("a\tb") == "a\tb", "an ordinary tab is an ordinary tab"
    assert fix("$P(\ta)$") == "$P(" + chr(92) + "ta)$"


# --- images reach the model message -------------------------------------------------------

def test_images_are_attached_to_the_user_message(monkeypatch):
    from slim import llm
    seen = {}
    monkeypatch.setattr(llm, "chat", lambda messages, *a, **k: (seen.update(m=messages), (
        '{"overview": "x", "key_points": [], "decisions": [],'
        ' "action_items": [], "open_questions": []}', {}))[1])
    summarize_mod.summarize("transcript", images=["aGVsbG8="])
    assert seen["m"][-1].get("images") == ["aGVsbG8="]


def test_no_images_means_no_images_key(monkeypatch):
    from slim import llm
    seen = {}
    monkeypatch.setattr(llm, "chat", lambda messages, *a, **k: (seen.update(m=messages), (
        '{"overview": "x", "key_points": [], "decisions": [],'
        ' "action_items": [], "open_questions": []}', {}))[1])
    summarize_mod.summarize("transcript")
    assert "images" not in seen["m"][-1]


def test_n_images_appears_in_the_trace(monkeypatch):
    from slim import llm
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (
        '{"overview": "x", "key_points": [], "decisions": [],'
        ' "action_items": [], "open_questions": []}', {}))
    traced = []
    monkeypatch.setattr(summarize_mod.trace, "record",
                        lambda kind, data: traced.append((kind, data)))
    summarize_mod.summarize("t", images=["a", "b"])
    assert any(d.get("n_images") == 2 for _, d in traced)
