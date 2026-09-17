"""Spoken routing: what they SAYS a recording is decides where it lands (2026-08-01).

The gap this closes: the spoken tag already reached frontmatter, but `_note_path` routed on
which INBOX FOLDER the audio came from — so every recording landed in `Journal/` regardless of
what they said it was. Tagging without routing is filing metadata into a pile.
"""
from datetime import datetime

import pytest

from slim import inbox, voicetags

REGISTRY = {
    "beacon": {"name": "Beacon", "vault_path": "Notes/beacon"},
    "school": {"name": "School", "vault_path": "Notes/school"},
    "work": {"name": "Work (Acme)", "aliases": ["acme", "my job"],
             "vault_path": "Notes/work"},
    "personal-intelligence": {"name": "Personal Intelligence (SLIM)", "aliases": ["slim"],
                              "vault_path": "Notes/personal-intelligence"},
}
WHEN = datetime(2026, 8, 1, 9, 30)


def route(text, default_dir="Journal", **kw):
    return voicetags.route(text, default_dir=default_dir, registry=REGISTRY, **kw)


# --- TYPE ---------------------------------------------------------------------------------

@pytest.mark.parametrize("opening,expected", [
    ("This is a journal, today was rough.", voicetags.JOURNAL),
    ("Just journaling for a minute.", voicetags.JOURNAL),
    ("Meeting with Jordan about the roster.", voicetags.MEETING),
    ("On a call with Sam right now.", voicetags.MEETING),
    ("Debrief from the interview.", voicetags.MEETING),
    ("Lecture note, week six.", voicetags.LECTURE),
    ("Class notes for the ML course.", voicetags.LECTURE),
    ("An idea for a scheduler.", voicetags.IDEA),
    ("Project idea, might be nothing.", voicetags.IDEA),
    ("Note to self, renew the domain.", voicetags.NOTE),
    ("Quick note about work.", voicetags.NOTE),
])
def test_spoken_type_is_detected_from_the_opening(opening, expected):
    assert voicetags.detect_type(opening) == expected


def test_a_type_word_used_later_does_not_retro_label_the_recording():
    """The tag is what they say FIRST. A memo whose CONTENT mentions a meeting is not a
    meeting note — otherwise half their journals would file themselves as meetings."""
    text = "x" * voicetags.OPENING_CHARS + " and then the meeting with Jordan ran long"
    assert voicetags.detect_type(text) is None


def test_no_spoken_type_returns_none_rather_than_guessing():
    assert voicetags.detect_type("So the thing about the roster is that it keeps breaking") is None


# --- TALKING ABOUT A JOURNAL vs MAKING ONE (2026-08-04) ------------------------------------
# `journal` is a bare noun and was first in `_TYPE_VOCAB`, and `detect_type` returned the first
# match in LIST order rather than the earliest match in the SENTENCE. So the word vetoed every
# other type: "meeting with Sam about the journal tab" was a journal, and worse, it landed in
# the privacy floor, where nothing carries it out. Position is the fix, because it is
# how the declaration actually works in speech — announce what a thing IS, then discuss it.

@pytest.mark.parametrize("opening,expected", [
    # journal as an OBJECT, after an explicit declaration of some other form
    ("I have an idea for the personal journal feature in SLIM.", voicetags.IDEA),
    ("An idea for journaling in SLIM: auto-prompt after a walk.", voicetags.IDEA),
    ("Meeting with Sam about the journal tab redesign.", voicetags.MEETING),
    ("Lecture note on journaling as a research method.", voicetags.LECTURE),
    ("Quick note: the journal view is broken.", voicetags.NOTE),
    # journal as the DECLARATION, even when the same opening also names another form
    ("This is a journal. Today was rough.", voicetags.JOURNAL),
    ("Journal entry. Thinking about the SLIM journal feature.", voicetags.JOURNAL),
    ("Journaling about my interview with Northwind.", voicetags.JOURNAL),
])
def test_the_earliest_declaration_wins_not_the_first_in_the_list(opening, expected):
    assert voicetags.detect_type(opening) == expected


def test_the_same_two_words_in_the_other_order_give_the_other_type():
    """The pair that shows this is position and not a keyword list: identical vocabulary, and
    only the order they said it in decides."""
    assert voicetags.detect_type("Journaling about my interview with Northwind.") \
        == voicetags.JOURNAL
    assert voicetags.detect_type("Interview with Northwind. I want to journal about it after.") \
        == voicetags.MEETING


def test_a_dead_heat_still_resolves_to_journal():
    """The two errors are NOT symmetric. A misfiled journal ends up outside the privacy floor
    and becomes synthesis-eligible; a misfiled idea is merely stuck inside it. So a genuine tie
    keeps list order, where journal leads — only an EARLIER declaration may override it."""
    assert voicetags.detect_type("journal") == voicetags.JOURNAL
    # same start offset for both patterns -> list order decides -> journal
    assert voicetags.detect_type("journal idea for a scheduler") == voicetags.JOURNAL


# --- SUBJECT ------------------------------------------------------------------------------

@pytest.mark.parametrize("opening,expected", [
    ("An idea for Beacon, about the Dagster DAG.", "beacon"),
    ("Lecture note for school.", "school"),
    ("Quick note, this is an acme thing.", "work"),
    ("An idea for SLIM.", "personal-intelligence"),
])
def test_subject_matches_registry_ids_aliases_and_names(opening, expected):
    assert voicetags.detect_subject(opening, REGISTRY) == expected


def test_subject_is_a_closed_set_so_a_memo_cannot_invent_a_project():
    assert voicetags.detect_subject("An idea for Photoshop.", REGISTRY) is None


def test_subject_matching_is_word_bounded():
    """"reworking" is not "work". A substring match would subject half their memos to their job."""
    assert voicetags.detect_subject("Journaling about reworking my morning.", REGISTRY) is None


# --- ROUTING ------------------------------------------------------------------------------

def test_a_journal_is_pinned_to_the_journal_floor_even_when_it_names_a_project():
    """`Journal/` is the permanent privacy floor and the floor is a path, so routing OUT of
    Journal is a privacy decision. A journal entry never routes, whatever project it happens
    to mention."""
    r = route("Journal entry. Thinking about Beacon again.")
    assert r.folder == "Journal"
    assert (r.subject, r.kind, r.routed_by) == ("beacon", voicetags.JOURNAL, "spoken")


def test_every_routing_type_lands_in_exactly_one_owned_segment_pair():
    """THE CEILING, at the emitter. Type used to be a path segment and meetings also carried a
    year, which crossed with subject into a Cartesian product — 11 folders for 116 notes. There
    are now exactly two destinations a type can produce."""
    for kind in voicetags.ROUTING_TYPES:
        assert voicetags.folder_for(kind, "beacon") in ("Journal", "Capture/beacon")
        assert len(voicetags.folder_for(kind, "beacon").split("/")) <= 2


def test_a_journal_is_the_floor_itself_not_a_capture_subfolder():
    """The floor is a top-level SIBLING of Capture/, which is what lets Capture/ carry one
    blanket grant: no grant can reach Journal/ at any depth."""
    assert voicetags.folder_for(voicetags.JOURNAL, "beacon") == "Journal"
    assert not voicetags.folder_for(voicetags.JOURNAL, None).startswith(voicetags.NAMESPACE)


def test_a_subjectless_capture_gets_the_sentinel_not_the_capture_root():
    """A bare `Capture/x.md` would make the invariant 'at most two segments', and 'at most'
    invariants rot. The sentinel keeps every capture at exactly the same depth."""
    r = route("Lecture note, week six, on attention.")
    assert r.folder == "Capture/_unfiled"
    assert (r.subject, r.kind, r.subject_by) == (None, voicetags.LECTURE, "none")


def test_no_registered_project_can_shadow_the_unfiled_sentinel():
    """`_unfiled` is a path segment the router emits, so a project registered under that id
    would collide with it and quietly capture every unsubjected recording."""
    from slim import config
    assert voicetags.UNFILED not in config.registry()


def test_a_named_subject_routes_into_that_project():
    r = route("An idea for Beacon: cache the macro series.")
    assert r.folder == "Capture/beacon"
    assert (r.subject, r.routed_by, r.subject_by) == ("beacon", "spoken", "spoken")


def test_a_meeting_and_a_lecture_on_one_subject_share_one_folder():
    """The point of dropping the type segment: `school` used to live in 6 directories at once.
    Type is still on the note, in frontmatter, where it is queryable."""
    assert route("Meeting with Jordan about school.").folder == \
        route("Lecture note for school on attention.").folder == "Capture/school"


def test_saying_nothing_moves_nothing():
    """The safety property that lets this ship: a recording with no spoken type keeps exactly
    the destination it had before routing existed."""
    assert route("So the roster broke again this morning.") == \
        ("Journal", None, "", "default", "none")
    assert route("So the roster broke again.",
                 default_dir="Capture/work").folder == "Capture/work"


def test_the_subject_subfolder_is_the_project_id_so_folder_and_tag_agree():
    """The folder name matches the `topics:` entry, so browsing and searching say the same
    thing. A registry entry needs no vault_path — the subject segment IS the id, which is what
    makes `config.project_prefixes` derivable rather than configured."""
    r = voicetags.route("An idea for ghost.", default_dir="Journal",
                        registry={"ghost": {"name": "Ghost"}})
    assert (r.folder, r.subject) == ("Capture/ghost", "ghost")


# --- THE NOTE ON DISK ---------------------------------------------------------------------

def test_note_path_honours_the_routed_folder():
    p = inbox._note_path(WHEN, "Cache the macro series", "deadbeef" * 8,
                         folder="Capture/beacon")
    assert p.parent.name == "beacon"
    assert p.name.startswith("2026-08-01--")


def test_note_path_without_routing_lands_in_the_floor():
    """The phone lane writes journals, so an undecidable drop keeps the most protected place
    it could have had rather than being pushed out to a sentinel."""
    assert inbox._note_path(WHEN, "t", "d" * 64).parent.name == voicetags.JOURNAL_FOLDER


def test_the_type_and_the_tag_are_one_word_by_construction():
    """ONE spoken vocabulary (2026-09-03). A lecture used to render `type: lecture` alongside
    `tags: [needs-triage]`, because the old tag vocabulary had no word for "lecture" and the
    classifier looked like it had run and failed. There is one list now, so they cannot part."""
    from types import SimpleNamespace
    t = SimpleNamespace(text="Lecture note, week six, on attention.",
                        model="parakeet", audio_seconds=1.0)
    body = inbox._render_note(
        "Attention", WHEN, t, "ab" * 32,
        {"title": "", "type_tag": "", "topics": []},
        {"folder": "Capture/school", "subject": None,
         "kind": voicetags.LECTURE, "routed_by": "spoken"})
    assert "type: lecture" in body
    assert "tags: [lecture]" in body
    assert "tagged_by: spoken" in body


def test_an_untyped_recording_writes_no_tags_line_and_routed_by_default():
    """The mirror: with nothing said and nothing inferred there is no type to write. The note
    says so by omission and by `tagged_by: none` — never by inventing a label nothing triaged."""
    from types import SimpleNamespace
    t = SimpleNamespace(text="So the roster broke again.", model="parakeet", audio_seconds=1.0)
    body = inbox._render_note(
        "Roster", WHEN, t, "ab" * 32,
        {"title": "", "type_tag": "", "topics": []},
        {"folder": "Journal", "subject": None, "kind": "", "routed_by": "default"})
    assert "tags:" not in body
    assert "tagged_by: none" in body
    assert "routed_by: default" in body


def test_an_inferred_type_is_written_as_the_tag_and_marked_inferred():
    """The middle case: they said nothing, enrichment did. `route` folds the inferred type into
    the same field a spoken one lands in, and `tagged_by: inferred` is what separates them."""
    from types import SimpleNamespace
    t = SimpleNamespace(text="So the roster broke again.", model="parakeet", audio_seconds=1.0)
    body = inbox._render_note(
        "Roster", WHEN, t, "ab" * 32,
        {"title": "", "type_tag": "meeting", "topics": []},
        {"folder": "Capture/_unfiled", "subject": None,
         "kind": "meeting", "routed_by": "inferred"})
    assert "type: meeting-note" in body
    assert "tags: [meeting]" in body
    assert "tagged_by: inferred" in body


# --- INFERRED ROUTING (the owner's call, 2026-08-01) -------------------------------------------------
# They record mid-meeting and forgets to announce it, so the model's guess routes when they said
# nothing. They accepted the egress consequence knowingly; these pin the guardrails that remain.

def test_the_models_guess_routes_when_they_said_nothing():
    """The whole point: 'so anyway Jordan thinks the roster is fine' names no type, but the
    enrichment model can tell it is a meeting. Before this it sat in frontmatter unused."""
    r = route("So anyway, Jordan thinks the roster is fine and we should ship it.",
              inferred_type="meeting")
    assert (r.folder, r.kind, r.routed_by) == \
        ("Capture/_unfiled", voicetags.MEETING, "inferred")


def test_spoken_still_beats_inferred():
    """Their stated intent is testimony; the model's is a guess. A guess never overrides them."""
    r = route("This is a journal. Today was rough.", inferred_type="meeting")
    assert (r.folder, r.kind, r.routed_by) == ("Journal", voicetags.JOURNAL, "spoken")


def test_an_inferred_journal_still_never_routes():
    r = route("Rough day and I keep circling the same thing.",
              inferred_type="journal", inferred_topics=["beacon"])
    assert (r.folder, r.routed_by) == ("Journal", "inferred")


def test_the_subject_comes_from_topics_but_stays_a_closed_set():
    """The model proposes open-vocabulary topics; CODE decides which of them names a real
    project. So a model can file a note under a project by NAMING one, never by inventing one."""
    r = route("Anyway that caching thing.", inferred_type="idea",
              inferred_topics=["macro", "beacon", "caching"])
    assert (r.subject, r.subject_by) == ("beacon", "topic")
    r2 = route("Anyway that thing.", inferred_type="idea",
               inferred_topics=["photoshop", "gardening"])
    assert (r2.subject, r2.subject_by, r2.folder) == (None, "none", "Capture/_unfiled")


def test_a_junk_inferred_type_is_ignored_rather_than_routed_on():
    """The enum is enforced in code, not trusted from the model's output."""
    assert route("Some recording.", inferred_type="wharrgarbl") == \
        ("Journal", None, "", "default", "none")


# --- THE DESCRIPTION LAYER (2026-08-03) ---------------------------------------------------
# 36 of 134 recordings carried no subject, because both earlier layers are LITERAL matches:
# `detect_subject` on words they spoke, `subject_from_topics` on words the model emitted. Neither
# reaches `school` from a transcript that only ever says "kernel trick". `enrich` now matches the
# recording against the registry's own descriptions, and this is the only layer that is a
# judgement — so it is last, and it is recorded.

def test_a_description_match_fills_a_subject_nothing_else_could():
    r = route("So the kernel trick maps the inputs into a higher dimensional space.",
              inferred_type="lecture", inferred_project="school")
    assert (r.subject, r.subject_by, r.folder) == ("school", "description", "Capture/school")


def test_a_spoken_subject_beats_a_description_match():
    """Deterministic first, always. The model reading a description never overrides a project
    they named out loud — same rule as type, on the other axis."""
    r = route("Lecture note for school on kernels.", inferred_project="work")
    assert (r.subject, r.subject_by) == ("school", "spoken")


def test_a_topic_match_beats_a_description_match():
    """Both are model output, but a topic match is decided in CODE against a closed set; the
    description match is the model choosing. Cheaper and more traceable wins."""
    r = route("Anyway that thing.", inferred_type="idea",
              inferred_topics=["beacon"], inferred_project="work")
    assert (r.subject, r.subject_by) == ("beacon", "topic")


def test_an_invented_project_is_rejected_rather_than_creating_a_folder():
    """The grammar enum makes the closed set structural, and this check is the redundant half:
    it is what survives a grammar that silently degrades."""
    r = route("Some rambling.", inferred_type="lecture", inferred_project="not-a-project")
    assert (r.subject, r.subject_by, r.folder) == (None, "none", "Capture/_unfiled")


def test_the_models_none_is_honoured_instead_of_being_guessed_past():
    """`none` must be a real option. A required enum with no null forces a project onto every
    recording — grammar-guaranteed false positives no prompt wording can prevent."""
    from slim import enrich
    assert enrich.NO_PROJECT in enrich._schema(REGISTRY)["properties"]["project"]["enum"]
    r = route("Some rambling.", inferred_type="lecture", inferred_project=enrich.NO_PROJECT)
    assert (r.subject, r.folder) == (None, "Capture/_unfiled")


def test_the_project_enum_is_exactly_the_registry_plus_none():
    """Same drift guard as the type vocabulary below: the candidates the model may choose from
    are the registry itself, not a hand-maintained copy of it."""
    from slim import enrich
    assert enrich._schema(REGISTRY)["properties"]["project"]["enum"] == \
        [*sorted(REGISTRY), enrich.NO_PROJECT]


def test_the_inferred_and_spoken_vocabularies_are_literally_the_same_list():
    """enrich-v1 claimed this and shipped a different list, which is why the model's inferred
    type sat in frontmatter unusable for two weeks. Pin it so they cannot drift again."""
    from slim import enrich
    assert enrich.TYPE_TAGS == list(voicetags.ROUTING_TYPES)
