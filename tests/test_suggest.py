from pathlib import Path

from slim import suggest


def _note(vault: Path, rel: str, topics: list[str] | None = None):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    tl = f"topics: [{', '.join(topics)}]\n" if topics else ""
    p.write_text(f"---\ntitle: \"x\"\ntype: lecture\norigin: recorded\n{tl}---\n\nbody\n",
                 encoding="utf-8")
    return p


def test_the_tree_reports_real_folders_with_counts(tmp_path):
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    _note(tmp_path, "Notes/school/CS201 ML/b.md")
    _note(tmp_path, "Notes/school/CS202 RL/c.md")
    got = {f.path: f.count for f in suggest.folder_tree(tmp_path)}
    assert got["Notes/school/CS201 ML"] == 2
    assert got["Notes/school/CS202 RL"] == 1
    assert got["Notes/school"] == 3          # counts are cumulative, for orientation


def test_derived_and_excluded_folders_never_appear_as_destinations(tmp_path):
    """_Reflections/ and Profile/ are derived. Offering them as a filing target would invite
    a recording into the one place the brain must not retrieve from."""
    _note(tmp_path, "_Reflections/r.md")
    _note(tmp_path, "Profile/me.md")
    _note(tmp_path, "Attachments/x.md")
    assert [f.path for f in suggest.folder_tree(tmp_path)] == []


def test_the_journal_floor_is_offered_but_never_with_children(tmp_path):
    """Journal/ is a legitimate destination for a journal recording and is FLAT — one stream.
    A subfolder under it would be a taxonomy slot inside a privacy floor."""
    _note(tmp_path, "Journal/j.md")
    _note(tmp_path, "Journal/nested/k.md")
    paths = [f.path for f in suggest.folder_tree(tmp_path)]
    assert "Journal" in paths
    assert not any(p.startswith("Journal/") for p in paths)


def test_tag_counts_come_from_what_is_already_used(tmp_path):
    """'Few and reused' is enforced by SHOWING the count, not by a rule. A tag with no
    history must be visibly new — including one they type themselves."""
    _note(tmp_path, "Notes/a/a.md", topics=["school", "svm"])
    _note(tmp_path, "Notes/a/b.md", topics=["school"])
    counts = suggest.tag_counts(tmp_path)
    assert counts["school"] == 2
    assert counts["svm"] == 1
    assert counts.get("mercer-theorem", 0) == 0


def test_max_depth_is_a_display_bound_not_a_filing_ceiling(tmp_path):
    """CLAUDE.md grants the owner any depth they like below level 2 under Notes/. A note four
    levels deep must still count toward its ancestors up to max_depth — it just stops being
    listed as its own browsable row past the bound."""
    _note(tmp_path, "Notes/school/CS201 ML/Kernels/deep.md")
    got = {f.path: f.count for f in suggest.folder_tree(tmp_path, max_depth=2)}
    assert got["Notes"] == 1
    assert got["Notes/school"] == 1
    # deeper than max_depth is not listed for browsing, but does not vanish or double count
    assert "Notes/school/CS201 ML" not in got
    assert "Notes/school/CS201 ML/Kernels" not in got

    # with a deeper bound the same note surfaces its deeper ancestor too
    got_deep = {f.path: f.count for f in suggest.folder_tree(tmp_path, max_depth=3)}
    assert got_deep["Notes/school/CS201 ML"] == 1
    assert "Notes/school/CS201 ML/Kernels" not in got_deep


def test_dot_prefixed_paths_are_excluded(tmp_path):
    """`Notes` itself is now listed once it exists as a directory — the tree walks directories,
    not just files. What must never appear is the dot-prefixed child."""
    _note(tmp_path, "Notes/.trash/gone.md")
    paths = [f.path for f in suggest.folder_tree(tmp_path)]
    assert not any(".trash" in p for p in paths)


# --- directories, not just the folders that happen to contain notes ----------------------

def test_a_newly_created_empty_folder_is_offered(tmp_path):
    """They create `Notes/school/CS204 Ethics/` in Obsidian and immediately records its first
    lecture. Deriving folders from FILES meant the folder they just made was the one folder the
    card could not offer — visible in Obsidian, absent from the recorder."""
    (tmp_path / "Notes" / "school" / "CS204 Ethics").mkdir(parents=True)
    got = {f.path: f.count for f in suggest.folder_tree(tmp_path)}
    assert got["Notes/school/CS204 Ethics"] == 0
    assert got["Notes/school"] == 0


def test_an_empty_folder_under_capture_is_offered_too(tmp_path):
    (tmp_path / "Capture" / "job-search").mkdir(parents=True)
    assert "Capture/job-search" in {f.path for f in suggest.folder_tree(tmp_path)}


def test_an_empty_folder_sorts_beside_populated_ones_with_a_zero_count(tmp_path):
    """The count is orientation, not a filter. A brand-new folder reads as empty rather than
    being hidden, which is the honest signal — 'nothing here yet', not 'does not exist'."""
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    (tmp_path / "Notes" / "school" / "CS204 Ethics").mkdir(parents=True)
    got = {f.path: f.count for f in suggest.folder_tree(tmp_path)}
    assert got["Notes/school/CS201 ML"] == 1
    assert got["Notes/school/CS204 Ethics"] == 0
    assert got["Notes/school"] == 1


def test_an_empty_dot_directory_is_still_excluded(tmp_path):
    (tmp_path / "Notes" / ".obsidian" / "plugins").mkdir(parents=True)
    assert not any(".obsidian" in f.path for f in suggest.folder_tree(tmp_path))


def test_empty_derived_directories_are_still_never_offered(tmp_path):
    """Walking directories must not quietly widen the exclusion hole: `_Reflections/` and
    `Profile/` are derived and `Attachments/` is binary, empty or not."""
    for d in ("_Reflections", "Profile", "Attachments", "Attachments/_incoming"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    assert [f.path for f in suggest.folder_tree(tmp_path)] == []


def test_an_empty_folder_deeper_than_max_depth_is_not_listed(tmp_path):
    (tmp_path / "Notes" / "school" / "CS201 ML" / "Kernels").mkdir(parents=True)
    paths = {f.path for f in suggest.folder_tree(tmp_path, max_depth=2)}
    assert "Notes/school" in paths
    assert "Notes/school/CS201 ML" not in paths
    assert "Notes/school/CS201 ML/Kernels" not in paths


def test_the_journal_floor_stays_flat_even_with_empty_subdirectories(tmp_path):
    """⚠ A directory walk is exactly how a subfolder could sneak into the floor. `Journal/` is
    one stream; an empty `Journal/2026/` on disk must still never be offered as a destination."""
    (tmp_path / "Journal" / "2026").mkdir(parents=True)
    paths = [f.path for f in suggest.folder_tree(tmp_path)]
    assert "Journal" in paths
    assert not any(p.startswith("Journal/") for p in paths)


def test_tag_counts_reads_comma_separated_string_spelling(tmp_path):
    """Both YAML spellings of topics must be honored — a plain comma-separated string, not
    only a real list."""
    p = tmp_path / "Notes/a/a.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\ntitle: \"x\"\ntopics: school, svm\n---\n\nbody\n", encoding="utf-8")
    counts = suggest.tag_counts(tmp_path)
    assert counts["school"] == 1
    assert counts["svm"] == 1


def test_tag_counts_excludes_derived_folders(tmp_path):
    """_Reflections/ is derived and Attachments/ is binary — neither may seed a tag
    suggestion."""
    _note(tmp_path, "_Reflections/r.md", topics=["derived-thing"])
    _note(tmp_path, "Attachments/x.md", topics=["binary-thing"])
    counts = suggest.tag_counts(tmp_path)
    assert "derived-thing" not in counts
    assert "binary-thing" not in counts


# --- the model layer: it may only CHOOSE, never invent ----------------------------------

import json


def _fake_model(monkeypatch, payload):
    """Patch `llm.chat`, which `chat_json` calls internally, so the fake covers both."""
    from slim import llm
    monkeypatch.setattr(llm, "chat", lambda *a, **k: (json.dumps(payload), {}))


def _payload(**kw):
    base = {"dest_dir": "Notes/school/CS201 ML", "confident_depth": 2,
            "type_tag": "lecture", "topics": ["school"]}
    base.update(kw)
    return base


def test_a_declared_type_is_never_overridden_by_the_model(tmp_path, monkeypatch):
    """They picked the type before speaking. That is testimony — the pipeline already treats a
    declared type as final, and the card must not quietly re-judge it."""
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    _fake_model(monkeypatch, _payload(type_tag="meeting-note"))
    card = suggest.build_card("...", declared_type="lecture", vault=tmp_path)
    assert card.type_tag == "lecture"


def test_a_destination_the_model_invented_is_rejected(tmp_path, monkeypatch):
    """The model may only CHOOSE among folders that exist. An invented path would create a
    folder nobody asked for — the sprawl the subject-only axis exists to prevent."""
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    _fake_model(monkeypatch, _payload(dest_dir="Notes/school/CS9999 Imaginary"))
    card = suggest.build_card("...", vault=tmp_path)
    assert card.dest_dir in {f.path for f in suggest.folder_tree(tmp_path)}


def test_an_invented_leaf_falls_back_to_its_real_ancestor(tmp_path, monkeypatch):
    """Walking back beats discarding: they still gets `Notes/school`, which is right as far as
    it goes, rather than being dumped in _unfiled."""
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    _fake_model(monkeypatch, _payload(dest_dir="Notes/school/CS9999 Imaginary"))
    assert suggest.build_card("...", vault=tmp_path).dest_dir == "Notes/school"


def test_the_uncertain_segment_offers_real_siblings(tmp_path, monkeypatch):
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    _note(tmp_path, "Notes/school/CS202 RL/b.md")
    _fake_model(monkeypatch, _payload())
    card = suggest.build_card("...", vault=tmp_path)
    assert set(card.candidates) == {"CS201 ML", "CS202 RL"}
    assert card.confident_depth == 2


def test_a_model_failure_still_yields_a_filable_card(tmp_path, monkeypatch):
    """⚠ The note must be FILED even when the model dies. Filing may never depend on a model
    call succeeding — an unfiled recording is the Notion failure this replaces."""
    from slim import llm
    _note(tmp_path, "Notes/school/CS201 ML/a.md")

    def boom(*a, **k):
        raise llm.LLMError("model is down")
    monkeypatch.setattr(llm, "chat", boom)

    card = suggest.build_card("...", declared_type="lecture", vault=tmp_path)
    assert card.dest_dir                       # a real destination, not empty
    assert card.type_tag == "lecture"          # their declaration survives the failure


def test_a_journal_recording_never_routes_out_of_the_floor(tmp_path, monkeypatch):
    """⚠ The floor is a path and no layer may move a journal off it. The model here proposes a
    course folder; the card must pin the destination regardless. Topics ride along as proposed:
    they are frontmatter for Obsidian, and since schema v12 nothing promotes them anywhere."""
    _note(tmp_path, "Journal/j.md")
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    _fake_model(monkeypatch, _payload(dest_dir="Notes/school/CS201 ML",
                                      type_tag="journal", topics=["school"]))
    card = suggest.build_card("...", declared_type="journal", vault=tmp_path)
    assert card.dest_dir == "Journal"
    assert card.topics == ["school"]


def test_the_default_tree_goes_all_the_way_down(tmp_path):
    """⚠ The default was 3, which sounded modest and was not: measured on the real vault, 48 of
    84 folders sat at depth 4-7, so the cap hid MOST of the tree — including the pattern in
    active use, `Notes/job-search/Companies/<company>`. "Type the path yourself" is not a
    substitute for a browser whose whole job is showing what you do not remember."""
    _note(tmp_path, "Notes/job-search/Companies/Contoso/a.md")
    paths = {f.path for f in suggest.folder_tree(tmp_path)}
    assert "Notes/job-search/Companies/Contoso" in paths      # depth 4
    assert "Notes/job-search/Companies" in paths


def test_a_deep_empty_folder_is_offered_too(tmp_path):
    """Both fixes together: directories rather than files, and no depth cap."""
    (tmp_path / "Notes/job-search/Companies/Contoso").mkdir(parents=True)
    got = {f.path: f.count for f in suggest.folder_tree(tmp_path)}
    assert got["Notes/job-search/Companies/Contoso"] == 0


def test_an_explicit_max_depth_still_bounds_browsing(tmp_path):
    """The parameter survives for a caller that wants a shallow slice."""
    _note(tmp_path, "Notes/a/b/c/d.md")
    shallow = {f.path for f in suggest.folder_tree(tmp_path, max_depth=2)}
    assert "Notes/a" in shallow and "Notes/a/b" not in shallow


# --- context size (2026-08-22) ---------------------------------------------------------------

def test_the_card_runs_at_record_ctx_not_background_ctx(tmp_path, monkeypatch):
    """⚠ Both calls a recording makes must use the SAME size. Split them and one recording asks
    Ollama for the model twice, spawning a second runner and reloading 19 GB mid-request.

    At a 131k window a dense model's KV cache alone is ~11 GB: measured 29.9 GB RSS on 2026-08-22, and
    macOS killed the pass with zero bytes of output.
    """
    from slim import llm
    seen = {}

    def spy(messages, schema, **kw):
        seen.update(kw)
        return _payload(), {}

    monkeypatch.setattr(llm, "chat_json", spy)
    monkeypatch.setattr(suggest.llm, "chat_json", spy, raising=False)
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    suggest.build_card("a transcript", declared_type="lecture", vault=tmp_path)
    assert seen["num_ctx"] == llm.RECORD_CTX
    # The recording path runs on the RESIDENT (RECORD_MODEL == MODEL since 2026-08-25), so
    # these two sizes are the same runner. They were separate numbers back when the recorder
    # used the dense model and had to stay clear of the big window; RECORD_CTX is now an
    # alias, and this is the invariant that keeps a recording from spawning a second runner.
    assert llm.RECORD_CTX == llm.RESIDENT_CTX


def test_the_model_is_never_asked_for_a_link(tmp_path, monkeypatch):
    """⚠ 0 for 2 on real data, both times the SAME invented path. The model is shown FOLDERS
    and asked to cite NOTES, so it could not do better by construction (2026-08-23). The ask
    itself is gone since 2026-09-03 — nothing in the grammar or the prompt requests one, so a
    link cannot reappear in a card by accident."""
    seen = {}

    def spy(messages, schema, **kw):
        seen["schema"], seen["system"] = schema, messages[0]["content"]
        return {"dest_dir": "Notes", "confident_depth": 1, "type_tag": "lecture",
                "topics": []}, {}

    from slim import llm
    monkeypatch.setattr(llm, "chat_json", spy)
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    card = suggest.build_card("a transcript", declared_type="lecture", vault=tmp_path)
    assert "links" not in seen["schema"]["properties"]
    assert "links" not in seen["schema"]["required"]
    assert "link" not in seen["system"].lower()
    assert not hasattr(card, "links")


def test_a_title_they_typed_is_given_to_the_card_as_their_words(tmp_path, monkeypatch):
    """A title they typed is TESTIMONY, like a declared type — and often the strongest hint about
    what a recording is about, since ASR over a lecture they only listened to says little."""
    from slim import llm
    seen = {}

    def spy(messages, schema, **kw):
        seen["user"] = messages[-1]["content"]
        return _payload(), {}

    monkeypatch.setattr(llm, "chat_json", spy)
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    suggest.build_card("some transcript", declared_type="lecture",
                       title="K-means for CS201", vault=tmp_path)
    assert "They titled this: K-means for CS201" in seen["user"]
    assert seen["user"].index("They titled this") < seen["user"].index("Transcript:")


def test_no_title_means_nothing_is_claimed_about_one(tmp_path, monkeypatch):
    from slim import llm
    seen = {}
    monkeypatch.setattr(llm, "chat_json",
                        lambda m, s, **kw: (seen.update(user=m[-1]["content"]), (_payload(), {}))[1])
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    suggest.build_card("some transcript", declared_type="lecture", title="  ", vault=tmp_path)
    assert "They titled this" not in seen["user"]


def test_the_notes_they_typed_while_listening_are_given_to_the_card(tmp_path, monkeypatch):
    """Their notes are the part they CHOSE to write down. A transcript records what was said; the
    notes record what they thought mattered about it."""
    from slim import llm
    seen = {}
    monkeypatch.setattr(llm, "chat_json",
                        lambda m, s, **kw: (seen.update(user=m[-1]["content"]), (_payload(), {}))[1])
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    suggest.build_card("some transcript", declared_type="lecture",
                       notes="- elbow plot picks k\n- compare to hierarchical", vault=tmp_path)
    assert "Their notes while listening:" in seen["user"]
    assert "elbow plot picks k" in seen["user"]
    assert seen["user"].index("Their notes") < seen["user"].index("Transcript:")


def test_long_notes_cannot_crowd_out_the_transcript(tmp_path, monkeypatch):
    from slim import llm
    seen = {}
    monkeypatch.setattr(llm, "chat_json",
                        lambda m, s, **kw: (seen.update(user=m[-1]["content"]), (_payload(), {}))[1])
    _note(tmp_path, "Notes/school/CS201 ML/a.md")
    suggest.build_card("the transcript", declared_type="lecture",
                       notes="x" * 10_000, vault=tmp_path)
    assert "x" * 4000 in seen["user"]
    assert "x" * 4001 not in seen["user"]     # truncated, not merely present
    assert "the transcript" in seen["user"]
