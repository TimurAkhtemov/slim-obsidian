"""The note as the editing tools see it: blocks with handles, the tools that change a draft, what
code refuses, and the splices the plugin applies. Nothing here writes a note."""

import pytest

from slim import edits
from slim.edits import Note, ToolError, apply, apply_splices, ollama_tools, splices, strip_labels

NOTE = "# Title\n\nIntro para.\n\n## A\n\nText a.\n- one\n- two\n\n## B\n\nText b.\n"
RECORDING = "---\ntitle: x\n---\n\n````slim-meeting\nid: 1\n````\n\nMy notes.\n"
MEETING = """---
title: Standup
type: meeting
---

```slim-meeting
<!-- slim:summary prompt="record-v2" -->
## Summary

Talked about the release.
<!-- /slim:summary -->

## Notes

ship friday

## Transcript

Me: we ship friday
```

Afterwards I wrote this line.
"""
MEMO = "---\ntype: meeting-note\n---\n\n# Call\n\none\ntwo\n\n## Transcript\n\nMe: hi\n"


# --- what a recording keeps, and which paths are notes --------------------------------------

def test_a_recordings_own_parts_are_its_frontmatter_its_block_and_its_transcript():
    take = lambda text: [text[begin:end] for begin, end in edits.protected_ranges(text)]
    assert take(MEETING) == [MEETING[:MEETING.index("---\n\n") + 4],
                             MEETING[MEETING.index("```slim"):MEETING.index("\nAfterwards")]]
    assert take(MEMO) == ["---\ntype: meeting-note\n---\n", "## Transcript\n\nMe: hi\n"]
    assert edits.protected_ranges("---\ntags: [a]\n---\n\n## Transcript of a talk\n") == []


def test_backslash_colon_and_case_variant_paths_are_refused():
    for raw in ["Notes\\..\\..\\escaped.md", "Attachments\\x.md", "C:x.md", "attachments/x.md",
                "_reflections/x.md", "Notes/.obsidian/x.md", "Notes/café.md"]:
        assert edits.note_path(raw) is None, raw
    assert edits.note_path("Notes/work/a note.md") == "Notes/work/a note.md"


# --- blocks and handles ----------------------------------------------------------------------

def test_blocks_rebuild_the_note_and_render_handles():
    note = Note(NOTE)
    assert note.text() == NOTE
    rendered = note.render()
    assert rendered.startswith("[b1] # Title\n")
    assert "[b3] ## A\n" in rendered and "[b4] Text a.\n" in rendered and "[b5] - one\n" in rendered
    assert " s" not in "".join(line.split("]")[0] for line in rendered.splitlines())


def test_a_section_runs_to_the_next_heading_of_its_level():
    note = Note(NOTE)
    i, j = note.span("s3")
    assert "".join(b.text for b in note.blocks[i:j]) == "## A\n\nText a.\n- one\n- two\n\n"
    assert note.span("s1") == (0, len(note.blocks))


def test_unknown_or_malformed_handles_are_refused_with_the_valid_range():
    note = Note(NOTE)
    with pytest.raises(ToolError, match="b1–b8"):
        note.span("b99")
    with pytest.raises(ToolError, match="not a handle"):
        note.span("Text a.")


def test_labels_copied_into_markdown_are_stripped():
    assert strip_labels("[b3 s3] ## A\n[b4] Text\n[locked] x\n") == "## A\nText\nx"


def test_splices_turn_before_into_after_in_utf16_units():
    before = "😀 one\nkeep\nthree\n"
    after = "😀 one\nkeep\nthree changed\nnew\n"
    found = splices(before, after)
    assert apply_splices(before, found) == after
    assert found[0]["from"] == 12          # the emoji is 2 UTF-16 units: 2 + len(" one\nkeep\n")


def test_splices_diff_only_the_lines_between_what_both_texts_share_at_start_and_end(monkeypatch):
    # difflib is quadratic in repeated lines: over a whole long note (4,000 blank lines) every
    # change took a second, inside the model lock.
    before = "".join(f"## S{k}\n\nPara {k}.\n\n" for k in range(500))
    seen, real = [], edits.difflib.SequenceMatcher
    monkeypatch.setattr(edits.difflib, "SequenceMatcher",
                        lambda junk, a, b, **kw: seen.append((len(a), len(b))) or real(junk, a, b, **kw))
    for after in (before.replace("Para 250.\n", "Para 250, longer.\nAnd a line.\n"),
                  "New top.\n\n" + before, before + "New end.\n", before[:-1], before):
        found = splices(before, after)
        assert apply_splices(before, found) == after
        assert max(seen.pop()) <= 2, "diffed lines both texts share"
    assert splices(before, before) == []
    edges = [("a\nb", "a\nc"), ("a\n", "a"), ("", "x\n"), ("x\n\ny\n", ""), ("x\n\n\ny\n", "x\n\ny\n"),
             ("p\n\n", "p\n\n\n")]
    for old, new in edges:
        assert apply_splices(old, splices(old, new)) == new


def test_replace_keeps_spacing_and_one_block_keeps_its_handle():
    note = Note(NOTE)
    i, j = note.span("b3")
    new = note.replace(i, j, "## A renamed", owner="c1", keep="b3")
    assert note.text() == NOTE.replace("## A\n", "## A renamed\n")
    assert new[0].handle == "b3" and new[0].owner == "c1"


def test_delete_insert_and_move_keep_blocks_apart():
    note = Note(NOTE)
    i, j = note.span("s7")
    note.replace(i, j, "", owner="c1")
    assert note.text() == "# Title\n\nIntro para.\n\n## A\n\nText a.\n- one\n- two\n\n"
    note = Note(NOTE)
    note.insert(note.span("b4")[1], "New para.", owner="c2")
    assert note.text() == NOTE.replace("Text a.\n- one", "Text a.\n\nNew para.\n\n- one")
    note = Note(NOTE)
    i, j = note.span("s7")
    note.move(i, j, note.span("s3")[0])
    assert note.text() == "# Title\n\nIntro para.\n\n## B\n\nText b.\n\n## A\n\nText a.\n- one\n- two\n\n"


def test_a_recordings_protected_blocks_are_locked_and_unlabelled():
    note = Note(RECORDING)
    assert [b.locked for b in note.blocks] == [True, True, False]
    assert note.render().count("[locked]") == 2
    with pytest.raises(ToolError, match="part of the recording"):
        note.target("b1")


def test_nothing_goes_after_a_transcript_that_ends_the_file():
    note = Note(f"# Memo\n\nSummary.\n\n{edits.TRANSCRIPT_MARKER}\nwords words\n")
    assert note.tail_locked
    with pytest.raises(ToolError, match="after the transcript"):
        note.check_insert(len(note.blocks))
    note.check_insert(1)


def test_a_target_is_read_from_its_first_line_and_its_handles():
    note = Note(NOTE)
    assert note.span("b4]\n- **Use cases**: pasted text") == note.span("b4")
    assert note.span("[b4] Text a.") == note.span("b4")
    assert note.span("b4-b5") == (note.span("b4")[0], note.span("b5")[1])
    assert note.span("b2 – s3") == (note.span("b2")[0], note.span("s3")[1])
    with pytest.raises(ToolError, match="from the earlier handle to the later one"):
        note.span("b5-b4")


def test_handles_listed_are_one_target_only_when_nothing_lies_between_them():
    note = Note(NOTE)
    for listed in ("b4, b8", "b4 and b8", "[b4] [b8]", "b2, b5-b6"):
        with pytest.raises(ToolError, match="not next to each other: make one call for each") as refused:
            note.name(listed)
        assert refused.value.code == "list"
    assert note.span("b4, b5") == note.span("b4 b5") == note.span("b4-b5")
    assert note.span("b4 through b6") == note.span("b4 to b6") == note.span("b4–b6")
    assert note.span("s3 (b3–b6)") == note.span("s3  b3–b6  ## A") == note.span("s3")    # a section map line
    assert note.span("s3, s7") == (note.span("s3")[0], len(note.blocks))                 # two sections in a row


def test_a_heading_label_with_both_handles_is_refused_not_guessed():
    note = Note(NOTE)
    with pytest.raises(ToolError, match="b3 is the heading line alone and s3 is the heading with everything under it"):
        note.span("b3 s3")


def test_s_on_a_block_that_is_not_a_heading_means_that_block():
    note = Note(NOTE)
    assert note.span("s4") == note.span("b4") and note.name("s4") == "b4" and note.name("s3") == "s3"
    assert note.span("s4-b5") == (note.span("b4")[0], note.span("b5")[1]) and note.name("s4-b5") == "b4-b5"
    assert note.span("b4 s4") == note.span("b4")


LIST = ("## Why is NLP Hard?\n- **Lexical**: bank/rivier vs bank/money.\n- **Pragmatics**: salt.\n"
        "- **Syntactic**: attachment.\n  - Example: the woman with the sandwich.\n  continued line.\n"
        "- **Open vocabulary**: doomscrolling.\n1. **Numbered**: too.\n\n## Next\n")


def test_every_top_level_list_item_is_its_own_block():
    note = Note(LIST)
    items = [b for b in note.blocks if b.kind == "list"]
    assert note.text() == LIST and len(items) == 5 and len({b.handle for b in items}) == 5
    assert items[2].text == "- **Syntactic**: attachment.\n  - Example: the woman with the sandwich.\n  continued line.\n"
    assert items[-1].text == "1. **Numbered**: too.\n\n"


def test_fixing_one_item_keeps_the_others():
    note = Note(LIST)
    i, j = note.span("b2")
    note.replace(i, j, "- **Lexical**: bank/river vs bank/money.", owner="c1", keep="b2")
    assert note.text() == LIST.replace("rivier", "river")


def test_a_list_stays_tight_when_an_item_comes_or_goes():
    text = "## L\n\n- a\n- b\n- c\n\n## Next\n"
    note = Note(text)
    i, j = note.span("b3")
    note.replace(i, j, "", owner="c1")
    assert note.text() == "## L\n\n- a\n- c\n\n## Next\n"
    note.insert(note.span("b2")[1], "- a2", owner="c2")
    note.insert(note.span("b4")[1], "- d", owner="c3")
    assert note.text() == "## L\n\n- a\n- a2\n- c\n- d\n\n## Next\n"
    note.insert(note.span("b2")[1], "A paragraph.", owner="c4")
    assert note.text() == "## L\n\n- a\n\nA paragraph.\n\n- a2\n- c\n- d\n\n## Next\n"


def test_an_item_moved_within_its_list_keeps_the_lists_spacing(monkeypatch):
    # A move's splices are its own (`_move_splices`), never a line diff of the note.
    monkeypatch.setattr(edits, "splices", lambda *_: pytest.fail("the move's own splices did not rebuild it"))
    text = "## L\n\n- one\n- two\n- three\n\n## Next\n"
    for args, moved in (({"target": "b3", "after": "b4"}, "- one\n- three\n- two\n"),      # to the end
                        ({"target": "b4", "before": "b2"}, "- three\n- one\n- two\n"),     # to the top
                        ({"target": "b2", "after": "b3"}, "- two\n- one\n- three\n")):
        note = Note(text)
        change, _ = apply(note, "move", args, change_id="c1", step=1)
        assert note.text() == f"## L\n\n{moved}\n## Next\n"
        assert apply_splices(text, change.splices) == note.text()
    note = Note("- one\n- two\n- three")                    # the list ends the note: only the order changes
    apply(note, "move", {"target": "b2", "after": "b3"}, change_id="c1", step=1)
    assert note.text() == "- one\n- three\n- two"
    loose = "- a\n\n- b\n\n- c\n\n## Next\n"                  # a loose list stays loose
    note = Note(loose)
    apply(note, "move", {"target": "b3", "before": "b2"}, change_id="c1", step=1)
    assert note.text() == "- a\n\n- c\n\n- b\n\n## Next\n"
    out = "## L\n\n- a\n- b\n\nPara.\n\n## Next\n"           # out of its list: a block of its own, as before
    note = Note(out)
    apply(note, "move", {"target": "b3", "after": "b4"}, change_id="c1", step=1)
    assert note.text() == "## L\n\n- a\n\nPara.\n\n- b\n\n## Next\n"


def test_a_range_past_the_end_of_the_note_runs_to_its_end():
    note = Note(NOTE)
    assert note.span("b7-b9") == (note.span("b7")[0], len(note.blocks)) and note.name("b7-b9") == "b7-b8"
    assert note.span("s7-b25") == note.span("s7") and note.name("s7-b25") == "s7-b8"
    with pytest.raises(ToolError, match="there is no b98"):
        note.span("b98-b99")
    note.replace(*note.span("b5"), "", owner="c1")
    with pytest.raises(ToolError, match="there is no b5"):      # gone, not past the end
        note.span("b4-b5")


def test_a_handle_the_run_made_and_replaced_is_gone_not_past_the_end():
    note = Note(NOTE)                                               # b1–b8
    note.replace(*note.span("b4"), "One.\n\nTwo.", owner="c1")     # b9, b10
    note.replace(*note.span("b10"), "Three.\n\nFour.", owner="c2")  # b11, b12: b10 is gone
    with pytest.raises(ToolError, match="there is no b10"):
        note.span("b2-b10")
    assert note.span("b2-b13") == (note.span("b2")[0], len(note.blocks))    # never made: to the end


def test_the_sections_map_names_each_heading_with_its_first_and_last_block():
    assert Note(NOTE).sections() == "s1  b1–b8  # Title\n  s3  b3–b6  ## A\n  s7  b7–b8  ## B"
    assert Note("## Alone\n\n## Two\n\nx\n").sections() == "s1  b1  ## Alone\ns2  b2–b3  ## Two"
    assert Note(RECORDING).sections() == ""
    memo = Note(f"# Memo\n\nSummary.\n\n{edits.TRANSCRIPT_MARKER}\nwords\n")
    assert memo.sections() == "s1  b1–b4  # Memo"             # the transcript's own heading is locked: not listed


def test_headings_are_compared_by_level_and_words_without_a_trailing_note():
    assert edits.headings("# T\n\ntext # not\n## A  \n```\n") == ["# T", "## A"]
    assert edits.heading_key("##  Toolkit (New Section)") == edits.heading_key("## toolkit") == (2, "toolkit")
    assert edits.heading_key("## Bag of Words (BoW)") == (2, "bag of words (bow)")


# --- the tools -------------------------------------------------------------------------------

def test_a_rename_is_one_change_whose_splices_rebuild_the_text():
    note = Note(NOTE)
    before = note.text()
    change, result = apply(note, "replace", {"target": "b3", "markdown": "[b3 s3] ## A renamed"},
                           change_id="c1", step=1)
    assert note.text() == before.replace("## A\n", "## A renamed\n")
    assert apply_splices(before, change.splices) == note.text()
    assert change.needs == [] and "[b3] ## A renamed" in result


def test_a_later_change_inside_new_text_needs_the_change_that_made_it():
    note = Note(NOTE)
    apply(note, "insert", {"after": "s7", "markdown": "## C\n\nText c."}, change_id="c1", step=1)
    heading = next(b.handle for b in note.blocks if b.text.startswith("## C"))
    change, _ = apply(note, "replace", {"target": heading, "markdown": "## C, renamed"}, change_id="c2", step=1)
    assert change.needs == ["c1"]


def test_moving_a_section_into_itself_or_onto_itself_is_refused():
    note = Note(NOTE)
    with pytest.raises(ToolError, match="inside"):
        apply(note, "move", {"target": "s1", "after": "b4"}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="already there"):
        apply(note, "move", {"target": "s3", "before": "s3"}, change_id="c1", step=1)


def test_insert_needs_exactly_one_place_and_some_text():
    note = Note(NOTE)
    with pytest.raises(ToolError, match="exactly one"):
        apply(note, "insert", {"markdown": "x"}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="empty"):
        apply(note, "insert", {"after": "b2", "markdown": "  "}, change_id="c1", step=1)


def test_a_place_written_null_is_no_place():
    # The reader keeps a text parameter's "null" (a note may say null); a place that says it names no
    # block, as when Ollama's parser made it None.
    note = Note(NOTE)
    apply(note, "insert", {"after": "null", "before": "b3", "markdown": "New."}, change_id="c1", step=1)
    assert note.text() == NOTE.replace("## A", "New.\n\n## A")
    with pytest.raises(ToolError, match="target is missing"):
        apply(Note(NOTE), "replace", {"target": "Null", "markdown": "x"}, change_id="c1", step=1)


def test_the_selection_limits_what_may_change():
    note = Note(NOTE)
    with pytest.raises(ToolError, match="outside the selected"):
        apply(note, "replace", {"target": "b2", "markdown": "x"}, change_id="c1", step=1, only={"b4", "b6"})
    apply(note, "replace", {"target": "b6", "markdown": "- **two**"}, change_id="c1", step=1, only={"b4", "b6"})
    assert "- one\n- **two**\n\n## B" in note.text()


def test_a_recordings_parts_are_refused_and_new_text_goes_after_the_meeting_block():
    note = Note(RECORDING)
    with pytest.raises(ToolError, match="part of the recording"):
        apply(note, "replace", {"target": "b2", "markdown": "x"}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="inside the recording"):
        apply(note, "insert", {"before": "b1", "markdown": "x"}, change_id="c1", step=1)
    before = note.text()
    apply(note, "insert", {"after": "b3", "markdown": "## Takeaways\n\n- one"}, change_id="c1", step=1)
    take = lambda t: [t[s:e] for s, e in edits.protected_ranges(t)]
    assert take(before) == take(note.text()) and note.text().endswith("## Takeaways\n\n- one\n")


PROPS = "---\ntags: [study]\n---\n\n# Title\n\nBody.\n"


def test_new_text_put_above_or_inside_a_notes_properties_goes_right_after_them():
    # Obsidian reads a note's properties only at its very top.
    note = Note(PROPS)
    _, said = apply(note, "insert", {"before": "b1", "markdown": "TL;DR."}, change_id="c1", step=1)
    assert note.text() == PROPS.replace("# Title", "TL;DR.\n\n# Title")
    assert said.startswith("inserted b4 before b1; placed after b1 (the end of the note's properties")
    listed = "---\ntags:\n  - a\n  - b\n---\n\n# Title\n"            # a property list is blocks of its own
    note = Note(listed)
    apply(note, "insert", {"after": "b2", "markdown": "TL;DR."}, change_id="c1", step=1)
    assert note.text() == listed.replace("# Title", "TL;DR.\n\n# Title")
    note = Note(PROPS)
    _, said = apply(note, "move", {"target": "b3", "before": "b1"}, change_id="c1", step=1)
    assert note.text() == "---\ntags: [study]\n---\n\nBody.\n\n# Title\n\n" and "end of the note's properties" in said
    with pytest.raises(ToolError, match="already right after the note's properties"):
        apply(Note(PROPS), "move", {"target": "b2", "before": "b1"}, change_id="c1", step=1)


def test_the_table_gives_ollama_function_schemas():
    [tool] = ollama_tools(["replace"])
    assert tool["type"] == "function" and tool["function"]["name"] == "replace"
    assert tool["function"]["parameters"]["required"] == ["target", "markdown"]
    plan, create = ollama_tools(["plan", "create_note"])
    assert plan["function"]["parameters"]["required"] == ["intro", "outline", "steps"]
    assert create["function"]["parameters"]["required"] == ["path", "markdown"]


def test_a_range_is_replaced_whole_and_a_pasted_label_still_finds_its_block():
    note = Note(NOTE)
    change, result = apply(note, "replace", {"target": "b4-b6", "markdown": "Text a, one and two."},
                           change_id="c1", step=1)
    assert note.text() == NOTE.replace("Text a.\n- one\n- two\n", "Text a, one and two.\n")
    assert change.label == "Rewrote “Text a.”" and "b4-b6" not in [b.handle for b in note.blocks]
    apply(note, "replace", {"target": "b2]\nIntro para.", "markdown": "Intro, shorter."}, change_id="c2", step=1)
    assert "Intro, shorter.\n" in note.text() and any(b.handle == "b2" for b in note.blocks)


def test_every_result_shows_what_follows_the_change():
    note = Note(NOTE)
    _, said = apply(note, "replace", {"target": "b3", "markdown": "## A renamed"}, change_id="c1", step=1)
    assert said.endswith("\nthen: [b4] Text a.")
    _, said = apply(note, "insert", {"after": "b4", "markdown": "New para."}, change_id="c2", step=1)
    assert said.endswith("\nthen: [b5] - one")
    _, said = apply(note, "move", {"target": "s7", "before": "s3"}, change_id="c3", step=1)
    assert said == "moved s7 before s3\nthen: [b3] ## A renamed"
    _, said = apply(note, "replace", {"target": "b8", "markdown": ""}, change_id="c4", step=1)
    assert said.endswith("\nthen: [b3] ## A renamed")
    _, said = apply(note, "replace", {"target": "s3", "markdown": ""}, change_id="c5", step=1)
    assert said == "deleted s3\nthen: (end of the note)"


def test_what_follows_is_cut_to_one_line_of_100_characters():
    note = Note("Short.\n\n" + "x" * 150 + "\nsecond line\n")
    _, said = apply(note, "replace", {"target": "b1", "markdown": "Shorter."}, change_id="c1", step=1)
    assert said.endswith("\nthen: [b2] " + "x" * 100)


def test_s_on_a_paragraph_is_named_back_as_the_block():
    note = Note(NOTE)
    change, said = apply(note, "replace", {"target": "s4", "markdown": "Text a, edited."}, change_id="c1", step=1)
    assert change.label == "Rewrote “Text a.”" and said.startswith("replaced b4;") and "Text a, edited.\n" in note.text()


CUT = "that would cut s3: its text from b4 on would end up under your new heading. Put it after s3 (after that whole section) or before s3."


def test_a_new_heading_before_a_sections_body_is_refused():
    note = Note(NOTE)
    with pytest.raises(ToolError) as refused:
        apply(note, "insert", {"before": "b4", "markdown": "## X\n\nx."}, change_id="c1", step=1)
    assert str(refused.value) == CUT


def test_a_new_heading_after_a_sections_heading_goes_after_the_whole_section():
    note = Note(NOTE)
    change, said = apply(note, "insert", {"after": "b3", "markdown": "## X\n\nx."}, change_id="c1", step=1)
    assert "- two\n\n## X\n\nx.\n\n## B" in note.text() and change.label == "Added “X”"
    assert said.startswith("inserted b9 b10 after b3; placed after s3 (the end of that section) so it doesn't cut it:\n")


def test_a_lower_heading_or_a_paragraph_may_go_inside_a_section():
    note = Note(NOTE)
    apply(note, "insert", {"after": "b4", "markdown": "### X"}, change_id="c1", step=1)
    apply(note, "insert", {"after": "b3", "markdown": "A paragraph."}, change_id="c2", step=1)
    assert "## A\n\nA paragraph.\n\nText a.\n\n### X\n\n- one" in note.text()


THREE = "# Title\n\n## A\n\na\n\n## B\n\nb\n\n## C\n\nc\n"


def test_a_section_moved_after_a_heading_goes_after_that_headings_section():
    note = Note(THREE)
    with pytest.raises(ToolError, match="that would cut s2: its text from b3 on"):
        apply(note, "move", {"target": "s6", "before": "b3"}, change_id="c1", step=1)
    _, said = apply(note, "move", {"target": "s6", "after": "b2"}, change_id="c1", step=1)
    assert note.text() == "# Title\n\n## A\n\na\n\n## C\n\nc\n\n## B\n\nb\n\n"     # (a move off the end leaves a blank)
    assert said == "moved s6 after b2; placed after s2 (the end of that section) so it doesn't cut it\nthen: [b4] ## B"


def test_every_enclosing_section_is_kept_whole_and_the_outermost_is_named():
    nested = "## A\n\nt\n\n### A1\n\nt1\n\n### A2\n\nt2\n\n## B\n\nb\n"
    note = Note(nested)
    with pytest.raises(ToolError, match="that would cut s1: its text from b5 on"):
        apply(note, "insert", {"before": "b5", "markdown": "## X"}, change_id="c1", step=1)
    _, said = apply(note, "insert", {"after": "s3", "markdown": "## X"}, change_id="c1", step=1)
    assert "placed after s1" in said and "t2\n\n## X\n\n## B" in note.text()
    _, said = apply(note, "insert", {"after": "b2", "markdown": "Para.\n\n## Y"}, change_id="c2", step=1)
    assert "placed after s1" in said and "t2\n\nPara.\n\n## Y\n\n## X" in note.text()
    apply(note, "insert", {"after": "s3", "markdown": "### Z"}, change_id="c3", step=1)
    assert "t1\n\n### Z\n\n### A2" in note.text()


def test_an_insert_that_copies_a_section_is_refused_with_the_move_to_use():
    note = Note(NOTE)
    for heading in ("## A", "##  a (New Section)"):
        with pytest.raises(ToolError) as refused:
            apply(note, "insert", {"after": "s7", "markdown": f"{heading}\n\nText a.\n- one\n- two"},
                  change_id="c1", step=1)
        assert str(refused.value) == (f"{heading.strip()} is already in the note as s3; to put that section here, use "
                                      "move s3 after s7, and replace it afterwards if it needs new text.")
    apply(note, "insert", {"before": "s7", "markdown": "## C\n\nText c."}, change_id="c1", step=1)
    assert "## C\n\nText c.\n\n## B" in note.text()


def test_a_repeated_subheading_with_its_own_text_is_not_a_copy():
    """A generic subheading repeats on purpose ("### Example" under each section); only a heading
    that brings most of its twin section's lines with it is a copy (build notes, round 4 K)."""
    text = "# T\n\n## Sigmoid\n\nS.\n\n### Example\n\nsigmoid(0) = 0.5\nit is symmetric\n\n## Softmax\n\nSm.\n"
    note = Note(text)
    apply(note, "insert", {"after": "s6", "markdown": "### Example\n\nsoftmax([0, 0]) = [0.5, 0.5]"},
          change_id="c1", step=1)
    assert note.text().endswith("Sm.\n\n### Example\n\nsoftmax([0, 0]) = [0.5, 0.5]\n")
    note = Note(text)
    with pytest.raises(ToolError, match="### Example is already in the note as s4"):
        apply(note, "insert", {"after": "s6", "markdown": "### Example\n\nsigmoid(0) = 0.5\nit is symmetric"},
              change_id="c1", step=1)
    with pytest.raises(ToolError, match="### Example is already in the note as s4"):   # 2 of 3 lines is most
        apply(note, "insert", {"after": "s6", "markdown": "### Example\n\nsigmoid(0) = 0.5\nnew line"},
              change_id="c1", step=1)


def test_a_replace_may_keep_its_own_heading_but_not_bring_in_another():
    note = Note(NOTE)
    apply(note, "replace", {"target": "s3", "markdown": "## A\n\nText a, shorter."}, change_id="c1", step=1)
    with pytest.raises(ToolError) as refused:
        apply(note, "replace", {"target": "b2", "markdown": "Intro.\n\n## B\n\nText b."}, change_id="c2", step=1)
    assert str(refused.value) == ("## B is already in the note as s7; to put that section here, use move s7 to where "
                                  "you want it, and replace it afterwards if it needs new text.")


def test_a_range_ending_on_the_next_sections_heading_leaves_it_out():
    note = Note(NOTE)
    change, said = apply(note, "replace", {"target": "b4-b7", "markdown": ""}, change_id="c1", step=1)
    assert note.text() == "# Title\n\nIntro para.\n\n## A\n\n## B\n\nText b.\n"
    assert change.label == "Removed “Text a.”"
    assert said == "deleted b4-b6 (b7 is the heading of the next section; left it out)\nthen: [b7] ## B"
    empty = Note("## A\n\nx\n\n## B\n\n## C\n\ny\n")         # ## B's section ends where the range does
    _, said = apply(empty, "replace", {"target": "b2-b3", "markdown": ""}, change_id="c1", step=1)
    assert said == "deleted b2-b3\nthen: [b4] ## C"


def test_new_text_may_not_add_a_recordings_marker_or_skip_lines_with_dots():
    """A `## Transcript` or a meeting block would lock the owner's text below it; a lone `...`
    is the model's "the rest as before", and it would delete the rest."""
    for markdown in (f"Notes.\n\n{edits.TRANSCRIPT_MARKER}\n\nMe: made up", "```slim-meeting\nx\n```"):
        note = Note(NOTE)
        with pytest.raises(ToolError, match="recording's block or transcript heading"):
            apply(note, "insert", {"after": "s7", "markdown": markdown}, change_id="c1", step=1)
        assert note.text() == NOTE
    note = Note(NOTE)
    with pytest.raises(ToolError, match=r"skips lines with `\.\.\.`"):
        apply(note, "replace", {"target": "s3", "markdown": "## A\n\nText a, shorter.\n\n..."}, change_id="c1", step=1)
    assert note.text() == NOTE
    kept = Note("a\n\n...\n\nb\n")                     # dots the note already had may stay
    apply(kept, "replace", {"target": "b1-b3", "markdown": "A\n\n...\n\nb"}, change_id="c1", step=1)
    assert kept.text() == "A\n\n...\n\nb\n"



ITEM = ("## Hard\n\n- **Lexical ambiguity**: multiple meanings (bank/rivier vs bank/money), compounds/idioms "
        "(“take out”), portmanteaus, metaphors.\n- **Open vocabulary**: doomscrolling.\n")


def test_edit_text_fixes_a_word_and_keeps_the_rest_of_the_block_byte_for_byte():
    note = Note(ITEM)
    change, said = apply(note, "edit_text", {"target": "b2", "old": "rivier", "new": "river"}, change_id="c1", step=1)
    assert note.text() == ITEM.replace("rivier", "river") and change.label == "Changed “rivier” to “river”"
    assert said.startswith("edited b2; it now reads:\n[b2] - **Lexical ambiguity**: multiple meanings (bank/river")
    assert said.endswith("\nthen: [b3] - **Open vocabulary**: doomscrolling.")


def test_edit_text_refuses_text_it_cannot_find_once():
    note = Note(ITEM)
    with pytest.raises(ToolError) as refused:
        apply(note, "edit_text", {"target": "b2", "old": "rivers", "new": "x"}, change_id="c1", step=1)
    assert str(refused.value) == f"'rivers' is not in b2; it reads: {ITEM.split(chr(10))[2]}"[:len("'rivers' is not in b2; it reads: ") + 300]
    assert refused.value.code == "quote"
    with pytest.raises(ToolError, match="'bank' appears 2 times in b2; quote a little more around it"):
        apply(note, "edit_text", {"target": "b2", "old": "bank", "new": "shore"}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="old is empty"):
        apply(note, "edit_text", {"target": "b2", "old": "", "new": "x"}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="part of the recording"):
        apply(Note(RECORDING), "edit_text", {"target": "b2", "old": "id: 1", "new": "id: 2"}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="outside the selected"):
        apply(note, "edit_text", {"target": "b3", "old": "doom", "new": "x"}, change_id="c1", step=1, only={"b2"})
    assert note.text() == ITEM


def test_edit_text_may_delete_words_and_is_offered_beside_replace():
    note = Note(ITEM)
    apply(note, "edit_text", {"target": "b2", "old": ", portmanteaus", "new": ""}, change_id="c1", step=1)
    assert "(“take out”), metaphors." in note.text()
    [tool] = ollama_tools(["edit_text"])
    assert tool["function"]["parameters"]["required"] == ["target", "old", "new"]
    assert tool["function"]["description"].startswith("Change a few words inside one block")
    assert list(edits.TOOLS).index("edit_text") == list(edits.TOOLS).index("replace") + 1


def test_a_replace_or_edit_text_without_its_text_is_refused_never_a_deletion():
    note = Note(NOTE)
    with pytest.raises(ToolError, match="^markdown is missing; this call gave target, content$"):
        apply(note, "replace", {"target": "b3", "content": "## A fixed"}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="^new is missing; this call gave target, old, replacement$"):
        apply(note, "edit_text", {"target": "b4", "old": "Text a.", "replacement": "Text A."}, change_id="c1", step=1)
    with pytest.raises(ToolError, match="^markdown is missing; this call gave target$"):
        apply(note, "replace", {"target": "b3", "markdown": None}, change_id="c1", step=1)
    assert note.text() == NOTE
    apply(note, "edit_text", {"target": "b4", "old": " a.", "new": ""}, change_id="c1", step=1)     # empty on purpose
    apply(note, "replace", {"target": "b2", "markdown": ""}, change_id="c2", step=1)
    assert note.text() == "# Title\n\n## A\n\nText\n- one\n- two\n\n## B\n\nText b.\n"


def test_a_section_replaced_by_only_a_heading_is_a_rename_and_keeps_its_text():
    for target in ("s3", "b3-b6"):
        note = Note(NOTE)
        change, said = apply(note, "replace", {"target": target, "markdown": "## New name"}, change_id="c1", step=1)
        assert note.text() == NOTE.replace("## A\n", "## New name\n") and change.label == "Renamed “A” to “New name”", target
        assert said == "renamed the heading; the section's text is kept:\n[b3] ## New name\nthen: [b4] Text a."
    note = Note(NOTE)
    _, said = apply(note, "replace", {"target": "b3", "markdown": "## A2"}, change_id="c1", step=1)
    assert note.text() == NOTE.replace("## A\n", "## A2\n") and said.startswith("replaced b3; it now reads:")
    note = Note(NOTE)
    apply(note, "replace", {"target": "s3", "markdown": "## A\n\nAll new."}, change_id="c1", step=1)
    assert note.text() == "# Title\n\nIntro para.\n\n## A\n\nAll new.\n\n## B\n\nText b.\n"

# --- a new note -------------------------------------------------------------------------------

@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Notes/school").mkdir(parents=True)
    (root / "Notes/school/a.md").write_text("alpha\nbeta\n", encoding="utf-8")
    (root / "Journal").mkdir()
    (root / "Inbox/Journal").mkdir(parents=True)
    return root


def test_a_new_note_goes_in_an_existing_folder_and_says_who_made_it(vault):
    made = edits.create_note(vault, "Notes/school/new.md", "# New\n\nhello", folder="Notes/school")
    assert made == {"path": "Notes/school/new.md", "text": "---\norigin: copilot\n---\n\n# New\n\nhello\n"}
    assert edits.create_note(vault, "`Topic brief.md`", "x", folder="Notes/school")["path"] == "Notes/school/Topic brief.md"
    assert edits.create_note(vault, "Notes/x.md", "---\ntags: [a]\n---\n\nx\n", folder="Notes/school")["text"] == \
        "---\ntags: [a]\norigin: copilot\n---\n\nx\n"
    assert not (vault / "Notes/school/new.md").exists()                 # nothing is written here


@pytest.mark.parametrize("raw, folder, reason", [
    ("../escape.md", "Notes/school", "not a note path in this vault"),
    ("Notes/school/pic.png", "Notes/school", "not a note path in this vault"),
    ("Notes/school/a.md", "Notes/school", "already exists"),
    ("Notes/school/A.md", "Notes/school", "already exists"),
    ("Notes/elsewhere/new.md", "Notes/school", "the folder Notes/elsewhere/ does not exist"),
    ("notes/School/new.md", "Notes/school", "the folder notes/School/ does not exist"),
    ("Journal/new.md", "Notes/school", "never creates a note in Journal/"),
    ("Inbox/Journal/new.md", "Notes/school", "never creates a note in Journal/"),
    ("Notes/school/new.md", "Journal", "a Journal note's text stays in Journal/"),
    ("Notes/school/new.md", "Inbox/Journal", "a Journal note's text stays in Journal/"),
])
def test_a_new_note_is_refused_where_it_may_not_go(vault, raw, folder, reason):
    with pytest.raises(ToolError, match=reason):
        edits.create_note(vault, raw, "x", folder=folder)


def test_a_new_note_needs_text_and_a_size_the_sidebar_can_carry(vault):
    with pytest.raises(ToolError, match="empty"):
        edits.create_note(vault, "Notes/school/new.md", " \n", folder="Notes/school")
    with pytest.raises(ToolError, match="too large"):
        edits.create_note(vault, "Notes/school/new.md", "x" * (edits.MAX_EDIT_CHARS + 1), folder="Notes/school")


# --- I2: a change never touches a recording's own parts, whatever a diff would make of it -------

LONG = "---\ntitle: x\n---\n\n## Top\n\ntop\n\n```slim-meeting\nid: 1\nkind: x\n```\n\n## Mine\n\n" + "".join(
    f"line {i}\n" for i in range(30))


def _u16_ranges(text):
    return [(edits.u16(text[:s]), edits.u16(text[:e])) for s, e in edits.protected_ranges(text)]


def test_a_move_is_its_chunk_taken_out_and_put_back_never_a_neighbour_moved_round_it():
    note = Note(LONG)
    assert [b.handle for b in note.blocks if b.locked] == ["b1", "b4"]
    before = note.text()
    change, _ = apply(note, "move", {"target": "s5", "before": "b2"}, change_id="c1", step=1)
    assert note.text().startswith("---\ntitle: x\n---\n\n## Mine\n\nline 0\n")
    assert apply_splices(before, change.splices) == note.text()
    (meeting_from, meeting_to) = _u16_ranges(before)[1]
    for splice in change.splices:            # a line diff kept the 30 lines and moved the meeting block
        assert splice["to"] <= meeting_from or splice["from"] >= meeting_to, splice


def test_any_change_whose_splices_touch_a_recordings_parts_is_refused_and_the_note_left_as_it_was(monkeypatch):
    note = Note(LONG)
    (meeting_from, meeting_to) = _u16_ranges(LONG)[1]
    monkeypatch.setattr(edits, "splices", lambda before, after: [
        {"from": meeting_from, "to": meeting_to + 6, "insert": "```slim-meeting\nid: 1\nkind: x\n```\n\ntop!\n"}])
    with pytest.raises(ToolError, match="part of the recording") as refused:
        apply(note, "replace", {"target": "b3", "markdown": "top!"}, change_id="c1", step=1)
    assert refused.value.code == "locked" and note.text() == LONG and [b.handle for b in note.blocks][:3] == ["b1", "b2", "b3"]


def test_a_change_that_changes_nothing_leaves_the_note_and_its_handles_as_they_were():
    note = Note(NOTE)
    with pytest.raises(ToolError, match="changes nothing"):
        apply(note, "replace", {"target": "b4-b6", "markdown": "Text a.\n- one\n- two"}, change_id="c1", step=1)
    assert [b.handle for b in note.blocks] == [f"b{n}" for n in range(1, 9)]


# --- a run never rewrites its own earlier steps; a rewrite of its own text is part of that change ---

def _added(note, change_id="c1", step=1):
    apply(note, "insert", {"after": "s7", "markdown": "## C\n\nText c."}, change_id=change_id, step=step)
    return next(b.handle for b in note.blocks if b.text.startswith("Text c."))


def test_a_later_step_may_not_rewrite_or_delete_what_an_earlier_step_wrote():
    for tool, args in (("replace", {"markdown": "Text c, longer."}), ("replace", {"markdown": ""}),
                       ("edit_text", {"old": "Text c", "new": "Text C"})):
        note = Note(NOTE)
        para = _added(note)
        before = note.text()
        with pytest.raises(ToolError) as refused:
            apply(note, tool, {"target": para, **args}, change_id="c2", step=2)
        assert refused.value.code == "own" and note.text() == before
        assert str(refused.value) == (f"{para} is what you wrote in step 1; it stays as written. Make only what "
                                      "this step still needs; if the note already has it, reply without a "
                                      "tool call.")


def test_the_same_step_may_rework_its_own_text_and_that_change_is_part_of_the_first():
    note = Note(NOTE)
    para = _added(note)
    change, _ = apply(note, "replace", {"target": para, "markdown": "Text c, longer."}, change_id="c2", step=1)
    assert change.part_of == "c1" and change.needs == ["c1"]
    again = next(b.handle for b in note.blocks if b.text.startswith("Text c, longer."))
    change, _ = apply(note, "edit_text", {"target": again, "old": "longer", "new": "long"}, change_id="c3", step=1)
    assert change.part_of == "c1"           # the first change it grew from, never the one between


def test_owner_text_in_the_target_makes_a_change_of_its_own():
    note = Note(NOTE)
    _added(note)
    mixed = next(b.handle for b in note.blocks if b.text.startswith("Text b."))
    last = next(b.handle for b in note.blocks if b.text.startswith("Text c."))
    change, _ = apply(note, "replace", {"target": f"{mixed}-{last}", "markdown": "B and C."}, change_id="c2", step=1)
    assert change.part_of is None
    change, _ = apply(Note(NOTE), "insert", {"after": "s7", "markdown": "More."}, change_id="c1", step=1)
    assert change.part_of is None


def test_moving_what_an_earlier_step_wrote_is_allowed_and_part_of_it():
    note = Note(NOTE)
    _added(note)
    heading = next(b.handle for b in note.blocks if b.text.startswith("## C"))
    change, _ = apply(note, "move", {"target": "s" + heading[1:], "before": "s3"}, change_id="c2", step=2)
    assert change.part_of == "c1" and note.text().index("## C") < note.text().index("## A")


HANDLE = __import__("re").compile(r"\b[bs]\d+\b")


def test_labels_name_the_text_never_a_handle():
    note = Note(NOTE)
    made = [apply(note, "insert", {"after": "s7", "markdown": "> [!note]- TVM explanation\n> Short."},
                  change_id="c1", step=1)[0],
            apply(note, "replace", {"target": "b4", "markdown": "Text a, simpler."}, change_id="c2", step=1)[0],
            apply(note, "replace", {"target": "b3", "markdown": "## A renamed"}, change_id="c3", step=1)[0],
            apply(note, "edit_text", {"target": "b2", "old": "Intro", "new": "Opening"}, change_id="c4", step=1)[0],
            apply(note, "move", {"target": "s7", "before": "s3"}, change_id="c5", step=1)[0],
            apply(note, "replace", {"target": "b5-b6", "markdown": ""}, change_id="c6", step=1)[0]]
    assert [c.label for c in made] == ["Added “TVM explanation”", "Rewrote “Text a.”", "Renamed “A” to “A renamed”",
                                       "Changed “Intro” to “Opening”", "Moved “B”", "Removed “one”"]
    assert not any(HANDLE.search(c.label) for c in made)
    long = apply(Note("![[Screenshot 1.png|597]]\n\n" + "word " * 30 + "\n"), "replace",
                 {"target": "b2", "markdown": "x"}, change_id="c1", step=1)[0]
    assert long.label == "Rewrote “word word word word word word word word…”"


def test_handles_in_words_for_the_owner_become_the_text_they_point_at():
    note = Note("![[Screenshot 1.png|597]]\n\n## Quick Checks\n\nText.\n")
    assert note.plain("Insert the header and opening tag after block b1") == "Insert the header and opening tag after the image"
    assert note.plain("Move s2 before [b1], then fix b3-b3") == "Move “Quick Checks” before the image, then fix “Text.”"
    assert note.plain("Tidy b99 and the rest") == "Tidy b99 and the rest"     # bare and unknown: a word, kept
    assert note.plain("Tidy block b99 and [b98] the rest") == "Tidy and the rest"
    assert note.plain("Use web3 and s2 buckets") == "Use web3 and “Quick Checks” buckets"


def test_new_html_blocks_are_refused_with_obsidians_own_ways_to_fold():
    note = Note(NOTE)
    with pytest.raises(ToolError) as refused:
        apply(note, "insert", {"after": "b1", "markdown": "<div class=\"why\">\n\nText\n</div>"},
              change_id="c1", step=1)
    assert refused.value.code == "html" and note.text() == NOTE
    assert "> [!note]- Title" in str(refused.value) and "heading already folds" in str(refused.value)
    kept = Note("<div>a</div>\n\nb\n")                # HTML the note already has may stay
    apply(kept, "replace", {"target": "b1", "markdown": "<div>A</div>"}, change_id="c1", step=1)
    assert kept.text() == "<div>A</div>\n\nb\n"
    apply(Note(NOTE), "insert", {"after": "b1", "markdown": "Line one<br>line two <kbd>x</kbd>."}, change_id="c1", step=1)


def test_a_quote_of_long_words_is_cut_at_a_word_by_characters_too():
    said = edits.quote("Use cases: sentiment/hate-speech detection, knowledge discovery, dialogue/chatbots, documents")
    assert said == "“Use cases: sentiment/hate-speech detection, knowledge…”"
    assert len(said) <= edits.QUOTE_CHARS + 3


def test_html_shown_as_code_is_text_not_an_html_block():
    note = Note(NOTE)
    apply(note, "insert", {"after": "b1", "markdown": "Use `<details>` to fold.\n\n```html\n<div>x</div>\n```"},
          change_id="c1", step=1)
    assert "```html\n<div>x</div>\n```" in note.text()



# --- review (2026-10-13): what a change actually changes decides; the owner's text is never a draft ---

def test_a_range_over_owner_text_still_may_not_rewrite_an_earlier_steps_draft():
    note = Note("Owner.\n")
    apply(note, "insert", {"after": "b1", "markdown": "Draft."}, change_id="c1", step=1)
    with pytest.raises(ToolError) as refused:
        apply(note, "replace", {"target": "b1-b2", "markdown": "Owner.\n\nDifferent draft."}, change_id="c2", step=2)
    assert refused.value.code == "own"


def test_renaming_a_heading_an_earlier_step_wrote_is_refused_whatever_its_body():
    note = Note(NOTE)
    _added(note)
    heading = next(b.handle for b in note.blocks if b.text.startswith("## C"))
    with pytest.raises(ToolError, match="is what you wrote in step 1"):
        apply(note, "replace", {"target": "s" + heading[1:], "markdown": "## C2"}, change_id="c2", step=2)


def test_a_retyped_range_keeps_the_owner_s_unchanged_blocks_theirs():
    note = Note("## H\n\nOwner body.\n")
    apply(note, "edit_text", {"target": "s1", "old": "## H", "new": "## Title"}, change_id="c1", step=1)
    body = next(b.handle for b in note.blocks if b.text.startswith("Owner body."))
    change, _ = apply(note, "replace", {"target": body, "markdown": ""}, change_id="c2", step=1)
    assert change.part_of is None               # the owner's own text went: never hidden as SLIM's draft


def test_two_steps_may_each_fix_a_word_in_the_same_owner_paragraph():
    note = Note(ITEM)
    apply(note, "edit_text", {"target": "b2", "old": "rivier", "new": "river"}, change_id="c1", step=1)
    change, _ = apply(note, "edit_text", {"target": "b2", "old": "portmanteaus", "new": "blends"}, change_id="c2", step=2)
    assert change.part_of is None and "river" in note.text() and "blends" in note.text()


def test_rewording_text_slim_edited_from_the_owner_s_is_not_part_of_that_edit():
    note = Note(NOTE)
    apply(note, "replace", {"target": "b4", "markdown": "Text a, simpler."}, change_id="c1", step=1)
    change, _ = apply(note, "replace", {"target": "b4", "markdown": "Text a, simplest."}, change_id="c2", step=1)
    assert change.part_of is None and change.needs == ["c1"]


def test_ordinary_words_that_look_like_handles_are_left_alone():
    note = Note("One.\n\nTwo.\n")
    assert note.plain("Use an s3 bucket and vitamin b12; web3.") == "Use an s3 bucket and vitamin b12; web3."


def test_existing_html_is_no_allowance_for_a_different_block():
    note = Note("<div>x</div>\n\nb\n")
    with pytest.raises(ToolError) as refused:
        apply(note, "replace", {"target": "b1", "markdown": "<div>\nraw markdown"}, change_id="c1", step=1)
    assert refused.value.code == "html"


@pytest.mark.parametrize("markdown", ["```html\n<div>x</div>\n````", "  ```\n  <div>x</div>\n  ```",
                                      "```\n<details>never closed", "Use ``<div>`` here."])
def test_html_inside_any_kind_of_code_is_text(markdown):
    note = Note(NOTE)
    apply(note, "insert", {"after": "b1", "markdown": markdown}, change_id="c1", step=1)


def test_html_after_an_unmatched_backtick_is_not_code():
    with pytest.raises(ToolError, match="HTML tags"):
        apply(Note(NOTE), "insert", {"after": "b1", "markdown": "Use ``<div>` here."}, change_id="c1", step=1)


def test_a_quote_of_one_long_word_is_cut_too():
    assert len(edits.quote("x" * 100)) <= edits.QUOTE_CHARS + 3 and edits.quote("x" * 100).endswith("…”")


# --- second review (2026-10-13) ---

def test_a_change_to_owner_text_and_an_earlier_steps_draft_together_is_refused_naming_the_draft():
    note = Note("Owner.\n")
    apply(note, "insert", {"after": "b1", "markdown": "Draft."}, change_id="c1", step=1)
    draft = next(b.handle for b in note.blocks if b.text.startswith("Draft."))
    with pytest.raises(ToolError) as refused:
        apply(note, "replace", {"target": f"b1-{draft}", "markdown": "Owner changed.\n\nDraft changed."},
              change_id="c2", step=2)
    assert refused.value.code == "own" and str(refused.value).startswith(f"{draft} is what you wrote in step 1;")
    change, _ = apply(note, "replace", {"target": f"b1-{draft}", "markdown": "Owner changed.\n\nDraft."},
                      change_id="c2", step=2)                       # the draft kept as written: fine
    assert "Owner changed.\n\nDraft.\n" in note.text()


def test_a_new_html_line_needs_an_old_line_with_the_same_tags():
    with pytest.raises(ToolError, match="HTML tags"):
        apply(Note("<div>x</div>\n\nKeep.\n"), "replace", {"target": "b1-b2", "markdown": "Keep.\n\n<div>**new**"},
              change_id="c1", step=1)


def test_a_backtick_in_a_fences_info_string_makes_it_no_fence_and_a_span_may_run_over_a_line():
    with pytest.raises(ToolError, match="HTML tags"):
        apply(Note(NOTE), "insert", {"after": "b1", "markdown": "```bad`info\n\n<div>raw</div>"}, change_id="c1", step=1)
    apply(Note(NOTE), "insert", {"after": "b1", "markdown": "Use `<div>\nexample` here."}, change_id="c1", step=1)


def test_an_answer_keeps_bare_words_that_look_like_handles():
    note = Note("".join(f"Paragraph {n}.\n\n" for n in range(1, 13)))
    assert note.plain("Use an s3 bucket.", bare=False) == "Use an s3 bucket."
    assert note.plain("See block b3 and [b4].", bare=False) == "See “Paragraph 3.” and “Paragraph 4.”."
    assert note.plain("Move s3.") == "Move “Paragraph 3.”."


# --- a new <details> is written as the callout Obsidian folds (2026-10-13) ---

TVM = ("<details>\n<summary>TVM Explanation</summary>\n\nMoney now is worth more than later.\n\n"
       "$$PV = \\frac{FV}{(1+r)^n}$$\n\n</details>")


def test_a_new_details_is_written_as_a_callout_that_starts_folded():
    note = Note(NOTE)
    change, said = apply(note, "insert", {"after": "b1", "markdown": TVM}, change_id="c1", step=1)
    assert note.text().startswith("# Title\n\n> [!note]- TVM Explanation\n> Money now is worth more than later.\n>\n"
                                  "> $$PV = \\frac{FV}{(1+r)^n}$$\n\nIntro para.\n")
    assert "callout" in said and "<details>" not in said
    assert change.label == "Added “TVM Explanation”"


def test_details_open_starts_unfolded_and_one_never_closed_runs_to_the_end():
    note = Note(NOTE)
    apply(note, "insert", {"after": "b1", "markdown": "<details open><summary>Why</summary>\nA.\n\nB."},
          change_id="c1", step=1)
    assert "# Title\n\n> [!note]+ Why\n> A.\n>\n> B.\n\nIntro para." in note.text()


def test_a_heading_that_opens_a_details_gives_the_callout_its_title():
    note = Note(NOTE)                       # as the model wrote it live
    apply(note, "insert", {"after": "b2", "markdown": "### <details><summary>Why</summary>\n\nText\n</details>"},
          change_id="c1", step=1)
    assert "Intro para.\n\n> [!note]- Why\n> Text\n\n## A" in note.text()


def test_edit_text_that_wraps_words_in_details_writes_a_callout():
    note = Note(NOTE)
    apply(note, "edit_text", {"target": "b2", "old": "Intro para.",
                              "new": "<details><summary>Intro</summary>\nIntro para.\n</details>"}, change_id="c1", step=1)
    assert "# Title\n\n> [!note]- Intro\n> Intro para.\n\n## A" in note.text()


def test_details_the_old_text_had_and_details_in_code_stay_as_written():
    mine = "<details>\n<summary>Mine</summary>\n\nOwner text.\n\n</details>\n"
    note = Note(mine)
    edited = mine.replace("Owner text.", "Owner text, edited.")
    apply(note, "replace", {"target": "b1-b3", "markdown": edited.rstrip("\n")}, change_id="c1", step=1)
    assert note.text() == edited
    shown = "Use `<details>` to fold.\n\n```html\n<details><summary>x</summary></details>\n```"
    assert edits.fold(shown) == (shown, [])


def test_nested_details_nest_callouts_and_a_summary_keeps_only_its_words():
    out, titles = edits.fold("<details><summary><b>Outer</b> part</summary>\nx\n"
                             "<details><summary>### Inner</summary>\ny\n</details>\n</details>")
    assert out == "> [!note]- Outer part\n> x\n>\n> > [!note]- Inner\n> > y"
    assert titles == ["Outer part", "Inner"]


def test_a_callout_is_set_apart_so_the_lines_around_it_stay_outside_it():
    assert edits.fold("Intro\n<details><summary>T</summary>\nbody\n</details> After.\nNext")[0] == \
        "Intro\n\n> [!note]- T\n> body\n\nAfter.\nNext"
    assert edits.fold("<details><summary>T</summary>\nbody\n</details>\nNext")[0] == "> [!note]- T\n> body\n\nNext"


def test_a_details_keeps_its_place_in_a_list_and_loses_the_indent_of_its_body():
    assert edits.fold("- item\n  <details><summary>T</summary>\n    body\n      nested\n  </details>")[0] == \
        "- item\n\n  > [!note]- T\n  > body\n  >   nested"
    assert edits.fold("<details>\nbody\n</details>")[0] == "> [!note]- Details\n> body"


def test_insert_asks_for_its_place_before_its_text():
    # The model writes parameters in the order the schema lists them; with the place first, an
    # insert's text can be shown where it goes as it is written (measured 2026-10-13).
    [tool] = ollama_tools(["insert"])
    assert list(tool["function"]["parameters"]["properties"]) == ["after", "before", "markdown"]


def test_a_details_cannot_hide_a_skip_or_a_recordings_heading_from_the_checks():
    with pytest.raises(ToolError) as refused:
        apply(Note(NOTE), "replace", {"target": "s3", "markdown": "<details><summary>A</summary>\n\n...\n\n</details>"},
              change_id="c1", step=1)
    assert refused.value.code == "elision"
    with pytest.raises(ToolError) as refused:
        apply(Note(NOTE), "insert", {"after": "b2", "markdown": "<details><summary>T</summary>\n\n## Transcript\n\nx\n</details>"},
              change_id="c1", step=1)
    assert refused.value.code == "marker"


def test_two_details_on_one_line_are_both_callouts():
    out, titles = edits.fold("<details><summary>A</summary>a</details><details><summary>B</summary>b</details>")
    assert out == "> [!note]- A\n> a\n\n> [!note]- B\n> b" and titles == ["A", "B"]
