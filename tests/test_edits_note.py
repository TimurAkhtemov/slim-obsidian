"""The note as the editing tools see it: blocks with handles, which of them a recording locks, and
how a target is read. Nothing here writes a note."""

import pytest

from slim.copilot import blocks as edits_note
from slim.index import chunk
from slim.copilot.blocks import Note, ToolError, strip_labels

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


# --- what a recording keeps --------------------------------------------------------------------

def test_a_recordings_own_parts_are_its_frontmatter_its_block_and_its_transcript():
    take = lambda text: [text[begin:end] for begin, end in edits_note.protected_ranges(text)]
    assert take(MEETING) == [MEETING[:MEETING.index("---\n\n") + 4],
                             MEETING[MEETING.index("```slim"):MEETING.index("\nAfterwards")]]
    assert take(MEMO) == ["---\ntype: meeting-note\n---\n", "## Transcript\n\nMe: hi\n"]
    assert edits_note.protected_ranges("---\ntags: [a]\n---\n\n## Transcript of a talk\n") == []


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
    note = Note(f"# Memo\n\nSummary.\n\n{chunk.TRANSCRIPT_MARKER}\nwords words\n")
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
    memo = Note(f"# Memo\n\nSummary.\n\n{chunk.TRANSCRIPT_MARKER}\nwords\n")
    assert memo.sections() == "s1  b1–b4  # Memo"             # the transcript's own heading is locked: not listed


def test_headings_are_compared_by_level_and_words_without_a_trailing_note():
    assert edits_note.headings("# T\n\ntext # not\n## A  \n```\n") == ["# T", "## A"]
    assert edits_note.heading_key("##  Toolkit (New Section)") == edits_note.heading_key("## toolkit") == (2, "toolkit")
    assert edits_note.heading_key("## Bag of Words (BoW)") == (2, "bag of words (bow)")
