"""Copilot edit proposals: parsing the model's blocks, applying them in code, and what may be
touched. Nothing here writes a file — the plugin writes only what the owner accepts."""

from pathlib import Path

import pytest

from slim import edits

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


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}=======\n{replace}>>>>>>> REPLACE\n"


def test_parse_splits_prose_from_blocks_and_keeps_blocks_in_order():
    answer = ("I tightened the intro and added a list.\n\n"
              + block("Notes/a.md", "old line\n", "new line\n")
              + "\n"
              + block("Notes/a.md", "", "- item\n")
              + "\nThat's it.")
    prose, blocks = edits.parse(answer)
    assert prose == "I tightened the intro and added a list.\n\nThat's it."
    assert [(b.path, b.search, b.replace, b.complete) for b in blocks] == [
        ("Notes/a.md", "old line\n", "new line\n", True),
        ("Notes/a.md", "", "- item\n", True),
    ]


def test_parse_drops_a_code_fence_wrapped_around_the_blocks():
    answer = ("Here you go.\n```markdown\n"
              + block("`Notes/a.md`", "x\n", "y\n")
              + "```\nDone.")
    prose, blocks = edits.parse(answer)
    assert prose == "Here you go.\nDone."
    assert blocks[0].path == "Notes/a.md"


def test_parse_keeps_a_parent_escape_visible_for_propose_to_refuse():
    _prose, blocks = edits.parse(block("../outside.md", "", "x\n") + block("./Notes/a.md", "", "y\n"))
    assert [b.path for b in blocks] == ["../outside.md", "Notes/a.md"]


def test_parse_marks_an_unfinished_block_incomplete():
    _prose, blocks = edits.parse("FILE: Notes/a.md\n<<<<<<< SEARCH\nx\n=======\ny")
    assert blocks[0].complete is False


def test_apply_replaces_an_exact_unique_match():
    after, results = edits.apply_blocks("one\ntwo\nthree\n", [edits.Block("p", "two\n", "TWO\n")])
    assert after == "one\nTWO\nthree\n"
    assert results == [{"ok": True}]


def test_apply_tolerates_trailing_whitespace_differences():
    after, results = edits.apply_blocks("one  \ntwo\t\nthree\n",
                                        [edits.Block("p", "one\ntwo\n", "joined\n")])
    assert after == "joined\nthree\n"
    assert results[0]["ok"]


def test_apply_refuses_a_missing_or_ambiguous_search():
    text = "same\nsame\n"
    after, results = edits.apply_blocks(text, [edits.Block("p", "same\n", "x\n"),
                                               edits.Block("p", "absent\n", "x\n")])
    assert after == text
    assert results[0] == {"ok": False, "reason": "matches 2 places; include more lines"}
    assert results[1] == {"ok": False, "reason": "text not found in the note"}


def test_empty_search_appends_to_an_existing_note():
    after, _ = edits.apply_blocks("body\n\n\n", [edits.Block("p", "", "added\n")])
    assert after == "body\n\nadded\n"


def test_recorded_note_protects_frontmatter_block_and_transcript_but_not_what_follows():
    for search, replace in [("type: meeting\n", "type: lecture\n"), ("ship friday\n", "ship monday\n"),
                            ("Me: we ship friday\n", "Me: we ship monday\n")]:
        after, results = edits.apply_blocks(MEETING, [edits.Block("p", search, replace)])
        assert after == MEETING and results[0]["reason"] == edits.PROTECTED
    after, results = edits.apply_blocks(
        MEETING, [edits.Block("p", "Afterwards I wrote this line.\n", "Afterwards, a better line.\n")])
    assert results == [{"ok": True}] and after.endswith("Afterwards, a better line.\n")


def test_a_file_applies_all_or_nothing():
    """A move is a delete plus an insert: the delete alone lost the section (review 2026-09-23)."""
    text = "# A\nsection A body\n# B\nbody b\n"
    after, results = edits.apply_blocks(text, [
        edits.Block("p", "# A\nsection A body\n", ""),
        edits.Block("p", "body B TYPO\n", "body b\n# A\nsection A body\n")])
    assert after == text
    assert results[1] == {"ok": False, "reason": "text not found in the note"}
    assert results[0]["ok"] is False and results[0]["reason"].startswith("held back")


def test_a_repeated_block_applies_once():
    change = block("Notes/a.md", "## Tasks\n", "## Tasks\n- [ ] call Bob\n")
    _prose, blocks = edits.parse(change * 3 + block("Notes/a.md", "", "Summary: done\n") * 2)
    assert len(blocks) == 2


def test_a_match_starts_at_a_line_start():
    after, _ = edits.apply_blocks("Grand Total: 5\nTotal: 5  \nend\n",
                                  [edits.Block("p", "Total: 5\n", "Total: 6\n")])
    assert after == "Grand Total: 5\nTotal: 6\nend\n"
    after, results = edits.apply_blocks("Grand Total: 5\n", [edits.Block("p", "Total: 5\n", "x\n")])
    assert results[0]["reason"] == "text not found in the note"


def test_an_append_lands_before_a_transcript_that_runs_to_the_end():
    memo = "---\ntype: meeting-note\n---\n\nmy line\n\n## Transcript\n\nMe: we ship friday\n"
    after, results = edits.apply_blocks(memo, [edits.Block("p", "", "- [ ] follow up\n")])
    assert results == [{"ok": True}]
    assert after == ("---\ntype: meeting-note\n---\n\nmy line\n\n- [ ] follow up\n\n"
                     "## Transcript\n\nMe: we ship friday\n")


def test_a_proposal_cannot_unlock_the_transcript_with_a_decoy_fence():
    memo = "my own line\n\n## Transcript\n\nMe: we ship friday\n"
    after, results = edits.apply_blocks(memo, [
        edits.Block("p", "my own line\n", "my own line\n````slim-meeting\n````\n"),
        edits.Block("p", "Me: we ship friday\n", "Me: we ship NEVER\n")])
    assert after == memo and not any(r["ok"] for r in results)
    # Even without adding markers, a change that alters a protected slice is refused whole.
    after, results = edits.apply_blocks(MEETING, [
        edits.Block("p", "Afterwards I wrote this line.\n", "x\n"),
        edits.Block("p", "Me: we ship friday\n", "Me: no\n")])
    assert after == MEETING


def test_a_divider_inside_search_is_not_the_split():
    note = "Title\n=======\n\nBody\n"
    answer = ("FILE: Notes/a.md\n<<<<<<< SEARCH\nTitle\n=====\n=======\nBetter Title\n=====\n"
              ">>>>>>> REPLACE\n")
    _prose, (only,) = edits.parse(answer)
    assert (only.search, only.replace) == ("Title\n=====\n", "Better Title\n=====\n")
    ambiguous = "FILE: Notes/a.md\n<<<<<<< SEARCH\nTitle\n=======\n=======\nx\n>>>>>>> REPLACE\n"
    _prose, (bad,) = edits.parse(ambiguous)
    assert bad.problem == "the change was malformed"
    assert edits.apply_blocks(note, [bad])[0] == note


def test_a_divider_echoed_before_replace_is_not_a_second_divider():
    """Measured 2026-10-12: 3 rewrites in 20 ended their new text with `=======`."""
    _prose, (only,) = edits.parse("FILE: Notes/a.md\n<<<<<<< SEARCH\nold\n=======\nnew\n=======\n"
                                  ">>>>>>> REPLACE\n")
    assert (only.search, only.replace, only.problem) == ("old\n", "new\n", "")


def test_a_bare_path_right_above_a_search_names_the_file():
    answer = ("I shortened it.\n\nNotes/my note — v2.md\n<<<<<<< SEARCH\nold\n=======\nnew\n"
              ">>>>>>> REPLACE\n")
    prose, (only,) = edits.parse(answer)
    assert prose == "I shortened it." and only.path == "Notes/my note — v2.md"
    prose, _ = edits.parse("See Notes/a.md\n\nfor more.\n")
    assert "See Notes/a.md" in prose


def test_the_file_path_does_not_carry_across_prose_and_prose_file_lines_stay_prose():
    answer = (block("Notes/a.md", "x\n", "y\n") + "Now the second part.\n"
              "<<<<<<< SEARCH\nz\n=======\nw\n>>>>>>> REPLACE\n"
              "File: the quarterly report is attached.\n")
    prose, blocks = edits.parse(answer)
    assert [b.path for b in blocks] == ["Notes/a.md", None]
    assert "File: the quarterly report is attached." in prose


def test_backslash_colon_and_case_variant_paths_are_refused():
    for raw in ["Notes\\..\\..\\escaped.md", "Attachments\\x.md", "C:x.md", "attachments/x.md",
                "_reflections/x.md", "Notes/.obsidian/x.md", "Notes/cafe\u0301.md"]:
        assert edits.note_path(raw) is None, raw
    assert edits.note_path("Notes/work/a note.md") == "Notes/work/a note.md"


def test_an_empty_result_and_a_deleted_line_behave():
    after, results = edits.apply_blocks("only\n", [edits.Block("p", "only\n", "")])
    assert after == "" and results == [{"ok": True}]          # propose() refuses the empty file
    after, _ = edits.apply_blocks("a  \nb\n", [edits.Block("p", "a\n", "")])
    assert after == "b\n"
    after, _ = edits.apply_blocks("a\nlast", [edits.Block("p", "last\n", "LAST\n")])
    assert after == "a\nLAST"                                   # no final newline stays so


def test_an_ordinary_note_is_editable_everywhere_including_frontmatter():
    text = "---\ntags: [a]\n---\n\nBody.\n"
    after, results = edits.apply_blocks(text, [edits.Block("p", "tags: [a]\n", "tags: [a, b]\n")])
    assert results[0]["ok"] and "tags: [a, b]" in after


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Notes/school").mkdir(parents=True)
    (root / "Notes/school/a.md").write_text("alpha\nbeta\n", encoding="utf-8")
    (root / "Notes/school/b.md").write_text("gamma\n", encoding="utf-8")
    return root


def test_propose_edits_notes_in_view_and_creates_notes_in_existing_folders(vault):
    blocks = [
        edits.Block("Notes/school/a.md", "beta\n", "BETA\n"),
        edits.Block("Notes/school/new.md", "", "# New\n\nhello\n"),
    ]
    proposal = edits.propose(vault, blocks, in_view={"Notes/school/a.md"})
    edit, create = proposal["files"]
    assert edit["kind"] == "edit" and edit["after"] == "alpha\nBETA\n"
    assert edit["base_hash"] == edits.digest("alpha\nbeta\n")
    assert create["kind"] == "create" and create["after"] == "---\norigin: copilot\n---\n\n# New\n\nhello\n"
    assert create["base_hash"] is None
    assert proposal["dropped"] == []


def test_propose_drops_paths_the_owner_did_not_put_in_view(vault):
    blocks = [
        edits.Block("Notes/school/b.md", "gamma\n", "G\n"),
        edits.Block("Notes/elsewhere/new.md", "", "x\n"),
        edits.Block("../escape.md", "", "x\n"),
        edits.Block(".obsidian/app.md", "", "x\n"),
        edits.Block("Notes/school/pic.png", "", "x\n"),
        edits.Block(None, "a", "b"),
    ]
    proposal = edits.propose(vault, blocks, in_view={"Notes/school/a.md"})
    assert proposal["files"] == []
    reasons = {item["path"]: item["reason"] for item in proposal["dropped"]}
    assert reasons["Notes/school/b.md"] == "not part of this request"
    assert reasons["Notes/elsewhere/new.md"] == "folder does not exist"
    assert reasons["../escape.md"] == "not a note path in this vault"
    assert reasons[".obsidian/app.md"] == "not a note path in this vault"
    assert reasons["Notes/school/pic.png"] == "not a note path in this vault"
    assert reasons[""] == "no FILE line before the change"


def test_propose_keeps_a_file_whose_blocks_all_failed_so_the_card_can_say_why(vault):
    proposal = edits.propose(vault, [edits.Block("Notes/school/a.md", "zzz\n", "y\n")],
                             in_view={"Notes/school/a.md"})
    (only,) = proposal["files"]
    assert only["after"] is None
    assert only["blocks"] == [{"ok": False, "reason": "text not found in the note"}]


def test_propose_refuses_an_incomplete_block_rather_than_applying_half(vault):
    blocks = [edits.Block("Notes/school/a.md", "beta\n", "BE", complete=False)]
    proposal = edits.propose(vault, blocks, in_view={"Notes/school/a.md"})
    assert proposal["files"][0]["after"] is None
    assert proposal["files"][0]["blocks"][0]["reason"] == "cut off before the change ended"


def test_propose_refuses_to_empty_a_note(vault):
    proposal = edits.propose(vault, [edits.Block("Notes/school/a.md", "alpha\nbeta\n", "")],
                             in_view={"Notes/school/a.md"})
    assert proposal["files"][0]["after"] is None
    assert proposal["files"][0]["blocks"][0]["reason"] == "would leave the note empty"


def test_propose_keeps_journal_text_in_journal(vault):
    (vault / "Journal").mkdir()
    (vault / "Journal/day.md").write_text("private\n", encoding="utf-8")
    proposal = edits.propose(vault, [edits.Block("Notes/school/copy.md", "", "private\n"),
                                     edits.Block("Journal/day.md", "private\n", "private!\n")],
                             in_view={"Journal/day.md"})
    assert [f["path"] for f in proposal["files"]] == ["Journal/day.md"]
    assert proposal["dropped"] == [{"path": "Notes/school/copy.md",
                                    "reason": "a Journal note's text stays in Journal/"}]


def test_propose_refuses_case_variants_crlf_and_huge_notes(vault):
    (vault / "Notes/school/crlf.md").write_bytes(b"one\r\ntwo\r\n")
    (vault / "Notes/school/big.md").write_text("x" * (edits.MAX_EDIT_CHARS + 1), encoding="utf-8")
    proposal = edits.propose(vault, [
        edits.Block("notes/school/A.md", "alpha\n", "x\n"),
        edits.Block("notes/School/new.md", "", "x\n"),
        edits.Block("Notes/school/crlf.md", "one\n", "1\n"),
        edits.Block("Notes/school/big.md", "", "more\n"),
    ], in_view={"notes/school/A.md", "Notes/school/crlf.md", "Notes/school/big.md"})
    assert proposal["files"] == []
    reasons = {item["path"]: item["reason"] for item in proposal["dropped"]}
    assert reasons["notes/school/A.md"] == "a note with this name exists in different letter case"
    assert reasons["notes/School/new.md"] == "folder does not exist"
    assert reasons["Notes/school/crlf.md"].startswith("uses Windows line endings")
    assert reasons["Notes/school/big.md"] == "too large to edit from the sidebar"


def test_a_fuzzy_match_keeps_hard_breaks_on_unchanged_lines():
    text = "first line  \nsecond line  \nthird\n"
    after, _ = edits.apply_blocks(text, [edits.Block("p", "first line\nsecond line\n",
                                                     "first line\nsecond line, fixed\n")])
    assert after == "first line  \nsecond line, fixed\nthird\n"


def test_a_new_note_skill_only_creates_and_a_selection_skill_stays_inside_the_selection(vault):
    only_new = edits.propose(vault, [edits.Block("Notes/school/a.md", "beta\n", "B\n"),
                                     edits.Block("Notes/school/n.md", "", "x\n")],
                             in_view={"Notes/school/a.md"}, creates_only=True)
    assert [f["path"] for f in only_new["files"]] == ["Notes/school/n.md"]
    assert only_new["dropped"] == [{"path": "Notes/school/a.md", "reason": "this skill only creates new notes"}]
    inside = edits.propose(vault, [edits.Block("Notes/school/a.md", "beta\n", "B\n"),
                                   edits.Block("Notes/school/a.md", "alpha\n", "A\n")],
                           in_view={"Notes/school/a.md"}, within=("Notes/school/a.md", "beta  \n"))
    assert inside["files"][0]["after"] == "alpha\nB\n"
    assert inside["dropped"] == [{"path": "Notes/school/a.md", "reason": "outside the text you selected"}]


def test_a_long_fragment_matches_mid_line_but_a_short_one_does_not():
    text = "We use $\\eta$ is the learnign rate. It scales steps.\n"
    after, results = edits.apply_blocks(text, [edits.Block(
        "p", "$\\eta$ is the learnign rate.", "$\\eta$ is the learning rate.")])
    assert results == [{"ok": True}] and "learning rate." in after
    _, results = edits.apply_blocks("Grand Total: 5\n", [edits.Block("p", "Total: 5\n", "x\n")])
    assert results[0]["reason"] == "text not found in the note"


def test_protected_text_copied_through_unchanged_is_allowed():
    notion = "---\ntype: meeting-note\n---\n\n# Call\n\n## Transcript\n\nMe: hi\n"
    after, results = edits.apply_blocks(notion, [edits.Block(
        "p", "---\ntype: meeting-note\n---\n\n# Call\n", "---\ntype: meeting-note\n---\n\n# Call\n\nSummary.\n")])
    assert results == [{"ok": True}] and "# Call\n\nSummary.\n" in after


def test_text_inserted_above_a_kept_transcript_heading_is_allowed():
    """Live: SEARCH `## Transcript` -> REPLACE `<brief>\\n\\n## Transcript` was refused as adding
    a heading it only kept. Only MORE markers than the SEARCH held is refused."""
    memo = "---\ntype: meeting-note\n---\n\n# Call\n\n## Transcript\n\nMe: hi\n"
    after, results = edits.apply_blocks(memo, [edits.Block(
        "p", "## Transcript\n", "## Brief\n\nShip friday.\n\n## Transcript\n")])
    assert results == [{"ok": True}]
    assert after == memo.replace("# Call\n\n", "# Call\n\n## Brief\n\nShip friday.\n\n")
    for replace in ("## Brief\n", "## Transcript\n\n## Transcript\n"):
        after, results = edits.apply_blocks(memo, [edits.Block("p", "## Transcript\n", replace)])
        assert after == memo and not results[0]["ok"]
    # Above a recorder block is the owner's; inside it is still record.py's.
    after, results = edits.apply_blocks(MEETING, [edits.Block(
        "p", "```slim-meeting\n", "## Brief\n\n```slim-meeting\n")])
    assert results == [{"ok": True}] and after == MEETING.replace("\n```slim-meeting", "\n## Brief\n\n```slim-meeting")
    after, results = edits.apply_blocks(MEETING, [edits.Block(
        "p", "## Transcript\n", "## Brief\n\n## Transcript\n")])
    assert after == MEETING and results[0]["reason"] == edits.PROTECTED


SOFTMAX = "\n# Softmax\n\nintro line\n\n## Formula\n\n$$\n\\sigma(z)_i = e^{z_i}\n$$\n\nwhere $z$ is\n\n## Example\n\nend\n"
MEMO = "---\ntype: meeting-note\n---\n\n# Call\n\none\ntwo\n\n## Transcript\n\nMe: hi\n"


def note_block(path, text, end=">>>>>>> NOTE"):
    return f"FILE: {path}\n<<<<<<< NOTE\n{text}{end}\n"


def test_a_note_block_is_the_whole_new_text():
    """Copied into a SEARCH first, a 40-line note cost twice its length and one mistyped
    character refused the change (13 of 20 rewrites, 2026-10-12). A NOTE block writes it once."""
    for end in (">>>>>>> NOTE", ">>>>>>> REPLACE", "<<<<<<< NOTE", "NOTE"):
        prose, (only,) = edits.parse("Simpler.\n\n" + note_block("Notes/a.md", "# Softmax\n\nshort\n", end))
        assert prose == "Simpler." and only.whole and only.replace == "# Softmax\n\nshort\n"
    after, results = edits.apply_blocks(SOFTMAX, [only])
    assert results == [{"ok": True}] and after == "\n# Softmax\n\nshort\n"
    _prose, (cut,) = edits.parse("FILE: Notes/a.md\n<<<<<<< NOTE\n# Softmax\n")
    assert cut.whole and not cut.complete
    assert edits.apply_blocks(SOFTMAX, [cut])[1][0]["reason"] == "cut off before the change ended"


def test_a_note_block_written_like_a_search_replace_is_read_as_one():
    """Measured 2026-10-12, 3 rewrites in 8: the old note (or a first draft), a divider, the
    new note; or the new note closed by a divider."""
    section = "## Formula\n\n$$\n\\sigma(z)_i = e^{z_i}\n$$\n\nwhere $z$ is\n"
    for body, after in [
            (SOFTMAX + "=======\n# Softmax\n\nshort\n", "# Softmax\n\nshort\n"),        # old, new
            ("# Soft\n\ndraft\n=======\n# Softmax\n\nshort\n", "\n# Softmax\n\nshort\n"),  # draft, new
            ("# Softmax\n\nshort\n=======\n", "\n# Softmax\n\nshort\n"),                    # closed
            (section + "=======\n## Formula\n\n$x$\n", SOFTMAX.replace(section, "## Formula\n\n$x$\n"))]:
        after_text, results = edits.apply_blocks(SOFTMAX, [edits.Block("p", "", body, whole=True)])
        assert results == [{"ok": True}] and after_text == after, body
    _prose, (closed,) = edits.parse("FILE: Notes/a.md\n<<<<<<< NOTE\n# Softmax\n=======\n")
    assert closed.complete
    setext = "Title\n=======\n\nbody\n"
    after, _ = edits.apply_blocks(setext, [edits.Block("p", "", "Title\n=======\n\nnew\n", whole=True)])
    assert after == "Title\n=======\n\nnew\n"


def test_a_note_block_keeps_frontmatter_it_leaves_out_and_takes_frontmatter_it_writes():
    text = "---\ntags: [a]\n---\n\n# T\n\nold body\n"
    after, _ = edits.apply_blocks(text, [edits.Block("p", "", "# T\n\nnew body\n", whole=True)])
    assert after == "---\ntags: [a]\n---\n\n# T\n\nnew body\n"
    after, _ = edits.apply_blocks(text, [edits.Block("p", "", "---\ntags: [b]\n---\n\n# T\n", whole=True)])
    assert after == "---\ntags: [b]\n---\n\n# T\n"


def test_a_note_block_on_a_recording_replaces_only_the_owners_text():
    # The model rewrote the frontmatter and the transcript too: both are dropped, never applied.
    model = "---\ntype: lecture\n---\n\n# Call\n\nOne line.\n\n## Transcript\n\nMe: bye\n"
    after, results = edits.apply_blocks(MEMO, [edits.Block("p", "", model, whole=True)])
    assert results == [{"ok": True}]
    assert after == "---\ntype: meeting-note\n---\n\n# Call\n\nOne line.\n\n## Transcript\n\nMe: hi\n"
    after, results = edits.apply_blocks(MEETING, [edits.Block("p", "", "A better line.\n", whole=True)])
    assert results == [{"ok": True}] and after == MEETING.replace("Afterwards I wrote this line.", "A better line.")
    fresh = MEETING.replace("\nAfterwards I wrote this line.\n", "")
    after, _ = edits.apply_blocks(fresh, [edits.Block("p", "", "Next steps.\n", whole=True)])
    assert after == fresh.replace("\n```slim-meeting", "\nNext steps.\n\n```slim-meeting")   # where an append goes


def test_a_note_block_refuses_what_it_cannot_place_or_must_not_write():
    two = MEETING.replace("```slim-meeting", "Before the block.\n\n```slim-meeting")
    after, results = edits.apply_blocks(two, [edits.Block("p", "", "x\n", whole=True)])
    assert after == two and "more than one place" in results[0]["reason"]
    for new, reason in [("# T\n\n## Transcript\n\nMe: made up\n", "adds a recording's block or transcript heading"),
                        ("# Softmax\n\n...\n", edits.SKIPS)]:
        after, results = edits.apply_blocks(SOFTMAX, [edits.Block("p", "", new, whole=True)])
        assert after == SOFTMAX and results == [{"ok": False, "reason": reason}]


def test_a_note_block_for_a_new_path_creates_the_note(vault):
    proposal = edits.propose(vault, edits.parse(note_block("Notes/school/new.md", "# New\n"))[1],
                             in_view={"Notes/school/a.md"})
    assert proposal["files"][0]["kind"] == "create"
    assert proposal["files"][0]["after"] == "---\norigin: copilot\n---\n\n# New\n"
    inside = edits.propose(vault, [edits.Block("Notes/school/a.md", "", "x\n", whole=True)],
                           in_view={"Notes/school/a.md"}, within=("Notes/school/a.md", "beta\n"))
    assert inside["dropped"] == [{"path": "Notes/school/a.md", "reason": "outside the text you selected"}]


def test_a_long_search_copied_with_a_mistake_still_matches():
    """Measured 2026-10-12: every refused rewrite had the note's line count and 95-99.7% of its
    lines exact; one LaTeX subscript (`e^{z_j}` copied as `e^{j}`) refused the whole change."""
    wrong = SOFTMAX.lstrip("\n").replace("e^{z_i}", "e^{i}").replace("## Example\n\nend\n", "")
    after, results = edits.apply_blocks(SOFTMAX, [edits.Block("p", wrong, "# Softmax\n\nshort\n")])
    assert results == [{"ok": True}] and after == "\n# Softmax\n\nshort\n\n## Example\n\nend\n"
    for search in ["intro line\n$$\nx\n",                                    # too short to trust
                   wrong.replace("where $z$ is", "$$"),                      # last line twice
                   "# Softmax\n\na\nb\nc\nd\ne\nf\nwhere $z$ is\n"]:         # mostly not the note's
        _, results = edits.apply_blocks(SOFTMAX, [edits.Block("p", search, "x\n")])
        assert results == [{"ok": False, "reason": "text not found in the note"}], search


def test_a_replacement_that_skips_lines_with_dots_is_refused():
    after, results = edits.apply_blocks(SOFTMAX, [edits.Block(
        "p", "## Formula\n", "## Formula\n...\n")])
    assert after == SOFTMAX and results == [{"ok": False, "reason": edits.SKIPS}]
    after, results = edits.apply_blocks("a\n...\nb\n", [edits.Block("p", "a\n...\nb\n", "A\n...\nb\n")])
    assert results == [{"ok": True}] and after == "A\n...\nb\n"


def stream(*lines):
    return "".join(line + "\n" for line in lines)


def test_a_search_copying_a_note_from_its_top_turns_into_a_note_block():
    notes = {"Notes/a.md": SOFTMAX, "Notes/b.md": "# Other\n\nbody\n"}
    head = stream("I'll make it simpler.", "", "FILE: Notes/a.md · the open note", "<<<<<<< SEARCH")
    copy = SOFTMAX.lstrip("\n").splitlines()
    assert edits.rewrite_prefill(head, notes) == (False, None)
    assert edits.rewrite_prefill(head + stream(*copy[:5]), notes) == (False, None)
    decided, prefill = edits.rewrite_prefill(head + stream(*copy[:8]), notes)
    assert decided and prefill == stream("I'll make it simpler.", "", "FILE: Notes/a.md", "<<<<<<< NOTE")
    # A copied frontmatter is skipped; with no FILE line the note is found by its top.
    fm = {"Notes/c.md": "---\ntags: [a]\n---\n" + SOFTMAX}
    decided, prefill = edits.rewrite_prefill(stream("Here.", "<<<<<<< SEARCH", "---", "tags: [a]", "---", *copy[:8]), fm)
    assert prefill == stream("Here.", "FILE: Notes/c.md", "<<<<<<< NOTE", "---", "tags: [a]", "---")
    # Not from the top, or closed before it got long: left alone, decided.
    assert edits.rewrite_prefill(head + stream("## Formula"), notes) == (True, None)
    assert edits.rewrite_prefill(head + stream(*copy[:3], "======="), notes) == (True, None)
    assert edits.rewrite_prefill(stream("FILE: Notes/b.md", "<<<<<<< SEARCH", *copy[:8]), notes) == (True, None)


def test_a_long_note_waits_for_a_quarter_of_it_and_a_recording_starts_after_its_block():
    long = "# Long\n\n" + "".join(f"line {i}\n" for i in range(100))
    head = stream("FILE: Notes/l.md", "<<<<<<< SEARCH")
    copy = long.splitlines()
    assert edits.rewrite_prefill(head + stream(*copy[:20]), {"Notes/l.md": long}) == (False, None)
    assert edits.rewrite_prefill(head + stream(*copy[:26]), {"Notes/l.md": long})[1]
    rec = {"Capture/m.md": MEETING}
    _decided, prefill = edits.rewrite_prefill(
        stream("FILE: Capture/m.md", "<<<<<<< SEARCH", "Afterwards I wrote this line.", *[""] * 8), rec)
    assert prefill == "FILE: Capture/m.md\n<<<<<<< NOTE\n" + MEETING[:MEETING.index("\nAfterwards")]
    assert edits.rewrite_prefill(stream("FILE: Capture/m.md", "<<<<<<< SEARCH", "Me: we ship friday", *[""] * 8), rec) == (True, None)


def test_a_note_block_gets_a_recordings_parts_from_disk_instead_of_retyping_them():
    """The model retyped a whole meeting block and transcript into a NOTE: 32k characters, cut
    off at the output cap (2026-10-12). Code puts the block in, and ends the NOTE at a transcript."""
    notes = {"Capture/m.md": MEETING, "Capture/memo.md": MEMO, "Notes/a.md": SOFTMAX}
    open_m = stream("Tidier.", "FILE: Capture/m.md", "<<<<<<< NOTE", "---", "title: Standup", "---", "")
    fence = MEETING[MEETING.index("```slim-meeting"):MEETING.index("\nAfterwards")]
    assert edits.recording_prefill(open_m + "````slim-meeting\n", notes) == (open_m + fence, False)
    open_memo = stream("FILE: Capture/memo.md", "<<<<<<< NOTE", "# Call", "", "One line.", "")
    assert edits.recording_prefill(open_memo + "## Transcript\n", notes) == (open_memo + ">>>>>>> NOTE\n", True)
    after, _ = edits.apply_blocks(MEMO, edits.parse(open_memo + ">>>>>>> NOTE\n")[1])
    assert after == MEMO.replace("one\ntwo\n", "One line.\n")
    for answer in [open_memo + "## Transcrip",                                   # not a whole line
                   stream("FILE: Capture/memo.md", "<<<<<<< SEARCH", "## Transcript"),   # a SEARCH
                   stream("FILE: Notes/a.md", "<<<<<<< NOTE", "## Transcript"),          # not a recording
                   open_memo + stream(">>>>>>> NOTE", "## Transcript")]:                  # closed
        assert edits.recording_prefill(answer, notes) is None, answer


def test_a_divider_closing_an_unfinished_answer_closes_its_block():
    _prose, (only,) = edits.parse("FILE: Notes/a.md\n<<<<<<< SEARCH\nold\n=======\nnew\n=======\n")
    assert (only.search, only.replace, only.complete) == ("old\n", "new\n", True)
    _prose, (cut,) = edits.parse("FILE: Notes/a.md\n<<<<<<< SEARCH\nold\n=======\n")
    assert cut.complete is False                      # one divider: cut off before the new text


def test_inbox_journal_is_on_the_floor_too(vault):
    (vault / "Inbox/Journal").mkdir(parents=True)
    (vault / "Inbox/Journal/memo.md").write_text("private\n", encoding="utf-8")
    proposal = edits.propose(vault, [edits.Block("Notes/school/a.md", "beta\n", "private\n")],
                             in_view={"Inbox/Journal/memo.md", "Notes/school/a.md"})
    assert proposal["dropped"] == [{"path": "Notes/school/a.md",
                                    "reason": "a Journal note's text stays in Journal/"}]
