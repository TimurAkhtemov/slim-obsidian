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
    blocks = [
        edits.Block("p", "type: meeting\n", "type: lecture\n"),
        edits.Block("p", "ship friday\n", "ship monday\n"),
        edits.Block("p", "Me: we ship friday\n", "Me: we ship monday\n"),
        edits.Block("p", "Afterwards I wrote this line.\n", "Afterwards, a better line.\n"),
    ]
    after, results = edits.apply_blocks(MEETING, blocks)
    assert [r["ok"] for r in results] == [False, False, False, True]
    assert results[0]["reason"] == "part of the recording; SLIM never edits it"
    assert "type: meeting" in after and "Me: we ship friday" in after
    assert after.endswith("Afterwards, a better line.\n")


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
    assert create["kind"] == "create" and create["after"] == "# New\n\nhello\n"
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
    assert reasons["Notes/school/b.md"] == "not in view; link it as [[b]] to let SLIM edit it"
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
