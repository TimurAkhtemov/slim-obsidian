"""frontmatter: parsing and writing."""
import pytest

from slim.vault import frontmatter


# --- the frontmatter writer ---------------------------------------------------------------

def test_setting_a_field_replaces_it_in_place_and_leaves_everything_else():
    text = '---\ntitle: "x"\ntype: meeting-note\norigin: imported\n---\n\nbody\n'
    out = frontmatter.set_frontmatter(text, {"type": "lecture"})
    assert out == '---\ntitle: "x"\ntype: lecture\norigin: imported\n---\n\nbody\n'


def test_a_new_field_is_appended_before_the_closing_fence():
    text = "---\ntype: meeting-note\n---\n\nbody\n"
    out = frontmatter.set_frontmatter(text, {"topics": ["school", "svm"]})
    assert out == "---\ntype: meeting-note\ntopics: [school, svm]\n---\n\nbody\n"


def test_the_body_is_never_touched():
    """Raw sources are never rewritten. Only frontmatter changes, byte-for-byte below."""
    text = ('---\ntitle: "x"\ntype: meeting-note\n---\n\n# x\n\n## Transcript\n\n'
            'Today we derive the kernel trick.\n')
    before = text.split("---", 2)[2]
    out = frontmatter.set_frontmatter(text, {"type": "lecture", "tagged_by": "llm"})
    assert out.split("---", 2)[2] == before


def test_a_note_without_frontmatter_is_refused_rather_than_mangled():
    with pytest.raises(ValueError):
        frontmatter.set_frontmatter("# just a note\n\nbody\n", {"type": "lecture"})


def test_set_frontmatter_replaces_a_block_list_instead_of_appending_a_second_key(tmp_path):
    """`^key:\\s` does not match a bare `topics:` heading a block list, so the key looked
    absent and a SECOND one was appended. Obsidian shows a duplicate property and every
    last-wins reader silently drops the hand-curated one."""
    text = ('---\ntitle: "x"\ntopics:\n  - kernel methods\n  - svm\ndate: 2026-06-01\n---\n'
            '\nbody\n')
    out = frontmatter.set_frontmatter(text, {"topics": ["a", "b"]})
    assert out.count("topics:") == 1
    assert "kernel methods" not in out and "- svm" not in out
    assert "date: 2026-06-01" in out and out.endswith("\nbody\n")


def test_a_block_list_reads_as_a_list():
    """Obsidian's property editor writes every list as a block list. The reader took the bare
    `tags:` for an empty string and skipped the `- item` lines, so a note edited there lost its
    tags, topics and audio to every reader."""
    text = ('---\ntitle: "x"\ntags:\n  - lecture\n  - "ml"\ntopics:\n- svm\n'
            'date: 2026-06-01\n---\n\nbody\n')
    fm, skip = frontmatter.parse_frontmatter(text)
    assert fm["tags"] == ["lecture", "ml"]
    assert fm["topics"] == ["svm"]            # YAML allows the dash at the key's own indent
    assert fm["date"] == "2026-06-01" and skip == 9


def test_a_bare_key_with_no_list_is_still_empty():
    fm, _ = frontmatter.parse_frontmatter("---\ntitle: x\ntags:\ndate: 2026-06-01\n---\n")
    assert fm["tags"] == "" and fm["date"] == "2026-06-01"


def test_a_hash_tag_item_survives_a_write_and_a_read():
    """`[idea, #todo]` is YAML for `[idea]` plus a comment: Obsidian would show one tag."""
    out = frontmatter.set_frontmatter("---\ntitle: x\n---\n", {"tags": ["idea", "#todo"]})
    fm, _ = frontmatter.parse_frontmatter(out)
    assert fm["tags"] == ["idea", "#todo"]
    assert ' #' not in out.split("tags:", 1)[1].splitlines()[0]


# --- key order (2026-08-06) -----------------------------------------------------------------

def test_a_new_key_lands_at_its_canonical_position_not_at_the_end():
    """27 recorded notes grew 15 frontmatter shapes because every pass appended its fields
    wherever it ran. Nothing parses key order — they read it."""
    text = "---\ntitle: \"X\"\ntype: idea\norigin: recorded\nfiled_by: slim\n---\n\nbody\n"
    out = frontmatter.set_frontmatter(text, {"tags": ["idea"], "review_status": "pending"})
    keys = [l.split(":", 1)[0] for l in out.split("---")[1].strip().splitlines()]
    assert keys == ["title", "type", "tags", "origin", "review_status", "filed_by"]


def test_an_existing_key_is_still_replaced_in_place():
    """⚠ Only NEW keys move. Reordering a note that already has the field would let a
    retagging pass silently rewrite the shape of 265 imported notes."""
    text = "---\ntitle: \"X\"\ntopics: [a]\ndate: 2026-08-06\norigin: imported\n---\n\nbody\n"
    out = frontmatter.set_frontmatter(text, {"topics": ["b"]})
    keys = [l.split(":", 1)[0] for l in out.split("---")[1].strip().splitlines()]
    assert keys == ["title", "topics", "date", "origin"]   # date stays after topics
    assert "topics: [b]" in out


def test_an_unknown_key_still_goes_to_the_end():
    text = "---\ntitle: \"X\"\norigin: recorded\n---\n\nbody\n"
    out = frontmatter.set_frontmatter(text, {"some_future_field": "v"})
    keys = [l.split(":", 1)[0] for l in out.split("---")[1].strip().splitlines()]
    assert keys == ["title", "origin", "some_future_field"]


def test_record_renders_a_subsequence_of_the_canonical_order():
    """One statement of the order. If record.py drifts from it, this fails rather than two
    modules quietly disagreeing about what a recording looks like."""
    from datetime import datetime, timezone

    from slim.recorder import note

    text = note.render_note(
        title="t", when=datetime(2026, 8, 6, tzinfo=timezone.utc), type_tag="idea",
        transcript="x", notes_md="", summary_md="", topics=["b"],
        audio_rel="Attachments/Recordings/x.webm", filed_by_slim=True,
        audio_sha256="d" * 64, asr_model="parakeet", speakers_by="channels",
        audio_seconds=1.0,
        transcribed_at=datetime(2026, 8, 6, tzinfo=timezone.utc))
    keys = [l.split(":", 1)[0] for l in text.split("---")[1].strip().splitlines()]
    assert set(keys) - set(frontmatter.FRONTMATTER_ORDER) == set(), keys
    assert keys == sorted(keys, key=frontmatter.FRONTMATTER_ORDER.index), keys
