"""chunk: frontmatter parsing and writing, fragment splitting."""
import pytest

from slim import chunk


# --- prose with no line breaks to split on -----------------------------------------------

def test_an_unbroken_transcript_is_split_instead_of_becoming_one_fragment():
    """⚠ Measured on a real 37-minute meeting: parakeet returns ONE unbroken string, so a
    21,800-char transcript held 3 newlines and every blank-line split above found nothing. It
    became a single fragment — unpointable by retrieval and far past the embedder's context, so
    its vector described the opening minute and nothing after it."""
    from slim.chunk import chunk_markdown
    from slim.config import CHUNK_MAX
    body = " ".join(f"This is sentence number {i} about the dashboard project." for i in range(600))
    frags = chunk_markdown(f"---\ntitle: \"x\"\n---\n\n## Transcript\n\n{body}\n")
    assert len(frags) > 1
    assert all(len(f.text) <= CHUNK_MAX for f in frags)


def test_splitting_loses_no_words():
    """A split that drops text is worse than an oversized fragment: the note would look
    indexed and be quietly incomplete."""
    from slim.chunk import chunk_markdown
    body = " ".join(f"Sentence {i} mentions Playwright and Tableau." for i in range(400))
    frags = chunk_markdown(f"---\ntitle: \"x\"\n---\n\n## Transcript\n\n{body}\n")
    joined = " ".join(f.text for f in frags)
    assert joined.count("Playwright") == 400
    assert "Sentence 0 " in joined and "Sentence 399 " in joined


def test_pieces_break_at_sentence_boundaries():
    """Cutting mid-clause embeds badly and reads worse when quoted back as evidence."""
    from slim.chunk import _split_prose
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    for piece in _split_prose(text):
        assert piece.endswith(".")


def test_text_with_no_punctuation_still_gets_bounded():
    """An ASR run with no sentence punctuation at all must not defeat the cap."""
    from slim.chunk import _split_prose
    from slim.config import CHUNK_MAX
    assert all(len(p) <= CHUNK_MAX for p in _split_prose("word " * 4000))


def test_normal_notes_are_unaffected():
    """Everything that already chunked well must chunk identically — this is a LAST resort,
    not a new primary strategy."""
    from slim.chunk import chunk_markdown
    doc = "---\ntitle: \"x\"\n---\n\n## A\n\nShort para.\n\nAnother para.\n\n## B\n\nMore text.\n"
    frags = chunk_markdown(doc)
    assert [f.heading_path for f in frags] == ["A", "B"]


# --- the frontmatter writer ---------------------------------------------------------------

def test_setting_a_field_replaces_it_in_place_and_leaves_everything_else():
    text = '---\ntitle: "x"\ntype: meeting-note\norigin: imported\n---\n\nbody\n'
    out = chunk.set_frontmatter(text, {"type": "lecture"})
    assert out == '---\ntitle: "x"\ntype: lecture\norigin: imported\n---\n\nbody\n'


def test_a_new_field_is_appended_before_the_closing_fence():
    text = "---\ntype: meeting-note\n---\n\nbody\n"
    out = chunk.set_frontmatter(text, {"topics": ["school", "svm"]})
    assert out == "---\ntype: meeting-note\ntopics: [school, svm]\n---\n\nbody\n"


def test_the_body_is_never_touched():
    """Raw sources are never rewritten. Only frontmatter changes, byte-for-byte below."""
    text = ('---\ntitle: "x"\ntype: meeting-note\n---\n\n# x\n\n## Transcript\n\n'
            'Today we derive the kernel trick.\n')
    before = text.split("---", 2)[2]
    out = chunk.set_frontmatter(text, {"type": "lecture", "tagged_by": "llm"})
    assert out.split("---", 2)[2] == before


def test_a_note_without_frontmatter_is_refused_rather_than_mangled():
    with pytest.raises(ValueError):
        chunk.set_frontmatter("# just a note\n\nbody\n", {"type": "lecture"})


def test_set_frontmatter_replaces_a_block_list_instead_of_appending_a_second_key(tmp_path):
    """`^key:\\s` does not match a bare `topics:` heading a block list, so the key looked
    absent and a SECOND one was appended. Obsidian shows a duplicate property and every
    last-wins reader silently drops the hand-curated one."""
    text = ('---\ntitle: "x"\ntopics:\n  - kernel methods\n  - svm\ndate: 2026-06-01\n---\n'
            '\nbody\n')
    out = chunk.set_frontmatter(text, {"topics": ["a", "b"]})
    assert out.count("topics:") == 1
    assert "kernel methods" not in out and "- svm" not in out
    assert "date: 2026-06-01" in out and out.endswith("\nbody\n")


# --- key order (2026-08-06) -----------------------------------------------------------------

def test_a_new_key_lands_at_its_canonical_position_not_at_the_end():
    """27 recorded notes grew 15 frontmatter shapes because every pass appended its fields
    wherever it ran. Nothing parses key order — they read it."""
    text = "---\ntitle: \"X\"\ntype: idea\norigin: recorded\nfiled_by: slim\n---\n\nbody\n"
    out = chunk.set_frontmatter(text, {"tags": ["idea"], "review_status": "pending"})
    keys = [l.split(":", 1)[0] for l in out.split("---")[1].strip().splitlines()]
    assert keys == ["title", "type", "tags", "origin", "review_status", "filed_by"]


def test_an_existing_key_is_still_replaced_in_place():
    """⚠ Only NEW keys move. Reordering a note that already has the field would let a
    retagging pass silently rewrite the shape of 265 imported notes."""
    text = "---\ntitle: \"X\"\ntopics: [a]\ndate: 2026-08-06\norigin: imported\n---\n\nbody\n"
    out = chunk.set_frontmatter(text, {"topics": ["b"]})
    keys = [l.split(":", 1)[0] for l in out.split("---")[1].strip().splitlines()]
    assert keys == ["title", "topics", "date", "origin"]   # date stays after topics
    assert "topics: [b]" in out


def test_an_unknown_key_still_goes_to_the_end():
    text = "---\ntitle: \"X\"\norigin: recorded\n---\n\nbody\n"
    out = chunk.set_frontmatter(text, {"some_future_field": "v"})
    keys = [l.split(":", 1)[0] for l in out.split("---")[1].strip().splitlines()]
    assert keys == ["title", "origin", "some_future_field"]


def test_record_renders_a_subsequence_of_the_canonical_order():
    """One statement of the order. If record.py drifts from it, this fails rather than two
    modules quietly disagreeing about what a recording looks like."""
    from datetime import datetime, timezone

    from slim import record

    text = record.render_note(
        title="t", when=datetime(2026, 8, 6, tzinfo=timezone.utc), type_tag="idea",
        transcript="x", notes_md="", summary_md="", topics=["b"],
        audio_rel="Attachments/Recordings/x.webm", filed_by_slim=True,
        audio_sha256="d" * 64, asr_model="parakeet", audio_seconds=1.0,
        transcribed_at=datetime(2026, 8, 6, tzinfo=timezone.utc))
    keys = [l.split(":", 1)[0] for l in text.split("---")[1].strip().splitlines()]
    assert set(keys) - set(chunk.FRONTMATTER_ORDER) == set(), keys
    assert keys == sorted(keys, key=chunk.FRONTMATTER_ORDER.index), keys
