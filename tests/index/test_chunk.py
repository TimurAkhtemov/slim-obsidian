"""chunk: fragment splitting."""


# --- prose with no line breaks to split on -----------------------------------------------

def test_an_unbroken_transcript_is_split_instead_of_becoming_one_fragment():
    """⚠ Measured on a real 37-minute meeting: parakeet returns ONE unbroken string, so a
    21,800-char transcript held 3 newlines and every blank-line split above found nothing. It
    became a single fragment — unpointable by retrieval and far past the embedder's context, so
    its vector described the opening minute and nothing after it."""
    from slim.index.chunk import chunk_markdown
    from slim.config import CHUNK_MAX
    body = " ".join(f"This is sentence number {i} about the dashboard project." for i in range(600))
    frags = chunk_markdown(f"---\ntitle: \"x\"\n---\n\n## Transcript\n\n{body}\n")
    assert len(frags) > 1
    assert all(len(f.text) <= CHUNK_MAX for f in frags)


def test_splitting_loses_no_words():
    """A split that drops text is worse than an oversized fragment: the note would look
    indexed and be quietly incomplete."""
    from slim.index.chunk import chunk_markdown
    body = " ".join(f"Sentence {i} mentions Playwright and Tableau." for i in range(400))
    frags = chunk_markdown(f"---\ntitle: \"x\"\n---\n\n## Transcript\n\n{body}\n")
    joined = " ".join(f.text for f in frags)
    assert joined.count("Playwright") == 400
    assert "Sentence 0 " in joined and "Sentence 399 " in joined


def test_pieces_break_at_sentence_boundaries():
    """Cutting mid-clause embeds badly and reads worse when quoted back as evidence."""
    from slim.index.chunk import _split_prose
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    for piece in _split_prose(text):
        assert piece.endswith(".")


def test_text_with_no_punctuation_still_gets_bounded():
    """An ASR run with no sentence punctuation at all must not defeat the cap."""
    from slim.index.chunk import _split_prose
    from slim.config import CHUNK_MAX
    assert all(len(p) <= CHUNK_MAX for p in _split_prose("word " * 4000))


def test_normal_notes_are_unaffected():
    """Everything that already chunked well must chunk identically — this is a LAST resort,
    not a new primary strategy."""
    from slim.index.chunk import chunk_markdown
    doc = "---\ntitle: \"x\"\n---\n\n## A\n\nShort para.\n\nAnother para.\n\n## B\n\nMore text.\n"
    frags = chunk_markdown(doc)
    assert [f.heading_path for f in frags] == ["A", "B"]
