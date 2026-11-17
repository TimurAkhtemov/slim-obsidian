import pytest

from slim import db
from slim.copilot import blocks, context, edits, lookup
from slim.index import search


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "copilot.db")


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Notes/school/cs-201").mkdir(parents=True)
    (root / "Notes/school/cs-202").mkdir(parents=True)
    (root / "_Reflections").mkdir()
    return root


@pytest.fixture(autouse=True)
def no_vectors(monkeypatch):
    monkeypatch.setattr(search, "vector", lambda *a, **k: [])


def write(root, path, body):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"---\ntitle: {target.stem}\n---\n\n{body}\n", encoding="utf-8")
    return path


def ingest(con, vault, path):
    context.sync_note(con, vault, path, embed=False)


@pytest.fixture
def two(con, vault):
    a = write(vault, "Notes/school/cs-201/graphs.md", "# Graphs\n\nDijkstra shortest path algorithm.\n")
    b = write(vault, "Notes/school/cs-202/sorting.md", "# Sorting\n\nDijkstra is not a sort, but quicksort is.\n")
    for p in (a, b):
        ingest(con, vault, p)
    return a, b


def test_search_returns_paths_and_passages(con, vault, two):
    r = lookup.Reader(con, vault)
    out = r.call("search", {"query": "Dijkstra"})
    assert "Notes/school/cs-201/graphs.md" in out and "shortest path" in out
    assert "Notes/school/cs-201/graphs.md" in r.searched


def test_search_folder_scopes(con, vault, two):
    out = lookup.Reader(con, vault).call("search", {"query": "Dijkstra", "folder": "Notes/school/cs-202"})
    assert "cs-201" not in out and "cs-202" in out
    out = lookup.Reader(con, vault, default_folder="Notes/school/cs-202").call("search", {"query": "Dijkstra"})
    assert "cs-201" not in out and "cs-202" in out


@pytest.mark.parametrize("whole", ["", "/"])
def test_an_explicit_empty_folder_searches_the_whole_vault(con, vault, two, whole):
    reader = lookup.Reader(con, vault, default_folder="Notes/school/cs-202")
    out = reader.call("search", {"query": "Dijkstra", "folder": whole})
    assert not out.startswith("refused:") and "cs-201" in out and "cs-202" in out
    assert '""' in lookup.TOOLS["search"]["parameters"]["properties"]["folder"]["description"]


def test_search_excludes_excluded_top(con, vault, two):
    write(vault, "_Reflections/r.md", "# R\n\nDijkstra reflection.\n")
    try:
        ingest(con, vault, "_Reflections/r.md")
    except Exception:
        pass
    assert "_Reflections" not in lookup.Reader(con, vault).call("search", {"query": "Dijkstra"})


def test_search_without_con_is_refused(vault):
    assert lookup.Reader(None, vault).call("search", {"query": "x"}) == "refused: search is not available"


def test_read_note_whole_and_section(vault):
    path = write(vault, "Notes/a.md", "# T\n\n## A\n\nalpha text\n\n## B\n\nbeta text\n")
    r = lookup.Reader(None, vault)
    whole = r.call("read_note", {"path": path})
    assert whole.startswith(f"=== {path} ===") and "alpha text" in whole and "beta text" in whole
    part = r.call("read_note", {"path": path, "section": "A"})
    assert "alpha text" in part and "beta text" not in part
    assert r.read == [path]


def test_read_note_long_gives_outline_and_head(vault):
    body = "# Big\n\n## Part\n\n" + ("word " * 10_000)
    path = write(vault, "Notes/big.md", body)
    out = lookup.Reader(None, vault).call("read_note", {"path": path})
    assert out.startswith(f"=== {path} ===\n# Big\n## Part\n")
    assert out.endswith("[… the rest: read_note(path, section) for one section]")
    assert len(out) > lookup.READ_MAX_CHARS


def test_read_note_open_note_section_has_handles(vault):
    text = "# Title\n\n## A\n\na\n\n## B\n\nb\n"
    path = write(vault, "Notes/o.md", "x")
    r = lookup.Reader(None, vault, open_note=(path, blocks.Note(text)))
    out = r.call("read_note", {"path": path, "section": "s2"})
    assert "[b2]" in out and "[b3]" in out and "[b4]" not in out
    assert {"b2", "b3"} <= r.shown and "b4" not in r.shown
    assert "[b4]" in r.call("read_note", {"path": path, "section": "B"})


@pytest.mark.parametrize("bad", ["../x.md", "/etc/passwd", "Notes", "Notes/missing.md", "_Reflections/r.md"])
def test_read_note_refuses_bad_paths(vault, bad):
    write(vault, "_Reflections/r.md", "r")
    assert lookup.Reader(None, vault).call("read_note", {"path": bad}).startswith("refused:")


def test_list_folder(vault):
    write(vault, "Notes/b.md", "b")
    write(vault, "Notes/a.md", "a")
    (vault / "Notes/z.txt").write_text("z")
    r = lookup.Reader(None, vault)
    assert r.call("list_folder", {"path": "Notes"}).splitlines() == ["school/", "a.md", "b.md"]
    assert "Notes/" in r.call("list_folder", {"path": ""}) and "_Reflections" not in r.call("list_folder", {"path": ""})
    assert r.call("list_folder", {"path": "Nope"}).startswith("refused:")


def test_ollama_tools_includes_lookups():
    tools = edits.ollama_tools(["search", "replace"])
    assert [t["function"]["name"] for t in tools] == ["search", "replace"]


def test_search_failure_is_refused_not_raised(con, vault, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("embedder down")
    monkeypatch.setattr(search, "hybrid", boom)
    out = lookup.Reader(con, vault).call("search", {"query": "x"})
    assert out.startswith("refused:") and "RuntimeError" in out


def test_a_long_open_note_shows_whole_blocks_and_shown_is_what_it_showed(vault):
    text = "# Big\n\n" + "".join(f"## Part {n}\n\n" + "word " * 2000 + "\n\n" for n in range(8))
    path = write(vault, "Notes/o.md", "x")
    note = blocks.Note(text)
    r = lookup.Reader(None, vault, open_note=(path, note))
    out = r.call("read_note", {"path": path})
    labels = set(__import__("re").findall(r"^\[(b\d+)\] ", out, __import__("re").M))
    assert labels and labels == r.shown and len(labels) < len(note.blocks)
    for handle in labels:                                    # every block it shows, it shows whole
        block = next(b for b in note.blocks if b.handle == handle)
        assert f"[{handle}] {block.text}" in out
    assert "Sections:\n" in out and out.endswith("[… the rest: read_note(path, section) for one section]")


def test_a_long_range_of_a_headingless_open_note_names_the_blocks_not_shown(vault):
    text = "".join(f"Paragraph {n}. " + "word " * 1000 + "\n\n" for n in range(20))
    path = write(vault, "Notes/o.md", "x")
    note = blocks.Note(text)
    r = lookup.Reader(None, vault, open_note=(path, note))
    last = next(b.handle for b in reversed(note.blocks) if b.kind != "blank")
    out = r.call("read_note", {"path": path, "section": f"b1-{last}"})
    hint = __import__("re").search(r'read_note\("Notes/o\.md", "(b\d+)-(b\d+)"\)\]$', out)
    assert hint and hint[1] not in r.shown and hint[2] == last
    assert r.call("read_note", {"path": path, "section": f"{hint[1]}-{last}"}) != out


def test_a_page_of_a_block_the_open_note_lacks_is_refused_as_missing(vault):
    path = write(vault, "Notes/o.md", "x")
    r = lookup.Reader(None, vault, open_note=(path, blocks.Note("# T\n\nText.\n")))
    assert r.call("read_note", {"path": path, "section": "b99#2"}) == "refused: the open note has no block b99"


FILLER = " ".join(f"Pancake batter step detail number {i}." for i in range(40))


def test_a_search_passage_is_where_the_query_matched_not_the_section_head(con, vault):
    # Live 2026-11-14: the figures asked about sat 1,300 characters into the chunk; a head-only
    # passage hid them and the model answered from a neighbouring note's figure.
    path = write(vault, "Notes/school/cs-201/pancakes.md", f"# Notes\n\n{FILLER} Oven: 180 degrees, 25 minutes.\n")
    ingest(con, vault, path)
    out = lookup.Reader(con, vault).call("search", {"query": "pancakes oven minutes"})
    assert "180 degrees" in out and "25 minutes" in out
    assert "… " in out                      # says the passage starts mid-section


def test_a_short_section_is_shown_whole(con, vault, two):
    out = lookup.Reader(con, vault).call("search", {"query": "shortest"})
    assert "Dijkstra shortest path algorithm." in out and "…" not in out


@pytest.mark.parametrize("given", ["school/cs-201/graphs.md", "cs-201/graphs.md", "graphs.md",
                                   "notes/school/CS-201/graphs.md"])
def test_read_note_finds_the_one_note_a_shortened_path_names(con, vault, two, given):
    out = lookup.Reader(con, vault).call("read_note", {"path": given})
    assert out.startswith("=== Notes/school/cs-201/graphs.md ===") and "Dijkstra shortest" in out


def test_read_note_names_the_candidates_when_a_shortened_path_is_ambiguous(con, vault, two):
    other = write(vault, "Notes/work/graphs.md", "# Graphs\n\nwork graphs\n")
    ingest(con, vault, other)
    out = lookup.Reader(con, vault).call("read_note", {"path": "graphs.md"})
    assert out.startswith("refused:")
    assert "Notes/school/cs-201/graphs.md" in out and "Notes/work/graphs.md" in out


def test_a_shortened_path_never_reaches_an_excluded_folder(con, vault):
    write(vault, "_Reflections/r.md", "# R\n\nsecret\n")
    assert lookup.Reader(con, vault).call("read_note", {"path": "r.md"}).startswith("refused:")


def test_a_note_under_a_pinned_excluded_folder_may_be_read_and_listed(con, vault):
    """What the owner points at is fair game (the owner's ruling, 2026-11-13): a note under a
    pinned folder opens even in an excluded top folder. Search still never sees one."""
    write(vault, "Profile/about/me.md", "# Me\n\nI like graphs.\n")
    write(vault, "Skills/s.md", "# S\n\nx\n")
    reader = lookup.Reader(con, vault, pinned=("Profile",))
    out = reader.call("read_note", {"path": "Profile/about/me.md"})
    assert out.startswith("=== Profile/about/me.md ===") and "I like graphs." in out
    assert reader.call("list_folder", {"path": "Profile/about"}) == "me.md"
    assert reader.call("read_note", {"path": "Skills/s.md"}).startswith("refused:")    # not pinned
    assert reader.call("search", {"query": "graphs", "folder": "Profile"}).startswith("refused:")
    assert lookup.Reader(con, vault).call("read_note", {"path": "Profile/about/me.md"}).startswith("refused:")


def test_a_head_that_reached_every_block_does_not_promise_a_rest(vault):
    """A paged block makes the section too long, but nothing is left unread after the head."""
    from slim.copilot import blocks
    note = blocks.Note("## S\n\nintro\n\n" + "word " * 9000 + "\n")
    path = write(vault, "Notes/a.md", "x")
    r = lookup.Reader(None, vault, open_note=(path, note))
    out = r.call("read_note", {"path": path, "section": "s1"})
    assert "too long to show" in out and "the rest:" not in out


def test_a_block_just_under_the_limit_is_shown_whole_never_paged(vault):
    from slim.copilot import blocks
    note = blocks.Note("para " * (lookup.READ_MAX_CHARS // 5 - 1) + "\n")
    assert len(note.blocks[0].text) < lookup.READ_MAX_CHARS < len(note.render())
    path = write(vault, "Notes/a.md", "x")
    r = lookup.Reader(None, vault, open_note=(path, note))
    out = r.call("read_note", {"path": path, "section": "b1"})
    assert "too long" not in out and "para para" in out
    assert not note.paged and note.blocks[0].handle in r.shown
