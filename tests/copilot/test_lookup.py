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
