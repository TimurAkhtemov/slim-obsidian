"""Retrieval behavior: ranking, weighted fusion, per-source cap, prefix scope, degenerate
queries.

Vector-arm tests skip when Ollama isn't reachable — the lexical and fusion logic must
stay testable without a running model server.
"""

import numpy as np
import pytest

from slim import db, embed as embed_mod, search as search_mod
from slim.ingest import ingest
from slim.search import Hit, _rrf
from tests.conftest import ollama_available


@pytest.fixture
def con(tmp_path):
    v = tmp_path / "vault"
    (v / "Projects/A").mkdir(parents=True)
    (v / "Projects/B").mkdir(parents=True)
    (v / "Projects/A/alpha.md").write_text(
        "---\ntitle: \"Alpha\"\ndate: 2026-07-01\n---\n# Alpha\n\n"
        "The quarterly perceptyx survey launches July 15.\n")
    (v / "Projects/B/beta.md").write_text(
        "---\ntitle: \"Beta\"\ndate: 2020-01-01\n---\n# Beta\n\n"
        "Survey tooling comparison for the beta project.\n")
    # Scope fixtures. All three carry "shared term"; B repeats it, so B outranks both A
    # notes on that query and a top-k taken before scoping would return nothing from A.
    (v / "Projects/A/one.md").write_text(
        "---\ntitle: \"One\"\n---\n# One\n\nA shared term, filed under lathe maintenance.\n")
    (v / "Projects/A/two.md").write_text(
        "---\ntitle: \"Two\"\n---\n# Two\n\nAnother shared term, filed under chisel sharpening.\n")
    (v / "Projects/B/one.md").write_text(
        "---\ntitle: \"B One\"\n---\n# B One\n\n"
        "shared term shared term shared term shared term shared term\n")
    con = db.connect(tmp_path / "t.db")
    ingest(con, v)
    return con


def hit(fid: int, source: str = "s1", path: str = "p.md") -> Hit:
    return Hit(fragment_id=fid, source_id=source, path=path, title=None,
               heading_path=None, text="t")


def test_lexical_ranks_the_best_match_first(con):
    r = search_mod.lexical(con, "perceptyx survey launch")
    assert r and r[0].path == "Projects/A/alpha.md"


def test_lexical_falls_back_to_or_matching(con):
    # no document contains all three terms; a strict AND would return nothing
    r = search_mod.lexical(con, "perceptyx beta nonexistentword")
    assert r, "lexical must fall back to OR-matching rather than return empty"


def test_degenerate_query(con):
    assert search_mod.lexical(con, "!!! ???") == []


def test_prefix_scope_applies_in_sql_before_top_k(con):
    # Three notes; scope to Projects/A only; k=1 must return an A hit even when B ranks higher.
    assert search_mod.lexical(con, "shared term", k=1)[0].path == "Projects/B/one.md"
    hits = search_mod.lexical(con, "shared term", k=1, prefixes=("Projects/A",))
    assert hits and all(h.path.startswith("Projects/A/") for h in hits)


def test_an_exact_note_path_is_a_prefix(con):
    hits = search_mod.lexical(con, "shared term", k=5, prefixes=("Projects/A/one.md",))
    assert {h.path for h in hits} == {"Projects/A/one.md"}


def test_prefix_matching_is_case_insensitive_and_escapes_like_wildcards(con):
    assert search_mod.lexical(con, "shared term", k=5, prefixes=("projects/a",))
    # An exact note path in the wrong case reaches the `=` arm alone — the LIKE arm asks for
    # children — so this is the one assertion that fails if `COLLATE NOCASE` is dropped there.
    hits = search_mod.lexical(con, "shared term", k=5, prefixes=("projects/a/one.md",))
    assert {h.path for h in hits} == {"Projects/A/one.md"}
    # `_unfiled` must not match `xunfiled`: `_` is a LIKE wildcard unless escaped
    assert not search_mod.lexical(con, "shared term", k=5, prefixes=("Projects/_",))


def test_load_matrix_is_cached_until_embeddings_change(con):
    ids1, m1 = embed_mod.load_matrix(con)
    ids2, m2 = embed_mod.load_matrix(con)
    assert m1 is m2, "every search re-reading the whole matrix from SQLite was the defect"
    source_id = con.execute("SELECT id FROM sources LIMIT 1").fetchone()[0]
    con.execute("INSERT INTO fragments (source_id, content_hash, seq, heading_path, text) "
                "VALUES (?,'h9',9,'','new')", (source_id,))
    fid = con.execute("SELECT max(id) FROM fragments").fetchone()[0]
    con.execute("INSERT INTO embeddings (fragment_id, content_hash, vector) VALUES (?,?,?)",
                (fid, "h9", (np.ones(768, dtype=np.float32) / 28).tobytes()))
    con.commit()
    ids3, m3 = embed_mod.load_matrix(con)
    assert m3 is not m1 and len(ids3) == len(ids1) + 1


def test_re_embedding_the_newest_fragment_refreshes_the_cache(con, monkeypatch):
    """SQLite reuses a freed maximum rowid, so (count, max id) alone cannot see this edit."""
    first, second = np.zeros(768, dtype=np.float32), np.zeros(768, dtype=np.float32)
    first[0], second[1] = 1.0, 1.0
    monkeypatch.setattr(embed_mod, "_embed", lambda texts, timeout=120: [list(first)] * len(texts))
    embed_mod.embed_missing(con)
    ids1, _m1 = embed_mod.load_matrix(con)
    top = max(ids1)
    source_id = con.execute("SELECT source_id FROM fragments WHERE id = ?", (top,)).fetchone()[0]

    # Rewrite the note that owns the highest fragment id, keeping its fragment count.
    con.execute("DELETE FROM fragments WHERE id = ?", (top,))     # cascades its embedding
    con.execute("INSERT INTO fragments (source_id, content_hash, seq, heading_path, text) "
                "VALUES (?,'edited',99,'','edited text')", (source_id,))
    con.commit()
    assert con.execute("SELECT max(id) FROM fragments").fetchone()[0] == top, "id must be reused"
    monkeypatch.setattr(embed_mod, "_embed", lambda texts, timeout=120: [list(second)] * len(texts))
    embed_mod.embed_source(con, source_id)

    ids2, m2 = embed_mod.load_matrix(con)
    assert (len(ids2), max(ids2)) == (len(ids1), top), "the SQL half of the key did not move"
    assert np.allclose(m2[ids2.index(top)], second), "the cache must still have refreshed"


def test_vector_outweighs_lexical_in_fusion():
    """The M2 finding: at equal weight, lexical junk outranked a correct vector hit."""
    lex = [hit(1, "s1", "junk.md"), hit(2, "s2", "junk2.md")]
    vec = [hit(3, "s3", "correct.md")]
    fused = _rrf({"lex": lex, "vec": vec}, cap=3, k=3)
    assert fused[0].path == "correct.md", "vector arm must outrank lexical noise"


def test_per_source_cap_prevents_flooding():
    flood = [hit(i, "s1", "flood.md") for i in range(1, 6)]
    other = [hit(9, "s2", "other.md")]
    fused = _rrf({"lex": flood + other}, cap=2, k=10)
    assert sum(1 for h in fused if h.source_id == "s1") == 2
    assert any(h.source_id == "s2" for h in fused)


@pytest.mark.skipif(not ollama_available(), reason="ollama or the embedder is not available")
def test_vector_arm_finds_paraphrase(con):
    """Semantic retrieval must find content whose wording differs from the query."""
    embed_mod.embed_missing(con)
    r = search_mod.vector(con, "when does the employee questionnaire go out?", k=2)
    assert r and r[0].path == "Projects/A/alpha.md"


@pytest.mark.skipif(not ollama_available(), reason="ollama or the embedder is not available")
def test_embedding_idempotent_and_normalized(con):
    total = con.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
    s1 = embed_mod.embed_missing(con)
    s2 = embed_mod.embed_missing(con)
    assert s1["embedded"] == total and s2["embedded"] == 0
    ids, mat = embed_mod.load_matrix(con)
    assert len(ids) == total
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0), "vectors must be stored normalized"


