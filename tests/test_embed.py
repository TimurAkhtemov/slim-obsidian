"""Embeddings: generation, normalization, and the gate that keeps a no-model run off the
embedder. The vector cache and retrieval are tested with `search`."""

import numpy as np
import pytest

from slim import db, embed as embed_mod, search as search_mod
from slim.ingest import ingest
from tests.conftest import ollama_available


@pytest.fixture
def con(tmp_path):
    v = tmp_path / "vault"
    (v / "Projects").mkdir(parents=True)
    (v / "Projects/alpha.md").write_text(
        "---\ntitle: \"Alpha\"\n---\n# Alpha\n\nThe quarterly survey launches July 15.\n")
    (v / "Projects/beta.md").write_text(
        "---\ntitle: \"Beta\"\n---\n# Beta\n\nSurvey tooling comparison for the beta project.\n")
    con = db.connect(tmp_path / "t.db")
    ingest(con, v)
    return con


def test_a_run_that_must_not_load_a_model_skips_the_embedder(monkeypatch):
    """SLIM_TEST_NO_MODELS=1 is for a run beside a job that has the model loaded: the embedder
    tests skip instead of loading a second model. Unset, they run whenever the embedder is there."""
    monkeypatch.setenv("SLIM_TEST_NO_MODELS", "1")
    assert ollama_available() is False


@pytest.mark.skipif(not ollama_available(), reason="ollama or the embedder is not available")
def test_embedding_idempotent_and_normalized(con):
    total = con.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
    s1 = embed_mod.embed_missing(con)
    s2 = embed_mod.embed_missing(con)
    assert s1["embedded"] == total and s2["embedded"] == 0
    ids, mat = search_mod.load_matrix(con)
    assert len(ids) == total
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0), "vectors must be stored normalized"
