"""Fragment embeddings via Ollama (embeddinggemma; see MODEL).

Embeddings are derived state: a float32 blob per fragment, keyed by the fragment's
content hash so re-embedding is skipped when content is unchanged. The whole table
is disposable — `slim ingest` rebuilds it from the vault in minutes.

Brute-force cosine over a numpy matrix is exact and fast at personal scale
(3k fragments x 768 dims = 9 MB; a query scans it in ~2ms). No vector DB, no ANN
index — neither is justified until measured need.
"""

import json
import sqlite3
import threading
import urllib.request

import numpy as np

OLLAMA = "http://localhost:11434/api/embed"

# Measured on paraphrases, 2026-07-13: recall@5=0.70 / separation=0.247, against 0.40 / 0.139
# for the alternative. Paraphrases decide it, never a question set drafted while reading the
# corpus — those reuse its vocabulary, so lexical search wins by default.
MODEL = "embeddinggemma"
DIM = 768
BATCH = 32

# these models are trained WITH task prefixes; omitting them measurably degrades recall
DOC_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "


def _embed(texts: list[str], timeout: int = 120) -> list[list[float]]:
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp["embeddings"]


def embed_query(text: str) -> np.ndarray:
    vec = np.array(_embed([QUERY_PREFIX + text])[0], dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def embed_missing(con: sqlite3.Connection, verbose: bool = False) -> dict:
    """Embed every fragment lacking a current-hash embedding. Idempotent."""
    rows = con.execute("""
        SELECT f.id, f.content_hash, f.text
        FROM fragments f
        LEFT JOIN embeddings e ON e.fragment_id = f.id AND e.content_hash = f.content_hash
        WHERE e.fragment_id IS NULL
    """).fetchall()
    stats = {"embedded": 0, "already": 0}
    stats["already"] = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        vectors = _embed([DOC_PREFIX + r["text"] for r in batch])
        for row, vec in zip(batch, vectors):
            v = np.array(vec, dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-9  # store normalized: cosine becomes a dot product
            con.execute(
                "INSERT OR REPLACE INTO embeddings (fragment_id, content_hash, vector) "
                "VALUES (?, ?, ?)", (row["id"], row["content_hash"], v.tobytes()))
        stats["embedded"] += len(batch)
        if verbose and i % (BATCH * 10) == 0:
            print(f"  {stats['embedded']}/{len(rows)}")
    # drop embeddings for fragments that no longer exist
    con.execute("DELETE FROM embeddings WHERE fragment_id NOT IN (SELECT id FROM fragments)")
    con.commit()
    _bump_generation()
    return stats


def embed_source(con: sqlite3.Connection, source_id: str) -> dict:
    """Embed only the current fragments belonging to one source."""
    rows = con.execute("""
        SELECT f.id, f.content_hash, f.text FROM fragments f
        LEFT JOIN embeddings e ON e.fragment_id = f.id AND e.content_hash = f.content_hash
        WHERE f.source_id = ? AND e.fragment_id IS NULL ORDER BY f.seq
    """, (source_id,)).fetchall()
    embedded = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        vectors = _embed([DOC_PREFIX + row["text"] for row in batch])
        for row, vec in zip(batch, vectors):
            value = np.array(vec, dtype=np.float32)
            value /= np.linalg.norm(value) + 1e-9
            con.execute("INSERT OR REPLACE INTO embeddings (fragment_id, content_hash, vector) VALUES (?, ?, ?)", (row["id"], row["content_hash"], value.tobytes()))
        embedded += len(batch)
    con.commit()
    _bump_generation()
    return {"embedded": embedded}


_MATRIX = {"key": None, "ids": [], "mat": None}
# `chat` is a ThreadingHTTPServer and every handler opens its own connection, so two copilot
# turns reach this cache at once. Without the lock a reader can take `ids` from one generation
# and `mat` from the next, and the mismatched pair indexes out of range in `search.vector`.
_MATRIX_LOCK = threading.Lock()
# Bumped by every write below, because the SQL half of the cache key cannot see one case:
# SQLite reuses a freed maximum rowid, so re-ingesting the note that owns the highest fragment
# id leaves (count, max id) unchanged — the copilot's own path on the note they have open.
_GENERATION = 0


def _bump_generation() -> None:
    global _GENERATION
    with _MATRIX_LOCK:
        _GENERATION += 1


def load_matrix(con: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """All current embeddings as (fragment_ids, normalized matrix). Cached per process and
    invalidated by (db file, row count, max id, this process's write count): an edit rewrites
    its fragments under new ids and cascades their old vectors away, so any change moves the
    key. A write from ANOTHER process that both reuses the maximum fragment id and leaves the
    row count unchanged is the one edit the key cannot see; that note keeps its previous
    vectors here until the next change."""
    dbfile = con.execute("PRAGMA database_list").fetchone()[2]
    n, top = con.execute("SELECT count(*), max(fragment_id) FROM embeddings").fetchone()
    with _MATRIX_LOCK:
        key = (dbfile, n, top, _GENERATION)
        if key == _MATRIX["key"]:
            return _MATRIX["ids"], _MATRIX["mat"]
    rows = con.execute("""
        SELECT e.fragment_id, e.vector FROM embeddings e
        JOIN fragments f ON f.id = e.fragment_id AND f.content_hash = e.content_hash
        JOIN sources s ON s.id = f.source_id AND s.deleted = 0
        ORDER BY e.fragment_id
    """).fetchall()
    ids = [r["fragment_id"] for r in rows]
    mat = (np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
           if rows else np.zeros((0, DIM), dtype=np.float32))
    # Every caller shares this buffer, so an in-place write would corrupt every later turn.
    mat.flags.writeable = False
    with _MATRIX_LOCK:
        _MATRIX.update(key=key, ids=ids, mat=mat)
    return ids, mat
