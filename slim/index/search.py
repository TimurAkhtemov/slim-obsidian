"""Retrieval: lexical (FTS5/BM25), vector (brute-force cosine), and `hybrid` — the two
fused by reciprocal rank with a per-source cap. Hybrid is what the copilot calls.

Scope is a `prefixes` argument. The lexical arm applies it in SQL; the vector arm scores the
whole matrix either way, so it narrows the ROWS READ and then walks the ranking until `k`
eligible fragments are found (see `vector`). Either way scope decides eligibility, never
score. There is no vault-wide Q&A lane: the copilot is a note- and folder-scoped assistant
by decision, so there is no boost arm and no dispatcher.
"""

import re
import sqlite3
import threading
from dataclasses import dataclass

import numpy as np

from . import embed as embed_mod

# A tuple of path prefixes. Applied in SQL, because a predicate run AFTER ranking forced both
# arms to read the whole corpus on every scoped call, and the copilot scopes every call.
Prefixes = tuple[str, ...]

RRF_K = 60          # standard reciprocal-rank-fusion damping constant
PER_SOURCE_CAP = 3  # no single note may flood the evidence bundle
POOL = 60           # candidates pulled per arm before fusion

# Vector arm outweighs lexical 4:1 in fusion. Measured, not guessed (2026-07-13):
# at equal weight, a paraphrased query (no term overlap) still gets lexical's top-5
# JUNK at full RRF weight, which outranks the vector arm's correct hit — paraphrase
# recall@5 collapsed to 0.30 while vector alone scored 0.70. Sweeping w_vec:
# 1.0 -> para 0.30, 2.0 -> 0.40, 4.0 -> 0.60, 6.0+ -> gold recall starts degrading.
# 4.0 keeps gold recall@5 at its max (0.926) and doubles paraphrase recall.
VEC_WEIGHT = 4.0


@dataclass
class Hit:
    fragment_id: int
    source_id: str
    path: str
    title: str | None
    heading_path: str | None
    text: str


FRAG_SELECT = """
    SELECT f.id AS fragment_id, s.id AS source_id, s.path, s.title, f.heading_path, f.text
    FROM fragments f JOIN sources s ON s.id = f.source_id
"""


_MATRIX = {"key": None, "ids": [], "mat": None}
# `server` is a ThreadingHTTPServer and every handler opens its own connection, so two copilot
# turns reach this cache at once. Without the lock a reader can take `ids` from one generation
# and `mat` from the next, and the mismatched pair indexes out of range in `vector`.
_MATRIX_LOCK = threading.Lock()


def load_matrix(con: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """All current embeddings as (fragment_ids, normalized matrix). Cached per process and
    invalidated by (db file, row count, max id, this process's write count from
    `embed.generation`): an edit rewrites its fragments under new ids and cascades their old
    vectors away, so any change moves the key. A write from ANOTHER process that both reuses
    the maximum fragment id and leaves the row count unchanged is the one edit the key cannot
    see; that note keeps its previous vectors here until the next change."""
    dbfile = con.execute("PRAGMA database_list").fetchone()[2]
    n, top = con.execute("SELECT count(*), max(fragment_id) FROM embeddings").fetchone()
    with _MATRIX_LOCK:
        key = (dbfile, n, top, embed_mod.generation())
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
           if rows else np.zeros((0, embed_mod.DIM), dtype=np.float32))
    # Every caller shares this buffer, so an in-place write would corrupt every later turn.
    mat.flags.writeable = False
    with _MATRIX_LOCK:
        _MATRIX.update(key=key, ids=ids, mat=mat)
    return ids, mat


def fts_query(query: str, any_terms: bool = False) -> str:
    """Sanitize a natural query into an FTS5 expression (quoted terms)."""
    terms = re.findall(r"[A-Za-z0-9]+", query)
    if not terms:
        return '""'
    joiner = " OR " if any_terms else " "
    return joiner.join(f'"{t}"' for t in terms)


def _rows_to_hits(rows) -> list[Hit]:
    return [Hit(fragment_id=r["fragment_id"], source_id=r["source_id"], path=r["path"],
                title=r["title"], heading_path=r["heading_path"], text=r["text"]) for r in rows]


def _like(p: str) -> str:
    """Escape the LIKE wildcards in a literal path. `Capture/_unfiled` is a real folder."""
    return p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _scope(prefixes: Prefixes | None) -> tuple[str, tuple]:
    """SQL for 'path is one of these, or under one of these'. Case-insensitive like APFS."""
    if not prefixes:
        return "", ()
    clauses, params = [], []
    # Deduped on the key the SQL matches on, so ("Notes/school", "notes/School") is one clause
    # and not two. The first spelling is the one kept, for a readable query.
    unique: dict[str, str] = {}
    for x in prefixes:
        unique.setdefault(x.rstrip("/").casefold(), x.rstrip("/"))
    for p in unique.values():
        clauses.append("(s.path = ? COLLATE NOCASE OR s.path LIKE ? ESCAPE '\\')")
        params += [p, _like(p) + "/%"]
    return " AND (" + " OR ".join(clauses) + ")", tuple(params)


def lexical(con: sqlite3.Connection, query: str, k: int = POOL,
            any_terms: bool = False,
            prefixes: Prefixes | None = None) -> list[Hit]:
    """Lexical arm: BM25 over FTS5; retries with OR-matching when AND finds nothing."""
    where, scope_params = _scope(prefixes)
    sql = f"""
        SELECT f.id AS fragment_id, s.id AS source_id, s.path, s.title,
               f.heading_path, f.text, bm25(fragments_fts) AS bm25
        FROM fragments_fts
        JOIN fragments f ON f.id = fragments_fts.rowid
        JOIN sources s ON s.id = f.source_id
        WHERE fragments_fts MATCH ? AND s.deleted = 0{where}
        ORDER BY bm25
        LIMIT ?
    """
    try:
        rows = con.execute(sql, (fts_query(query, any_terms), *scope_params, k)).fetchall()
    except sqlite3.OperationalError:   # degenerate query after sanitization
        return []
    if not rows and not any_terms:
        return lexical(con, query, k, any_terms=True, prefixes=prefixes)
    return _rows_to_hits(rows)


def vector(con: sqlite3.Connection, query: str, k: int = POOL,
           prefixes: Prefixes | None = None) -> list[Hit]:
    """Vector arm: brute-force cosine over normalized embeddings (exact, no ANN)."""
    ids, mat = load_matrix(con)
    if not ids:
        return []
    qv = embed_mod.embed_query(query)
    sims = mat @ qv                                    # normalized vectors: dot == cosine
    if prefixes:
        # The arm scores the whole matrix either way, so scope narrows the ROWS READ, not the
        # arithmetic: ids only, then walk the ranking until k eligible ones are found. Text is
        # loaded for the survivors alone.
        where, params = _scope(prefixes)
        eligible = {r[0] for r in con.execute(
            "SELECT f.id FROM fragments f JOIN sources s ON s.id = f.source_id "
            f"WHERE s.deleted = 0{where}", params)}
        order = []
        for i in np.argsort(-sims):
            if ids[i] in eligible:
                order.append(i)
                if len(order) >= k:
                    break
    else:
        order = list(np.argsort(-sims)[:k])
    if not order:
        return []
    frag_ids = [ids[i] for i in order]
    rows = con.execute(
        f"{FRAG_SELECT} WHERE f.id IN ({','.join('?' * len(frag_ids))}) AND s.deleted = 0",
        frag_ids).fetchall()
    by_id = {r["fragment_id"]: r for r in rows}
    return _rows_to_hits([by_id[fid] for fid in frag_ids if fid in by_id])


def _rrf(rankings: dict[str, list[Hit]], cap: int, k: int) -> list[Hit]:
    """Weighted reciprocal-rank fusion across arms, then per-source capping."""
    weights = {"vec": VEC_WEIGHT, "lex": 1.0}
    fused: dict[int, Hit] = {}
    scores: dict[int, float] = {}
    for arm, hits in rankings.items():
        w = weights.get(arm, 1.0)
        for rank, h in enumerate(hits, start=1):
            scores[h.fragment_id] = scores.get(h.fragment_id, 0.0) + w / (RRF_K + rank)
            fused.setdefault(h.fragment_id, h)
    ordered = sorted(fused.values(), key=lambda h: -scores[h.fragment_id])
    out: list[Hit] = []
    per_source: dict[str, int] = {}
    for h in ordered:
        if per_source.get(h.source_id, 0) >= cap:
            continue
        per_source[h.source_id] = per_source.get(h.source_id, 0) + 1
        out.append(h)
        if len(out) >= k:
            break
    return out


# No similarity threshold, by measurement (2026-07-13): a real v2/v3 pair of one document
# scored 0.845 and two unrelated documents on one topic 0.836. A cutoff at 0.86 misses
# supersession; at 0.84 it demotes honest corroboration. Rank; never threshold.
def hybrid(con: sqlite3.Connection, query: str, k: int = 10,
           cap: int = PER_SOURCE_CAP,
           prefixes: Prefixes | None = None) -> list[Hit]:
    """Hybrid: lexical + vector fused by reciprocal rank, per-source capped — the one
    caller-facing entry point."""
    return _rrf({"lex": lexical(con, query, prefixes=prefixes),
                 "vec": vector(con, query, prefixes=prefixes)}, cap, k)
