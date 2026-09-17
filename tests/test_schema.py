"""Schema migrations reach the current version atomically, from any older brain."""
import sqlite3

import pytest

from slim import db

MEMORY_TABLES = ("memory_extractions", "memory_evidence", "memory_candidates", "memories",
                 "ledger_state", "corrections", "corrections_fts")

# The v11 DDL, copied from `git show c7528b4:slim/db.py`. `sources` and `fragments` kept this
# exact shape from v3 to v11, so every helper below builds from it and only the memory tables
# and `PRAGMA user_version` say which era a fixture brain belongs to.
V11_SCHEMA = """
CREATE TABLE sources (
    id              TEXT PRIMARY KEY,
    path            TEXT UNIQUE NOT NULL,
    title           TEXT,
    origin          TEXT,
    authority       TEXT,
    status          TEXT,
    superseded_by   TEXT,
    privacy         TEXT,
    projects        TEXT NOT NULL DEFAULT '[]',
    type            TEXT,
    authored_at     TEXT,
    current_hash    TEXT NOT NULL,
    first_seen_at   TEXT NOT NULL,
    last_ingested_at TEXT NOT NULL,
    deleted         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE source_versions (
    id           INTEGER PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(id),
    content_hash TEXT NOT NULL,
    size         INTEGER,
    mtime        REAL,
    ingested_at  TEXT NOT NULL,
    UNIQUE (source_id, content_hash)
);

CREATE TABLE fragments (
    id           INTEGER PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(id),
    content_hash TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    heading_path TEXT,
    start_line   INTEGER,
    end_line     INTEGER,
    text         TEXT NOT NULL
);
CREATE INDEX idx_fragments_source ON fragments(source_id);

CREATE VIRTUAL TABLE fragments_fts USING fts5(
    text, heading_path,
    content='fragments', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER fragments_ai AFTER INSERT ON fragments BEGIN
    INSERT INTO fragments_fts(rowid, text, heading_path)
    VALUES (new.id, new.text, new.heading_path);
END;
CREATE TRIGGER fragments_ad AFTER DELETE ON fragments BEGIN
    INSERT INTO fragments_fts(fragments_fts, rowid, text, heading_path)
    VALUES ('delete', old.id, old.text, old.heading_path);
END;

CREATE TABLE embeddings (
    fragment_id  INTEGER PRIMARY KEY REFERENCES fragments(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    vector       BLOB NOT NULL
);
"""

_MEMORY_TABLES_DDL = """
CREATE UNIQUE INDEX idx_fragments_id_source ON fragments(id, source_id);
CREATE TABLE memories (id TEXT PRIMARY KEY, state TEXT);
CREATE TABLE memory_evidence (id INTEGER PRIMARY KEY,
                              memory_id TEXT REFERENCES memories(id));
CREATE TABLE memory_candidates (id TEXT PRIMARY KEY);
CREATE TABLE memory_extractions (id TEXT PRIMARY KEY);
CREATE TABLE ledger_state (id INTEGER PRIMARY KEY);
CREATE TABLE corrections (id TEXT PRIMARY KEY, text TEXT);
CREATE VIRTUAL TABLE corrections_fts USING fts5(correction_id UNINDEXED, text);
"""


def _seed(con, marker: str) -> None:
    con.execute("INSERT INTO sources (id, path, current_hash, first_seen_at, last_ingested_at) "
                "VALUES ('s1', 'Notes/school/a.md', 'hash', 't', 't')")
    con.execute("INSERT INTO fragments (source_id, content_hash, seq, text) "
                "VALUES ('s1', 'hash', 0, ?)", (marker,))


def _v3_database(path):
    con = sqlite3.connect(path)
    con.executescript(V11_SCHEMA)
    _seed(con, "representative v3 data")
    con.execute("PRAGMA user_version = 3")
    con.commit()
    con.close()


def _v10_database(path):
    """A brain as the last curated-memory release left it: the seven derived tables present."""
    con = sqlite3.connect(path)
    con.executescript(V11_SCHEMA + _MEMORY_TABLES_DDL)
    _seed(con, "representative v10 data")
    con.execute("INSERT INTO memories VALUES ('m1', 'ACTIVE')")
    con.execute("INSERT INTO memory_evidence (memory_id) VALUES ('m1')")
    con.execute("PRAGMA user_version = 10")
    con.commit()
    con.close()


def _v11_database(path):
    """A v11 brain with one source, one version row, one fragment, one embedding."""
    con = sqlite3.connect(path)
    con.executescript(V11_SCHEMA)
    con.execute("INSERT INTO sources (id, path, title, origin, authority, status, superseded_by, privacy, "
                "projects, type, authored_at, current_hash, first_seen_at, last_ingested_at) VALUES "
                "('s1','Notes/a.md','A','recorded','evidence',NULL,NULL,'LOCAL_ONLY','[\"school\"]','lecture','2026-08-01','h1','t','t')")
    con.execute("INSERT INTO source_versions (source_id, content_hash, size, mtime, ingested_at) VALUES ('s1','h1',1,1.0,'t')")
    con.execute("INSERT INTO fragments (source_id, content_hash, seq, heading_path, start_line, end_line, text) "
                "VALUES ('s1','h1',0,'',1,1,'hello')")
    con.execute("INSERT INTO embeddings (fragment_id, content_hash, vector) VALUES (1,'h1',x'00')")
    con.execute("PRAGMA user_version = 11")
    con.commit()
    con.close()


def _tables(con) -> set[str]:
    return {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_v3_migration_preserves_data_and_reaches_the_current_version(tmp_path):
    path = tmp_path / "brain.db"
    _v3_database(path)
    con = db.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert con.execute("SELECT text FROM fragments").fetchone()[0] == "representative v3 data"
    assert not (set(MEMORY_TABLES) & _tables(con)), \
        "a brain that never had the memory tables must not acquire them on the way to v11"
    con.close()


def test_v11_drops_the_curated_memory_tables_and_keeps_everything_else(tmp_path):
    """The owner's live brain was at v10 with the seven derived tables when curated memory was
    deleted (2026-08-27). One step, children before parents under foreign_keys=ON, and the
    sources/fragments it sits next to are untouched."""
    path = tmp_path / "brain.db"
    _v10_database(path)
    con = db.connect(path)
    assert not (set(MEMORY_TABLES) & _tables(con))
    objects = {row[0] for row in con.execute("SELECT name FROM sqlite_master")}
    assert "idx_fragments_id_source" not in objects, "v4's index must go with its tables"
    assert con.execute("SELECT text FROM fragments").fetchone()[0] == "representative v10 data"
    assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1
    con.close()


def test_v12_drops_dead_columns_and_source_versions(tmp_path):
    path = tmp_path / "v11.db"
    _v11_database(path)
    con = db.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 12
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "source_versions" not in tables
    cols = {r[1] for r in con.execute("PRAGMA table_info(sources)")}
    assert cols == {"id", "path", "title", "type", "authored_at", "current_hash", "deleted"}
    fcols = {r[1] for r in con.execute("PRAGMA table_info(fragments)")}
    assert fcols == {"id", "source_id", "content_hash", "seq", "heading_path", "text"}
    # data survives
    # `db.connect` sets row_factory, so unwrap before comparing.
    assert tuple(con.execute("SELECT title, type FROM sources").fetchone()) == ("A", "lecture")
    # The id is the point of the migration: a saved chat's `source_id` must still find its note.
    assert con.execute("SELECT id FROM sources").fetchone()[0] == "s1"
    assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 1
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    # a fresh brain has the same shape as a migrated one
    fresh = db.connect(tmp_path / "fresh.db")
    assert {r[1] for r in fresh.execute("PRAGMA table_info(sources)")} == cols
    assert {r[1] for r in fresh.execute("PRAGMA table_info(fragments)")} == fcols
    fresh.close()
    con.close()


def test_failed_migration_rolls_back_and_is_retryable(tmp_path, monkeypatch):
    path = tmp_path / "brain.db"
    _v3_database(path)
    original = db.MIGRATIONS[4]
    monkeypatch.setitem(
        db.MIGRATIONS, 4,
        original + "\nCREATE TABLE migration_marker (id INTEGER);\nINSERT INTO missing_table VALUES (1);",
    )
    with pytest.raises(sqlite3.OperationalError):
        db.connect(path)

    raw = sqlite3.connect(path)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 3
    objects = {row[0] for row in raw.execute("SELECT name FROM sqlite_master")}
    assert "migration_marker" not in objects
    assert "idx_fragments_id_source" not in objects
    assert raw.execute("SELECT COUNT(*) FROM fragments").fetchone()[0] == 1
    raw.close()

    monkeypatch.setitem(db.MIGRATIONS, 4, original)
    con = db.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert con.execute("SELECT COUNT(*) FROM fragments").fetchone()[0] == 1
    con.close()


def test_failed_fresh_schema_creation_rolls_back_and_is_retryable(tmp_path, monkeypatch):
    path = tmp_path / "fresh.db"
    original = db.SCHEMA
    monkeypatch.setattr(
        db, "SCHEMA",
        original + "\nCREATE TABLE fresh_marker (id INTEGER);\nINSERT INTO missing_table VALUES (1);",
    )
    with pytest.raises(sqlite3.OperationalError):
        db.connect(path)
    raw = sqlite3.connect(path)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    assert raw.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('sources','fresh_marker')",
    ).fetchone()[0] == 0
    raw.close()

    monkeypatch.setattr(db, "SCHEMA", original)
    con = db.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    con.close()
