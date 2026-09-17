"""SQLite connection and schema. Everything here is derived state — rebuildable
from the vault. Schema changes bump PRAGMA user_version with a migration step."""

import sqlite3
from pathlib import Path

from .config import DB_PATH

SCHEMA_VERSION = 12

SCHEMA = """
CREATE TABLE sources (
    id           TEXT PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,
    title        TEXT,
    type         TEXT,           -- frontmatter `type:`; `reflect` selects journals by it
    authored_at  TEXT,           -- frontmatter `date:`; `reflect`'s window
    current_hash TEXT NOT NULL,  -- content hash is identity; path is an attribute
    deleted      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE fragments (
    id           INTEGER PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(id),
    content_hash TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    heading_path TEXT,
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

-- derived, disposable: rebuilt by `slim ingest` from the vault
CREATE TABLE embeddings (
    fragment_id  INTEGER PRIMARY KEY REFERENCES fragments(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    vector       BLOB NOT NULL
);
"""


MIGRATIONS = {
    2: """
    CREATE TABLE IF NOT EXISTS embeddings (
        fragment_id  INTEGER PRIMARY KEY REFERENCES fragments(id) ON DELETE CASCADE,
        content_hash TEXT NOT NULL,
        vector       BLOB NOT NULL
    );
    """,
    3: """
    ALTER TABLE sources ADD COLUMN status TEXT;
    ALTER TABLE sources ADD COLUMN superseded_by TEXT;
    """,
    # v4-v10 are empty: they built tables for two subsystems that are gone. `connect()` needs a
    # key for every intermediate version, and v11 drops whatever they created.
    4: "",
    5: "",
    6: "",
    7: "",
    8: "",
    9: "",
    10: "",
    # v11: drop those tables. Children first — the connection runs with foreign keys ON, so a
    # parent-first drop can fail inside the transaction. IF EXISTS, because a brain that skipped
    # v4-v10 never had them. Dropping the inert index keeps a migrated brain identical to a
    # fresh one.
    11: """
    DROP TABLE IF EXISTS memory_extractions;
    DROP TABLE IF EXISTS memory_evidence;
    DROP TABLE IF EXISTS memory_candidates;
    DROP TABLE IF EXISTS memories;
    DROP TABLE IF EXISTS ledger_state;
    DROP TABLE IF EXISTS corrections_fts;
    DROP TABLE IF EXISTS corrections;
    DROP INDEX IF EXISTS idx_fragments_id_source;
    """,
    # v12: nothing read these columns. `sources.id` stays the uuid the copilot threads are
    # keyed on — a migration, not a rebuild, so saved chats keep their note.
    12: """
    DROP TABLE IF EXISTS source_versions;
    ALTER TABLE sources DROP COLUMN origin;
    ALTER TABLE sources DROP COLUMN authority;
    ALTER TABLE sources DROP COLUMN status;
    ALTER TABLE sources DROP COLUMN superseded_by;
    ALTER TABLE sources DROP COLUMN privacy;
    ALTER TABLE sources DROP COLUMN projects;
    ALTER TABLE sources DROP COLUMN first_seen_at;
    ALTER TABLE sources DROP COLUMN last_ingested_at;
    ALTER TABLE fragments DROP COLUMN start_line;
    ALTER TABLE fragments DROP COLUMN end_line;
    """,
}


def connect(db_path=None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            _migrate_atomically(con, SCHEMA, SCHEMA_VERSION)
        elif version < SCHEMA_VERSION:
            scripts = []
            for v in range(version + 1, SCHEMA_VERSION + 1):
                if v not in MIGRATIONS:
                    raise RuntimeError(f"no migration to schema v{v} — delete the DB and re-run `slim ingest`")
                scripts.append(MIGRATIONS[v])
            _migrate_atomically(con, "\n".join(scripts), SCHEMA_VERSION)
        elif version > SCHEMA_VERSION:
            raise RuntimeError(f"db schema v{version} is newer than code (v{SCHEMA_VERSION})")
        return con
    except BaseException:
        con.close()
        raise


def _migrate_atomically(con: sqlite3.Connection, script: str, target_version: int) -> None:
    """Apply schema and user_version as one retryable SQLite transaction."""
    try:
        con.executescript(
            "BEGIN IMMEDIATE;\n"
            + script
            + f"\nPRAGMA user_version = {target_version};\nCOMMIT;"
        )
    except BaseException:
        if con.in_transaction:
            con.rollback()
        raise
