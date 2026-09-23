"""M1 acceptance tests: idempotent import, edit/rename identity, exclusions."""

import pytest

from slim import db
from slim.ingest import ingest, ingest_note


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    (v / "Projects/Demo").mkdir(parents=True)
    (v / ".obsidian").mkdir(parents=True)
    (v / "Projects/Demo/note.md").write_text(
        "---\ntitle: \"Demo Note\"\ntype: meeting-note\ndate: 2026-07-01\n---\n\n"
        "# Demo Note\n\n## Decisions\n\n"
        "We decided to use SQLite with FTS5 for the index.\n")
    (v / ".obsidian/junk.md").write_text("# quarantined junk\n")
    return v


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "test.db")


def counts(con):
    return {
        "sources": con.execute("SELECT COUNT(*) FROM sources WHERE deleted=0").fetchone()[0],
        "fragments": con.execute("SELECT COUNT(*) FROM fragments").fetchone()[0],
    }


def test_idempotent(con, vault):
    s1 = ingest(con, vault)
    assert s1["new"] == 1
    before = counts(con)
    s2 = ingest(con, vault)
    assert s2 == {"unchanged": 1, "new": 0, "edited": 0, "renamed": 0, "removed": 0}
    assert counts(con) == before


def test_an_excluded_dir_is_never_indexed(con, vault):
    ingest(con, vault)
    paths = [r[0] for r in con.execute("SELECT path FROM sources WHERE deleted=0")]
    assert paths == ["Projects/Demo/note.md"]
    hits = con.execute("SELECT COUNT(*) FROM fragments_fts WHERE fragments_fts MATCH '\"quarantined\"'").fetchone()[0]
    assert hits == 0


def test_edit_keeps_the_same_identity_and_rebuilds_fragments(con, vault):
    ingest(con, vault)
    sid = con.execute("SELECT id FROM sources WHERE path LIKE '%note.md'").fetchone()[0]
    f = vault / "Projects/Demo/note.md"
    f.write_text(f.read_text() + "\n## Update\n\nSwitched the reranker off.\n")
    s = ingest(con, vault)
    assert s["edited"] == 1
    assert con.execute("SELECT id FROM sources WHERE path LIKE '%note.md'").fetchone()[0] == sid
    hit = con.execute("SELECT COUNT(*) FROM fragments_fts WHERE fragments_fts MATCH '\"reranker\"'").fetchone()[0]
    assert hit >= 1


def test_rename_keeps_identity(con, vault):
    ingest(con, vault)
    sid = con.execute("SELECT id FROM sources WHERE path LIKE '%note.md'").fetchone()[0]
    (vault / "Projects/Demo/note.md").rename(vault / "Projects/Demo/renamed note.md")
    s = ingest(con, vault)
    assert s["renamed"] == 1 and s["new"] == 0 and s["removed"] == 0
    row = con.execute("SELECT id, path FROM sources WHERE deleted=0").fetchone()
    assert row["id"] == sid and row["path"] == "Projects/Demo/renamed note.md"


def test_delete_sweeps(con, vault):
    ingest(con, vault)
    (vault / "Projects/Demo/note.md").unlink()
    s = ingest(con, vault)
    assert s["removed"] == 1
    assert con.execute("SELECT COUNT(*) FROM fragments").fetchone()[0] == 0


def test_profile_namespace_is_never_ingested(con, vault):
    """Profile/ is injected context, not a source (decision-log 2026-07-17): indexing it
    would make the brain retrieve its own injected opinions as evidence."""
    (vault / "Profile").mkdir()
    (vault / "Profile/me.md").write_text("I prefer terse answers.\n")
    (vault / "Profile/other.md").write_text("Anything in the namespace is not a source.\n")
    ingest(con, vault)
    profile_rows = con.execute(
        "SELECT COUNT(*) FROM sources WHERE path LIKE 'Profile/%'").fetchone()[0]
    assert profile_rows == 0
    hits = con.execute(
        "SELECT COUNT(*) FROM fragments_fts WHERE fragments_fts MATCH '\"terse\"'"
    ).fetchone()[0]
    assert hits == 0


def test_lowercase_profile_folder_is_excluded_too(tmp_path):
    """Exclusion casefolds: on case-insensitive APFS a `profile/` folder IS `Profile/`."""
    from slim import db as db_mod
    v = tmp_path / "vault2"
    (v / "profile").mkdir(parents=True)
    (v / "profile/me.md").write_text("casefolded exclusion\n")
    con2 = db_mod.connect(tmp_path / "t2.db")
    ingest(con2, v)
    assert con2.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0


def test_managed_recorder_drafts_are_not_ingested(tmp_path):
    """A live draft changes on every keystroke and has no transcript yet. Indexing it would
    create source versions mid-recording and let an incomplete journal reach reflection."""
    from slim import ingest as ingest_mod

    vault = tmp_path / "draft-vault"
    journal = vault / "Journal" / "2026-08-23--recording--abc.slim-draft.md"
    capture = vault / "Capture" / "_unfiled" / "2026-08-23--recording--def.slim-draft.md"
    journal.parent.mkdir(parents=True)
    capture.parent.mkdir(parents=True)
    journal.write_text("private words still being typed\n")
    capture.write_text("meeting notes still being typed\n")

    assert list(ingest_mod.eligible_files(vault)) == []


def test_draft_exclusion_is_narrow_to_the_two_managed_locations(tmp_path):
    """The suffix is lifecycle state only where the recorder owns it. A similarly named
    human note elsewhere must not become an undocumented opt-out of indexing."""
    from slim import ingest as ingest_mod

    vault = tmp_path / "draft-vault"
    ordinary = vault / "Notes" / "work" / "idea.slim-draft.md"
    completed = vault / "Capture" / "_unfiled" / "completed.md"
    nested_journal = vault / "Journal" / "nested" / "idea.slim-draft.md"
    for path in (ordinary, completed, nested_journal):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name + "\n")

    got = [p.relative_to(vault).as_posix() for p in ingest_mod.eligible_files(vault)]
    assert got == [
        "Capture/_unfiled/completed.md",
        "Journal/nested/idea.slim-draft.md",
        "Notes/work/idea.slim-draft.md",
    ]


# --- the index keeps three frontmatter facts and invents none ---------------------------------

def test_source_fields_keeps_only_what_something_reads():
    """Schema v12 cut every other column. `reflect` selects on `type` and `authored_at`, the
    copilot shows `title`; anything else a note declares is Obsidian's business, so it must not
    silently reappear as a source column."""
    from slim import ingest as ingest_mod
    out = ingest_mod.source_fields({
        "title": "A", "type": "journal", "date": "2026-07-22",
        "topics": ["work"], "tags": ["idea"], "origin": "recorded"})
    assert out == {"title": "A", "type": "journal", "authored_at": "2026-07-22"}


def test_authored_at_falls_back_to_created_time():
    """Imported notes carry `created_time` where recorded ones carry `date`; `reflect`'s window
    reads one column, so the fallback lives here."""
    from slim import ingest as ingest_mod
    assert ingest_mod.source_fields({"created_time": "2026-07-01"})["authored_at"] == "2026-07-01"
    assert ingest_mod.source_fields({})["authored_at"] is None


# --- the single-note path (`ingest_note`, used by the open-note copilot) must make the SAME
# identity decisions as the sweep, or a note opened in Obsidian can rewrite the archive.

def test_a_leftover_roots_source_is_swept_on_the_next_ingest(tmp_path):
    """`ingest_roots` was deleted on 2026-09-01 with 85 `Roots/` rows still in the live
    brain. A path that maps to no file under the vault is confirmed gone and swept."""
    vault = tmp_path / "vault"
    (vault / "Notes").mkdir(parents=True)
    (vault / "Notes/kept.md").write_text("---\ntitle: kept\n---\n\nbody\n")
    con = db.connect(tmp_path / "t.db")
    con.execute(
        "INSERT INTO sources (id, path, current_hash) "
        "VALUES ('root-1', 'Roots/slim-repo/README.md', 'h')")
    con.commit()
    stats = ingest(con, vault)
    assert stats["removed"] == 1
    assert con.execute("SELECT deleted FROM sources WHERE id='root-1'").fetchone()[0] == 1


def test_a_source_under_an_excluded_dir_is_evicted_even_though_the_file_exists(vault, con):
    # Obsidian's .trash/ was not excluded until v12; rows from that era must be swept on the
    # next ingest although the files are still on disk.
    trash = vault / ".trash"
    trash.mkdir()
    (trash / "old.md").write_text("---\ntitle: old\n---\n\nsuperseded text\n")
    con.execute(
        "INSERT INTO sources (id, path, title, current_hash) "
        "VALUES ('t1', '.trash/old.md', 'old', 'deadbeef')")
    con.execute(
        "INSERT INTO fragments (source_id, content_hash, seq, heading_path, text) "
        "VALUES ('t1', 'deadbeef', 0, '', 'superseded text')")
    con.commit()
    stats = ingest(con, vault)
    assert stats["removed"] == 1
    assert con.execute("SELECT deleted FROM sources WHERE id='t1'").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM fragments WHERE source_id='t1'").fetchone()[0] == 0


def test_ingest_note_claims_a_moved_vault_note_like_the_sweep(tmp_path):
    from slim.ingest import ingest_note
    vault = tmp_path / "vault"
    (vault / "Notes/a").mkdir(parents=True)
    (vault / "Notes/b").mkdir(parents=True)
    con = db.connect(tmp_path / "t.db")
    note = vault / "Notes/a/moved.md"
    note.write_text("---\ntitle: moved\n---\n\nsame bytes\n")
    first = ingest_note(con, vault, "Notes/a/moved.md")
    note.rename(vault / "Notes/b/moved.md")
    second = ingest_note(con, vault, "Notes/b/moved.md")
    assert second["id"] == first["id"]
    assert con.execute("SELECT COUNT(*) FROM sources WHERE deleted=0").fetchone()[0] == 1


def test_sweep_skips_managed_recorder_drafts_like_the_single_note_path(tmp_path):
    from slim.ingest import ingest_note, IngestError
    vault = tmp_path / "vault"
    for rel in ["Journal/2026-08-25.slim-draft.md", "Capture/_unfiled/x.slim-draft.md",
                "Notes/own.slim-draft.md", "Notes/kept.md"]:
        target = vault / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {rel}\n\nbody\n")
    con = db.connect(tmp_path / "t.db")
    ingest(con, vault)
    indexed = {row[0] for row in con.execute("SELECT path FROM sources WHERE deleted=0")}
    # The recorder's working copy under Journal/ or Capture/_unfiled/ becomes a real note
    # when filed; indexing the draft gives that note a same-content rival. A `.slim-draft.md`
    # anywhere else is the user's own file.
    assert indexed == {"Notes/own.slim-draft.md", "Notes/kept.md"}
    with pytest.raises(IngestError, match="not indexable"):
        ingest_note(con, vault, "Journal/2026-08-25.slim-draft.md")


def test_status_reports_fragments_without_a_current_vector(con, vault, monkeypatch, capsys):
    """`embedded:` counts rows in `embeddings`, which includes STALE vectors, so after a content
    pass it can equal `fragments:` while retrieval runs on a fraction of the corpus (measured
    2026-08-04: 2,802 of 3,403). `slim status` must show the counter that catches it."""
    from slim import cli
    ingest(con, vault)
    fragments = counts(con)["fragments"]
    assert fragments > 0
    monkeypatch.setattr(cli.db, "connect", lambda *a, **k: con)
    cli.cmd_status(None)
    out = capsys.readouterr().out
    assert f"unembedded: {fragments}" in out
    assert "slim ingest" in out


def test_a_note_edited_and_moved_in_one_step_keeps_identity_when_told_where_it_came_from(con, vault):
    """Filing a recording rewrites the note AND moves it before one ingest. The hash rule
    cannot see a rename through an edit, so the mover names the old path and the row is
    retargeted instead of replaced — a copilot chat opened on the draft survives filing."""
    ingest(con, vault)
    sid = con.execute("SELECT id FROM sources WHERE path LIKE '%note.md'").fetchone()[0]
    old = vault / "Projects/Demo/note.md"
    new = vault / "Projects/Filed/note.md"
    new.parent.mkdir()
    new.write_text(old.read_text() + "\n## Filed\n\nEdited on the way.\n")
    old.unlink()
    got = ingest_note(con, vault, "Projects/Filed/note.md", moved_from="Projects/Demo/note.md")
    assert got["id"] == sid
    rows = con.execute("SELECT id, path FROM sources WHERE deleted=0").fetchall()
    assert [(r["id"], r["path"]) for r in rows] == [(sid, "Projects/Filed/note.md")]
    hit = con.execute("SELECT COUNT(*) FROM fragments_fts WHERE fragments_fts MATCH '\"Edited\"'").fetchone()[0]
    assert hit >= 1


def test_the_moved_from_hint_is_ignored_while_the_old_file_still_exists(con, vault):
    """A copy is not a move: the hint only claims a row whose file is gone."""
    ingest(con, vault)
    sid = con.execute("SELECT id FROM sources WHERE path LIKE '%note.md'").fetchone()[0]
    old = vault / "Projects/Demo/note.md"
    new = vault / "Projects/Demo/copy.md"
    new.write_text(old.read_text() + "\nchanged\n")
    got = ingest_note(con, vault, "Projects/Demo/copy.md", moved_from="Projects/Demo/note.md")
    assert got["id"] != sid
    assert con.execute("SELECT COUNT(*) FROM sources WHERE deleted=0").fetchone()[0] == 2
