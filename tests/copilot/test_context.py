"""What the copilot reads: targeted indexing, deterministic scope, nearby notes and the `@` pins."""

from pathlib import Path

import pytest

from slim import db
from slim.copilot import context


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "copilot.db")


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Notes/school/cs-201/week-01").mkdir(parents=True)
    (root / "Notes/school/cs-201/week-02").mkdir(parents=True)
    (root / "Notes/school/cs-202").mkdir(parents=True)
    (root / "Capture/school/cs-201").mkdir(parents=True)
    (root / "Journal").mkdir()
    return root


def write(root: Path, path: str, body: str) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"---\ntitle: {target.stem}\n---\n\n{body}\n", encoding="utf-8")
    return target



def test_sync_note_indexes_only_the_requested_file_and_preserves_identity(con, vault, monkeypatch):
    active = write(vault, "Notes/school/cs-201/week-01/lesson.md", "gradient descent")
    other = write(vault, "Notes/school/cs-202/graphs.md", "minimum spanning trees")
    monkeypatch.setattr(context.embed_mod, "embed_source", lambda _con, sid: {"embedded": 1})
    first = context.sync_note(con, vault, "Notes/school/cs-201/week-01/lesson.md")
    assert first["path"] == "Notes/school/cs-201/week-01/lesson.md"
    assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1
    active.write_text(active.read_text() + "\nNewton updates.\n", encoding="utf-8")
    second = context.sync_note(con, vault, "Notes/school/cs-201/week-01/lesson.md")
    assert second["id"] == first["id"]
    assert second["current_hash"] != first["current_hash"]
    assert other.exists()
    assert con.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1


def test_sync_note_refuses_a_path_escape(con, vault):
    with pytest.raises(context.CopilotError, match="vault-relative"):
        context.sync_note(con, vault, "../outside.md")


def test_scope_levels_come_only_from_the_open_path(monkeypatch):
    monkeypatch.setattr(context.config_mod, "registry", lambda: {"school": {"vault_path": "Notes/school"}})
    assert context.scope_levels("Notes/school/cs-201/week-01/lesson.md") == [
        ("folder", ("Notes/school/cs-201/week-01",)),
        ("course", ("Notes/school/cs-201", "Capture/school/cs-201")),
        ("subject", ("Notes/school", "Capture/school")),
    ]
    assert context.scope_levels("Journal/2026-08-23.md") == [("folder", ("Journal",))]


def test_related_notes_widens_dedupes_and_drops_missing_paths(con, vault, monkeypatch):
    paths = [
        "Notes/school/cs-201/week-01/lesson.md",
        "Notes/school/cs-201/week-01/exercises.md",
        "Notes/school/cs-201/week-02/review.md",
        "Notes/school/cs-202/graphs.md",
    ]
    sources = {}
    for path in paths:
        write(vault, path, f"shared optimization material for {path}")
        sources[path] = context.sync_note(con, vault, path, embed=False)
    (vault / paths[1]).unlink()
    monkeypatch.setattr(context.config_mod, "registry", lambda: {"school": {"vault_path": "Notes/school"}})

    class Hit:
        def __init__(self, source_id, path, title):
            self.source_id, self.path, self.title = source_id, path, title

    # Ranked vault-wide, as a real hybrid() would: the subject-level hit outranks the
    # course-level one, yet the folder/course/subject ORDER must still decide the output.
    ranked = [Hit(sources[paths[3]]["id"], paths[3], "graphs"),
              Hit(sources[paths[0]]["id"], paths[0], "lesson"),
              Hit(sources[paths[1]]["id"], paths[1], "exercises"),
              Hit(sources[paths[2]]["id"], paths[2], "review")]
    calls = []

    def fake_hybrid(_con, query, *, k, prefixes, **_kw):
        calls.append(query)
        return [hit for hit in ranked
                if any(context._matches(hit.path, prefix) for prefix in prefixes)][:k]

    monkeypatch.setattr(context.search_mod, "hybrid", fake_hybrid)
    got = context.related_notes(con, vault, sources[paths[0]], limit=5)
    assert [(item["path"], item["scope"]) for item in got] == [
        (paths[2], "course"), (paths[3], "subject")]
    # One search over the widest scope, then labelled in code — not one full hybrid()
    # (an embedding call plus an FTS scan of the same 5,000-char query) per level.
    assert len(calls) == 1
    assert len(calls[0]) <= context.RELATED_QUERY_CHARS


def test_scope_levels_survive_a_case_only_folder_rename(monkeypatch):
    # APFS is case-insensitive: `notes/School/` IS `Notes/school/`. `_matches` already casefolds;
    # the relative-path step must not raise on the same input.
    monkeypatch.setattr(context.config_mod, "registry", lambda: {"school": {"vault_path": "Notes/school"}})
    assert context.scope_levels("notes/School/cs-201/week-01/lesson.md") == [
        ("folder", ("notes/School/cs-201/week-01",)),
        ("course", ("Notes/school/cs-201", "Capture/school/cs-201")),
        ("subject", ("Notes/school", "Capture/school")),
    ]


def test_context_refresh_without_nearby_notes_still_embeds(con, vault, monkeypatch):
    # The plugin re-syncs the open note before each question with related=False. The loop's
    # search then needs the edited fragments EMBEDDED — a fragment without a vector is
    # invisible to it (CLAUDE.md: `embed` is part of the order).
    embedded = []
    monkeypatch.setattr(context.embed_mod, "embed_source", lambda _con, sid: embedded.append(sid))
    write(vault, "Notes/school/lesson.md", "gradient descent")
    context.context_payload(con, vault, "Notes/school/lesson.md", related=False)
    assert len(embedded) == 1


# --- `@` pins: the context a chat pinned ------------------------------------------------------

def test_pinned_notes_and_folders(con, vault):
    write(vault, "Notes/school/b.md", "Buzzing.")
    write(vault, "Profile/me.md", "I like graphs.")
    write(vault, "Notes/school/cs-202/sorting.md", "Quicksort.")
    text, read, left = context.pinned_message(con, vault, [
        {"kind": "note", "path": "Notes/school/b.md"}, {"kind": "note", "path": "Profile/me.md"},
        {"kind": "folder", "path": "Notes/school/cs-202"}])
    assert text.startswith("Context I've pinned:\n")
    assert "Buzzing." in text and "I like graphs." in text            # a pin is read whatever its folder
    assert "sorting.md" in text
    assert 'Searches default to Notes/school/cs-202/; pass folder "" to search the whole vault.' in text
    assert read == ["Notes/school/b.md", "Profile/me.md", "Notes/school/cs-202"] and left == []


def test_a_pinned_excluded_folder_still_lists(con, vault):
    write(vault, "Profile/me.md", "x")
    text, read, _left = context.pinned_message(con, vault, [{"kind": "folder", "path": "Profile"}])
    assert "me.md" in text and read == ["Profile"]
    assert "Searches default" not in text                             # search never looks there


def test_searches_default_to_the_first_searchable_folder(con, vault):
    write(vault, "Profile/me.md", "x")
    write(vault, "Notes/school/cs-202/sorting.md", "Quicksort.")
    text, _read, _left = context.pinned_message(con, vault, [
        {"kind": "folder", "path": "Profile"}, {"kind": "folder", "path": "Notes/school/cs-202"}])
    assert "Searches default to Notes/school/cs-202/" in text and "Searches default to Profile" not in text


def test_pins_share_the_budget_and_one_that_does_not_fit_is_named(con, vault, monkeypatch):
    monkeypatch.setattr(context, "INPUT_BUDGET_CHARS", 200)
    write(vault, "Notes/a.md", "short")
    write(vault, "Notes/big.md", "y" * 500)
    text, read, left = context.pinned_message(con, vault, [
        {"kind": "note", "path": "Notes/big.md"}, {"kind": "note", "path": "Notes/a.md"}])
    assert read == ["Notes/a.md"] and left == ["Notes/big.md"]
    assert "Left out (too long): Notes/big.md" in text and "y" * 100 not in text


def test_a_pin_that_is_gone_or_outside_the_vault_is_left_out(con, vault):
    text, read, left = context.pinned_message(con, vault, [
        {"kind": "note", "path": "Notes/gone.md"}, {"kind": "note", "path": "../secret.md"},
        {"kind": "folder", "path": "Nope"}])
    assert read == [] and left == ["Notes/gone.md", "../secret.md", "Nope"]
    assert "Not found: Notes/gone.md, ../secret.md, Nope" in text


def test_no_pins_no_message(con, vault):
    assert context.pinned_message(con, vault, []) == ("", [], [])
