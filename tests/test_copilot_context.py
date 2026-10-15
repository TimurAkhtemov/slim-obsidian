"""What the copilot reads: targeted indexing, deterministic scope, nearby notes and the evidence
a question retrieves."""

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


def test_collect_evidence_takes_the_open_note_before_any_sibling(con, monkeypatch):
    # PER_SOURCE_CAP (3) exists so no single note floods a VAULT-WIDE pack. The open note is
    # the subject of every copilot question, so it must never be capped below the pack size.
    class Hit:
        def __init__(self, fragment_id, source_id, path):
            self.fragment_id, self.source_id, self.path = fragment_id, source_id, path
            self.title, self.heading_path, self.text = path, None, f"text {fragment_id}"

    own = [Hit(f"own-{i}", "s1", "Notes/school/a.md") for i in range(12)]
    sibling = [Hit(f"sib-{i}", "s2", "Notes/school/b.md") for i in range(12)]

    def fake_hybrid(_con, _query, *, k, cap=context.search_mod.PER_SOURCE_CAP, prefixes):
        per_source, out = {}, []
        for hit in own + sibling:
            if not any(context._matches(hit.path, prefix) for prefix in prefixes):
                continue
            per_source[hit.source_id] = per_source.get(hit.source_id, 0) + 1
            if per_source[hit.source_id] <= cap:
                out.append(hit)
        return out[:k]

    monkeypatch.setattr(context.search_mod, "hybrid", fake_hybrid)
    monkeypatch.setattr(context, "scope_levels", lambda _path: [("folder", ("Notes/school",))])
    evidence, levels = context.collect_evidence(
        con, {"id": "s1", "path": "Notes/school/a.md"}, "summarize this note", limit=12)
    assert [item["source_id"] for item in evidence] == ["s1"] * 12
    assert levels == ["note"]


def test_context_refresh_without_nearby_notes_still_embeds(con, vault, monkeypatch):
    # The plugin re-syncs the open note before each question with related=False. The
    # question's vector search then needs the edited fragments EMBEDDED — a fragment
    # without a vector is invisible to retrieval (CLAUDE.md: `embed` is part of the order).
    embedded = []
    monkeypatch.setattr(context.embed_mod, "embed_source", lambda _con, sid: embedded.append(sid))
    write(vault, "Notes/school/lesson.md", "gradient descent")
    context.context_payload(con, vault, "Notes/school/lesson.md", related=False)
    assert len(embedded) == 1
