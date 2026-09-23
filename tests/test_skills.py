"""Copilot skills: prompt templates as Markdown files, built-in or the owner's own in `Skills/`."""

from pathlib import Path

from slim import skills


def write_skill(vault: Path, name: str, text: str) -> None:
    folder = vault / skills.VAULT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.md").write_text(text, encoding="utf-8")


def test_parse_reads_frontmatter_and_body():
    skill = skills.parse_skill("digest", (
        "---\ndescription: Digest the folder\ninput: folder\nfilter: {type: meeting}\n"
        "section: Notes\noutput: new-note\nmode: deep\n---\nConsolidate these. {{args}}\n"), "vault")
    assert (skill.name, skill.description, skill.input, skill.filter, skill.section,
            skill.output, skill.mode, skill.origin, skill.error) == (
        "digest", "Digest the folder", "folder", {"type": "meeting"}, "Notes",
        "new-note", "deep", "vault", "")
    assert skill.prompt == "Consolidate these. {{args}}"


def test_parse_defaults_and_errors():
    plain = skills.parse_skill("tldr", "Summarize this note in three lines.", "vault")
    assert (plain.input, plain.output, plain.mode, plain.error) == ("note", "chat", "", "")
    bad = skills.parse_skill("x", "---\ninput: everything\n---\nbody", "vault")
    assert bad.error == "input must be one of note, selection, folder"
    empty = skills.parse_skill("y", "---\ndescription: nothing\n---\n", "vault")
    assert empty.error == "the prompt is empty"
    broken = skills.parse_skill("z", "---\nfilter: [unclosed\n---\nbody", "vault")
    assert broken.error.startswith("frontmatter is not valid YAML")


def test_builtins_parse_cleanly_and_include_the_consolidation(tmp_path):
    loaded = skills.load(tmp_path)
    assert all(not skill.error for skill in loaded.values()), {
        name: skill.error for name, skill in loaded.items() if skill.error}
    consolidate = loaded["consolidate-my-notes"]
    assert (consolidate.input, consolidate.section, consolidate.output) == ("folder", "Notes", "new-note")
    assert consolidate.filter == {"type": "meeting"}


def test_a_vault_skill_overrides_a_builtin_of_the_same_name(tmp_path):
    write_skill(tmp_path, "actions", "My own action items prompt.")
    write_skill(tmp_path, "Bad Name", "ignored: not a command name")
    loaded = skills.load(tmp_path)
    assert loaded["actions"].origin == "vault"
    assert loaded["actions"].prompt == "My own action items prompt."
    # Listed with its error, so the sidebar can say why "/Bad Name" does nothing.
    assert loaded["Bad Name"].error == "a command name is lowercase letters, digits and dashes"


def test_invocation_parses_only_a_known_command():
    loaded = {"quiz": skills.parse_skill("quiz", "Quiz me.", "builtin")}
    skill, args = skills.invocation("/quiz  on chapter 2 ", loaded)
    assert (skill.name, args) == ("quiz", "on chapter 2")
    assert skills.invocation("/usr/bin is a path", loaded) is None
    assert skills.invocation("quiz me", loaded) is None


def test_render_prompt_fills_args_or_appends_them():
    with_slot = skills.parse_skill("a", "Explain {{args}} simply.", "builtin")
    without = skills.parse_skill("b", "Make flashcards.", "builtin")
    assert skills.render_prompt(with_slot, "entropy") == "Explain entropy simply."
    assert skills.render_prompt(with_slot, "") == "Explain simply."
    assert skills.render_prompt(without, "only chapter 3") == "Make flashcards.\n\nonly chapter 3"
