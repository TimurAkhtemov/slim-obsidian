"""SKILLS — the copilot's slash commands: prompt templates kept as Markdown files.

`/name args` in the sidebar runs one. The built-ins ship in `slim/skills/`, and the owner's own
live in the vault's `Skills/` folder (excluded from ingest: a prompt is not evidence), where a
file overrides the built-in of the same name. The file name is the command; the body is the
prompt, with `{{args}}` standing for whatever follows the command.

Frontmatter says what the model reads. Code builds that input (`copilot.gather`), never
retrieval, and never the model:

    input:   note | selection | folder   the open note (default), the editor selection, or
                                         the notes in the open note's folder
    filter:  {type: meeting}             folder only: frontmatter equality
    section: Notes                       keep only this heading's section of each note
    output:  chat | edit | new-note      edit and new-note always propose changes to accept
    mode:    quick | deep                the default depth; the sidebar's selector still wins
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VAULT_DIR = "Skills"
BUILTIN_DIR = Path(__file__).with_name("skills")
INPUTS = ("note", "selection", "folder")
OUTPUTS = ("chat", "edit", "new-note")
MODES = ("quick", "deep")
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)


@dataclass
class Skill:
    name: str
    prompt: str = ""
    description: str = ""
    input: str = "note"
    filter: dict = field(default_factory=dict)
    section: str = ""
    output: str = "chat"
    mode: str = ""
    origin: str = "builtin"
    error: str = ""

    @property
    def proposes(self) -> bool:
        return self.output in ("edit", "new-note")

    def listing(self) -> dict:
        return {"name": self.name, "description": self.description, "input": self.input,
                "output": self.output, "mode": self.mode, "origin": self.origin,
                "error": self.error}


def parse_skill(name: str, text: str, origin: str) -> Skill:
    skill = Skill(name=name, origin=origin)
    if not _NAME.fullmatch(name):
        skill.error = "a command name is lowercase letters, digits and dashes"
        return skill
    meta, body = {}, text
    match = _FRONTMATTER.match(text)
    if match:
        body = text[match.end():]
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            skill.error = f"frontmatter is not valid YAML ({type(exc).__name__})"
            return skill
        if not isinstance(meta, dict):
            skill.error = "frontmatter must be key: value lines"
            return skill
    skill.prompt = body.strip()
    skill.description = str(meta.get("description") or "").strip()
    skill.input = str(meta.get("input") or "note").strip().lower()
    skill.section = str(meta.get("section") or "").strip()
    skill.output = str(meta.get("output") or "chat").strip().lower()
    skill.mode = str(meta.get("mode") or "").strip().lower()
    raw_filter = meta.get("filter") or {}
    if skill.input not in INPUTS:
        skill.error = f"input must be one of {', '.join(INPUTS)}"
    elif skill.output not in OUTPUTS:
        skill.error = f"output must be one of {', '.join(OUTPUTS)}"
    elif skill.mode and skill.mode not in MODES:
        skill.error = f"mode must be one of {', '.join(MODES)}"
    elif not isinstance(raw_filter, dict):
        skill.error = "filter must be key: value pairs"
    elif not skill.prompt:
        skill.error = "the prompt is empty"
    skill.filter = {str(k): str(v) for k, v in raw_filter.items()} if isinstance(raw_filter, dict) else {}
    return skill


def load(vault: Path) -> dict[str, Skill]:
    """Every skill by name: the built-ins, then the vault's, which win on a clash."""
    found: dict[str, Skill] = {}
    for origin, folder in (("builtin", BUILTIN_DIR), ("vault", Path(vault) / VAULT_DIR)):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                found[path.stem] = Skill(name=path.stem, origin=origin,
                                         error=f"unreadable ({type(exc).__name__})")
                continue
            found[path.stem] = parse_skill(path.stem, text, origin)
    return dict(sorted(found.items()))


def invocation(question: str, loaded: dict[str, Skill]) -> tuple[Skill, str] | None:
    """`/name args` → (skill, args), or None when the text does not name a known command."""
    match = re.match(r"^/([a-z0-9][a-z0-9-]*)(?:\s+(.*))?$", question.strip(), re.DOTALL)
    if not match or match.group(1) not in loaded:
        return None
    return loaded[match.group(1)], (match.group(2) or "").strip()


def render_prompt(skill: Skill, args: str) -> str:
    if "{{args}}" in skill.prompt:
        if args:
            return skill.prompt.replace("{{args}}", args)
        return skill.prompt.replace(" {{args}}", "").replace("{{args}}", "")
    return f"{skill.prompt}\n\n{args}" if args else skill.prompt
