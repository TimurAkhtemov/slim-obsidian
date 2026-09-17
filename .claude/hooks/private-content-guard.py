#!/usr/bin/env python3
"""Refuse to hand journal transcripts and the Notion export to an agent.

CLAUDE.md: "Journal transcripts are PRIVATE — do not read them. Every OTHER transcript is
ordinary working material."

That rule has been enforced by the agent remembering to obey it. This makes it structural,
the same way `Journal/` became a top-level sibling of `Capture/` so no grant could reach it.

METADATA STAYS ALLOWED, deliberately: `ls`, `find`, `wc`, `stat` and `sqlite3` over the brain
DB all pass. Only commands that put file CONTENT on screen are refused. Blocking counts and
dates would break the verification CLAUDE.md explicitly permits, and an agent that cannot
verify anything routes around the guard instead of respecting it.
"""
# `from __future__` keeps this runnable on the system python 3.9 that a hook inherits —
# a hook must never depend on the project venv being active.
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path


def _vault() -> Path:
    """Where the vault is, asked of the SAME code the rest of SLIM asks.

    ⚠ Loaded BY PATH, not imported: a hook runs on the system python with no venv, so
    `import slim.config` would fail on PyYAML. `slim/vaultpath.py` is stdlib-only for this.
    A second copy of the discovery here is a copy that goes stale — and a guard pointing at a
    vault that has moved protects nothing while looking exactly like it works.
    """
    module_path = Path(__file__).resolve().parent.parent.parent / "slim" / "vaultpath.py"
    spec = importlib.util.spec_from_file_location("slim_vaultpath", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load vault discovery from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.discover_vault()


try:
    VAULT = _vault()
    VAULT_ERROR = ""
except Exception as exc:  # A privacy guard with no vault identity must fail CLOSED.
    VAULT = None
    VAULT_ERROR = f"{type(exc).__name__}: {exc}"

# Journal/ only. Capture/ holds meeting and project recordings — the owner's call, 2026-08-23:
# those are not journals.
PROTECTED_DIRS = [VAULT / "Journal"] if VAULT is not None else []

# 376 real transcripts, gitignored but one glob from a push (CLAUDE.md, "Private git remote").
PROTECTED_GLOBS = [re.compile(r"Export-[0-9a-f-]{8,}")]

# Commands that put file content on screen. `ls`/`find`/`wc`/`stat`/`sqlite3` are absent on
# purpose — see the docstring.
READERS = re.compile(
    r"(?:^|[|;&]|\s)(?:cat|bat|head|tail|less|more|nl|od|xxd|strings|sed|awk|cut|sort|uniq|"
    r"grep|egrep|fgrep|rg|ag|ack|open|pbcopy|python3?|perl|ruby|node)\b"
)

REASON = (
    "BLOCKED by private-content-guard: {what} is private (CLAUDE.md — journal "
    "transcripts are for the local model, not for an agent).\n"
    "Metadata is still available to you: `ls`, `find`, `wc -l`, `stat`, frontmatter counts, "
    "and `sqlite3` over ~/Library/Application Support/slim/. Use those instead, and if you "
    "need transcript content, ask the owner to run it locally."
)


# A Bash command must name BOTH the vault's own folder and the protected one, because a
# heredoc whose PROSE says "Journal/" reads nothing, and matching prose alone blocked an audit
# report in 2026-09 (register D12).
def _hit(text: str, *, bash: bool = False) -> str | None:
    """Return a human name for the protected thing `text` refers to, or None."""
    if not text:
        return None
    expanded = os.path.expanduser(text)
    # A shell escapes the space in the vault's name, so the same path arrives as
    # `Obsidian\ Vault/...`. Match both spellings or one backslash walks through the guard.
    spellings = (expanded, expanded.replace("\\ ", " "))
    for d in PROTECTED_DIRS:
        # Built from the directory itself, not written out: renaming the protected folder
        # must not leave a stale literal here that matches the wrong thing.
        under_vault = f"{d.parent.name}/{d.name}"
        if any(str(d) in one or under_vault in one for one in spellings):
            return f"{d.name}/ in the vault"
        if not bash or any(d.parent.name in one for one in spellings):
            if any(f"/{d.name}/" in one or one.rstrip("/").endswith(f"/{d.name}")
                   for one in spellings):
                return f"{d.name}/ in the vault"
    for pattern in PROTECTED_GLOBS:
        if any(pattern.search(one) for one in spellings):
            return "the Notion export (Export-*/)"
    return None


def main() -> int:
    if VAULT_ERROR:
        print("BLOCKED by private-content-guard: could not determine the vault; refusing the "
              f"tool call instead of exposing Journal content. ({VAULT_ERROR})", file=sys.stderr)
        return 2
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # never block on a malformed event; the guard is not a gate on the harness

    tool = event.get("tool_name", "")
    args = event.get("tool_input", {}) or {}

    if tool == "Bash":
        command = args.get("command", "") or ""
        what = _hit(command, bash=True)
        if what and READERS.search(command):
            print(REASON.format(what=what), file=sys.stderr)
            return 2
        return 0

    if tool in {"Read", "Edit", "Write", "NotebookEdit", "Grep", "Glob"}:
        for key in ("file_path", "path", "notebook_path", "pattern"):
            what = _hit(str(args.get(key, "")))
            if what:
                print(REASON.format(what=what), file=sys.stderr)
                return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
