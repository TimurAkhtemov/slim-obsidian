"""SUGGEST — everything the review card offers, and the line between lookup and judgement.

Which folders exist, which tags they already uses, how many notes are in a folder: all LOOKUP,
deterministic in code. The model is only ever asked to CHOOSE among candidates found here,
which is what stops it inventing a folder. Folders come from DIRECTORIES on disk, not from
the folders that happen to contain notes, and the WHOLE tree is walked, not a shallow slice.

Never reads a note BODY — frontmatter and path only. Called by `chat.py` per recording.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .chunk import parse_frontmatter
from .config import VAULT
from . import llm, voicetags

# Derived or binary; never a filing destination and never a source of tags or counts.
_NEVER = {"_reflections", "profile", "attachments"}

# The trees a recording may be filed INTO, walked as directories so a folder they just created
# is offered before it holds anything. `Journal/` is handled separately in `folder_tree` —
# it is a destination but a FLAT one, so it must not be walked. `Inbox/` is staging, not a
# home, and is deliberately absent.
FILING_ROOTS = (voicetags.NAMESPACE, "Notes")


@dataclass
class Folder:
    path: str      # vault-relative, e.g. "Notes/school/CS201 ML"
    count: int     # notes inside, cumulative, for orientation


def _eligible(vault: Path):
    """Every note path that may seed a folder or tag candidate. Excludes derived/binary
    top-level folders and any dot-prefixed path part, casefolded."""
    for f in sorted(vault.rglob("*.md")):
        rel = f.relative_to(vault)
        if not rel.parts or rel.parts[0].casefold() in _NEVER:
            continue
        if any(p.startswith(".") for p in rel.parts):
            continue
        yield rel


def folder_tree(vault: Path = VAULT, *, max_depth: int | None = None) -> list[Folder]:
    """Every real folder that may receive a recording, with cumulative note counts —
    `Notes/school` counts everything beneath it, not just its direct children.

    ⚠ **Folders come from DIRECTORIES on disk, not from the folders that happen to contain
    notes.** A folder they just created to hold a course they are about to record is empty, so
    deriving folders from files hid it exactly when they needed it. An empty folder reports
    `count: 0`, which is the honest signal.

    ⚠ **`max_depth` defaults to None — the WHOLE tree.** At 3 it hid MOST of the tree: measured
    on the real vault, 48 of 84 folders sit at depth 4–7, including the pattern actually in use.
    Passing a number bounds BROWSING only, never filing — notes below the bound keep counting
    toward their visible ancestors.

    ⚠ The floor appears but is FLAT. Walking directories is precisely how a subfolder could
    sneak into it, so an empty dated folder on disk is still never emitted.
    """
    counts: Counter[str] = Counter()
    for rel in _eligible(vault):
        parts = list(rel.parts[:-1])
        if not parts:
            continue
        if parts[0] == voicetags.JOURNAL_FOLDER:
            counts[voicetags.JOURNAL_FOLDER] += 1
            continue
        deepest = len(parts) if max_depth is None else min(len(parts), max_depth)
        for depth in range(1, deepest + 1):
            counts["/".join(parts[:depth])] += 1

    found: set[str] = set()
    for root in FILING_ROOTS:
        base = vault / root
        if not base.is_dir():
            continue
        found.add(root)
        for d in base.rglob("*"):
            if not d.is_dir():
                continue
            rel = d.relative_to(vault)
            if any(p.startswith(".") for p in rel.parts):
                continue
            if max_depth is not None and len(rel.parts) > max_depth:
                continue
            found.add("/".join(rel.parts))
    if (vault / voicetags.JOURNAL_FOLDER).is_dir():
        found.add(voicetags.JOURNAL_FOLDER)      # the floor itself, never its children

    return [Folder(path=p, count=counts.get(p, 0)) for p in sorted(found)]


def tag_counts(vault: Path = VAULT) -> dict[str, int]:
    """How often each `topics:` value is already used across eligible notes, lowercased.

    Handles both YAML spellings: a comma-separated string and a real list. A tag with no
    history is absent, not zero-valued, which is what lets the UI show 'this would be new.'
    """
    counts: Counter[str] = Counter()
    for rel in _eligible(vault):
        fm, _ = parse_frontmatter((vault / rel).read_text(encoding="utf-8"))
        raw = (fm or {}).get("topics")
        if isinstance(raw, str):
            values = [p.strip() for p in raw.split(",")]
        else:
            values = [str(v).strip() for v in (raw or [])]
        for v in values:
            if v:
                counts[v.lower()] += 1
    return dict(counts)


# --------------------------------------------------------------------------
# the model layer — it may only CHOOSE among what the lookups above found
# --------------------------------------------------------------------------

NUM_PREDICT = 700       # bounded: greedy decoding into a grammar loops on unbounded strings

_SCHEMA = {
    "type": "object",
    "properties": {
        "dest_dir": {"type": "string", "maxLength": 200},
        "confident_depth": {"type": "integer"},
        "type_tag": {"type": "string", "maxLength": 40},
        "topics": {"type": "array", "items": {"type": "string", "maxLength": 40}},
    },
    "required": ["dest_dir", "confident_depth", "type_tag", "topics"],
}

SYSTEM = """You are filing one recording into an existing vault.

You may ONLY choose a destination from the folder list given to you. Never invent a folder.

`confident_depth` is how many leading path segments you are SURE of. If you know which subject a
lecture belongs to but not which course, say 2 for `Notes/<subject>/...` and still give your best
full guess in `dest_dir`.

Prefer topics that already appear in the vault over inventing new ones."""


@dataclass
class Card:
    dest_dir: str
    confident_depth: int
    candidates: list[str]
    type_tag: str
    topics: list[str]


def build_card(transcript: str, *, declared_type: str = "", title: str = "",
               notes: str = "", vault: Path = VAULT, model: str | None = None) -> Card:
    """What the card should offer for this recording.

    Every judgement here is a CHOICE among things `folder_tree` and `tag_counts` already found.
    A destination the model invents is walked back to its deepest real ancestor rather than
    created — they still gets `Notes/school`, which is right as far as it goes.

    ⚠ **A model failure must still produce a filable card.** Filing may never depend on a model
    call succeeding: the note is written BEFORE the card is shown, and a recording that failed
    to file is exactly the Notion failure this feature exists to remove.

    `title` is the one THEY TYPED, or empty, and `notes` are what they typed while listening. Both
    are testimony in the same sense a declared type is, so both are shown first and labelled as
    their. ⚠ Neither may ever be model-generated: that is derived from this same transcript, so it
    adds no evidence and lets one guess reinforce the next.
    """
    folders = folder_tree(vault)
    known = {f.path for f in folders}

    listing = "\n".join(f"{f.path}  ({f.count})" for f in folders) or "(vault is empty)"
    # First, and labelled, so their words read as statements rather than as more transcript.
    # Bounded: notes are typed live and can run long, and the transcript still has to fit.
    named = f"They titled this: {title.strip()}\n\n" if title.strip() else ""
    if notes.strip():
        named += f"Their notes while listening:\n{notes.strip()[:4000]}\n\n"
    vocab = ", ".join(f"{t} ({c})" for t, c in
                      sorted(tag_counts(vault).items(), key=lambda kv: -kv[1])[:40])
    try:
        raw, _stats = llm.chat_json(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": (f"{named}Folders:\n{listing}\n\n"
                                          f"Existing topics: {vocab}\n\n"
                                          f"Transcript:\n{transcript[:12000]}")}],
            _SCHEMA,
            model=model or llm.RECORD_MODEL,
            num_ctx=llm.RECORD_CTX,      # ⚠ moves with chat.handle_record's summarize call
            num_predict=NUM_PREDICT,
            temperature=0.0,      # structured work: determinism IS correctness
            think=False,          # the summary pass thinks; this classification does not
        )
    except Exception:             # noqa: BLE001 - a card is still owed, see the docstring
        raw = {}

    # A type they SPOKE is testimony; a model's guess never overrides it.
    type_tag = declared_type or str(raw.get("type_tag") or "")
    topics = [str(t).strip() for t in (raw.get("topics") or []) if str(t).strip()]

    if type_tag == "journal":
        # ⚠ The floor is FLAT and permanent, so the destination is fixed regardless of what
        # the model chose. Topics ride along as proposed: they are frontmatter for Obsidian's
        # tag pane, and since v12 nothing promotes them into anything retrieval reads.
        return Card(dest_dir=voicetags.JOURNAL_FOLDER, confident_depth=1, candidates=[],
                    type_tag=type_tag, topics=topics)

    dest = str(raw.get("dest_dir") or "").strip("/")
    if dest not in known:
        parts = dest.split("/") if dest else []
        while parts and "/".join(parts) not in known:
            parts.pop()
        dest = "/".join(parts) or f"{voicetags.NAMESPACE}/{voicetags.UNFILED}"

    depth = max(0, min(int(raw.get("confident_depth") or 0), len(dest.split("/"))))
    parent = "/".join(dest.split("/")[:depth]) if depth else ""
    candidates = sorted(
        f.path.split("/")[-1] for f in folders
        if parent and f.path.startswith(parent + "/")
        and f.path.count("/") == parent.count("/") + 1)

    # ⚠ THE CARD PROPOSES NO LINKS, and the reason is a measurement: the feature made two
    # proposals in its life and both were the same wrong path. It could not do better by
    # construction — this function shows the model FOLDER names and asked it to cite NOTES, so
    # every link was invented or lifted from the listing, and nothing checked the target
    # existed. Rebuilding it needs real note paths, an existence check, and `[[wikilinks]]`.
    return Card(dest_dir=dest, confident_depth=depth, candidates=candidates,
                type_tag=type_tag, topics=topics)
