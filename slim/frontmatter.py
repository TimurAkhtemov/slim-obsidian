"""The frontmatter reader and writer. Used by `chunk`, `ingest`, `record`, `record_pipeline`,
`record_review`, `reflect`, `suggest`, `edits`, `edits_note` and `copilot_context`."""

import re


def parse_frontmatter(text: str) -> tuple[dict, int]:
    """Return (frontmatter dict, number of lines consumed)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, 0
    fm = {}
    block = None    # a bare `key:` just read, which the `- item` lines below it belong to
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return fm, i + 1
        # A BLOCK LIST — `key:` then `  - item` lines — is how Obsidian's property editor
        # writes every list, so one edit there turns `audio: [a, b]` into this shape.
        item = re.match(r"^\s*-\s+(.*)$", line)
        if item and block is not None:
            fm[block] = [*(fm[block] or []), item.group(1).strip().strip("\"'")]
            continue
        m = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip().strip("\"'")
            if val.startswith("[") and val.endswith("]"):
                fm[key] = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
            else:
                fm[key] = val
            block = key if not val else None
    return {}, 0  # unterminated frontmatter: treat as body


def body_after_frontmatter(text: str) -> str:
    """The note body with any YAML frontmatter block removed. One implementation, so the
    'parse_frontmatter, then slice off `skip` lines' arithmetic lives in exactly one place."""
    _, skip = parse_frontmatter(text)
    return "\n".join(text.splitlines()[skip:])


# The one statement of frontmatter key order. Nothing PARSES it — the owner reads it, and 27
# recorded notes had grown 15 different shapes because every pass appended its fields at the
# end. Every key `record.render_note` writes appears here, in this order, enforced by
# `tests/test_frontmatter.py::test_record_renders_a_subsequence_of_the_canonical_order`.
FRONTMATTER_ORDER = (
    "title", "date", "type", "tags", "subject_by", "topics",
    "origin", "review_status", "source", "asr_model", "asr_selected", "speakers_by", "audio",
    "audio_sha256", "audio_seconds", "recorded_at", "transcribed_at", "routed_by",
    "tagged_by", "filed_by",
)


def _insert_at(head: list[str], key: str) -> int:
    """Where a NEW key belongs: before the first key that outranks it, else at the end.

    ⚠ Only new keys move. An existing key is replaced in place by `set_frontmatter`, so this
    never reorders a note that already has the field — which is what keeps a labeling pass from
    silently rewriting the shape of 265 imported notes it only meant to retag.
    """
    if key not in FRONTMATTER_ORDER:
        return len(head) - 1
    rank = FRONTMATTER_ORDER.index(key)
    for i, line in enumerate(head[1:-1], start=1):
        existing = line.split(":", 1)[0]
        if existing in FRONTMATTER_ORDER and FRONTMATTER_ORDER.index(existing) > rank:
            return i
    return len(head) - 1


def set_frontmatter(text: str, updates: dict[str, object]) -> str:
    """Return `text` with `updates` applied to its frontmatter, body untouched.

    Rewrites only the keys given: an existing key is replaced in place (so field order and
    every other line survive), a new one is inserted at its position in `FRONTMATTER_ORDER`.
    The body is sliced off by line count and re-attached unchanged, so nothing below the
    frontmatter can be disturbed by a labeling pass.

    New keys used to land at the end, which is how 27 recorded notes grew 15 different shapes.
    Both YAML spellings are replaced: a match requiring `^key:\\s` misses a bare `topics:`
    heading a BLOCK LIST, so the key looked absent and a second one was appended, leaving the
    file with two and every last-wins reader dropping the hand-curated one. A block's
    continuation lines are consumed with its key, or they strand under the next field.
    """
    fm, skip = parse_frontmatter(text)
    if not skip:
        raise ValueError("note has no parseable frontmatter")
    lines = text.splitlines()
    head, body = lines[:skip], lines[skip:]

    def render(key: str, value) -> str:
        if isinstance(value, (list, tuple)):
            # A bare `#` after a space opens a YAML comment: `[idea, #todo]` is one tag.
            items = (f'"{v}"' if str(v).startswith("#") else str(v) for v in value)
            return f"{key}: [{', '.join(items)}]"
        return f"{key}: {value}"

    for key, value in updates.items():
        rendered = render(key, value)
        for i, line in enumerate(head):
            if not re.match(rf"^{re.escape(key)}:(\s|$)", line):
                continue
            end = i + 1
            if not line[len(key) + 1:].strip():      # bare `key:` — a block list may follow
                while end < len(head) - 1:
                    nxt = head[end]
                    if nxt.strip().startswith("- ") or (nxt[:1].isspace() and nxt.strip()):
                        end += 1
                        continue
                    break
            head[i:end] = [rendered]
            break
        else:
            head.insert(_insert_at(head, key), rendered)   # at its canonical position
    return "\n".join(head + body) + ("\n" if text.endswith("\n") else "")
