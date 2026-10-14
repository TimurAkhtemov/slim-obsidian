"""Heading-aware markdown chunking. Every fragment stays traceable to its source location
(heading breadcrumb + line span). Called by `ingest`; also the transcript marker and the
derived-block stripping every reader of a note agrees on."""

import re
from dataclasses import dataclass

from .config import CHUNK_MAX, CHUNK_MIN, CHUNK_TARGET
from .frontmatter import parse_frontmatter

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Fragment:
    seq: int
    heading_path: str
    start_line: int
    end_line: int
    text: str


# The synthetic H2 between a note's derived summary and its raw transcript. Matched only as a
# WHOLE stripped line, so a mention in prose can never split a note. Defined ONCE — every
# reader must agree on where a transcript begins.
TRANSCRIPT_MARKER = "## Transcript"


# The fence the recorder wraps a DERIVED summary in, inside an otherwise source-class note.
# `ingest.write_fragments` chunks the WHOLE body, so without stripping this the summary would
# be indexed and retrieved as evidence. The open line carries `model=`/`prompt=` attributes and
# is matched by PREFIX; the close line is exact.
SUMMARY_OPEN = "<!-- slim:summary"        # opening delimiter PREFIX (carries provenance attrs)
SUMMARY_CLOSE = "<!-- /slim:summary -->"  # closing delimiter (exact)

_DERIVED_FENCES = ((SUMMARY_OPEN, SUMMARY_CLOSE),)


def strip_meeting_wrapper(text: str) -> str:
    """Blank only the outer ``slim-meeting`` fence, leaving its Markdown source untouched.

    The fence is a presentation anchor for Obsidian, not evidence and not a semantic code
    block. Removing its two lines before chunking lets Summary/Notes/Transcript keep their
    ordinary Markdown meaning and preserves line numbers for citations.
    """
    lines = text.splitlines()
    fence = None
    for i, line in enumerate(lines):
        match = re.fullmatch(r"\s*(`{3,})slim-meeting[^\n]*", line)
        if match is not None:
            fence = match.group(1)
            lines[i] = ""
            break
    if fence is None:
        return text
    for i in range(i + 1, len(lines)):
        if lines[i].strip() == fence:
            lines[i] = ""
            break
    result = "\n".join(lines)
    if text.endswith("\n"):
        result += "\n"
    return result


def strip_derived_blocks(text: str) -> str:
    """Blank out every DERIVED block so ingestion never indexes SLIM's own output as evidence.

    Every removed line is replaced by an EMPTY one, preserving the line count so the surviving
    transcript's fragment line numbers stay accurate. A note without the fence is returned
    byte-for-byte. An unterminated fence fails SAFE, blanking to the end: over-excluding a
    derived block is the correct direction.
    """
    if not any(open_ in text for open_, _ in _DERIVED_FENCES):
        return text
    out, closing = [], None
    for line in text.splitlines():
        s = line.strip()
        if closing is None:
            match = next((c for o, c in _DERIVED_FENCES if s.startswith(o)), None)
            if match is not None:
                closing = match
                out.append("")
                continue
            out.append(line)
            continue
        out.append("")
        if s == closing:
            closing = None
    result = "\n".join(out)
    if text.endswith("\n"):        # splitlines() drops it; reattach so the line count is exact
        result += "\n"
    return result


def chunk_markdown(text: str) -> list[Fragment]:
    """Split on headings, then pack sections toward CHUNK_TARGET chars,
    splitting oversized sections at paragraph boundaries."""
    _, skip = parse_frontmatter(text)
    lines = text.splitlines()

    # sections: (heading_path, start_line, lines)
    sections: list[tuple[str, int, list[str]]] = []
    breadcrumb: dict[int, str] = {}
    cur_head, cur_start, cur_lines = "", skip + 1, []
    in_code = False
    for n, line in enumerate(lines[skip:], start=skip + 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            if cur_lines and any(l.strip() for l in cur_lines):
                sections.append((cur_head, cur_start, cur_lines))
            level = len(m.group(1))
            breadcrumb[level] = m.group(2).strip()
            for deeper in list(breadcrumb):
                if deeper > level:
                    del breadcrumb[deeper]
            cur_head = " > ".join(breadcrumb[k] for k in sorted(breadcrumb))
            cur_start, cur_lines = n, [line]
        else:
            cur_lines.append(line)
    if cur_lines and any(l.strip() for l in cur_lines):
        sections.append((cur_head, cur_start, cur_lines))

    # pack sections into fragments
    frags: list[Fragment] = []

    def emit(head: str, start: int, block_lines: list[str]):
        body = "\n".join(block_lines).strip()
        if not body:
            return
        frags.append(Fragment(len(frags), head, start, start + len(block_lines) - 1, body))

    for head, start, sec_lines in sections:
        size = sum(len(l) + 1 for l in sec_lines)
        if size <= CHUNK_MAX:
            # merge small sections into the previous fragment when same-ish context
            if frags and size < CHUNK_MIN and len(frags[-1].text) + size < CHUNK_MAX \
                    and frags[-1].heading_path.split(" > ")[0] == head.split(" > ")[0]:
                prev = frags[-1]
                frags[-1] = Fragment(prev.seq, prev.heading_path, prev.start_line,
                                     start + len(sec_lines) - 1,
                                     prev.text + "\n\n" + "\n".join(sec_lines).strip())
            else:
                emit(head, start, sec_lines)
            continue
        # oversized: split at blank lines near the target
        block, block_start, acc = [], start, 0
        for i, line in enumerate(sec_lines):
            block.append(line)
            acc += len(line) + 1
            if acc >= CHUNK_TARGET and not line.strip():
                emit(head, block_start, block)
                block_start += len(block)
                block, acc = [], 0
        emit(head, block_start, block)

    # ⚠ LAST RESORT: split text that offers NO line breaks to split on. Everything above splits
    # at blank lines, and a machine transcript has none — one 37-minute meeting arrived as
    # 21,800 characters containing THREE newlines and became a single fragment whose vector
    # described its opening minute. The note looks indexed and is mostly unreachable. Sentence
    # boundaries rather than a hard cut: a fragment severed mid-clause embeds badly.
    out: list[Fragment] = []
    for f in frags:
        if len(f.text) <= CHUNK_MAX:
            out.append(f)
            continue
        for piece in _split_prose(f.text):
            out.append(Fragment(0, f.heading_path, f.start_line, f.end_line, piece))

    for i, f in enumerate(out):
        f.seq = i
    return out


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_prose(text: str) -> list[str]:
    """Unbroken prose into ~CHUNK_TARGET pieces, cut at sentence boundaries.

    A "sentence" longer than CHUNK_MAX means the ASR produced no usable punctuation at all; a
    hard cut is the honest fallback there, since the alternative is the oversized fragment this
    function exists to prevent.
    """
    pieces: list[str] = []
    buf = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > CHUNK_MAX:
            if buf:
                pieces.append(buf.strip())
                buf = ""
            pieces.append(sentence[:CHUNK_TARGET].strip())
            sentence = sentence[CHUNK_TARGET:]
        if buf and len(buf) + len(sentence) + 1 > CHUNK_TARGET:
            pieces.append(buf.strip())
            buf = ""
        buf = f"{buf} {sentence}".strip() if buf else sentence
    if buf.strip():
        pieces.append(buf.strip())
    return [p for p in pieces if p]
