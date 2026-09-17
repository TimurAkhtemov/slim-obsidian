# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf4llm>=0.0.20"]
# ///
"""Convert Canvas print-to-PDF course documents into Obsidian Markdown.

    uv run scripts/pdf2md.py PDF [PDF ...]
    uv run scripts/pdf2md.py --vault "~/Documents/Obsidian Vault" --images-dir Attachments/school PDF

For each PDF this writes `<stem>.md` beside it and its figures to
`<vault>/<images-dir>/<stem>/fig-NN.png`, embedded as `![[...]]` wikilinks.

pymupdf4llm does the extraction (headings, lists, links, code fences, tables).
This script only removes what the browser's print-to-PDF added — the page header
(title + timestamp), the page footer (Canvas URL + `n/N`), the `Published / Assign To /
Edit` toolbar, icon-font glyphs, and OCR'd text of screenshots — and normalizes what
Canvas printed oddly: links as `**<u>text (url)</u>**`, headings wrapped in `**`, code
blocks split at page breaks. Everything the author wrote stays.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

import pymupdf4llm

DATE = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M"
DATE_LINE = re.compile(rf"^\s*{DATE}\s*$")
DATE_INLINE = re.compile(DATE)
CANVAS_URL = r"https?://[\w.-]+\.instructure\.com/\S*"   # any Canvas instance
CANVAS_URL_LINE = re.compile(rf"^\s*{CANVAS_URL}\s*$")
PAGE_COUNTER = re.compile(r"^\s*\d+/\d+\s*$")
TOOLBAR = re.compile(r"^\s*(<sup>)?Published(</sup>)?.*Edit.*$")
PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
PRIVATE_USE = re.compile("[\ue000-\uf8ff]")
IMAGE_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^\s*```")


def outside_fences(md: str, fn: Callable[[str], str]) -> str:
    """Apply fn to every run of lines that is not inside a ``` fence."""
    out, buf, in_fence = [], [], False
    for line in md.split("\n"):
        if FENCE.match(line):
            if not in_fence:
                out.append(fn("\n".join(buf)))
                buf = []
            else:
                out.append("\n".join(buf))
                buf = []
            in_fence = not in_fence
            out.append(line)
            continue
        buf.append(line)
    out.append(fn("\n".join(buf)) if not in_fence else "\n".join(buf))
    return "\n".join(out)


def _url(u: str) -> str:
    """Print-to-PDF wraps long URLs with a space and toggles bold inside them."""
    return re.sub(r"[\s*]+", "", u).replace("?-", "?")


def strip_chrome(md: str) -> tuple[str, str | None, str | None]:
    """Remove the print header/footer. Returns (md, page_title, print_date)."""
    md = PRIVATE_USE.sub("", md)
    md = PICTURE_TEXT.sub("", md)
    lines = md.split("\n")
    title = next((l.strip() for l in lines if l.strip()), None)
    date = next((m.group(0) for l in lines if (m := DATE_INLINE.search(l))), None)
    out: list[str] = []
    for l in lines:
        s = l.strip()
        if title and s == title:
            continue
        if DATE_LINE.match(l) or CANVAS_URL_LINE.match(l) or PAGE_COUNTER.match(l) or TOOLBAR.match(l):
            continue
        if re.match(rf"^#+\s*{DATE}\s*$", l) or (title and re.match(r"^#+\s*" + re.escape(title) + r"\s*$", l)):
            continue
        if s in ("Create Rubric", "Find Rubric"):
            continue
        out.append(l)
    md = "\n".join(out)
    # a page break inside a paragraph leaves the header/footer inline: join the
    # sentence with a space, but keep a paragraph break when it stood on its own
    def gap(m: re.Match) -> str:
        return "\n\n" if "\n" in m.group(1) and "\n" in m.group(2) else " "

    if title and date:
        t = re.escape(title)
        md = re.sub(rf"(\s*){DATE}\s+{t}(\s*)", gap, md)
        md = re.sub(rf"(\s*){t}\s+{DATE}(\s*)", gap, md)
    md = re.sub(rf"(\s*){CANVAS_URL}\s+\d+/\d+(\s*)", gap, md)
    return md, title, date


def _junk_strike(m: re.Match) -> str:
    """Link icons print as short strikethrough runs; a real strikethrough has words."""
    x = m.group(1)
    return "" if len(x) <= 4 or not re.search(r"[a-z]{3,}", x) else m.group(0)


def convert_links(md: str) -> str:
    link = lambda m: f"[{m.group(1).strip()}]({_url(m.group(2))})"
    # **<u>text (url)</u>**  and  <u>text (url)</u>
    md = re.sub(r"\*\*<u>([^<\n]+?)\s*\((https?://[^)]*?)\)\s*</u>\*\*", link, md)
    md = re.sub(r"<u>([^<\n]+?)\s*\((https?://[^)]*?)\)\s*</u>", link, md)
    # **<u>text</u>** [icon] **<u>(url)</u>** — text and URL printed as two runs
    md = re.sub(
        r"\*\*<u>([^<\n]+?)</u>\*\*\s*(?:~~[^~\n]*~~)?\s*(?:\S{1,2}\s+)?\*\*<u>\((https?://[^)]*?)\)</u>\*\*",
        link,
        md,
    )

    # a bare **<u>(url)</u>** left over: Canvas prints a second "download" URL after file links
    def bare(m: re.Match) -> str:
        u = _url(m.group(1))
        return f"([download]({u}))" if "download" in u else f"<{u}>"

    md = re.sub(r"\*\*<u>\((https?://[^)]*?)\)</u>\*\*", bare, md)
    md = re.sub(r"\*\*\((https?://[^)]*?)\)\*\*", bare, md)
    md = re.sub(r"</?mark>", "", md)
    md = re.sub(r"~~([^~\n]*)~~", _junk_strike, md)
    md = re.sub(r"^#+\s*$\n?", "", md, flags=re.M)  # a heading that was only an icon
    md = re.sub(r"</?u>", "", md)
    md = re.sub(r"</?sup>", "", md)
    md = re.sub(r"\*\*[ \t]*\*\*", "", md)
    # link text and URL: no bold runs, no spaces in the URL
    def clean_link(m: re.Match) -> str:
        text = m.group(1).replace("**", "").strip()
        return f"[{text}]({_url(m.group(2))})"

    md = re.sub(r"\[([^\]\n]*)\]\((https?://[^)\n]+)\)", clean_link, md)
    # the download link belongs to the sentence before it
    md = re.sub(r"\)\n\n\(\[download\]", ") ([download]", md)
    # Canvas link previews print as: the link, a thumbnail, the bare URL again
    md = re.sub(r"(\[[^\]\n]+\]\((https?://[^)\n]+)\))\s*(?:!\[[^\]]*\]\([^)]+\)\s*)?<\2>[ \t]*", r"\1", md)
    md = re.sub(r"\)\n\n([,;.] )", r") \1", md)  # text that followed the preview continues the sentence
    return md


def fix_tables(md: str) -> str:
    """Canvas's due-date box prints with the page header as its first row and a
    'Create Rubric | Find Rubric' toolbar as its last; keep only the real rows."""
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            out.append(lines[i])
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        rows = [r for r in block if not DATE_INLINE.search(r) and "Create Rubric" not in r and "Find Rubric" not in r]
        if rows and re.match(r"^\s*\|(\s*-+\s*\|)+\s*$", rows[0]) and len(rows) > 1:
            rows[0], rows[1] = rows[1], rows[0]
        if out and out[-1].strip():
            out.append("")
        out.extend(rows)
    return "\n".join(out)


def fix_metadata(md: str) -> str:
    """'50 Points a file upload Submitting' / 'pdf File Types' are Canvas's assignment facts."""
    md = re.sub(r"^[ \t]*(\d+) Points\s+(.+?)\s+Submitting[ \t]*$", r"- **Points:** \1\n- **Submitting:** \2", md, flags=re.M)
    md = re.sub(r"^[ \t]*(\S+) File Types[ \t]*$", r"- **File types:** \1", md, flags=re.M)
    md = re.sub(r"(- \*\*Submitting:\*\* [^\n]+)\n\n(- \*\*File types:\*\*)", r"\1\n\2", md)
    return md


def fix_headings(md: str, title: str | None) -> str:
    """Un-bold headings; the page title becomes the single H1, any other H1 an H2."""
    lines = md.split("\n")
    in_fence = False
    first_heading_seen = False
    for n, l in enumerate(lines):
        if FENCE.match(l):
            in_fence = not in_fence
            continue
        if in_fence or not l.startswith("#"):
            continue
        m = re.match(r"^(#+)\s*(.*?)\s*$", l)
        level, text = m.group(1), m.group(2)
        tail = ""
        fused = re.match(r"^\*\*(.+?)\*\*([A-Za-z].+)$", text)  # **Heading**Paragraph text
        if fused:
            text, tail = fused.group(1), fused.group(2)
        text = text.replace("**", "").strip()
        text = re.sub(r"^_(.+?)_$", r"\1", text).strip()
        text = re.sub(r"\s+([.:])$", r"\1", text)
        text = re.sub(r"(\w) :(\s)", r"\1:\2", text)
        if not first_heading_seen:
            first_heading_seen = True
            level = "#"
        elif level == "#":
            level = "##"
        lines[n] = f"{level} {text}" + (f"\n\n{tail}" if tail else "")
    md = "\n".join(lines)
    if title and not first_heading_seen:
        md = f"# {title}\n\n{md}"
    return md


def fix_prose(md: str) -> str:
    """Punctuation the printer pushed away from its word; only outside code."""

    def fn(s: str) -> str:
        s = re.sub(r"_(\w+)_ (s|es)\b", r"*\1*\2", s)  # _feature_ s  ->  *feature*s
        s = re.sub(r"(\)|\*\*|`|_)\s+([.,;:])", r"\1\2", s)  # `os` .  ->  `os`.
        s = re.sub(r"(?<=\S)  +(?=\S)", " ", s)
        return s

    return outside_fences(md, fn)


ITEM_END = re.compile(r"(;( and| or)?|[.:])\s*$")
BULLET = re.compile(r"^(\s*)- (.*)$")
LOWER_WORD = re.compile(r"^[a-z][a-z']*$")
ITEM_SHAPE = re.compile(r"^(?:[a-z]\w*: \S|[\w./-]+\.(?:py|csv|txt|zip)(?: [–-] .*)?$)")  # "sd: A date…" / "foo.py – …"


def _is_item(l: str) -> bool:
    return bool(re.match(r"^[a-z]", l)) and bool(ITEM_END.search(l) or re.search(r"\d+ points?$", l) or ITEM_SHAPE.match(l))


def fix_lists(md: str, report: list[str]) -> str:
    """Bullets are drawn, not typed, so the converter loses some markers and adds
    others at wrapped lines; and a page break splits a paragraph in two. Repairs:
    a line that ends mid-sentence is joined with the lowercase line that follows;
    a lowercase line shaped like an item, next to bullets or in a run of such
    lines, gets its marker back."""

    def fn(seg: str) -> str:
        lines = seg.split("\n")
        out: list[str] = []
        for n, l in enumerate(lines):
            if not l.strip():
                out.append(l)
                continue
            prev_i = next((k for k in range(len(out) - 1, -1, -1) if out[k].strip()), None)
            prev = out[prev_i] if prev_i is not None else ""
            nxt = next((x for x in lines[n + 1 :] if x.strip()), "")
            pb, cb, nb = BULLET.match(prev), BULLET.match(l), BULLET.match(nxt)
            body = cb.group(2) if cb else l.strip()
            last = prev.rstrip().split(" ")[-1] if prev else ""
            prev_open = prev and not prev.startswith("#") and not ITEM_END.search(prev) and not re.search(r"\d+ points?\s*$", prev) and LOWER_WORD.match(last)
            if prev_open and re.match(r"^[a-z]", body) and not ITEM_SHAPE.match(body):
                del out[prev_i + 1 :]
                out[prev_i] = prev.rstrip() + " " + body
                report.append(f"join   : {prev.strip()[-40:]!r} + {body[:40]!r}")
                continue
            if not cb and not l.startswith("#") and _is_item(l.strip()) and (pb or nb or _is_item(nxt.strip()) or (prev and _is_item(prev.strip()))):
                indent = (pb or nb).group(1) if (pb or nb) else ""
                out.append(f"{indent}- {l.strip()}")
                report.append(f"bullet : {l.strip()[:60]!r}")
                continue
            out.append(l)
        return "\n".join(out)

    md = outside_fences(md, fn)
    return outside_fences(md, fn)  # a marker restored late makes a neighbour for the line before it


def fix_code_wraps(md: str, report: list[str]) -> str:
    """The printer wraps a long code line; the tail comes back as a short line of its own."""
    lines = md.split("\n")
    out: list[str] = []
    in_fence = False
    skip = False
    for n, l in enumerate(lines):
        if skip:
            skip = False
            continue
        if FENCE.match(l):
            in_fence = not in_fence
            out.append(l)
            continue
        nxt = lines[n + 1] if n + 1 < len(lines) else ""
        if (
            in_fence
            and len(l) >= 95
            and nxt.strip()
            and not FENCE.match(nxt)
            and len(nxt) <= 30
            and not nxt[0].isspace()
            and re.match(r"^[a-z0-9)\]]", nxt)
            and not re.match(r"^(def|import|from|for|if|elif|else|while|return|class|with|print|try|except|pass)\b", nxt)
            and not re.match(r"^\w+\s*=", nxt)
        ):
            out.append(l + nxt)
            report.append(f"code   : {l[-30:]!r} + {nxt!r}")
            skip = True
            continue
        out.append(l)
    return "\n".join(out)


def merge_fences(md: str) -> str:
    """A code block split at a page break comes back as two fences with nothing between."""
    return re.sub(r"```\n\s*\n```\n", "", md)


def place_images(md: str, stem: str, vault: Path, images_dir: str) -> tuple[str, int]:
    refs = IMAGE_REF.findall(md)
    if not refs:
        return md, 0
    dest = vault / images_dir / stem
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in refs:
        p = Path(src)
        if not p.exists():
            continue
        n += 1
        name = f"fig-{n:02d}{p.suffix}"
        shutil.copyfile(p, dest / name)
        md = md.replace(f"({src})", f"({(Path(images_dir) / stem / name).as_posix()})", 1)
    md = re.sub(r"!\[[^\]]*\]\(([^)]+)\)", lambda m: f"![[{m.group(1)}]]", md)
    return md, n


def repair_urls(md: str, pdf: Path) -> tuple[str, list[str]]:
    """The converter de-hyphenates words at line wraps, which eats real hyphens inside
    URLs. The PDF's own text keeps them, so any URL that matches one there once
    hyphens are ignored is restored to the PDF's spelling."""
    import fitz

    raw = "".join(page.get_text() for page in fitz.open(str(pdf))).replace("\n", "")
    by_key = {}
    for u in re.findall(r"https?://[^\s()<>\"]+", raw):
        u = u.rstrip(".,;:)]>")
        by_key.setdefault(u.replace("-", ""), u)
    fixed: list[str] = []

    def fix(m: re.Match) -> str:
        u = m.group(2)
        want = by_key.get(u.replace("-", ""))
        if want and want != u:
            fixed.append(f"url    : {u[-50:]!r} -> {want[-50:]!r}")
            return f"{m.group(1)}{want}{m.group(3)}"
        return m.group(0)

    md = re.sub(r"(\]\()(https?://[^)\s]+)(\))", fix, md)
    md = re.sub(r"(<)(https?://[^>\s]+)(>)", fix, md)
    return md, fixed


LIGATURES = ("ff", "fi", "fl", "ffi", "ffl")
EXTRA_WORDS = {  # in these documents, missing from the 1934 word list macOS ships
    "offline", "online", "tradeoff", "overfit", "overfitting", "underfit", "underfitting",
    "backoff", "cutoff", "dropoff", "payoff", "spinoff", "misclassification", "classifier",
    "coefficient", "workflow", "cashflow", "dataflow", "stopfile", "profile", "efficient",
    "offload", "offloading",
}


def fix_ligatures(md: str) -> tuple[str, int]:
    """Some fonts print ff/fi/fl/ffi/ffl as one glyph the extractor cannot read, and
    the word comes out as `di\ufffderent`. Try each ligature; keep the first spelling
    that is a dictionary word or a word the document itself uses cleanly."""
    if "\ufffd" not in md:
        return md, 0
    vocab = set()
    words_file = Path("/usr/share/dict/words")
    if words_file.exists():
        vocab.update(w.strip().lower() for w in words_file.read_text(encoding="utf-8", errors="ignore").split("\n"))
    vocab.update(w.lower() for w in re.findall(r"[A-Za-z]{3,}", md))
    vocab.update(EXTRA_WORDS)
    fixed = 0

    def known(w: str) -> bool:
        w = w.lower()
        if w in vocab:
            return True
        for suf in ("s", "es", "ed", "ing", "ly", "er", "est"):  # the word list has no inflections
            if w.endswith(suf):
                stem = w[: -len(suf)]
                if stem in vocab or (len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in vocab) or stem + "e" in vocab:
                    return True
        if w.endswith("ies") and w[:-3] + "y" in vocab:
            return True
        return False

    def repl(m: re.Match) -> str:
        nonlocal fixed
        word = m.group(0)
        n = word.count("\ufffd")
        import itertools

        for combo in itertools.product(LIGATURES, repeat=n):
            cand = word
            for lig in combo:
                cand = cand.replace("\ufffd", lig, 1)
            if known(cand):
                fixed += 1
                return cand
        if n == 1 and re.search(r"[A-Za-z]{2,}\ufffd(s|es|ed|ing)?$", word):  # payoff(s), tradeoff(s): a final glyph is ff
            fixed += 1
            return word.replace("\ufffd", "ff")
        return word

    md = re.sub(r"[A-Za-z]+\ufffd[A-Za-z\ufffd]*", repl, md)  # a letter before it: a word, not a formula symbol
    return md, fixed


def tidy(md: str) -> str:
    md = PICTURE_TEXT.sub("", md)
    md = "\n".join(l.rstrip() for l in md.split("\n"))
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def convert(pdf: Path, vault: Path, images_dir: str) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        md = pymupdf4llm.to_markdown(
            str(pdf), write_images=True, image_path=tmp, image_format="png", dpi=150, show_progress=False
        )
        md = md.replace("\n-----\n", "\n")
        md = re.sub(r"^[ \t]*```[ \t]*```[ \t]*$\n?", "", md, flags=re.M)  # an empty block
        md, title, date = strip_chrome(md)
        md = convert_links(md)
        md = fix_tables(md)
        md = fix_metadata(md)
        md = fix_headings(md, title)
        md = merge_fences(md)
        md = fix_prose(md)
        report: list[str] = []
        md = fix_lists(md, report)
        md = fix_code_wraps(md, report)
        md, n_img = place_images(md, pdf.stem, vault, images_dir)
        md, fixed = repair_urls(md, pdf)
        report.extend(fixed)
        md, n_lig = fix_ligatures(md)
        if n_lig:
            report.append(f"ligatures restored: {n_lig}")
        md = tidy(md)
    for r in report:
        print(f"    {r}", file=sys.stderr)
    import fitz  # bundled with pymupdf4llm

    n_pages = fitz.open(str(pdf)).page_count
    printed = f", printed from Canvas {date.split(',')[0]}" if date else ""
    note = (
        f"> [!note]\n> Converted from `{pdf.name}` ({n_pages} pages{printed}). "
        "Only the print header and footer on each page were removed; Canvas is the authoritative version.\n\n"
    )
    out = pdf.with_suffix(".md")
    out.write_text(note + md, encoding="utf-8")
    print(f"{out.name}: {n_pages} pages, {len(md.split())} words, {n_img} figures", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--vault", type=Path, default=Path("~/Documents/Obsidian Vault").expanduser())
    ap.add_argument("--images-dir", default="Attachments/school", help="vault-relative folder for figures")
    a = ap.parse_args()
    for pdf in a.pdfs:
        convert(pdf, a.vault.expanduser(), a.images_dir)


if __name__ == "__main__":
    main()
