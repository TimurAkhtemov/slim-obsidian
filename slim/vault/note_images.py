"""Images embedded in a vault note, found where Obsidian finds them and resized for the model.

Read from the vault on every call and never stored. Used by the copilot (the open note's images)
and the recorder (the images in the owner's notes, for the summary).
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


# Both embeds Obsidian renders: `![[name.png|size]]`, a vault path or a BARE NAME (Obsidian's
# own paste), and `![alt](Folder%20Name/shot.png)`, URL-encoded and relative to the note (Notion
# exports). ⚠ Obsidian's paste lands in `./Attachements` beside wherever the note was then, and
# filing moves the note away from it, so a bare name is found anywhere in the vault.
NOTE_EMBED_RE = re.compile(
    r'!\[\[([^\]|\n]+\.(?:png|jpe?g|webp))(?:\|[^\]\n]*)?\]\]'
    r'|!\[[^\]\n]*\]\(<?([^)>\n]+?\.(?:png|jpe?g|webp))>?(?:\s+"[^"\n]*")?\)', re.I)
MAX_NOTE_IMAGE_FILE_BYTES = 10_000_000
NOTE_IMAGE_MAX_EDGE = 1600


def _images_by_name(vault: Path) -> dict[str, list[Path]]:
    """Every image in the vault by file name, for embeds that name no folder. Hidden folders
    (`.obsidian`, `.trash`) are not the vault's content, so they are not searched."""
    found: dict[str, list[Path]] = {}
    for root, dirs, files in os.walk(vault):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                found.setdefault(name, []).append(Path(root) / name)
    return found


def _nearest(candidates: list[Path], vault: Path, folder: tuple[str, ...]) -> Path | None:
    """The candidate sharing the most folders with the note, then the shallowest. ⚠ A name is
    NOT unique: every converted PDF writes its own `fig-01.png`, ten of them in the real vault."""
    def shared(path: Path) -> int:
        parts = path.relative_to(vault).parent.parts
        return next((i for i, (a, b) in enumerate(zip(parts, folder)) if a != b),
                    min(len(parts), len(folder)))
    ranked = sorted(candidates, key=lambda p: (-shared(p), len(p.relative_to(vault).parts), str(p)))
    return ranked[0] if ranked else None


def _load_for_model(path: Path) -> str:
    """One image as Ollama's base64, its long edge at most NOTE_IMAGE_MAX_EDGE."""
    from io import BytesIO
    from PIL import Image, ImageOps

    raw_size = path.stat().st_size
    if raw_size > MAX_NOTE_IMAGE_FILE_BYTES:
        raise ValueError(f"too large ({raw_size} bytes)")
    img = Image.open(path)
    img.load()
    has_alpha = img.mode in ("RGBA", "LA", "PA")
    w, h = img.size
    if max(w, h) > NOTE_IMAGE_MAX_EDGE:
        scale = NOTE_IMAGE_MAX_EDGE / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    if hasattr(img, "_getexif") and img._getexif():
        img = ImageOps.exif_transpose(img)
    buf = BytesIO()
    if has_alpha:
        img.save(buf, format="PNG", optimize=True)
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load(text: str, vault: Path, *, note_rel: str | None = None,
         limit: int) -> tuple[list[tuple[str, str]], list[dict]]:
    """The images `text` embeds, found where Obsidian finds them, loaded for the model.

    Returns ([(embed target, base64)], skipped) in embed order, each file once, at most `limit`.
    An embed resolves beside the note (`note_rel`), then as a vault path, then as the file of
    that name nearest the note. Nothing outside the vault is ever read; a web image is not an
    embed of the vault's and is neither sent nor reported.
    """
    vault_root = vault.resolve()
    folder = PurePosixPath(note_rel).parent if note_rel else PurePosixPath()
    by_name: dict[str, list[Path]] | None = None
    images: list[tuple[str, str]] = []
    skipped: list[dict] = []
    seen: set[Path] = set()
    for wiki, markdown in NOTE_EMBED_RE.findall(text):
        if len(images) >= limit:
            break
        target = wiki or unquote(markdown)
        if "://" in target:
            continue
        path = None
        for rel in ([folder / target] if note_rel else []) + [PurePosixPath(target)]:
            candidate = (vault / rel).resolve()
            if candidate.is_relative_to(vault_root) and candidate.is_file():
                path = candidate
                break
        if path is None:
            if by_name is None:
                by_name = _images_by_name(vault_root)
            path = _nearest(by_name.get(PurePosixPath(target).name, []), vault_root, folder.parts)
        if path is None:
            skipped.append({"path": target, "reason": "missing"})
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            images.append((target, _load_for_model(path)))
        except Exception as exc:  # noqa: BLE001 — one unreadable image never costs the rest
            skipped.append({"path": target, "reason": str(exc)})
    return images, skipped
