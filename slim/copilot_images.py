"""Images the local model sees: the copilot's attachments, and the images embedded in a note.

Attachments are resumable chat state, not note evidence. They live beside the thread JSON under
``THREADS_DIR`` (outside the vault); the JSON stores only small, validated references.
Embedded images are read from the vault on every call and never stored (`note_images`).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import threading
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from . import threads

ALLOWED_MIME = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
MAX_IMAGES_PER_TURN = 3
MAX_IMAGE_BYTES = 2_000_000
MAX_HISTORY_IMAGES = 3
MAX_NAME_CHARS = 120
_IMAGE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


class ImageError(RuntimeError):
    """A user-facing refusal for an invalid or oversized image attachment."""


def _thread_dir(thread_id: str) -> Path:
    return threads.THREADS_DIR / f"{threads._validate_id(thread_id)}.images"


def request_lock(thread_id: str) -> threading.RLock:
    """Serialize one thread's save/use/rollback lifecycle across HTTP handlers."""
    tid = threads._validate_id(thread_id)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(tid, threading.RLock())


def _safe_name(value, mime: str) -> str:
    name = Path(str(value or "").replace("\\", "/")).name.strip()
    name = "".join(char for char in name if ord(char) >= 32)[:MAX_NAME_CHARS]
    return name or f"image.{ALLOWED_MIME[mime]}"


def clean_refs(raw) -> list[dict]:
    """Return bounded metadata only. Image bytes and caller-supplied paths never survive."""
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise ImageError("attachments must be a list")
    if len(raw) > MAX_IMAGES_PER_TURN:
        raise ImageError(f"attach at most {MAX_IMAGES_PER_TURN} images to one message")
    refs = []
    for item in raw:
        if not isinstance(item, dict):
            raise ImageError("each image attachment must be an object")
        image_id = str(item.get("id") or "")
        mime = str(item.get("mime") or "").lower()
        size = item.get("bytes")
        if not _IMAGE_ID_RE.fullmatch(image_id) or mime not in ALLOWED_MIME:
            raise ImageError("image attachment metadata is invalid")
        if not isinstance(size, int) or not 0 < size <= MAX_IMAGE_BYTES:
            raise ImageError("image attachment size is invalid")
        refs.append({"id": image_id, "name": _safe_name(item.get("name"), mime),
                     "mime": mime, "bytes": size})
    return refs


def _decode(item: dict) -> tuple[bytes, str, str]:
    mime = str(item.get("mime") or "").lower()
    if mime not in ALLOWED_MIME:
        raise ImageError("images must be PNG, JPEG, or WebP")
    encoded = item.get("data")
    if not isinstance(encoded, str) or not encoded:
        raise ImageError("image data is missing")
    prefix = f"data:{mime};base64,"
    if encoded.startswith("data:"):
        if not encoded.startswith(prefix):
            raise ImageError("image data type does not match its attachment type")
        encoded = encoded[len(prefix):]
    # Base64 expands by 4/3. Refuse before allocating a decoded object far over the cap.
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise ImageError(f"each image must be at most {MAX_IMAGE_BYTES // 1_000_000} MB after resizing")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("image data is not valid base64") from exc
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ImageError(f"each image must be at most {MAX_IMAGE_BYTES // 1_000_000} MB after resizing")
    signatures = {
        "image/png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": payload.startswith(b"\xff\xd8\xff"),
        "image/webp": payload.startswith(b"RIFF") and payload[8:12] == b"WEBP",
    }
    if not signatures[mime]:
        raise ImageError("image bytes do not match the declared type")
    return payload, mime, _safe_name(item.get("name"), mime)


def image_path(thread_id: str, image_id: str) -> tuple[Path, str] | None:
    """The stored file and mime for one image id, or None. Ids are validated, never joined raw."""
    if not isinstance(image_id, str) or not _IMAGE_ID_RE.fullmatch(image_id):
        return None
    directory = _thread_dir(thread_id)
    for mime, extension in ALLOWED_MIME.items():
        path = directory / f"{image_id}.{extension}"
        if path.is_file():
            return path, mime
    return None


def save_images(thread_id: str, raw) -> tuple[list[dict], list[Path]]:
    """Validate and atomically persist one message's images; return refs and new paths.

    An item without `data` is a stored ref sent back (the plugin's Regenerate re-asks a question
    with the images it had); it must already exist for this thread and creates nothing.
    """
    if raw in (None, []):
        return [], []
    if not isinstance(raw, list):
        raise ImageError("images must be a list")
    if len(raw) > MAX_IMAGES_PER_TURN:
        raise ImageError(f"attach at most {MAX_IMAGES_PER_TURN} images to one message")
    directory = _thread_dir(thread_id)
    refs, created = [], []
    try:
        for item in raw:
            if not isinstance(item, dict):
                raise ImageError("each image must be an object")
            if "data" not in item:
                ref, = clean_refs([item])
                if not (directory / f"{ref['id']}.{ALLOWED_MIME[ref['mime']]}").is_file():
                    raise ImageError(f"attached image is missing: {ref['name']}")
                refs.append(ref)
                continue
            payload, mime, name = _decode(item)
            image_id = hashlib.sha256(payload).hexdigest()
            target = directory / f"{image_id}.{ALLOWED_MIME[mime]}"
            if not target.exists():
                directory.mkdir(parents=True, exist_ok=True)
                # The HTTP server is threaded. A unique partial keeps two same-image requests
                # from sharing a staging path; replacing the identical hash target is atomic.
                token = hashlib.sha256(f"{id(payload)}-{name}".encode()).hexdigest()[:12]
                partial = target.with_name(f".{target.name}.{token}.part")
                partial.write_bytes(payload)
                partial.replace(target)
                created.append(target)
            refs.append({"id": image_id, "name": name, "mime": mime, "bytes": len(payload)})
    except Exception:
        discard_created(created)
        raise
    return refs, created


def discard_created(paths: list[Path]) -> None:
    """Roll back files created for a request that failed before it produced a turn."""
    for path in paths:
        try:
            path.unlink()
            if not any(path.parent.iterdir()):
                path.parent.rmdir()
        except (FileNotFoundError, OSError):
            pass


def load_refs(thread_id: str, raw_refs) -> tuple[list[dict], list[str]]:
    """Resolve validated refs to Ollama's base64 image strings."""
    refs = clean_refs(raw_refs)
    directory = _thread_dir(thread_id)
    images = []
    for ref in refs:
        path = directory / f"{ref['id']}.{ALLOWED_MIME[ref['mime']]}"
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise ImageError(f"attached image is missing: {ref['name']}") from exc
        if len(payload) != ref["bytes"] or len(payload) > MAX_IMAGE_BYTES:
            raise ImageError(f"attached image changed on disk: {ref['name']}")
        if hashlib.sha256(payload).hexdigest() != ref["id"]:
            raise ImageError(f"attached image failed its integrity check: {ref['name']}")
        images.append(base64.b64encode(payload).decode("ascii"))
    return refs, images


def hydrate_history(thread_id: str, history: list[dict]) -> list[dict]:
    """Attach bytes to only the most recent bounded image refs in model history."""
    hydrated = [dict(turn) for turn in history]
    remaining = MAX_HISTORY_IMAGES
    for turn in reversed(hydrated):
        if remaining <= 0 or turn.get("role") != "you":
            continue
        refs = clean_refs(turn.get("attachments"))
        if not refs:
            continue
        refs = refs[-remaining:]
        clean, images = load_refs(thread_id, refs)
        turn["attachments"] = clean
        turn["images"] = images
        remaining -= len(images)
    return hydrated


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


def note_images(text: str, vault: Path, *, note_rel: str | None = None,
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
