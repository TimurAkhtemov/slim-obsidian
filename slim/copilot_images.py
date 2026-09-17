"""Bounded image attachments for the open-note copilot.

Images are resumable chat state, not note evidence. They live beside the thread JSON under
``THREADS_DIR`` (outside the vault); the JSON stores only small, validated references.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
import threading
from pathlib import Path

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
