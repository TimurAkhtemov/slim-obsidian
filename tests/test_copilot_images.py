"""Vision attachments stay bounded, resumable, and outside the note/index surface."""
import base64

import pytest

from slim import copilot, copilot_images, threads


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


def raw_image(name="equation.png"):
    return {"name": name, "mime": "image/png", "data": base64.b64encode(PNG).decode()}


def test_images_round_trip_as_refs_and_thread_json_keeps_no_base64():
    refs, created = copilot_images.save_images("chat-1", [raw_image()])
    assert len(created) == 1
    clean, images = copilot_images.load_refs("chat-1", refs)
    assert clean == refs
    assert base64.b64decode(images[0]) == PNG

    threads.save_copilot(
        "chat-1", "Equation", [{"role": "you", "text": "Explain", "attachments": refs}],
        source_id="source-1", source_path="Notes/math.md", reasoning_mode="quick")
    stored = threads.load("chat-1")
    assert stored["turns"][0]["attachments"] == refs
    body = (threads.THREADS_DIR / "chat-1.json").read_text()
    assert refs[0]["id"] in body
    assert base64.b64encode(PNG).decode() not in body


def test_images_reject_wrong_mime_oversize_count_and_path_shaped_thread_id():
    with pytest.raises(copilot_images.ImageError, match="declared type"):
        copilot_images.save_images("chat-1", [{**raw_image(), "mime": "image/jpeg"}])
    with pytest.raises(copilot_images.ImageError, match="at most 3"):
        copilot_images.save_images("chat-1", [raw_image()] * 4)
    with pytest.raises(threads.ThreadError):
        copilot_images.save_images("../escape", [raw_image()])


def test_recent_image_refs_are_rehydrated_for_follow_up_turns():
    refs = []
    for i in range(4):
        payload = PNG + bytes([i])
        item = {"name": f"eq-{i}.png", "mime": "image/png",
                "data": base64.b64encode(payload).decode()}
        ref, _created = copilot_images.save_images("chat-1", [item])
        refs.append(ref[0])
    history = [{"role": "you", "text": str(i), "attachments": [ref]} for i, ref in enumerate(refs)]
    got = copilot_images.hydrate_history("chat-1", history)
    assert "images" not in got[0]
    assert [len(turn.get("images") or []) for turn in got] == [0, 1, 1, 1]


def test_a_stored_ref_can_be_sent_back_as_an_image_without_bytes():
    # Regenerate re-asks with the refs the question already has; nothing new is written.
    refs, _created = copilot_images.save_images("chat-1", [raw_image()])
    again, created = copilot_images.save_images("chat-1", [{k: refs[0][k] for k in ("id", "name", "mime", "bytes")}])
    assert again == refs and created == []
    _clean, images = copilot_images.load_refs("chat-1", again)
    assert base64.b64decode(images[0]) == PNG
    with pytest.raises(copilot_images.ImageError, match="missing"):
        copilot_images.save_images("chat-2", [{**refs[0]}])          # another thread's file
    with pytest.raises(copilot_images.ImageError, match="invalid"):
        copilot_images.save_images("chat-1", [{"id": "nope", "name": "x.png", "mime": "image/png", "bytes": 1}])


def test_image_path_finds_a_stored_file_by_validated_id_only():
    refs, _created = copilot_images.save_images("chat-1", [raw_image()])
    path, mime = copilot_images.image_path("chat-1", refs[0]["id"])
    assert path.read_bytes() == PNG and mime == "image/png"
    assert copilot_images.image_path("chat-1", "f" * 64) is None
    assert copilot_images.image_path("chat-1", "../../etc/passwd") is None
    with pytest.raises(threads.ThreadError):
        copilot_images.image_path("../escape", refs[0]["id"])


def test_thread_delete_removes_its_images_too():
    copilot_images.save_images("chat-1", [raw_image()])
    threads.save_copilot("chat-1", "Equation", [], source_id="src1",
                         source_path="Notes/x.md", reasoning_mode="quick")
    assert (threads.THREADS_DIR / "chat-1.images").is_dir()
    assert threads.delete_thread("chat-1") is True
    assert not (threads.THREADS_DIR / "chat-1.images").exists()


def test_generate_answer_puts_images_on_user_messages_only(monkeypatch):
    seen = {}

    def fake_stream(messages, **_kwargs):
        seen["messages"] = messages
        return "Answer", {"duration_s": 1}

    monkeypatch.setattr(copilot.llm, "chat_stream", fake_stream)
    copilot.generate_answer(
        "Explain", [{"role": "you", "text": "Earlier", "images": ["old"]},
                    {"role": "slim", "text": "Earlier answer", "images": ["never"]}],
        [], "quick", images=["current"])
    messages = seen["messages"]
    assert messages[1]["images"] == ["old"]
    assert "images" not in messages[2]
    assert messages[-1]["images"] == ["current"]
