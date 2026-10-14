"""Images embedded in a note: resolved where Obsidian finds them, never outside the vault."""
import base64

from slim import note_images


def note_image(root, rel, width=10):
    """A real image whose width tells the tests which file was sent."""
    from PIL import Image
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, 8), color="red").save(str(path))
    return path


def widths(images):
    from io import BytesIO
    from PIL import Image
    return [Image.open(BytesIO(base64.b64decode(data))).size[0] for _name, data in images]


def test_an_ambiguous_bare_name_is_the_file_nearest_the_note(tmp_path):
    """⚠ 84 image names occur more than once in the real vault (`fig-01.png` ten times: every
    converted PDF writes its own). The first match by name is some other lecture's figure."""
    note_image(tmp_path, "Notes/aaa/fig-01.png", width=11)
    note_image(tmp_path, "Notes/omscs/nlp/Module 4/fig-01.png", width=44)
    note_image(tmp_path, "Notes/omscs/ml/fig-01.png", width=22)
    images, _ = note_images.load(
        "![[fig-01.png]]", tmp_path, note_rel="Notes/omscs/nlp/Module 4/notes.md", limit=6)
    assert widths(images) == [44]
    images, _ = note_images.load(
        "![[fig-01.png]]", tmp_path, note_rel="Notes/omscs/nlp/Module 3/notes.md", limit=6)
    assert widths(images) == [44]      # a cousin folder beats one further away


def test_a_markdown_embed_resolves_beside_the_note_and_decodes_its_spaces(tmp_path):
    # Notion exports embed `![](Folder%20Name/Screenshot.png)`, relative to the note.
    note_image(tmp_path, "Notes/ml/SVMs/Screenshot 1.png", width=33)
    images, skipped = note_images.load(
        "Kernels ![kernel](SVMs/Screenshot%201.png) and ![remote](https://x.io/a.png)",
        tmp_path, note_rel="Notes/ml/SVMs.md", limit=6)
    assert widths(images) == [33]
    assert [name for name, _ in images] == ["SVMs/Screenshot 1.png"]
    assert skipped == []               # a web image is not the vault's to send, nor a miss


def test_a_note_image_never_comes_from_outside_the_vault(tmp_path):
    vault = tmp_path / "vault"
    note_image(tmp_path, "secret.png")
    images, _ = note_images.load(
        "![](../../secret.png) ![[../secret.png]]", vault, note_rel="Notes/a.md", limit=6)
    assert images == []


def test_note_images_are_sent_once_in_order_up_to_the_limit(tmp_path):
    for i in range(5):
        note_image(tmp_path, f"Notes/c/shot-{i}.png", width=10 + i)
    text = "\n".join(f"![[shot-{i}.png|300]]" for i in (3, 1, 3, 0, 4, 2))
    images, _ = note_images.load(text, tmp_path, note_rel="Notes/c/n.md", limit=3)
    assert widths(images) == [13, 11, 10]
