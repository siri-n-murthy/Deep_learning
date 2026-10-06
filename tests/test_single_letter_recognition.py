from PIL import Image
import torch

from app import validate_character_image
from recognize import build_reference_gallery


def test_reference_gallery_builds_class_samples():
    gallery, labels = build_reference_gallery(background=False, max_classes=5)

    assert gallery.ndim == 4
    assert gallery.shape[0] == len(labels) == 5
    assert gallery.shape[1] == 1
    assert gallery.shape[2] == 105
    assert gallery.shape[3] == 105
    assert all(isinstance(label, str) for label in labels)
    assert torch.is_tensor(gallery)


def test_validate_character_image_rejects_photo_like_images():
    photo = Image.new("RGB", (200, 200), color=(120, 80, 60))
    with open("tests/data_photo.png", "wb") as f:
        photo.save(f, format="PNG")

    assert validate_character_image(photo) is False


def test_validate_character_image_accepts_handwritten_style_character():
    char = Image.new("L", (120, 120), color=255)
    for x in range(20, 100):
        for y in range(20, 100):
            if abs(x - 60) < 18 or abs(y - 60) < 18:
                char.putpixel((x, y), 0)

    assert validate_character_image(char) is True
