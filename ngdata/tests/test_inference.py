from pathlib import Path

from detector.inference import discover_images


def test_discover_images_handles_single_file_and_directories(tmp_path: Path):
    image_path = tmp_path / "single.jpg"
    image_path.write_bytes(b"image")

    directory = tmp_path / "input"
    directory.mkdir()
    (directory / "a.jpg").write_bytes(b"image")
    (directory / "b.png").write_bytes(b"image")
    (directory / "ignore.txt").write_text("x", encoding="utf-8")

    assert discover_images(image_path) == [image_path]
    assert [path.name for path in discover_images(directory)] == ["a.jpg", "b.png"]

