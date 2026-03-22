import json
from pathlib import Path

from detector.coco import build_yolo_dataset


def test_build_yolo_dataset_writes_labels_and_dataset_yaml(tmp_path: Path):
    coco_root = tmp_path / "coco"
    product_root = tmp_path / "products"
    images_dir = coco_root / "train" / "images"
    images_dir.mkdir(parents=True)
    product_root.mkdir(parents=True)

    (images_dir / "img_00001.jpg").write_bytes(b"fake-image")
    (images_dir / "img_00002.jpg").write_bytes(b"fake-image")

    annotations = {
        "images": [
            {"id": 1, "file_name": "img_00001.jpg", "width": 200, "height": 100},
            {"id": 2, "file_name": "img_00002.jpg", "width": 200, "height": 100},
        ],
        "categories": [{"id": 0, "name": "PRODUCT A"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [20, 10, 40, 20], "area": 800, "iscrowd": 0},
        ],
    }
    (coco_root / "train" / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    (product_root / "metadata.json").write_text(json.dumps({"products": []}), encoding="utf-8")

    dataset_yaml_path, _ = build_yolo_dataset(coco_root, product_root, tmp_path / "prepared", val_ratio=0.5, seed=1)

    dataset_yaml = dataset_yaml_path.read_text(encoding="utf-8")
    label_files = list((tmp_path / "prepared" / "labels").rglob("*.txt"))

    assert "images/train" in dataset_yaml
    assert "images/val" in dataset_yaml
    assert len(label_files) == 2
    assert any("0 0.200000 0.200000 0.200000 0.200000" in path.read_text(encoding="utf-8") for path in label_files)

