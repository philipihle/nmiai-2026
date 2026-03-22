from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from detector.catalog import build_category_catalog, load_product_metadata, save_category_catalog


def load_coco_annotations(coco_root: Path) -> dict[str, Any]:
    annotations_path = coco_root / "train" / "annotations.json"
    return json.loads(annotations_path.read_text(encoding="utf-8"))


def _build_split_map(images: list[dict[str, Any]], val_ratio: float, seed: int) -> dict[int, str]:
    image_ids = [int(image["id"]) for image in images]
    random.Random(seed).shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * val_ratio))
    val_ids = set(image_ids[:val_count])
    return {image_id: ("val" if image_id in val_ids else "train") for image_id in image_ids}


def _to_yolo_line(annotation: dict[str, Any], image_lookup: dict[int, dict[str, Any]]) -> str:
    image = image_lookup[int(annotation["image_id"])]
    width = float(image["width"])
    height = float(image["height"])
    x, y, box_width, box_height = annotation["bbox"]
    x_center = (x + (box_width / 2.0)) / width
    y_center = (y + (box_height / 2.0)) / height
    normalized_width = box_width / width
    normalized_height = box_height / height
    return (
        f"{int(annotation['category_id'])} "
        f"{x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}"
    )


def build_yolo_dataset(
    coco_root: Path,
    product_root: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    data = load_coco_annotations(coco_root)
    image_lookup = {int(image["id"]): image for image in data["images"]}
    split_map = _build_split_map(data["images"], val_ratio=val_ratio, seed=seed)

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    source_images_dir = coco_root / "train" / "images"
    for image in data["images"]:
        image_id = int(image["id"])
        split = split_map[image_id]
        source_image = source_images_dir / image["file_name"]
        destination_image = output_root / "images" / split / image["file_name"]
        shutil.copy2(source_image, destination_image)

        label_path = output_root / "labels" / split / f"{Path(image['file_name']).stem}.txt"
        label_lines = [
            _to_yolo_line(annotation, image_lookup)
            for annotation in annotations_by_image.get(image_id, [])
        ]
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

    names = {int(category["id"]): category["name"] for category in data["categories"]}
    dataset_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=True, allow_unicode=True), encoding="utf-8")

    metadata_path = product_root / "metadata.json"
    if metadata_path.exists():
        metadata_by_code = load_product_metadata(metadata_path)
        catalog = build_category_catalog(data["categories"], metadata_by_code)
        save_category_catalog(catalog, output_root / "category_catalog.json")

    return dataset_yaml_path, output_root / "category_catalog.json"

