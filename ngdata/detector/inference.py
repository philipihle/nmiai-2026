from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from detector.catalog import CategoryRecord


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def _as_xyxy(boxes: Any, index: int) -> list[float]:
    values = boxes.xyxy[index].tolist()
    return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]


def _as_xywh(boxes: Any, index: int) -> list[float]:
    x1, y1, x2, y2 = _as_xyxy(boxes, index)
    return [x1, y1, x2 - x1, y2 - y1]


def _build_prediction_record(
    image_path: Path,
    boxes: Any,
    index: int,
    detection_id: int,
    category_catalog: dict[int, CategoryRecord],
) -> dict[str, Any]:
    category_id = int(boxes.cls[index].item())
    score = float(boxes.conf[index].item())
    category = category_catalog.get(category_id)
    return {
        "detection_id": f"D{detection_id:04d}",
        "image_name": image_path.name,
        "category_id": category_id,
        "category_name": category.category_name if category else None,
        "product_code": category.product_code if category else None,
        "score": score,
        "bbox_xyxy": _as_xyxy(boxes, index),
        "bbox_xywh": _as_xywh(boxes, index),
    }


def _load_model(weights_path: Path):
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _write_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    if output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"predictions": predictions}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    output_path.mkdir(parents=True, exist_ok=True)
    payload_path = output_path / "predictions.json"
    payload_path.write_text(
        json.dumps({"predictions": predictions}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_inference(
    input_path: Path,
    output_path: Path,
    weights_path: Path,
    category_catalog: dict[int, CategoryRecord],
    conf: float,
    iou: float,
    imgsz: int,
) -> list[dict[str, Any]]:
    images = discover_images(input_path)
    if not images:
        return []

    model = _load_model(weights_path)
    predictions: list[dict[str, Any]] = []
    detection_id = 1
    results = model.predict(
        source=[str(image) for image in images],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    for image_path, result in zip(images, results):
        boxes = result.boxes
        if boxes is None:
            continue
        for index in range(len(boxes)):
            predictions.append(
                _build_prediction_record(
                    image_path=image_path,
                    boxes=boxes,
                    index=index,
                    detection_id=detection_id,
                    category_catalog=category_catalog,
                )
            )
            detection_id += 1

    _write_predictions(predictions, output_path)
    return predictions

