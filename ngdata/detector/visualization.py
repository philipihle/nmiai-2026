from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from detector.validation import load_planogram, match_predictions_to_planogram


GREEN = (60, 180, 75)
RED = (40, 40, 220)
BLUE = (220, 140, 40)


def _color_for_prediction(prediction: dict[str, Any], use_validation: bool) -> tuple[int, int, int]:
    if not use_validation:
        return BLUE
    return GREEN if prediction.get("is_match") else RED


def annotate_image_from_predictions(
    image_path: Path,
    predictions_payload: dict[str, Any],
    output_path: Path,
    planogram_path: Path | None = None,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    predictions = [
        prediction
        for prediction in predictions_payload.get("predictions", [])
        if prediction.get("image_name") == image_path.name
    ]
    use_validation = planogram_path is not None
    if planogram_path is not None:
        slots = load_planogram(planogram_path)
        predictions = match_predictions_to_planogram(predictions, slots)

    for prediction in predictions:
        x1, y1, x2, y2 = [int(round(value)) for value in prediction["bbox_xyxy"]]
        color = _color_for_prediction(prediction, use_validation)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = prediction["detection_id"]
        if prediction.get("slot_id"):
            label = f"{label} {prediction['slot_id']}"
        if prediction.get("category_id") is not None:
            label = f"{label} c={prediction['category_id']}"
        if prediction.get("score") is not None:
            label = f"{label} {prediction['score']:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Could not write annotated image: {output_path}")
