from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Slot:
    slot_id: str
    bbox: tuple[float, float, float, float]
    expected_category_id: int | None = None
    expected_product_code: str | None = None


def load_planogram(planogram_path: Path) -> list[Slot]:
    payload = json.loads(planogram_path.read_text(encoding="utf-8"))
    slots = []
    for raw_slot in payload.get("slots", []):
        slots.append(
            Slot(
                slot_id=raw_slot["slot_id"],
                bbox=tuple(float(value) for value in raw_slot["bbox"]),
                expected_category_id=raw_slot.get("expected_category_id"),
                expected_product_code=raw_slot.get("expected_product_code"),
            )
        )
    return slots


def bbox_iou(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions_to_planogram(
    predictions: list[dict[str, Any]],
    slots: list[Slot],
    iou_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    remaining_slots = list(slots)
    sorted_predictions = sorted(predictions, key=lambda item: item.get("score", 0), reverse=True)
    enriched_predictions: list[dict[str, Any]] = []

    for prediction in sorted_predictions:
        best_slot = None
        best_iou = 0.0
        for slot in remaining_slots:
            iou = bbox_iou(prediction["bbox_xyxy"], slot.bbox)
            if iou > best_iou:
                best_iou = iou
                best_slot = slot

        enriched = dict(prediction)
        if best_slot and best_iou >= iou_threshold:
            is_match = False
            if best_slot.expected_product_code and prediction.get("product_code"):
                is_match = prediction["product_code"] == best_slot.expected_product_code
            elif best_slot.expected_category_id is not None:
                is_match = prediction["category_id"] == best_slot.expected_category_id

            enriched["slot_id"] = best_slot.slot_id
            enriched["iou_to_slot"] = best_iou
            enriched["is_match"] = is_match
            remaining_slots.remove(best_slot)
        else:
            enriched["slot_id"] = None
            enriched["iou_to_slot"] = 0.0
            enriched["is_match"] = False

        enriched_predictions.append(enriched)

    return enriched_predictions

