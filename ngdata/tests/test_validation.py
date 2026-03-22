from detector.validation import Slot, bbox_iou, match_predictions_to_planogram


def test_bbox_iou_returns_overlap_ratio():
    iou = bbox_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert round(iou, 3) == 0.143


def test_match_predictions_to_planogram_marks_correct_product():
    predictions = [
        {
            "detection_id": "D0001",
            "category_id": 42,
            "product_code": "8445291513365",
            "score": 0.9,
            "bbox_xyxy": [10, 10, 100, 100],
        }
    ]
    slots = [
        Slot(
            slot_id="S01",
            bbox=(12, 12, 98, 98),
            expected_category_id=42,
            expected_product_code="8445291513365",
        )
    ]

    matched = match_predictions_to_planogram(predictions, slots)

    assert matched[0]["slot_id"] == "S01"
    assert matched[0]["is_match"] is True
