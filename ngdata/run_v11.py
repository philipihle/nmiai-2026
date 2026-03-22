import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_image_id(stem: str) -> int:
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph = size - nh
    pw = size - nw
    top, left = ph // 2, pw // 2
    img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


def infer_single(tile, session, input_name, conf, imgsz, offset_x, offset_y, orig_w, orig_h):
    tile_lb, ratio, pad_w, pad_h = letterbox(tile, imgsz)
    blob = cv2.cvtColor(tile_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    output = session.run(None, {input_name: blob})[0]
    if output.ndim == 3 and output.shape[1] < output.shape[2]:
        output = output.transpose(0, 2, 1)
    rows = output[0]

    box_coords = rows[:, :4]
    class_scores = rows[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)

    mask = confs >= conf
    box_coords, confs, class_ids = box_coords[mask], confs[mask], class_ids[mask]
    if len(box_coords) == 0:
        return None, None, None

    cx, cy, bw, bh = box_coords[:, 0], box_coords[:, 1], box_coords[:, 2], box_coords[:, 3]
    x1 = np.clip((cx - bw / 2 - pad_w) / ratio + offset_x, 0, orig_w)
    y1 = np.clip((cy - bh / 2 - pad_h) / ratio + offset_y, 0, orig_h)
    x2 = np.clip((cx + bw / 2 - pad_w) / ratio + offset_x, 0, orig_w)
    y2 = np.clip((cy + bh / 2 - pad_h) / ratio + offset_y, 0, orig_h)

    return np.stack([x1, y1, x2, y2], axis=1), confs, class_ids


def detect(img, session, input_name, conf, iou, imgsz):
    orig_h, orig_w = img.shape[:2]

    ox, oy = int(orig_w * 0.2), int(orig_h * 0.2)
    mx, my = orig_w // 2, orig_h // 2

    tiles = [
        (0, 0, orig_w, orig_h),
        (0,       0,       mx + ox, my + oy),
        (mx - ox, 0,       orig_w,  my + oy),
        (0,       my - oy, mx + ox, orig_h),
        (mx - ox, my - oy, orig_w,  orig_h),
    ]

    boxes_list = []
    scores_list = []
    labels_list = []

    # Original image — 5 tiles
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = img[ty1:ty2, tx1:tx2]
        boxes, scores, class_ids = infer_single(tile, session, input_name, conf, imgsz,
                                                tx1, ty1, orig_w, orig_h)
        if boxes is not None:
            norm = boxes.copy()
            norm[:, 0] /= orig_w
            norm[:, 1] /= orig_h
            norm[:, 2] /= orig_w
            norm[:, 3] /= orig_h
            norm = np.clip(norm, 0.0, 1.0)
            boxes_list.append(norm)
            scores_list.append(scores)
            labels_list.append(class_ids)

    # TTA: horizontal flip — 5 tiles
    flipped = cv2.flip(img, 1)
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = flipped[ty1:ty2, tx1:tx2]
        boxes, scores, class_ids = infer_single(tile, session, input_name, conf, imgsz,
                                                tx1, ty1, orig_w, orig_h)
        if boxes is not None:
            # Flip x-coords back to original image space
            new_x1 = orig_w - boxes[:, 2]
            new_x2 = orig_w - boxes[:, 0]
            boxes[:, 0] = new_x1
            boxes[:, 2] = new_x2
            norm = boxes.copy()
            norm[:, 0] /= orig_w
            norm[:, 1] /= orig_h
            norm[:, 2] /= orig_w
            norm[:, 3] /= orig_h
            norm = np.clip(norm, 0.0, 1.0)
            boxes_list.append(norm)
            scores_list.append(scores)
            labels_list.append(class_ids)

    if not boxes_list:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Weighted Box Fusion — per-class by default
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou, skip_box_thr=conf
    )

    # Denormalize back to pixel coordinates
    fused_boxes[:, 0] *= orig_w
    fused_boxes[:, 1] *= orig_h
    fused_boxes[:, 2] *= orig_w
    fused_boxes[:, 3] *= orig_h

    return fused_boxes, fused_scores, fused_labels.astype(int)


def run(input_path: Path, output_path: Path, weights_path: Path,
        conf: float, iou: float, imgsz: int) -> None:
    images = discover_images(input_path)
    if not images:
        print("No images found.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        return

    session = ort.InferenceSession(
        str(weights_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Warmup
    _dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    session.run(None, {input_name: _dummy})

    predictions = []

    for img_path in images:
        image_id = extract_image_id(img_path.stem)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            boxes, scores, class_ids = detect(img, session, input_name, conf, iou, imgsz)
        except Exception:
            continue

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            predictions.append({
                "image_id": image_id,
                "category_id": int(class_ids[i]),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(scores[i]), 4),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"Done: {len(predictions)} predictions -> {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/data/images")
    parser.add_argument("--output", default="/output/predictions.json")
    parser.add_argument("--weights", default=str(base / "artifacts" / "best.onnx"))
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.weights),
            args.conf, args.iou, args.imgsz)
    except Exception:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("[]", encoding="utf-8")
