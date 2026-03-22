import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

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


def nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.where(union > 0, union, 1e-10)
        order = order[1:][iou <= iou_threshold]
    return keep


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
    all_boxes, all_scores, all_class_ids = [], [], []

    ox, oy = int(orig_w * 0.2), int(orig_h * 0.2)
    mx, my = orig_w // 2, orig_h // 2

    tiles = [
        (0, 0, orig_w, orig_h),
        (0,       0,       mx + ox, my + oy),
        (mx - ox, 0,       orig_w,  my + oy),
        (0,       my - oy, mx + ox, orig_h),
        (mx - ox, my - oy, orig_w,  orig_h),
    ]
    for (tx1, ty1, tx2, ty2) in tiles:
        tile = img[ty1:ty2, tx1:tx2]
        boxes, scores, class_ids = infer_single(tile, session, input_name, conf, imgsz,
                                                tx1, ty1, orig_w, orig_h)
        if boxes is not None:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_class_ids.append(class_ids)

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_class_ids = np.concatenate(all_class_ids)
    keep = nms(all_boxes, all_scores, iou)
    return all_boxes[keep], all_scores[keep], all_class_ids[keep]


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
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.output), Path(args.weights),
            args.conf, args.iou, args.imgsz)
    except Exception:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("[]", encoding="utf-8")
