"""Extract crops from training images using ground truth YOLO labels."""
import cv2
import json
from pathlib import Path
import numpy as np

IMAGES_DIR = Path('/home/devstar18471/ngd-object-detection/artifacts/prepared_dataset/images/train')
LABELS_DIR = Path('/home/devstar18471/ngd-object-detection/artifacts/prepared_dataset/labels/train')
OUT_DIR = Path('/home/devstar18471/crops')
PAD = 0.05  # 5% padding around each crop

OUT_DIR.mkdir(exist_ok=True)

total = 0
skipped = 0

for label_file in sorted(LABELS_DIR.glob('*.txt')):
    stem = label_file.stem
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        candidate = IMAGES_DIR / (stem + ext)
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        skipped += 1
        continue
    h, w = img.shape[:2]

    for line in label_file.read_text().strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cat_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Add padding
        pad_x = int((x2 - x1) * PAD)
        pad_y = int((y2 - y1) * PAD)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        cat_dir = OUT_DIR / str(cat_id)
        cat_dir.mkdir(exist_ok=True)
        out_path = cat_dir / f'{stem}_{total}.jpg'
        cv2.imwrite(str(out_path), crop)
        total += 1

print(f'Extracted {total} crops to {OUT_DIR}')
print(f'Skipped {skipped} images')
