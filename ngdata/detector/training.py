from __future__ import annotations

import shutil
from pathlib import Path

from detector.coco import build_yolo_dataset


def train_model(
    coco_root: Path,
    product_root: Path,
    artifacts_dir: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> None:
    prepared_dir = artifacts_dir / "prepared_dataset"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml_path, category_catalog_path = build_yolo_dataset(
        coco_root=coco_root,
        product_root=product_root,
        output_root=prepared_dir,
    )

    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(artifacts_dir / "runs"),
        name="ngd-yolo",
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, artifacts_dir / "best.pt")
    if category_catalog_path.exists():
        shutil.copy2(category_catalog_path, artifacts_dir / "category_catalog.json")

