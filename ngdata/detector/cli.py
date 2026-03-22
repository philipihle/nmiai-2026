from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from detector.catalog import load_category_catalog
from detector.inference import run_inference
from detector.training import train_model
from detector.visualization import annotate_image_from_predictions


def _default_input_path() -> str:
    return os.environ.get("INPUT_PATH", "/input")


def _default_output_path() -> str:
    return os.environ.get("OUTPUT_PATH", "/output/predictions.json")


def _default_weights_path() -> str:
    return os.environ.get("MODEL_WEIGHTS", "./artifacts/best.pt")


def _default_catalog_path() -> str:
    return os.environ.get("CATEGORY_CATALOG", "./artifacts/category_catalog.json")


def train_cli() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO model for NGD object detection.")
    parser.add_argument("--coco-root", required=True, help="Path to extracted NM_NGD_coco_dataset")
    parser.add_argument("--product-root", required=True, help="Path to extracted NM_NGD_product_images")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Directory for prepared data and weights")
    parser.add_argument("--model", default="yolo11m.pt", help="Ultralytics model checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="Training device, for example 0 or cpu")
    args = parser.parse_args()

    train_model(
        coco_root=Path(args.coco_root),
        product_root=Path(args.product_root),
        artifacts_dir=Path(args.artifacts_dir),
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    return 0


def run_inference_cli() -> int:
    parser = argparse.ArgumentParser(description="Run NGD shelf inference.")
    parser.add_argument("--input", default=_default_input_path(), help="Input image or directory")
    parser.add_argument("--output", default=_default_output_path(), help="Output file or directory")
    parser.add_argument("--weights", default=_default_weights_path(), help="Path to trained YOLO weights")
    parser.add_argument("--catalog", default=_default_catalog_path(), help="Path to category catalog json")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    args = parser.parse_args()

    catalog = load_category_catalog(Path(args.catalog)) if Path(args.catalog).exists() else {}
    predictions = run_inference(
        input_path=Path(args.input),
        output_path=Path(args.output),
        weights_path=Path(args.weights),
        category_catalog=catalog,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    if not predictions:
        print("No images were found for inference.")
    return 0


def visualize_cli() -> int:
    parser = argparse.ArgumentParser(description="Draw prediction boxes on an image.")
    parser.add_argument("--image", required=True, help="Shelf image path")
    parser.add_argument("--predictions", required=True, help="Prediction JSON path")
    parser.add_argument("--output", required=True, help="Output annotated image path")
    parser.add_argument("--planogram", help="Optional planogram json path")
    args = parser.parse_args()

    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    annotate_image_from_predictions(
        image_path=Path(args.image),
        predictions_payload=payload,
        output_path=Path(args.output),
        planogram_path=Path(args.planogram) if args.planogram else None,
    )
    return 0

