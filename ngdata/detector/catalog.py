from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductRecord:
    product_code: str
    product_name: str
    annotation_count: int = 0
    corrected_count: int = 0
    has_images: bool = False
    image_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryRecord:
    category_id: int
    category_name: str
    product_code: str | None
    product_name: str | None


ASCII_FRIENDLY_TRANSLATION = str.maketrans({
    "\u00c6": "AE",
    "\u00d8": "O",
    "\u00c5": "A",
})


def normalize_product_name(name: str) -> str:
    normalized = name.strip().upper().replace("&", " OG ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.replace("  ", " ")


def product_name_fingerprint(name: str) -> str:
    normalized = normalize_product_name(name).translate(ASCII_FRIENDLY_TRANSLATION)
    return re.sub(r"[^A-Z0-9]+", "", normalized)


def load_product_metadata(metadata_path: Path) -> dict[str, ProductRecord]:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    products = raw.get("products", [])
    result: dict[str, ProductRecord] = {}
    for entry in products:
        record = ProductRecord(
            product_code=str(entry["product_code"]),
            product_name=entry["product_name"],
            annotation_count=int(entry.get("annotation_count", 0)),
            corrected_count=int(entry.get("corrected_count", 0)),
            has_images=bool(entry.get("has_images", False)),
            image_types=tuple(entry.get("image_types", [])),
        )
        result[record.product_code] = record
    return result


def build_category_catalog(
    categories: list[dict[str, Any]],
    metadata_by_code: dict[str, ProductRecord],
) -> dict[int, CategoryRecord]:
    metadata_by_name = {
        normalize_product_name(record.product_name): record
        for record in metadata_by_code.values()
    }
    metadata_by_fingerprint = {
        product_name_fingerprint(record.product_name): record
        for record in metadata_by_code.values()
    }

    catalog: dict[int, CategoryRecord] = {}
    for category in categories:
        category_name = category["name"]
        match = metadata_by_name.get(normalize_product_name(category_name))
        if match is None:
            match = metadata_by_fingerprint.get(product_name_fingerprint(category_name))
        catalog[int(category["id"])] = CategoryRecord(
            category_id=int(category["id"]),
            category_name=category_name,
            product_code=match.product_code if match else None,
            product_name=match.product_name if match else None,
        )
    return catalog


def save_category_catalog(catalog: dict[int, CategoryRecord], output_path: Path) -> None:
    payload = {
        str(category_id): asdict(record)
        for category_id, record in sorted(catalog.items(), key=lambda item: item[0])
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_category_catalog(catalog_path: Path) -> dict[int, CategoryRecord]:
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {
        int(category_id): CategoryRecord(**record)
        for category_id, record in raw.items()
    }
