from pathlib import Path

from detector.catalog import (
    build_category_catalog,
    load_product_metadata,
    normalize_product_name,
    product_name_fingerprint,
)


def test_normalize_product_name_collapses_spacing():
    assert normalize_product_name("  Coffee   Mate 180g   Nestle ") == "COFFEE MATE 180G NESTLE"


def test_product_name_fingerprint_handles_symbols():
    assert product_name_fingerprint("KNEKKEBR\u00d8D 100 FR\u00d8&HAVSALT 245G WASA") == "KNEKKEBROD100FROOGHAVSALT245GWASA"


def test_build_category_catalog_matches_exact_metadata_name(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        """
        {
          "products": [
            {
              "product_code": "123",
              "product_name": "COFFEE MATE 180G NESTLE",
              "annotation_count": 10,
              "corrected_count": 3,
              "has_images": true,
              "image_types": ["main", "front"]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    metadata = load_product_metadata(metadata_path)
    catalog = build_category_catalog(
        categories=[{"id": 1, "name": "COFFEE MATE 180G NESTLE"}],
        metadata_by_code=metadata,
    )

    assert catalog[1].product_code == "123"
    assert catalog[1].product_name == "COFFEE MATE 180G NESTLE"


def test_build_category_catalog_falls_back_to_fingerprint_matching(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        """
        {
          "products": [
            {
              "product_code": "999",
              "product_name": "KNEKKEBR\u00d8D 100 FR\u00d8 OG HAVSALT 245G WASA",
              "annotation_count": 10,
              "corrected_count": 3,
              "has_images": true,
              "image_types": ["main", "front"]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    metadata = load_product_metadata(metadata_path)
    catalog = build_category_catalog(
        categories=[{"id": 7, "name": "KNEKKEBR\u00d8D 100 FR\u00d8&HAVSALT 245G WASA"}],
        metadata_by_code=metadata,
    )

    assert catalog[7].product_code == "999"
