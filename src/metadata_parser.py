"""Parse product metadata from image filenames and folder structure."""

import os
import re
from pathlib import Path

COLORS = [
    "Black", "Natural", "Grey", "Green", "White", "Terra", "Indigo",
    "Jute", "Brown", "Light Brown", "Dark Grey", "Light Grey", "Mid Grey",
    "Beige", "Rust Red", "Red Grape", "Ocean", "Forest", "Off White",
    "Midnight Black", "Pistache",
]

MATERIALS = [
    "Seagrass", "Abaca", "Hogla", "Jute", "Mendong", "Waterhyacinth",
    "Teak Root", "Teak", "Reclaimed", "Recycled Cotton", "Recycled",
    "Waste Cot", "Waste Cotton", "Waste Wood",
]

CATEGORY_MAP = {
    "Baskets-Trays-JPG-small-2023": "Baskets-Trays",
    "Furniture JPG small 2022": "Furniture",
    "Kitchen JPG small 2022": "Kitchen",
    "Throws and Cushions JPG small 2022": "Throws-Cushions",
}


def _extract_sku(filename: str) -> str:
    """Extract SKU code like BA0013.10 or BA0013.00_1 from filename."""
    match = re.match(r"([A-Z]{2}\d{4}[\.\d]*(?:_\d+)?)", filename)
    return match.group(1) if match else ""


def _extract_product_name(filename: str) -> str:
    """Extract descriptive product name from filename, stripping SKU and extension."""
    stem = Path(filename).stem
    # Remove SKU prefix (e.g. "BA0013.10 " or "BA0013.10-")
    name = re.sub(r"^[A-Z]{2}\d{4}[\.\d]*(?:_\d+)?\s*[-\s]*", "", stem)
    # Clean up multiple spaces/hyphens
    name = re.sub(r"\s+", " ", name).strip()
    name = name.strip("-").strip()
    return name


def _extract_color(product_name: str) -> str:
    """Match known color keywords in the product name."""
    name_lower = product_name.lower()
    # Check multi-word colors first (longer matches take priority)
    sorted_colors = sorted(COLORS, key=len, reverse=True)
    for color in sorted_colors:
        if color.lower() in name_lower:
            return color
    return ""


def _extract_material(product_name: str) -> str:
    """Match known material keywords in the product name."""
    name_lower = product_name.lower()
    sorted_materials = sorted(MATERIALS, key=len, reverse=True)
    for material in sorted_materials:
        if material.lower() in name_lower:
            return material
    return ""


def _generate_id(filename: str, category: str = "") -> str:
    """Generate a unique ID from category + filename to handle duplicates across folders."""
    stem = Path(filename).stem
    sanitized = re.sub(r"[^A-Za-z0-9_\-\.]", "_", stem)
    if category:
        return f"{category}__{sanitized}"
    return sanitized


def parse_image(image_path: str, datasources_root: str) -> dict:
    """Parse a single image file path into structured metadata.

    Args:
        image_path: Full path to the image file.
        datasources_root: Path to the datasources/ directory.

    Returns:
        Dict with keys: id, sku, category, product_name, color, material, image_path
    """
    rel_path = os.path.relpath(image_path, datasources_root)
    folder_name = Path(rel_path).parts[0]
    filename = os.path.basename(image_path)

    category = CATEGORY_MAP.get(folder_name, folder_name)
    sku = _extract_sku(filename)
    product_name = _extract_product_name(filename)
    color = _extract_color(product_name)
    material = _extract_material(product_name)

    return {
        "id": _generate_id(filename, category),
        "sku": sku,
        "category": category,
        "product_name": product_name,
        "color": color,
        "material": material,
        "image_path": image_path,
    }


def parse_all_images(datasources_root: str) -> list[dict]:
    """Scan all subdirectories in datasources_root and parse every .jpg image.

    Returns:
        List of metadata dicts, one per image.
    """
    results = []
    datasources = Path(datasources_root)

    for folder in sorted(datasources.iterdir()):
        if not folder.is_dir():
            continue
        for img_file in sorted(folder.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                metadata = parse_image(str(img_file), datasources_root)
                results.append(metadata)

    return results


if __name__ == "__main__":
    # Quick test
    import json

    root = os.path.join(os.path.dirname(__file__), "..", "datasources")
    items = parse_all_images(root)
    print(f"Parsed {len(items)} images")
    for item in items[:5]:
        print(json.dumps(item, indent=2))
