"""Unified search interface for text-to-image and image-to-image queries."""

import re

from src.embeddings import embed_text, embed_image
from src.metadata_parser import COLORS, MATERIALS, CATEGORY_MAP
from src.vectorstore import query_by_vector

# Build lookup: canonical category values (e.g. "Baskets-Trays", "Kitchen")
CATEGORIES = sorted(set(CATEGORY_MAP.values()))


def _extract_attributes(query: str) -> tuple[str, str | None, str | None, str | None]:
    """Extract color, material, and category from a query string.

    Returns:
        (cleaned_query, detected_color, detected_material, detected_category)
    """
    remaining = query
    detected_color = None
    detected_material = None
    detected_category = None

    # Match longer phrases first to avoid partial matches (e.g. "Light Brown" before "Brown")
    for color in sorted(COLORS, key=len, reverse=True):
        pattern = re.compile(re.escape(color), re.IGNORECASE)
        if pattern.search(remaining):
            detected_color = color
            remaining = pattern.sub("", remaining, count=1)
            break

    for material in sorted(MATERIALS, key=len, reverse=True):
        pattern = re.compile(re.escape(material), re.IGNORECASE)
        if pattern.search(remaining):
            detected_material = material
            remaining = pattern.sub("", remaining, count=1)
            break

    for category in sorted(CATEGORIES, key=len, reverse=True):
        pattern = re.compile(re.escape(category), re.IGNORECASE)
        if pattern.search(remaining):
            detected_category = category
            remaining = pattern.sub("", remaining, count=1)
            break

    # Clean up extra whitespace
    remaining = re.sub(r"\s+", " ", remaining).strip()

    return remaining, detected_color, detected_material, detected_category


def smart_search(
    query: str,
    n_results: int = 5,
    persist_dir: str = ".chroma",
) -> tuple[list[dict], dict]:
    """Search with automatic attribute extraction from the query.

    Detects color, material, and category keywords in the query,
    applies them as metadata filters, and uses the remaining text
    for embedding-based search.

    Args:
        query: Natural language query (e.g. "Black Teak Root Lunch Plate").
        n_results: Number of results to return.
        persist_dir: ChromaDB persistence directory.

    Returns:
        Tuple of (results, extracted_info) where extracted_info contains
        the cleaned query and detected filters.
    """
    cleaned, color, material, category = _extract_attributes(query)

    # Use cleaned query for embedding (focuses on product type)
    embed_query = cleaned if cleaned else query
    vector = embed_text(embed_query)
    where = _build_filter(category, color, material)

    raw = query_by_vector(vector, n_results=n_results, where_filter=where, persist_dir=persist_dir)
    results = _format_results(raw)

    extracted_info = {
        "original_query": query,
        "embed_query": embed_query,
        "color_filter": color,
        "material_filter": material,
        "category_filter": category,
    }

    return results, extracted_info


def text_search(
    query: str,
    n_results: int = 5,
    category_filter: str | None = None,
    color_filter: str | None = None,
    material_filter: str | None = None,
    persist_dir: str = ".chroma",
) -> list[dict]:
    """Search product images by text description.

    Args:
        query: Natural language query (e.g. "black cushion").
        n_results: Number of results to return.
        category_filter: Filter by category (e.g. "Kitchen").
        color_filter: Filter by color (e.g. "Black").
        material_filter: Filter by material (e.g. "Jute").
        persist_dir: ChromaDB persistence directory.

    Returns:
        List of result dicts with keys: id, metadata, distance.
    """
    vector = embed_text(query)
    where = _build_filter(category_filter, color_filter, material_filter)

    raw = query_by_vector(vector, n_results=n_results, where_filter=where, persist_dir=persist_dir)
    return _format_results(raw)


def image_search(
    image_path: str,
    n_results: int = 5,
    category_filter: str | None = None,
    color_filter: str | None = None,
    material_filter: str | None = None,
    exclude_self: bool = True,
    persist_dir: str = ".chroma",
) -> list[dict]:
    """Search for similar product images given a query image.

    Args:
        image_path: Path to the query image.
        n_results: Number of results to return.
        category_filter: Filter by category.
        color_filter: Filter by color.
        material_filter: Filter by material.
        exclude_self: If True, fetch one extra result and drop exact path match.
        persist_dir: ChromaDB persistence directory.

    Returns:
        List of result dicts with keys: id, metadata, distance.
    """
    vector = embed_image(image_path)
    where = _build_filter(category_filter, color_filter, material_filter)

    fetch_n = n_results + 1 if exclude_self else n_results
    raw = query_by_vector(vector, n_results=fetch_n, where_filter=where, persist_dir=persist_dir)
    results = _format_results(raw)

    if exclude_self:
        results = [r for r in results if r["metadata"].get("image_path") != image_path]
        results = results[:n_results]

    return results


def _build_filter(
    category: str | None, color: str | None, material: str | None
) -> dict | None:
    """Build a ChromaDB where-filter from optional field values."""
    conditions = []
    if category:
        conditions.append({"category": category})
    if color:
        conditions.append({"color": color})
    if material:
        conditions.append({"material": material})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _format_results(raw: dict) -> list[dict]:
    """Convert raw ChromaDB query output to a flat list of result dicts."""
    results = []
    if not raw or not raw.get("ids") or not raw["ids"][0]:
        return results

    ids = raw["ids"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    for doc_id, meta, dist in zip(ids, metadatas, distances):
        results.append({
            "id": doc_id,
            "metadata": meta,
            "distance": dist,
        })

    return results
