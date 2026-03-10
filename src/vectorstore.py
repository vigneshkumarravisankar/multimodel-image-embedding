"""ChromaDB vector store management for product catalog embeddings."""

import chromadb
import numpy as np


_client = None
_collection = None

COLLECTION_NAME = "product_catalog"


def get_collection(persist_dir: str = ".chroma") -> chromadb.Collection:
    """Get or create the persistent ChromaDB collection."""
    global _client, _collection

    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=persist_dir)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def index_all(
    embeddings: list[np.ndarray],
    metadata_list: list[dict],
    persist_dir: str = ".chroma",
    batch_size: int = 100,
) -> int:
    """Index all image embeddings + metadata into ChromaDB.

    Args:
        embeddings: List of 1024-dim numpy vectors.
        metadata_list: List of metadata dicts (must include 'id' key).
        persist_dir: ChromaDB persistence directory.
        batch_size: Number of records per upsert batch.

    Returns:
        Total number of records in the collection after indexing.
    """
    collection = get_collection(persist_dir)

    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i : i + batch_size]
        batch_meta = metadata_list[i : i + batch_size]

        ids = [m["id"] for m in batch_meta]
        # ChromaDB metadata must be flat str/int/float/bool
        store_meta = []
        for m in batch_meta:
            store_meta.append({
                "sku": m.get("sku", ""),
                "category": m.get("category", ""),
                "product_name": m.get("product_name", ""),
                "color": m.get("color", ""),
                "material": m.get("material", ""),
                "image_path": m.get("image_path", ""),
            })

        collection.upsert(
            ids=ids,
            embeddings=[emb.tolist() for emb in batch_emb],
            metadatas=store_meta,
        )

    return collection.count()


def query_by_vector(
    vector: np.ndarray,
    n_results: int = 5,
    where_filter: dict | None = None,
    persist_dir: str = ".chroma",
) -> dict:
    """Query ChromaDB with a vector, returning top-N results.

    Args:
        vector: 1024-dim query vector.
        n_results: Number of results to return.
        where_filter: Optional ChromaDB where filter, e.g. {"category": "Kitchen"}.
        persist_dir: ChromaDB persistence directory.

    Returns:
        ChromaDB query result dict with keys: ids, distances, metadatas.
    """
    collection = get_collection(persist_dir)

    kwargs = {
        "query_embeddings": [vector.tolist()],
        "n_results": n_results,
    }
    if where_filter:
        kwargs["where"] = where_filter

    return collection.query(**kwargs)


def get_count(persist_dir: str = ".chroma") -> int:
    """Return the number of records in the collection."""
    return get_collection(persist_dir).count()


def reset_collection(persist_dir: str = ".chroma"):
    """Delete and recreate the collection."""
    global _collection
    client = chromadb.PersistentClient(path=persist_dir)
    client.delete_collection(COLLECTION_NAME)
    _collection = None
    return get_collection(persist_dir)
