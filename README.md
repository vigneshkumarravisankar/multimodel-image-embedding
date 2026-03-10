# Multimodal Product Catalog Search

A multimodal search system for product catalogs that supports **text-to-image**, **image-to-image**, and **smart attribute-aware** search using OpenCLIP embeddings and ChromaDB.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                                                                 │
│   ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│   │  Text Search  │  │ Image Search  │  │  Browse Catalog    │  │
│   │  (Smart +     │  │ (Upload or    │  │  (Grid by category │  │
│   │   Manual)     │  │  click card)  │  │   → click to search│  │
│   └──────┬───────┘  └───────┬───────┘  └────────────────────┘  │
│          │                  │                                   │
│          │     Flask Web App (app.py) / Jupyter Notebook        │
└──────────┼──────────────────┼───────────────────────────────────┘
           │                  │
           ▼                  ▼
┌─────────────────────────────────────────────┐
│              Search Layer (src/search.py)    │
│                                             │
│  smart_search()    text_search()            │
│  ┌──────────────────────────────┐           │
│  │ 1. Extract attributes from   │           │
│  │    query (color, material,   │           │
│  │    category)                 │           │
│  │ 2. Strip attributes → clean  │           │
│  │    text for embedding        │           │
│  │ 3. Apply as metadata filters │           │
│  └──────────────────────────────┘           │
│                                             │
│  image_search()                             │
│  ┌──────────────────────────────┐           │
│  │ 1. Embed query image         │           │
│  │ 2. Find nearest neighbors    │           │
│  │ 3. Exclude self-match        │           │
│  └──────────────────────────────┘           │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌──────────────────────────────┐
│  Embedding  │  │      Vector Store             │
│  (src/      │  │   (src/vectorstore.py)        │
│  embeddings │  │                               │
│  .py)       │  │  ChromaDB (persistent)        │
│             │  │  ┌──────────────────────────┐ │
│ OpenCLIP    │  │  │ Collection:              │ │
│ ViT-B/32    │  │  │  product_catalog         │ │
│             │  │  │                          │ │
│ ┌─────────┐ │  │  │ • 512-dim embeddings     │ │
│ │ Image → │ │  │  │ • Cosine distance (HNSW) │ │
│ │ 512-dim │ │  │  │ • Metadata:              │ │
│ │ vector  │ │  │  │   sku, category, color,  │ │
│ ├─────────┤ │  │  │   material, product_name,│ │
│ │ Text  → │ │  │  │   image_path             │ │
│ │ 512-dim │ │  │  └──────────────────────────┘ │
│ │ vector  │ │  │                               │
│ └─────────┘ │  └──────────────────────────────┘
└─────────────┘
       ▲
       │
┌──────┴──────────────────────────────────────┐
│        Metadata Parser                       │
│        (src/metadata_parser.py)              │
│                                              │
│  datasources/                                │
│  ├── Baskets-Trays-JPG-small-2023/           │
│  ├── Furniture JPG small 2022/               │
│  ├── Kitchen JPG small 2022/                 │
│  └── Throws and Cushions JPG small 2022/     │
│                                              │
│  Filename → SKU, product name, color,        │
│             material, category               │
└──────────────────────────────────────────────┘
```

## How It Works

### Indexing Pipeline

1. **Parse metadata** — `metadata_parser.py` scans `datasources/` folders, extracting SKU, product name, color, material, and category from image filenames and folder structure.
2. **Generate embeddings** — `embeddings.py` passes each product image through **OpenCLIP ViT-B/32** to produce a normalized **512-dimensional vector**. Images that look similar produce vectors pointing in similar directions.
3. **Store in ChromaDB** — `vectorstore.py` upserts embeddings + metadata into a persistent ChromaDB collection using **cosine distance** with an **HNSW index** for fast approximate nearest neighbor search.

### Search Modes

| Mode | How it works |
|---|---|
| **Smart Search** | Auto-detects color, material, and category keywords in the query. Strips them out, applies as ChromaDB metadata filters, and sends the remaining text to the embedding model. E.g. `"Black Teak Root Lunch Plate"` → filters: color=Black, material=Teak Root; embedding query: `"Lunch Plate"`. |
| **Manual Search** | User types a text query and optionally selects filters from dropdowns. The full query text goes to the embedding model. |
| **Image Search** | Upload an image or click a catalog product. The image is embedded and the nearest vectors in ChromaDB are returned. |

### Key Design Decisions

- **Shared embedding space**: OpenCLIP maps both images and text into the same 512-dim space, enabling cross-modal search (text query → image results).
- **Metadata filters + vector search**: Pure vector search can miss specific attributes (e.g. color). Smart search combines structured metadata filtering with semantic similarity.
- **Persistent vector store**: ChromaDB persists to `.chroma/` — indexing only runs once. Delete the folder to re-index.
- **Category-prefixed IDs**: IDs include the category to handle duplicate filenames across folders.

## Project Structure

```
multimodal-embedding-rag/
├── app.py                  # Flask web app
├── templates/
│   └── index.html          # Search UI (text, image, browse)
├── notebooks/
│   └── search_demo.ipynb   # Jupyter notebook demo
├── src/
│   ├── embeddings.py       # OpenCLIP ViT-B/32 model wrapper
│   ├── metadata_parser.py  # Filename → structured metadata
│   ├── search.py           # Search API (smart, text, image)
│   └── vectorstore.py      # ChromaDB collection management
├── datasources/            # Product image folders
├── .chroma/                # Persistent vector store (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web App

```bash
python app.py
```

Open http://localhost:5000. On first text/image search, the model loads (~350 MB download, ~1.5 GB RAM).

### Jupyter Notebook

```bash
jupyter notebook notebooks/search_demo.ipynb
```

Run cells sequentially. Cell 2 (Indexing) downloads the model and indexes all product images — only needs to run once.

## Requirements

- **Python** 3.10+
- **RAM**: ~2 GB available (for model + ChromaDB)
- **Disk**: ~400 MB (model weights cached in `~/.cache/huggingface/`)

## Tech Stack

| Component | Technology |
|---|---|
| Embedding model | OpenCLIP ViT-B/32 (laion2b_s34b_b79k) |
| Vector store | ChromaDB (persistent, HNSW index) |
| Similarity metric | Cosine distance |
| Web framework | Flask |
| Notebook | Jupyter |
| Image processing | Pillow |
