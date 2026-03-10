"""Flask web app for multimodal product catalog search."""

import os
from flask import Flask, render_template, request, jsonify, send_file, abort
from werkzeug.utils import secure_filename

from src.search import text_search, image_search, smart_search, CATEGORIES
from src.metadata_parser import COLORS, MATERIALS, parse_all_images
from src.embeddings import embed_image
from src.vectorstore import get_count

app = Flask(__name__, template_folder="templates", static_folder="static")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASOURCES = os.path.join(PROJECT_ROOT, "datasources")
PERSIST_DIR = os.path.join(PROJECT_ROOT, ".chroma")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    count = get_count(PERSIST_DIR)
    return render_template(
        "index.html",
        colors=COLORS,
        materials=MATERIALS,
        categories=CATEGORIES,
        total_indexed=count,
    )


@app.route("/api/text-search", methods=["POST"])
def api_text_search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    n_results = min(int(data.get("n_results", 10)), 50)
    mode = data.get("mode", "smart")  # "smart" or "manual"

    if mode == "smart":
        results, info = smart_search(query, n_results=n_results, persist_dir=PERSIST_DIR)
        return jsonify({"results": results, "extracted": info})
    else:
        category = data.get("category") or None
        color = data.get("color") or None
        material = data.get("material") or None
        results = text_search(
            query, n_results=n_results,
            category_filter=category, color_filter=color, material_filter=material,
            persist_dir=PERSIST_DIR,
        )
        return jsonify({"results": results, "extracted": None})


@app.route("/api/image-search", methods=["POST"])
def api_image_search():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename or not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG or PNG."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    n_results = min(int(request.form.get("n_results", 10)), 50)

    try:
        results = image_search(
            filepath, n_results=n_results,
            exclude_self=False, persist_dir=PERSIST_DIR,
        )
        return jsonify({"results": results})
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/catalog-search", methods=["POST"])
def api_catalog_image_search():
    """Image search using a catalog image path."""
    data = request.get_json()
    image_path = data.get("image_path", "")

    # Validate the path is within datasources
    abs_path = os.path.normcase(os.path.abspath(image_path))
    if not abs_path.startswith(os.path.normcase(os.path.abspath(DATASOURCES))):
        return jsonify({"error": "Invalid image path"}), 400
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Image not found"}), 404

    n_results = min(int(data.get("n_results", 10)), 50)
    results = image_search(abs_path, n_results=n_results, persist_dir=PERSIST_DIR)
    return jsonify({"results": results})


@app.route("/image")
def serve_image():
    """Serve a product image by its path (must be under datasources/)."""
    path = request.args.get("path", "")
    abs_path = os.path.abspath(path)

    # Security: only serve files under datasources/ (normcase for Windows case-insensitivity)
    if not os.path.normcase(abs_path).startswith(os.path.normcase(os.path.abspath(DATASOURCES))):
        abort(403)
    if not os.path.isfile(abs_path):
        abort(404)

    return send_file(abs_path)


@app.route("/api/browse", methods=["GET"])
def api_browse():
    """Return all catalog images grouped by category for browsing."""
    metadata_list = parse_all_images(DATASOURCES)
    categories = {}
    for m in metadata_list:
        cat = m["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "id": m["id"],
            "sku": m["sku"],
            "product_name": m["product_name"],
            "color": m["color"],
            "material": m["material"],
            "image_path": m["image_path"],
        })
    return jsonify({"categories": categories})


if __name__ == "__main__":
    print(f"Indexed products: {get_count(PERSIST_DIR)}")
    print(f"Datasources: {DATASOURCES}")
    app.run(debug=True, port=5000)
