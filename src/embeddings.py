"""OpenCLIP ViT-B/32 wrapper for image and text embedding."""

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


_model = None
_preprocess = None
_tokenizer = None
_device = None


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    """Load OpenCLIP ViT-H/14 model, preprocessor, and tokenizer (cached)."""
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        return _model, _preprocess, _tokenizer

    _device = _get_device()
    print(f"Loading OpenCLIP ViT-B-32 on {_device}...")

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=_device
    )
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _model.eval()

    print("Model loaded.")
    return _model, _preprocess, _tokenizer


def embed_image(image_path: str) -> np.ndarray:
    """Embed a single image, returning a normalized 512-dim vector."""
    model, preprocess, _ = load_model()

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def embed_text(text: str) -> np.ndarray:
    """Embed a text query, returning a normalized 512-dim vector."""
    model, _, tokenizer = load_model()

    tokens = tokenizer([text]).to(_device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


def embed_images_batch(image_paths: list[str], batch_size: int = 16) -> list[np.ndarray]:
    """Embed a list of images in batches. Returns list of 512-dim vectors."""
    model, preprocess, _ = load_model()
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            img = preprocess(Image.open(p).convert("RGB"))
            images.append(img)

        batch_tensor = torch.stack(images).to(_device)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        embeddings = features.cpu().numpy()
        for emb in embeddings:
            all_embeddings.append(emb.flatten())

    return all_embeddings
