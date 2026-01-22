# vector_store.py
import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
try:
    import faiss
except Exception:
    faiss = None

from sentence_transformers import SentenceTransformer

STORAGE_DIR = "storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "faiss.index")
META_PATH = os.path.join(STORAGE_DIR, "metadata.json")


def _ensure_storage():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def _load_metadata() -> List[Dict[str, Any]]:
    if not os.path.exists(META_PATH):
        return []
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_metadata(metadata: List[Dict[str, Any]]):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def create_or_load_store(pages: List[Dict[str, Any]], force_rebuild: bool = False, model_name: str = "all-MiniLM-L6-v2") -> Tuple[SentenceTransformer, Any, List[Dict[str, Any]]]:
    """
    Build or load a FAISS index and metadata list.
    Returns: (sentence-transformer model, faiss index, metadata_list)
    """
    _ensure_storage()

    # load model
    model = SentenceTransformer(model_name)

    # if existing and not forcing rebuild, try load
    if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        if faiss is None:
            raise RuntimeError("faiss is not installed but index file exists. Install faiss to load index.")
        try:
            index = faiss.read_index(INDEX_PATH)
            metadata = _load_metadata()
            return model, index, metadata
        except Exception as e:
            # fallback to rebuild
            print("Failed to load existing index (will rebuild):", e)

    # build embeddings
    texts = [p.get("text") or "" for p in pages]
    # transform in batches to avoid mem spikes
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embs = embs.astype("float32")
    # normalize for cosine similarity using inner product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    # prepare faiss index
    if faiss is None:
        raise RuntimeError("faiss is required. Install 'faiss-cpu' or 'faiss-gpu'.")

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    # build metadata list
    metadata = []
    for p in pages:
        metadata.append({"text": p.get("text") or "", "source": p.get("source"), "page": p.get("page")})

    # save index & metadata
    try:
        faiss.write_index(index, INDEX_PATH)
        _save_metadata(metadata)
    except Exception as e:
        print("Warning: failed to save index/metadata:", e)

    return model, index, metadata
