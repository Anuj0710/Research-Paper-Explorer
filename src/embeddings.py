"""
    Embedding Model Wrapper + On disk cache
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src import config

def prefixes(model_name: str) -> Tuple[str,str]:

    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name and "m3" not in name:
        return "Represent this sentence for searching relevent message: ", ""
    return "", ""

class Embedder:
    def __init__(self, model_name : str = config.EMBED_MODEL,
                 batch_size: int = config.EMBED_BATCH_SIZE):
        
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Embedding model {model_name} on {device}.....")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_embedding_dimension())
        self.query_prefix, self.doc_prefix = prefixes(model_name)

    def embed_documents(self, texts : List[str], show_progress: bool = True) -> np.ndarray:
        vecs = self.model.encode(
            [self.doc_prefix + t for t in texts],
            batch_size = self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return vecs.astype("float32")
    
    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(self.query_prefix + query, normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype("float32")

# ------------------------------------------------------------------ cache

def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def load_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}

    data = np.load(path, allow_pickle=True)
    return dict(zip(data["keys"].tolist(), data["vectors"]))

def save_cache(path: Path, cache : Dict[str, np.ndarray]) -> None:
    if not cache:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array(list(cache.keys())), vectors = np.stack(list(cache.values())))

def embed_chunks_cached(embedder, chunks: List[dict], cache_path: Path, text_field: str = "embed_text", block: int = 1024) -> np.ndarray:

    #return a matrix, embedding only what is not cached

    cache = load_cache(cache_path)
    keys = [_key(c[text_field]) for c in chunks]

    missing : Dict[str, str] = {}
    for k,c in zip(keys, chunks):
        if k not in cache and k not in missing:
            missing[k] = c[text_field]
    print(f"Embeddings: {len(chunks) - len(missing)} cached, {len(missing)} to compute")

    items = list(missing.items())
    for i in range(0, len(items), block):
        part = items[i:i + block]
        vecs = embedder.embed_documents([t for _,t in part])
        for (k, _), v in zip(part,vecs):
            cache[k] = v
        save_cache(cache_path, cache)

    return np.stack([cache[k] for k in keys]).astype("float32")