#---- Thin helpers around Qdrant: create collection, upsert chunks, dense search.

import uuid
import numpy as np
from typing import Dict, List, Optional

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src import config

_NAMESPACE = uuid.UUID("6f1c2e4a-0b1d-4c5e-9a77-2d3f4b5c6d7e")
PAYLOAD_EXCLUDE = {"embed_text"}
KEYWORD_FIELDS = ["arxiv_di","section_type","kind","categories"]
INTEGER_FIELDS = ["year"]

def point_id(chunk_id: str) -> str:

    #deterministic uuid,  so re-running the indexer overwrites instead of duplicating

    return str(uuid.uuid5(_NAMESPACE, chunk_id))

def get_client(url: str = config.QDRANT_URL) -> QdrantClient:
    if url == "memory":
        return QdrantClient(":memory:")
    return QdrantClient(url=url)

def ensure_collection(client: QdrantClient, name : str, dim: int, recreate: bool = False) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if client.collection_exists(name):
        existing = client.get_collection(name).config.params.vectors
        existing_dim = getattr(existing,'size',None)
        if existing_dim and existing_dim != dim:
            raise SystemExit(
                f"Collection '{name}' has dimension {existing_dim} but the model produces"
                f"{dim}. Re-run with --create"
            )
        return  
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    for f in KEYWORD_FIELDS:
        client.create_payload_index(name, f, models.PayloadSchemaType.KEYWORD)
    for f in INTEGER_FIELDS:
        client.create_payload_index(name, f, models.PayloadSchemaType.INTEGER)

def upsert_chunks(client: QdrantClient, name: str, chunks: List[dict], vectors: np.ndarray, batch_size: int = config.UPSERT_BATCH_SIZE) -> None:

    assert len(chunks) == len(vectors), "chunks and vectors must align"
    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting to Qdrant"):
        points = [
            models.PointStruct(
                id=point_id(c["chunk_id"]),
                vector= vectors[i+j].tolist(),
                payload={k:v for k, v in c.items() if k not in PAYLOAD_EXCLUDE},
            )
            for j, c in enumerate(chunks[i:i + batch_size])
        ]
        client.upsert(collection_name=name, points=points, wait=True)

def build_filter(year_from: Optional[int] = None, year_to: Optional[int] = None, section_types: Optional[List[str]] = None,
                 kinds: Optional[List[str]] = None,
                  arxiv_ids: Optional[List[str]] = None) -> Optional[models.Filter]:
    must = []
    if year_from is not None or year_to is not None:
        must.append(models.FieldCondition(
            key="year", range=models.Range(gte=year_from, lte=year_to)
        ))
    if section_types:
        must.append(models.FieldCondition(key="kind",match=models.MatchAny(any=section_types)))

    if kinds:
        must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=kinds)))
    if arxiv_ids:
        must.append(models.FieldCondition(key="arxiv_id",
                                          match=models.MatchAny(any=arxiv_ids)))
    return models.Filter(must=must) if must else None

def dense_search(client: QdrantClient, name: str, query_vec: np.ndarray, k: int =10, flt:Optional[models.Filter] = None) -> List[Dict]:
    res = client.query_points(collection_name=name, query=query_vec.tolist(), limit=k, query_filter = flt, with_payload=True)

    return [{"chunk_id": p.payload["chunk_id"], "score": float(p.score), "payload":p.payload} for p in res.points] 