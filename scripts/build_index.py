# Embed all the chunks , load into Qdrant and Build the BM25 index

import argparse

from src.bm25_index import BM25Index
from src import config
from src.chunking import load_chunks
from src.embeddings import Embedder, embed_chunks_cached
from src.vector_store import ensure_collection, get_client, upsert_chunks

def slug(model_name : str) -> str:
    return model_name.replace("/","_").replace(" ","_")

def main(args) -> None:
    chunks = load_chunks()
    if args.limit:
        chunks = chunks[:args.limit]

    print(f"{len(chunks)} chunks loaded")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_qdrant:
        embedder = Embedder(args.model)
        cache_path = config.INDEX_DIR / f"embeddings_{slug(args.model)}.npz"
        vectors = embed_chunks_cached(embedder, chunks, cache_path)

        client = get_client()
        ensure_collection(client, config.COLLECTION_NAME, embedder.dim, recreate=args.recreate)
        upsert_chunks(client, config.COLLECTION_NAME, chunks, vectors)
        total = client.count(config.COLLECTION_NAME, exact=True).count
        print(f"Qdrant collection '{config.COLLECTION_NAME}' now holds {total} points"
              f"(dashbaord: {config.QDRANT_URL}/dashboard)")

    if not args.skip_bm25:
        index = BM25Index.build(chunks)
        index.save(config.BM25_PATH)
        print(f"BM25 index over {len(chunks)} chunks saved to {config.BM25_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",default=config.EMBED_MODEL)
    ap.add_argument("--limit",type=int)
    ap.add_argument("--recreate",action="store_true")
    ap.add_argument("--skip-bm25",action="store_true")
    ap.add_argument("--skip-qdrant",action="store_true")
    main(ap.parse_args())       