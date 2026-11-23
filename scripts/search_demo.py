# ------       sanity-check both indexes by searching them side by side.
import argparse
import json

from src import config
from src.bm25_index import BM25Index
from src.chunking import load_chunks
from src.embeddings import Embedder
from src.vector_store import build_filter, dense_search, get_client


def line(rank, score, arxiv_id, section, title):
    return f"   {rank}. {score:7.3f} | {arxiv_id:<11} | {section:<12} | {title[:55]}"


def run_query(q, args, embedder, client, bm25, by_id):
    print(f"\n### {q}")
    sections = [args.section] if args.section else None
    if args.mode in ("dense", "both"):
        flt = build_filter(year_from=args.year_from, section_types=sections)
        hits = dense_search(client, config.COLLECTION_NAME, embedder.embed_query(q), args.k, flt)
        print(" DENSE (meaning):")
        for r, h in enumerate(hits, 1):
            p = h["payload"]
            print(line(r, h["score"], p["arxiv_id"], p["section_type"], p["title"]))
            if args.full:
                print("      " + p["text"][:300].replace("\n", " ") + "...")
    if args.mode in ("bm25", "both"):
        hits = bm25.search(q, k=args.k, year_from=args.year_from, section_types=sections)
        print(" BM25 (keywords):")
        for r, h in enumerate(hits, 1):
            m = h["meta"]
            print(line(r, h["score"], m["arxiv_id"], m["section_type"], m["title"]))
            if args.full:
                print("      " + by_id[h["chunk_id"]]["text"][:300].replace("\n", " ") + "...")


def main(args) -> None:
    embedder = client = bm25 = None
    if args.mode in ("dense", "both"):
        embedder, client = Embedder(args.model), get_client()
    if args.mode in ("bm25", "both"):
        bm25 = BM25Index.load(config.BM25_PATH)
    by_id = {c["chunk_id"]: c for c in load_chunks()} if args.full else {}

    if args.query:
        queries = [args.query]
    else:
        qs = json.loads(config.QUESTIONS_DIR.read_text(encoding="utf-8"))
        queries = [q["question"] for q in qs[:args.n]]
    for q in queries:
        run_query(q, args, embedder, client, bm25, by_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query")
    ap.add_argument("--n", type=int, default=5, help="how many sample questions to run")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", choices=["dense", "bm25", "both"], default="both")
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--section")
    ap.add_argument("--full", action="store_true", help="show a text preview for each hit")
    ap.add_argument("--model", default=config.EMBED_MODEL)
    main(ap.parse_args())
