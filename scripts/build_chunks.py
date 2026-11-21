#Turned parse papers into chunks

import argparse
import json
import statistics
from collections import Counter

from src import config
from src.chunking import chunk_document, fallback_chunks

def main(args) -> None:
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    exclude = set() if args.include_appendix else set(config.EXCLUDE_SECTION_TYPES)
    all_chunks, stats = [], Counter()

    for arxiv_id, meta in metadata.items():
        if not meta.get("pdf_downloaded"):
            stats["skipped_no_pdf"] += 1
            continue
        parsed = config.PARSED_DIR / f"{arxiv_id.replace('/', '_')}.json"
        try:
            if parsed.exists():
                doc = json.loads(parsed.read_text(encoding="utf-8"))
                chunks = chunk_document(
                    doc, meta,
                    max_words = args.max_words, overlap_words=args.overlap,
                    exclude_types=exclude, include_tables=not args.no_tables,
                )
                stats["papers_chunked"] += 1
            else:
                chunks = fallback_chunks(meta)
                stats["papers_fallback"] += 1
        except Exception as e:
            print(f" Failed {arxiv_id}: {e}")
            stats["papers_failed"] +=1
            continue
        all_chunks.extend(chunks)

    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    words = [c["n_words"] for c in all_chunks] or [0]
    summary = {
        **stats,
        "total_chunks" : len(all_chunks),
        "chunks_per_paper_avg" : round(len(all_chunks) / max(1, stats["papers_chunked"] + stats["papers_fallback"]),1),
        "words_min" : min(words), "words_median" : int(statistics.median(words)),
        "words_mean": int(statistics.mean(words)), "words_max": max(words),
        "by_kind": dict(Counter(c["kind"] for c in all_chunks)),
        "by_section_type" : dict(Counter(c["section_type"]for c in all_chunks)),
        "oversized_chunks(>1.5x max)": sum(w > args.max_words * 1.5 for w in words), 
    }
    config.CHUNK_STATS_PATH.write_text(json.dumps(summary, indent=2),encoding="utf-8")

    print(json.dumps(summary,indent=2))
    print(f"\nWrote {len(all_chunks)} chunks to {config.CHUNKS_PATH}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=config.MAX_CHUNK_WORDS)
    ap.add_argument("--overlap",type=int,default=config.CHUNK_OVERLAP_WORDS)
    ap.add_argument("--include-appendix",action="store_true")
    ap.add_argument("--no-tables",action="store_true")
    main(ap.parse_args())