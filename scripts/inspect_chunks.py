# look at chunks before indexing them

import argparse
import json
import random
from collections import Counter

from src import config

def load()-> dict:
    if not config.CHUNKS_PATH.exists():
        raise SystemExit("No chunks Found")
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def show(c: dict, full: bool = True) -> None:
    print(f"\n--- {c['chunk_id']} | {c['kind']} / {c['section_type']} | "
          f"{c['n_words']}w | {c['year']}")
    print(f"    paper:   {c['title'][:80]}")
    print(f"    heading: {c['heading']}")
    text = c["text"] if full else c["text"][:150] + "..."
    print("    " + text.replace("\n", "\n    "))

def histogram(chunks: list) -> None:
    buckets = Counter()
    for c in chunks:
        buckets[min(c["n_words"] // 50 * 50, 450)] +=1
    print("Chunk Size distribution (words)")
    for b in sorted(buckets):
        label = f"{b}-{b + 49}" if b < 450 else "450+"
        print(f"  {label:<8} {'#' * (buckets[b] * 50 // len(chunks))} {buckets[b]}")


if __name__ == "__main__" :
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample",type=int)
    ap.add_argument("--section")
    ap.add_argument("--paper")
    a = ap.parse_args()

    chunks = load()
    if a.paper:
        mine = [c for c in chunks if c["arxiv_id"] == a.paper]
        print(f"{len(mine)} chunks for {a.paper}")
        for c in mine:
            show(c,full=False)
    elif a.sample:
        pool = [c for c in chunks if not a.section or c["section_type"] == a.section]
        for c in random.sample(pool, min(a.sample,len(pool))):
            show(c)
    else:
        print(f"Total chunks: {len(chunks)} from {len({c['arxiv_id'] for c in chunks})} papers")
        print(f"Fallback (abstract-only) papers:"
              f"{len({c['arxiv_id'] for c in chunks if c['fallback']})}\n")
        histogram(chunks)
        print("\nBy section types:")
        for t,n in Counter(c["section_type"] for c in chunks).most_common():
            print(f" {t:<14} {n}")