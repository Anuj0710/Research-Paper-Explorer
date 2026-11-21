# to check the parse quality 

import argparse
import json
import random 

from collections import Counter

from src import config

def load_all() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(config.PARSED_DIR.glob("*.json")) if not p.name.startswith("_")]

def report(docs: list[dict]) -> None:
    if not docs:
        raise SystemExit("No parsed papers found. Run scripts.parse_pdfs first.")
    flags = Counter(f for d in docs for f in d["quality_flags"])
    types = Counter(s["section_type"] for d in docs for s in d["sections"])
    words = [d["n_words"]for d in docs]
    clean = sum(1 for d in docs if not d["quality_flags"])


    print(f"Parsed papers : {len(docs)}")
    print(f"Fully clean:    {clean}({clean/ len(docs):.0%})")
    print(f"Avg words / paper: {sum(words) // len(words)}")
    print(f"Tables figures: {sum(len(d['captions'])for d in docs)} captions found")
    print("\n Quality flags:")
    for flag, n in flags.most_common():
        print(f" {flag:<18} {n}")
    print("\n Sections types (want 'other' to be small):")
    for t, n in types.most_common():
        print(f" {t:<14} {n}")

    bad = [d for d in docs if len(d["quality_flags"]) >= 2]
    if bad:
        print(f"\n worst papers (2+ flags), inspect these first:")
        for d in bad[:10]:
            print(f" {d['arxiv_id']:<14} {d['quality_flags']} {d['title'][:50]}")

def outline(d: dict, preview: bool = False) -> None:
    print(f"\n=== {d['arxiv_id']} | {d['title']}")
    print(f"    words={d['n_words']}  flags={d['quality_flags']}")
    print(f"    abstract: {d['abstract'][:150]}...")
    for s in d["sections"]:
        print(f"    [{s['section_type']:<12}] {s['heading'][:45]:<45} {s['n_words']:>5}w")
        if preview:
            print(f"        {s['text'][:200]}...")
    tables = [c for c in d["captions"] if c["kind"] == "table"]
    print(f"    tables={len(tables)} figures={len(d['captions']) - len(tables)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int)
    ap.add_argument("--id")
    a = ap.parse_args()
    docs = load_all()
    if a.id:
        match = [d for d in docs if d['arxiv_id'] == a.id]
        outline(match[0], preview=True) if match else print("Not found")

    elif a.sample:
        for d in random.sample(docs, min(a.sample, len(docs))):
            outline(d)
    else:
        report(docs)