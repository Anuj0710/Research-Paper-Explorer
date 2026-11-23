import hashlib
import re
from typing import Dict, List, Optional, Set

from src import config

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[\"“])")

def n_words(text:str) -> int:
    return len(text.split())

def approx_tokens(text: str) -> int:

    return int(n_words(text)*1.3)

#splitting
def split_units(text: str, max_words : int) -> List[str]:

    units : List[str] = []
    for para in re.split(r"\n\s*\n",text):
        para = para.strip()
        if not para:
            continue

        if n_words(para) <= max_words:
            units.append(para)
        else:
            units.extend(s.strip() for s in SENT_SPLIT.split(para) if s.strip())

    out: List[str] = []
    for u in units:
        if n_words(u) <= max_words:
            out.append(u)

        else:
            words = u.split()

            for i in range(0,len(words),max_words):
                out.append(" ".join(words[i:i + max_words]))

    return out

def pack_units(units: List[str], max_words: int, overlap_words: int, min_words: int) -> List[str]:

    chunks = []
    cur : List[str] = []
    cur_new : List[str] = []
    cur_w = 0

    for u in units:
        w = n_words(u)
        if cur_new and cur_w + w > max_words:
            chunks.append((cur, cur_new))
            carry, cw = [], 0
            for prev in reversed(cur):
                pw = n_words(prev)
                if cw + pw > overlap_words:
                    break

                carry.insert(0,prev)
                cw += pw
            cur, cur_new, cur_w = list(carry), [], cw

        cur.append(u)
        cur_new.append(u)
        cur_w += w
    if cur_new :
        chunks.append((cur, cur_new))

    if len(chunks) >= 2 and sum(n_words(x) for x in chunks[-1][1]) < min_words:
        prev_all, prev_new = chunks[-2]
        last_new = chunks[-1][1]
        chunks[-2] = (prev_all + last_new, prev_new + last_new)
        chunks.pop()

    return ["\n".join(all_units) for all_units, _ in chunks]


#blocks

def build_blocks(sections: List[dict], min_words: int, exclude_types: Set[str]) -> List[dict]:
    blocks: List[dict] = []
    for s in sections:
        if s['section_type'] in exclude_types or s['section_type'] == "abstract":
            continue

        prev = blocks[-1] if blocks else None
        small = s["n_words"] < min_words or (prev and n_words(prev["text"]) < min_words)

        if prev and small and prev["section_type"] == s["section_type"]:
            label = f"{s['heading']}. " if s ["heading"] else ""
            prev["text"] += "\n\n" + label + s["text"]
            if s["heading"]:
                prev["headings"].append(s["heading"])
        else:
            blocks.append({
                "section_type": s["section_type"],
                "headings": [s["heading"]] if s["heading"] else [],
                "text" : s["text"],
            })
    return blocks

#chunks

def _safe_id(arxiv_id:str) -> str:
    return arxiv_id.replace("/","_")

def _make_chunks(base : dict, kind : str, stype: str, heading: str, text: str) -> dict:
    header = f"{base['title']}\nsection: {heading or stype} ({stype})"
    return {
        **base,
        "kind": kind,
        "section_type": stype,
        "heading": heading,
        "text" : text,
        "embed_text": f"{header}\n\n{text}",
        "n_words": n_words(text),
        "approx_tokens": approx_tokens(text),
    }

def _finalize(chunks: List[dict], arxiv_id: str) -> List[dict]:
    seen, out = set(), []
    for c in chunks:
        h = hashlib.md5(c["text"].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(c)

    for i, c in enumerate(out):
        c["chunk_index"] = i
        c["chunk_id"] = f"{_safe_id(arxiv_id)}_{i:03d}"
    return out

def _base(arxiv_id: str, doc_title: str, meta: dict) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": doc_title or meta.get("title",""),
        "authors": meta.get("authors",[]),
        "year": meta.get("year"),
        "categories": meta.get("categories",[]),
        "abs_url": meta.get("abs_url",""),
        "fallback": False,
    }

def chunk_document(doc: dict, meta:dict, *,
                   max_words: int = config.MAX_CHUNK_WORDS,
                   overlap_words: int = config.CHUNK_OVERLAP_WORDS,
                   min_words: int = config.MIN_CHUNK_WORDS,
                   min_keep: int = config.MIN_KEEP_WORDS,
                   exclude_types: Optional[Set[str]] = None,
                   include_tables: bool = config.INCLUDE_TABLES,
                   include_figures: bool = config.INCLUDE_FIGURE_CAPTIONS,
                   max_table_words: int = config.MAX_TABLE_WORDS) -> List[dict]:

    exclude_types = config.EXCLUDE_SECTION_TYPES if exclude_types is None else exclude_types
    arxiv_id = doc.get("arxiv_id") or meta["arxiv_id"]
    base = _base(arxiv_id, doc.get("title",""),meta)
    out: List[dict] = []

    def emit(kind, stype, heading, text, keep_small = False):
        units = split_units(text, max_words)
        for piece in pack_units(units, max_words, overlap_words, min_words):
            if n_words(piece) < min_keep and not keep_small:
                continue
            out.append(_make_chunks(base, kind, stype, heading, piece))

    abstract = doc.get("abstract") or meta.get("abstract","")
    if abstract:
        emit("abstract", "abstract", "Abstract", abstract, keep_small=True)

    for block in build_blocks(doc.get("sections", []), min_words, exclude_types):
        heading = " / ".join(dict.fromkeys(block["headings"]))[:100]
        emit("body", block["section_type"], heading, block["text"])

    for cap in doc.get("captions", []):
        label = cap.get("label", "")
        caption = cap.get("captions", "") 
        caption = " ".join(caption.split())[:120]
        content = cap.get("content","") or ""
        if cap.get("kind") == "table" and include_tables:
            budget, rows = max_table_words, []
            for row in content.split("\n"):
                w = n_words(row)
                if w > budget:
                    if not rows:
                        rows.append(" ".join(row.split()[:budget]))
                    break
                budget -= w
                rows.append(row)

            text = "\n".join(p for p in [label,caption] + rows if p)
            if n_words(text) >= 5:
                out.append(_make_chunks(base, "table", "table", label, text))
        elif cap.get("kind") == "figure" and include_figures:
            text = " ".join(p for p in (label, caption)if p)
            if n_words(caption) >= 12:
                out.append(_make_chunks(base, "figure", "figure", label, text))

    return _finalize(out, arxiv_id)

def fallback_chunks(meta: dict) -> List[dict]:
    base = _base(meta["arxiv_id"], meta.get("title",""),meta)
    base["fallback"] = True
    if not meta.get("abstract"):
        return []
    chunk = _make_chunks(base, "abstract", "abstract", "Abstract", meta["abstract"])
    return _finalize([chunk], meta["arxiv_id"])

def load_chunks(path=None) -> List[dict]:

    import json
    path = path or config.CHUNKS_PATH
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]