""" Keyword Index
     -- Dense Search for exact terms such as model names, dataset names, and numbers
"""

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "by", "is", "are",
    "was", "were", "be", "been", "this", "that", "these", "those", "it", "its", "as", "at",
    "from", "we", "our", "can", "which", "what", "how", "do", "does", "not", "but", "also",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+|\.\d+)*")

def tokenize(text: str) -> List[str]:
    # Lowercase, drop stopwords, keep hyphenated terms whole. 

    tokens: List[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok or "_":
            tokens.extend(p for p in re.split(r"[-_]",tok) if p and not p in STOPWORDS)
    return tokens

META_FIELDS = ["chunk_id", "arxiv_id", "title", "year", "section_type", "kind", "heading"]

class BM25Index:

    def __init__(self,bm25: BM25Okapi, meta: List[dict]):
        self.bm25 = bm25
        self.meta = meta

    @classmethod
    def build(cls, chunks: List[dict], text_field: str = "embed_text") -> "BM25Index":
        corpus = [tokenize(c[text_field]) for c in chunks]
        meta = [{k: c.get(k) for k in META_FIELDS} for c in chunks]
        return cls(BM25Okapi(corpus), meta)

    def search(self,query:str, k:int = 10, year_from : Optional[int] = None,
               year_to: Optional[int] = None, section_types:Optional[List[str]] = None,
               kinds : Optional[List[str]] = None) -> List[Dict]:
        scores = self.bm25.get_scores(tokenize(query))
        out : List[Dict] = []
        for i in np.argsort(-scores):
            if scores[i] <= 0:
                break
            m = self.meta[i]
            if year_from is not None and (m["year"] or 0) < year_from:
                continue
            if year_to is not None and (m["year"] or 999) > year_to:
                continue
            if section_types and m["section_type"] not in section_types:
                continue
            if kinds and m["kind"] not in kinds:
                continue
            out.append({"chunk_id" : m["chunk_id"], "score" : float(scores[i]), "meta":m})

            if len(out) >= k:
                break
        return out

    def save(self,path:Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb")as f:
            pickle.dump({"bm25": self.bm25, "meta": self.meta}, f)

    @classmethod
    def load(cls,path:Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["bm25"], data["meta"])
