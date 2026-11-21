"""Run with:  python -m tests.test_chunking   (or pytest)"""
from src.chunking import chunk_document, fallback_chunks, n_words, pack_units, split_units

META = {"arxiv_id": "1234.5678", "title": "Agents Can Plan", "authors": ["A. Author"],
        "year": 2024, "categories": ["cs.AI"], "abs_url": "http://arxiv.org/abs/1234.5678",
        "abstract": "We study planning in agents."}


def para(prefix: str, n_sent: int) -> str:
    return " ".join(f"{prefix} sentence {i} has exactly seven words here." for i in range(n_sent))


def section(stype, heading, text):
    return {"heading": heading, "section_type": stype, "text": text, "n_words": n_words(text)}


def make_doc():
    return {
        "arxiv_id": "1234.5678", "title": "Agents Can Plan", "abstract": "We study planning.",
        "sections": [
            section("abstract", "Abstract", "Duplicate abstract in body " * 20),
            section("introduction", "Introduction", para("Intro", 30)),
            section("method", "Approach", para("Method", 80)),             # ~560 words
            section("method", "Planning Module", "Tiny subsection of text."),
            section("results", "Results", para("Results", 15)),
            section("appendix", "Prompts", para("Prompt", 30)),
        ],
        "captions": [
            {"kind": "table", "label": "Table 1", "caption": "Success rates on WebArena.",
             "content": "Model | SR\nOurs | 14.4"},
            {"kind": "figure", "label": "Figure 1", "caption": "Too short", "content": ""},
        ],
    }


def test_split_and_pack():
    units = split_units(para("X", 100), max_words=100)       # one 700-word paragraph
    assert all(n_words(u) <= 100 for u in units)
    chunks = pack_units(units, max_words=100, overlap_words=20, min_words=30)
    assert all(n_words(c) <= 130 for c in chunks)             # max + small tail merge
    assert len(chunks) >= 6
    # overlap: end of chunk i reappears at the start of chunk i+1
    last_sentence = chunks[0].split("\n")[-1]
    assert last_sentence in chunks[1]


def test_chunk_document():
    chunks = chunk_document(make_doc(), META, max_words=200, overlap_words=30)
    kinds = [c["kind"] for c in chunks]
    types = {c["section_type"] for c in chunks}
    assert kinds[0] == "abstract" and chunks[0]["text"] == "We study planning."
    assert "appendix" not in types                       # excluded by default
    assert sum(1 for c in chunks if c["text"].startswith("Duplicate")) == 0
    assert "table" in types and "figure" not in types     # short figure caption dropped
    assert all(c["arxiv_id"] == "1234.5678" and c["year"] == 2024 for c in chunks)
    assert all(c["title"] in c["embed_text"] for c in chunks)
    # tiny 'Planning Module' got merged into the method block, not its own chunk
    assert not any(c["heading"] == "Planning Module" for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert max(c["n_words"] for c in chunks if c["kind"] == "body") <= 260


def test_include_appendix_and_fallback():
    chunks = chunk_document(make_doc(), META, max_words=200, exclude_types=set())
    assert "appendix" in {c["section_type"] for c in chunks}
    fb = fallback_chunks(META)
    assert len(fb) == 1 and fb[0]["fallback"] is True


if __name__ == "__main__":
    test_split_and_pack()
    test_chunk_document()
    test_include_appendix_and_fallback()
    print("all tests passed")
