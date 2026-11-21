"""Convert Gorbid XML into Clean, section-labelled json document"""

import re
import xml.etree.ElementTree as ET

TEI = "{http://www.tei-c.org/ns/1.0}"

SECTION_PATTERNS = [
    ("abstract", r"^abstract"),
    ("limitations", r"limitation|threats to validity|failure (case|mode)|error analysis"),
    ("conclusion",r"conclusion|concluding|summary|future work"),
    ("related_word",r"related work|prior work|literature|background|preliminar"),
    ("introduction",r"introduction|overview|motivation"),
    ("results",  r"result|ablation|findings|analysis|performance|comparison"),
    ("experiments", r"experiment|evaluation|setup|setting|implementation|dataset|benchmark|baseline"),
    ("discussion", r"discussion"),
    ("method",r"method|approach|mdoel|framework|architecture|system|algorithm|design|agent|training"),
    ("appendix",r"appendix|supplementary"),
]

SKIP_BACK_TYPES = {"acknowledgement","acknowledgements","funding","availability","references","conflict","competing_interests"}
NUMBER_PREFIX = re.compile(r"^(?:\d+)*\.?|[IVX]+\.|[A-Z].\)\s+")


def _clean(text: str) -> str:
    return re.sub(r"\s+"," ", text or "").strip()

def _text(el) ->str:
    return _clean("".join(el.itertext())) if el is not None else ""

def classify_section(heading: str) -> str:
    h = NUMBER_PREFIX.sub("", _clean(heading)).lower()
    if not h :
        return "other"

    for section_type, pattern in SECTION_PATTERNS:
        if re.search(pattern, h):
            return section_type

    return "other"

def _is_subsection(n_attr: str | None) -> bool:

    return bool(n_attr) and "."in n_attr.strip(".")

def _parse_divs(divs, default_type : str = "other") -> list[dict]:
    sections, last_type = [], default_type
    for div in divs:
        div_type = (div.get("type") or "").lower()
        if div_type in SKIP_BACK_TYPES:
            continue

        head = div.find(f"{TEI}head")
        heading = _text(head)
        paragraphs = [_text(p) for p in div.findall(f"{TEI}p")]
        text = "\n\n".join(p for p in paragraphs if p)

        if not text:
            continue

        stype = classify_section(heading)

        if stype == "other" and (_is_subsection(head.get("n") if head is not None else None) or not heading):
            stype = last_type

        else:
            last_type = stype if stype != "other" else last_type

        sections.append({
            "heading": NUMBER_PREFIX.sub("",heading),
            "section_type": stype,
            "text": text,
            "n_words": len(text.split())
        })

    return sections


def _parse_captions(root) -> list[dict]:

    out = []
    for fig in root.iter(f"{TEI}figure"):
        kind = "table" if fig.get("type") == "table" else "figure"
        table = fig.find(f"{TEI}table")
        rows = []
        if table is not None:
            for row in table.findall(f"{TEI}row"):
                cells = [_text(c) for c in row.findall(f"{TEI}cell")]
                if any(cells):
                    rows.append(" | ".join(cells))
            out.append({
                "kind": kind,
                "label": _text(fig.find(f"{TEI}head")),
                 "captions" : _text(fig.find(f"{TEI}figDesc")),
                "content": "\n".join(rows),
            })
    return out

def quality_flags(doc: dict) -> list[str]:
    flags = []
    types = {s["section_type"] for s in doc["sections"]}
    if doc["n_words"] < 1500:
        flags.append("short_text")

    if len(doc["sections"]) < 3:
        flags.append("few_sections")

    if "introduction" not in types:
        flags.append("no_introduction")

    if "conclusion" not in types:
        flags.append("no_conclusion")
    if not doc["title"]:
        flags.append("no_title")

    if not doc["abstract"]:
        flags.append("no_abstract")
    return flags

def tei_to_doc(tei_xml: str | bytes, arxiv_id : str = "") -> dict:
    root = ET.fromstring(tei_xml)

    title_el = root.find(f".//{TEI}titleStmt/{TEI}title")
    abstract_el = root.find(f".//{TEI}profileDesc/{TEI}abstract")
    abstract_paras = ([_text(p) for p in abstract_el.iter(f"{TEI}p")]
                     if abstract_el is not None else [])
    abstract = "\n\n".join(p for p in abstract_paras if p) or _text(abstract_el)

    body = root.find(f".//{TEI}text/{TEI}body")
    back = root.find(f".//{TEI}text/{TEI}back")

    sections = _parse_divs(body.findall(f"{TEI}div")) if body is not None else []
    if back is not None:
        for s in _parse_divs(back.findall(f"{TEI}div"),default_type="appendix"):
            if s["section_type"] == "other":
                s["section_type"] == "appendix"

            sections.append(s)

    doc = {
        "arxiv_id" : arxiv_id,
        "title" : _text(title_el),
        "abstract" : abstract,
        "sections" : sections,
        "captions": _parse_captions(root),
        "n_words" : sum(s["n_words"] for s in sections),
    }

    doc["quality_flags"] = quality_flags(doc)
    return doc