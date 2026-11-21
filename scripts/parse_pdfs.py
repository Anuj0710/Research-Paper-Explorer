# Parse every download with GROBID into section labelled json

import argparse
import json 
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from src import config
from src.parsing import tei_to_doc

ENDPOINT = f"{config.GROBID_URL}/api/processFulltextDocument"

def grobid_alive() -> bool:
    try:
        return requests.get(f"{config.GROBID_URL}/api/isalive", timeout=5).status_code == 200
    except requests.RequestException:
        return False

def call_grobid(pdf_path, retries: int = 4) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with open(pdf_path, "rb") as fh:
                
                resp = requests.post(
                    ENDPOINT,
                    files={"input": (pdf_path.name, fh, "application/pdf")},
                    data={"consolidateHeader": "0", "consolidateCitations":"0"},
                    timeout=180,
                )
            if resp.status_code == 200:
                return resp.text
            last_err = f"HTTP {resp.status_code}"
            if resp.status_code != 503:
                break

        except requests.RequestException as e:
            last_err = str(e)

        time.sleep(3 * attempt)
    raise RuntimeError(last_err or "unknown error")


def process(arxiv_id: str, meta: dict) -> dict:
    pdf_path = config.PDF_DIR / meta["pdf_file"]
    tei = call_grobid(pdf_path)
    (config.TEI_DIR / f"{arxiv_id}.tei.xml").write_text(tei, encoding="utf-8")


    doc = tei_to_doc(tei, arxiv_id=arxiv_id)
    if not doc["title"]:
        doc["title"] = meta["title"]
        doc["quality_flags"] = [f for f in doc["quality_flags"] if f != "no_title"]

    if not doc["abstract"]:
        doc["abstract"] = meta["abstract"]
        doc["quality_flags"] = [f for f in doc["quality_flags"] if f != "no_abstract"]

    out = config.PARSED_DIR / f"{arxiv_id.replace('/', '_')}.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def main(limit: int | None, force: bool, workers: int) -> None:
    if not grobid_alive():
        raise SystemExit(f"Grobid is not reachable at {config.GROBID_URL}."
                         "Run docker compose up -d and wait ~30sec.")

    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TEI_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    todo = []
    for arxiv_id, meta in metadata.items():
        if not meta.get("pdf_downloaded"):
            continue

        out = config.PARSED_DIR / f"{arxiv_id.replace('/', '_')}.json"
        if out.exists() and not force:
            continue

        todo.append((arxiv_id, meta))

    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} PDFs parse with {workers} worker(s)")

    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, i ,m): i for i, m in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing"):
            arxiv_id = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failures[arxiv_id] = str(e)
                tqdm.write(f" Failed{arxiv_id}: {e}")

    (config.PARSED_DIR / "_failures.json").write_text(json.dumps(failures, indent=2),encoding="utf-8")
    print(f"\nDone. {len(todo) - len(failures)} parsed, {len(failures)} failed. ")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force",action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    main(a.limit,a.force,a.workers)