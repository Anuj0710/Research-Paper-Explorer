
# Fetches metadata and PDF reasearch papers from arXiv.org

import argparse
import json 
import re
import time

import arxiv
from tqdm import tqdm

from src import config

def clean_id(short_id: str) -> str:

    return re.sub(r"v\d+$","",short_id)

def safe_filename(arxiv_id: str) -> str:
    return arxiv_id.replace("/","_") + "pdf"

def load_metadata() -> dict:
    if config.METADATA_DIR.exists():
        return json.loads(config.METADATA_DIR.read_text())

    return {}

def save_metadata(metadata : dict) -> None:
    config.METADATA_DIR.write_text(json.dumps(metadata, indent = 2, ensure_ascii = False))

def result_to_record(r: arxiv.Result) -> dict:

    arxiv_id = clean_id(r.get_short_id())
    return {
        "arxiv_id": arxiv_id,
        "title" : r.title,
        "authors" : [a.name for a in r.authors],
        "abstract" : " ".join(r.summary.split()),
        "year" : r.published.year,
        "published" : r.published.isoformat(),
        "categories" : r.categories,
        "primary_category" : r.primary_category,
        "pdf_url" : r.pdf_url,
        "abs_url" : r.entry_id,
        "pdf_file": safe_filename(arxiv_id),
        "pdf_download" : False,
    }

def download_pdf(r: arxiv.Result, record: dict, retries : int = 3) -> bool:

    path = config.PDF_DIR / record['pdf_file']
    if path.exists() and path.stat().st_size > 10_000:
        return True

    for attempt in range(1, retries + 1):
        try: 
            r.download_pdf(dirpath=str(config.PDF_DIR), filename = record["pdf_file"])
            return True

        except Exception as e:
            wait = 5 * attempt
            tqdm.write(f" download failed for {record['arxiv_id']}"
                       f"(attempt {attempt}/{retries}) : {e}; retrying in {wait}s")

            time.sleep(wait)

    return False

def main(target: int, metadata_only : bool) -> None:

    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    print(f"Existing records : {len(metadata)}")

    client = arxiv.Client(page_size=100, delay_seconds= 3.0,num_retries= 5)
    search = arxiv.Search(
        query=config.ARXIV_QUERY,
        max_results=config.CANDIDATE_POOL,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    done = sum(1 for m in metadata.values() if m["pdf_downloaded"] or metadata_only)
    pbar = tqdm(total = target, initial=min(done, target), desc = "Papers")

    for r in client.results(search):
        if done >= target:
            break

        record = result_to_record(r)
        if record["year"] < config.MIN_YEAR:
            continue

        existing = metadata.get(record["arxiv_id"])
        if existing and (existing["pdf_download"] or metadata_only):
            continue

        if metadata_only:
            metadata[record["arxiv_id"]] = record

        else:
            record["pdf_download"] = download_pdf(r, record)
            metadata[record["arxiv_id"]] = record
            if not record["pdf_download"]:
                continue #keep the record but don't count in

        done += 1
        pbar.update(1)
        if done % 10 == 0:
            save_metadata(metadata)


    pbar.close()
    save_metadata(metadata)

    ok = sum(1 for m in metadata.values() if m["pdf_download"])
    print(f"\n Done. {len(metadata)} records, {ok} PDF's download")
    print(f"Metadata : {config.METADATA_PATH}")
    print(f"PDF's:  {config.PDF_DIR}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=config.TARGET_PAPERS)
    parser.add_argument("--metadata-only",action="store_true")
    args = parser.parse_args()
    main(args.target, args.metadata_only)

