import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
PDF_DIR = DATA_DIR / "pdfs"
METADATA_DIR = DATA_DIR / "metadata.json"
METADATA_PATH = METADATA_DIR
QUESTIONS_DIR = ROOT / "eval" /"sample_questions.json"

NICHE_NAME = "LLM Agents"
ARXIV_QUERY = (
    '(abs:"LLM agent" OR abs:"language model agent" OR abs:"tool use" '
    'OR abs:"agentic") AND (cat:cs.AI OR cat:cs.CL OR cat:cs.LG)'
)

TARGET_PAPERS = 200   #how many pdf we want
MIN_YEAR = 2023       #ignore old paper
CANDIDATE_POOL = 600   #how many search result to scan to find target paper

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = "papers"

PARSED_DIR = ROOT / "data" / "parsed"
TEI_DIR = ROOT / "data" / "parsed_tei"
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")

CHUNKS_DIR = ROOT / "data" / "chunks"
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"
CHUNK_STATS_PATH = CHUNKS_DIR / "chunk_stats.json"

MAX_CHUNK_WORDS = 300
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 60
MIN_KEEP_WORDS = 30
EXCLUDE_SECTION_TYPES = {"appendix"}
INCLUDE_TABLES = True
INCLUDE_FIGURE_CAPTIONS = True
MAX_TABLE_WORDS = 400
