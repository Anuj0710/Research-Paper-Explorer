import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" 
PDF_DIR = DATA_DIR / "pdfs"
METADATA_DIR = DATA_DIR / "metadata.json"
QUESTIONS_DIR = ROOT / "eval" /"sample_questions.json"

NICHE_NAME = "LLM Agents"
ARXIV_QUERY = (
    '(abs:"LLM agent" OR abs:"language model agent" OR abs:"tool use" '
    'OR abs:"agentic") AND (cat:cs.AI OR cat:cs.CL OR cat:cs.LG)'
)

TARGET_PAPERS = 200   #how many pdf we want
MIN_YEAR = 2020       #ignore old paper
CANDIDATE_POOL = 600   #how many search result to scan to find target paper

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = "papers"