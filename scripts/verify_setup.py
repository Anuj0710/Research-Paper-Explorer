import json 
import sys

from qdrant_client import QdrantClient

from src import config

def check_dirs() -> bool :
    for d in (config.DATA_DIR, config.PDF_DIR):
        d.mkdir(parents=True, exist_ok = True)
        print(f"[ok] directory exists: {d}")

    return True

def check_questions() -> bool:
    if not config.QUESTIONS_DIR.exists():
        print(f"[fail] missing {config.QUESTIONS_DIR}")
        return False
    questions = json.loads(config.QUESTIONS_DIR.read_text())
    print(f"[ok] {len(questions)} sample questions loaded")
    return True

def check_qdrant() -> bool:
    try:
        client = QdrantClient(url=config.QDRANT_URL, timeout=5)
        collections = client.get_collections().collections
        print(f"[ok] Qdrant reachable at {config.QDRANT_URL} "
              f"({len(collections)} collections)")
        return True
    except Exception as e:
        print(f"[fail] Qdrant not reachable at {config.QDRANT_URL}: {e}")
        print("       Did you run `docker compose up -d`?")
        return False

if __name__ == "__main__":
    results = [check_dirs(), check_questions(), check_qdrant()]
    sys.exit(0 if all(results) else 1)