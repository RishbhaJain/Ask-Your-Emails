from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-5-haiku-20241022"

DEFAULT_TOP_K = 10
MAX_CONTEXT_EMAILS = 10
RAG_TOP_K = 20

SUBSET_SIZE = 15000
EVAL_SIZE = 500
BATCH_SIZE = 100

COLLECTION_NAME = "enron_emails"

# API key must be provided via environment variable or Streamlit secrets
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY not found. "
        "Please set it in your .env file or Streamlit secrets.toml"
    )
