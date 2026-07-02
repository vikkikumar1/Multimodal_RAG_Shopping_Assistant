import os
from pathlib import Path
from dotenv import load_dotenv
from utils.logger import setup_logger

load_dotenv()  # load environment variables from .env if present

logger = setup_logger(__name__)

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UPLOAD_DIR = DATA_DIR / "uploads" / "user_uploaded_images"

EMBEDDINGS_DIR = BASE_DIR / "embeddings" / "generated"
FAISS_INDEX_DIR = BASE_DIR / "vector_store" / "faiss_index"
QDRANT_DIR = BASE_DIR / "vector_store" / "qdrant"

REPORTS_DIR = BASE_DIR / "reports" / "generated_reports"
ASSETS_DIR = BASE_DIR / "assets"

# --- Robust directory creation ---
directories = [
    PROCESSED_DATA_DIR,
    UPLOAD_DIR,
    EMBEDDINGS_DIR,
    FAISS_INDEX_DIR,
    QDRANT_DIR,
    REPORTS_DIR,
]

for d in directories:
    if d.exists() and d.is_file():
        # Avoid deleting user files silently; log a warning instead
        logger.warning(f"Path exists as a file where a directory is expected: {d}")
    # Now create the directory (with parents) – safe to call even if it already exists
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not ensure directory {d}: {e}")

# --- Model settings ---
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', "all-MiniLM-L6-v2")
VISION_MODEL = os.getenv('VISION_MODEL', "gemini-pro-vision")
LLM_MODEL = os.getenv('LLM_MODEL', "gemini-pro")