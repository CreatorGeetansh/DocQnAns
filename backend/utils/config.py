import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_indexes")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-pro")
OCR_LANGUAGES = [lang.strip() for lang in os.getenv("OCR_LANGUAGES", "en, hi").split(',') if lang.strip()]

# Ensure the directory for FAISS indexes exists
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")
if not OCR_LANGUAGES:
    raise ValueError("OCR_LANGUAGES cannot be empty.")