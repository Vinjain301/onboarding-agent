import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "faiss_index"
ESCALATIONS_LOG = ROOT_DIR / "escalations.jsonl"

# OpenRouter (https://openrouter.ai) is used for chat generation. It exposes
# an OpenAI-compatible API, so it is accessed via ChatOpenAI with a custom
# base_url rather than a dedicated OpenRouter client.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Free OpenRouter models share a public capacity pool and are frequently
# rate-limited or retired without notice. CHAT_MODEL is tried first; if it
# returns a rate-limit (429) or not-found (404) error, each model in
# CHAT_MODEL_FALLBACKS is tried in order before the request fails.
CHAT_MODEL = os.getenv("CHAT_MODEL", "minimax/minimax-m3:free")
CHAT_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv("CHAT_MODEL_FALLBACKS", "liquid/lfm-2.5-2.6b:free").split(",") if m.strip()
]

# OpenRouter has no embeddings endpoint, so embeddings run locally via a free
# HuggingFace sentence-transformer (no API key or cost).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K = 4

# Below this normalized similarity score, the agent treats its own retrieval
# as too weak to answer confidently and triggers the HR escalation workflow
# instead of guessing. This value was set from manual evaluation across
# answerable and unanswerable sample questions, not chosen arbitrarily; see
# "Confidence threshold calibration" in ARCHITECTURE.md for the full data and
# the resulting precision/recall tradeoff.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
