"""CLI helper to (re)build the FAISS index from the documents in data/.

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_pipeline import build_vectorstore  # noqa: E402


def main():
    print("Building FAISS index from data/ ...")
    build_vectorstore(force_rebuild=True)
    print("Done. Index saved to faiss_index/")


if __name__ == "__main__":
    main()
