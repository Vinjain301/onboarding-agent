"""RAG pipeline: load HR documents, embed and index them with FAISS, and
retrieve matches with a confidence score that the calling code uses to decide
whether it is safe to generate an answer.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# langchain-community is marked for gradual sunsetting in favor of standalone
# integration packages, but no dedicated FAISS package exists yet as of the
# pinned LangChain version in requirements.txt. This import path is current
# and correct; langchain_classic.vectorstores.FAISS is the deprecated one.
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Very small YAML-frontmatter parser (key: value pairs only)."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_meta, body = match.groups()
    meta = {}
    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_documents(data_dir: Path = config.DATA_DIR) -> list[Document]:
    documents = []
    for path in sorted(data_dir.rglob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw_text)
        meta.update(
            {
                "source": str(path.relative_to(data_dir)).replace("\\", "/"),
                "category": path.parent.name,
            }
        )
        documents.append(Document(page_content=body, metadata=meta))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.split_documents(documents)


def get_embeddings() -> HuggingFaceEmbeddings:
    # normalize_embeddings=True keeps vectors unit-length. This is required for
    # _l2_distance_to_cosine_similarity below to be mathematically valid: the
    # conversion from L2 distance to cosine similarity assumes unit vectors.
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(force_rebuild: bool = False) -> FAISS:
    """Build the FAISS index from source documents and persist it to disk."""
    if config.INDEX_DIR.exists() and not force_rebuild:
        return load_vectorstore()

    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.INDEX_DIR))
    return vectorstore


def load_vectorstore() -> FAISS:
    return FAISS.load_local(
        str(config.INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def get_or_build_vectorstore() -> FAISS:
    if config.INDEX_DIR.exists():
        return load_vectorstore()
    return build_vectorstore(force_rebuild=True)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    title: str
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    confidence: float

    @property
    def is_confident(self) -> bool:
        return self.confidence >= config.CONFIDENCE_THRESHOLD


def _l2_distance_to_cosine_similarity(distance: float) -> float:
    """Convert FAISS L2 distance between unit-normalized vectors into cosine
    similarity (0-1). For unit vectors, distance^2 == 2 * (1 - cosine_sim).

    LangChain's default relevance-score formula (a linear `1 - d/sqrt(2)`
    rescaling) is miscalibrated for this embedding model in practice — clearly
    relevant matches were scoring below the confidence threshold. This direct
    cosine conversion tracks true semantic similarity much more closely.
    """
    cosine_sim = 1.0 - (distance**2) / 2.0
    return max(0.0, min(1.0, cosine_sim))


def _doc_is_relevant_to_role(doc: Document, role: str | None) -> bool:
    applies_to = doc.metadata.get("applies_to", "all")
    if applies_to == "all":
        return True
    if role is None:
        return True
    return applies_to.lower() == f"role:{role}".lower()


def retrieve(
    vectorstore: FAISS,
    query: str,
    role: str | None = None,
    k: int = config.RETRIEVAL_K,
) -> RetrievalResult:
    """Retrieve the top-k relevant chunks with a 0-1 confidence score.

    Confidence is the top chunk's normalized relevance score. Retrieval pulls
    more candidates than needed so role-specific filtering doesn't starve the
    result set.
    """
    raw_hits = vectorstore.similarity_search_with_score(query, k=max(k * 3, 8))
    raw_results = [(doc, _l2_distance_to_cosine_similarity(distance)) for doc, distance in raw_hits]

    filtered = [(doc, score) for doc, score in raw_results if _doc_is_relevant_to_role(doc, role)]
    top_results = (filtered or raw_results)[:k]

    chunks = [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            title=doc.metadata.get("title", doc.metadata.get("source", "unknown")),
            score=round(float(score), 4),
        )
        for doc, score in top_results
    ]
    confidence = chunks[0].score if chunks else 0.0
    return RetrievalResult(chunks=chunks, confidence=confidence)
