"""Formatting helpers for turning retrieved chunks into LLM context and
human-readable source citations.
"""

from app.rag_pipeline import RetrievedChunk


def format_context_for_prompt(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] Source: {chunk.title} ({chunk.source})\n{chunk.content}")
    return "\n\n".join(blocks)


def format_citations(chunks: list[RetrievedChunk]) -> list[dict]:
    seen = set()
    citations = []
    for chunk in chunks:
        if chunk.source in seen:
            continue
        seen.add(chunk.source)
        citations.append({"title": chunk.title, "source": chunk.source, "confidence": chunk.score})
    return citations
