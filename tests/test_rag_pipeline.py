from app.rag_pipeline import _parse_frontmatter, load_documents, split_documents


def test_parse_frontmatter_extracts_metadata_and_body():
    text = "---\ndoc_type: policy\ntitle: Test Policy\n---\n# Heading\nBody text."
    meta, body = _parse_frontmatter(text)
    assert meta == {"doc_type": "policy", "title": "Test Policy"}
    assert body.startswith("# Heading")


def test_parse_frontmatter_handles_missing_frontmatter():
    text = "Just plain text, no frontmatter."
    meta, body = _parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_load_documents_tags_role_specific_metadata():
    documents = load_documents()
    role_docs = [d for d in documents if d.metadata.get("category") == "role_specific"]
    assert len(role_docs) >= 4
    swe_doc = next(d for d in role_docs if "Software Engineer" in d.metadata.get("applies_to", ""))
    assert swe_doc.metadata["applies_to"] == "role:Software Engineer"


def test_load_documents_tags_policy_docs_as_applies_to_all():
    documents = load_documents()
    policy_docs = [d for d in documents if d.metadata.get("category") == "hr_policies"]
    assert policy_docs
    assert all(d.metadata.get("applies_to") == "all" for d in policy_docs)


def test_split_documents_produces_chunks_with_preserved_metadata():
    documents = load_documents()
    chunks = split_documents(documents)
    assert len(chunks) >= len(documents)
    assert all("source" in c.metadata for c in chunks)
