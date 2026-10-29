"""
HuggingFace embeddings — free, local, no API key.

Model: paraphrase-MiniLM-L3-v2 (384 dims, ~17MB, faster download)
Runs on CPU. Cached after first download.
"""
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # 17 MB only
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts locally."""
    if not texts:
        return []
    return _embeddings.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return _embeddings.embed_query(query)