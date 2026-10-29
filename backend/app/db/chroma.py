"""ChromaDB vector store setup."""
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

# Persistent client — data survives restarts
# anonymized_telemetry=False disables the noisy telemetry warnings
client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

# One collection for all documents (filter by doc_id)
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},  # cosine similarity
)


def add_chunks(chunks: list[dict], embeddings: list[list[float]], doc_id: str):
    """Insert chunks + embeddings into ChromaDB."""
    collection.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "page_num": c["page_num"],
                "chunk_index": c["chunk_index"],
                "doc_id": doc_id,
            }
            for c in chunks
        ],
    )


def search(query_embedding: list[float], top_k: int = 5, doc_id: str | None = None):
    """Semantic search — return top_k chunks."""
    where = {"doc_id": doc_id} if doc_id else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )
    return results


def list_documents() -> list[dict]:
    """List unique documents in the collection."""
    result = collection.get(include=["metadatas"])
    seen = {}
    for meta in result["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "source": meta["source"]}
    return list(seen.values())