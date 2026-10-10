"""Docs-Query API — RAG-powered document search."""
from fastapi import FastAPI

app = FastAPI(
    title="Docs-Query API",
    description="RAG-powered document search with hybrid retrieval",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Liveness endpoint."""
    return {"status": "ok"}
