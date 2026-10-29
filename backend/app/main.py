"""Docs-Query API — RAG-powered document search."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import upload, query, documents

app = FastAPI(
    title="Docs-Query API",
    description="RAG-powered document search with hybrid retrieval and citations",
    version="0.1.0",
)

# CORS — allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}
