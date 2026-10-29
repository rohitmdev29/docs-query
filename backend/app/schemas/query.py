"""Pydantic schemas for API endpoints."""
from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Response after uploading and ingesting a PDF."""
    doc_id: str
    source: str
    pages: int
    chunks: int


class QueryRequest(BaseModel):
    """Request body for /query."""
    question: str
    doc_id: str | None = None
    top_k: int = 5


class SourceItem(BaseModel):
    """A single source citation in the answer."""
    index: int
    source: str
    page: int
    text: str


class QueryResponse(BaseModel):
    """Response body for /query."""
    answer: str
    sources: list[SourceItem]


class DocumentItem(BaseModel):
    """A single document in the list."""
    doc_id: str
    source: str
