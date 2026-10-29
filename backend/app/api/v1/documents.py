"""List documents endpoint."""
from fastapi import APIRouter
from app.schemas.query import DocumentItem
from app.db.chroma import list_documents

router = APIRouter()


@router.get("/documents", response_model=list[DocumentItem])
async def get_documents():
    """List all ingested documents."""
    return list_documents()
