"""Upload PDF and trigger ingestion."""
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.services.ingestion import ingest_pdf
from app.schemas.query import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and ingest it into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / file.filename

    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = await ingest_pdf(upload_path)
        return result
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")
