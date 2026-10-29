"""
Ingestion pipeline: PDF → text → chunks → embeddings → ChromaDB.
"""
import uuid
from pathlib import Path
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.chunker import chunk_pages
from app.services.embedding import embed_texts
from app.db.chroma import add_chunks


async def ingest_pdf(file_path: Path) -> dict:
    """Full pipeline: file → vector DB."""
    doc_id = str(uuid.uuid4())

    # Step 1: Extract text from PDF
    pages = extract_text_from_pdf(file_path)
    if not pages:
        raise ValueError("No text found in PDF")

    # Step 2: Chunk into ~800-char overlapping pieces
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No chunks produced")

    # Step 3: Embed in batches (OpenAI max 2048 per call)
    all_embeddings = []
    for i in range(0, len(chunks), 100):
        batch = [c["text"] for c in chunks[i:i + 100]]
        all_embeddings.extend(embed_texts(batch))

    # Step 4: Store in ChromaDB
    add_chunks(chunks, all_embeddings, doc_id)

    return {
        "doc_id": doc_id,
        "source": file_path.name,
        "pages": len(pages),
        "chunks": len(chunks),
    }