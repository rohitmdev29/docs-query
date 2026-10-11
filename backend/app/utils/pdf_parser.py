"""Extract text from PDF files page by page."""
from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(file_path: Path) -> list[dict]:
    """Extract text from PDF. Returns pages with metadata."""
    reader = PdfReader(str(file_path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({
                "page_num": i,
                "text": text,
                "source": file_path.name,
            })

    return pages
