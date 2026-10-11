"""Quick test of PDF parser + chunker."""
from pathlib import Path
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.chunker import chunk_pages


def main():
    pdf_path = Path("../data/uploads/Rohit_Mehra_Backend_AI_Engineer.pdf")

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    print(f"Processing: {pdf_path.name}")

    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages")

    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks")

    print("\nFirst 3 chunks preview:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} (page {chunk['page_num']}) ---")
        print(chunk["text"][:200] + "...")
        print(f"[length: {len(chunk['text'])} chars]")


if __name__ == "__main__":
    main()
