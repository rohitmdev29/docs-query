"""Split long text into overlapping chunks."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Take extracted pages and split into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for page in pages:
        for j, chunk_text in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "text": chunk_text.strip(),
                "source": page["source"],
                "page_num": page["page_num"],
                "chunk_index": j,
            })

    return chunks
