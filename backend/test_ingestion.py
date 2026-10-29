"""
Test the full RAG pipeline:
  PDF → chunks → embeddings → ChromaDB → query → LLM answer
"""
import asyncio
from pathlib import Path
from app.services.ingestion import ingest_pdf
from app.services.embedding import embed_query
from app.services.generation import generate_answer
from app.db.chroma import search


async def main():
    pdf_path = Path("../data/uploads/Rohit_Mehra_Backend_AI_Engineer.pdf")

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    # ---- Step 1: Ingest ----
    print(f"Ingesting: {pdf_path.name}")
    result = await ingest_pdf(pdf_path)
    print(f"Done. doc_id: {result['doc_id']}")
    print(f"Pages: {result['pages']}, Chunks: {result['chunks']}")

    # ---- Step 2: Search ----
    query = "What is Rohit's experience with LLMs?"
    print(f"\nQuery: {query}")

    query_emb = embed_query(query)
    results = search(query_emb, top_k=3, doc_id=result["doc_id"])

    # Build chunks list for generation
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i],
        })

    # ---- Step 3: Generate Answer ----
    print(f"\nGenerating answer...")
    answer = generate_answer(query, chunks)

    print(f"\n{'='*60}")
    print("ANSWER:")
    print('='*60)
    print(answer["answer"])

    print(f"\n{'='*60}")
    print("SOURCES:")
    print('='*60)
    for src in answer["sources"]:
        print(f"[{src['index']}] {src['source']} — page {src['page']}")


if __name__ == "__main__":
    asyncio.run(main())