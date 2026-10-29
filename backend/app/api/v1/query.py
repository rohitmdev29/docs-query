"""Query endpoint — RAG pipeline with both Semantic and Hybrid search."""
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.schemas.query import QueryRequest, QueryResponse
from app.services.embedding import embed_query
from app.services.generation import generate_answer, stream_answer
from app.db.chroma import search as semantic_search
from app.services.retrieval import hybrid_search

router = APIRouter()

# Helper function to get chunks based on user's choice
def get_chunks(req: QueryRequest) -> list[dict]:
    
    if req.search_type == "hybrid":
        # Run Hybrid (BM25 + Semantic)
        return hybrid_search(req.question, top_k=req.top_k, doc_id=req.doc_id)
    
    else:
        # Run Semantic Only (ChromaDB)
        query_emb = embed_query(req.question)
        results = semantic_search(query_emb, top_k=req.top_k, doc_id=req.doc_id)
        
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i],
            })
        return chunks


@router.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    """Ask a question and get an answer with citations."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # Get chunks (hybrid or semantic based on request)
    chunks = get_chunks(req)

    return generate_answer(req.question, chunks)


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Ask a question and stream the answer token-by-token."""
    
    # Get chunks
    chunks = get_chunks(req)

    async def event_generator():
        for token in stream_answer(req.question, chunks):
            yield {"data": token}

    return EventSourceResponse(event_generator())