"""Query endpoint — RAG pipeline."""
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from app.schemas.query import QueryRequest, QueryResponse
from app.services.embedding import embed_query
from app.services.generation import generate_answer, stream_answer
from app.db.chroma import search

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    """Ask a question and get an answer with citations."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    # 1. Embed the query
    query_emb = embed_query(req.question)

    # 2. Semantic search
    results = search(query_emb, top_k=req.top_k, doc_id=req.doc_id)

    # 3. Build chunks list
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i],
        })

    # 4. Generate answer
    return generate_answer(req.question, chunks)


@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Ask a question and stream the answer token-by-token."""
    query_emb = embed_query(req.question)
    results = search(query_emb, top_k=req.top_k, doc_id=req.doc_id)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i],
        })

    async def event_generator():
        for token in stream_answer(req.question, chunks):
            yield {"data": token}

    return EventSourceResponse(event_generator())
