"""
Hybrid Search Service.
Combines Semantic Search (ChromaDB) and Keyword Search (BM25) 
using Reciprocal Rank Fusion (RRF).
"""
from rank_bm25 import BM25Okapi
from app.db.chroma import collection
from app.db.chroma import search as semantic_search
from app.services.embedding import embed_query


def hybrid_search(query: str, top_k: int = 3, doc_id: str | None = None) -> list[dict]:
    """
    Runs both semantic and keyword search, then fuses the results.
    """
    # 1. Semantic Search (Finds meaning)
    query_emb = embed_query(query)
    semantic_results = semantic_search(query_emb, top_k=top_k * 2, doc_id=doc_id)
    
    sem_ids = semantic_results["ids"][0]
    
    # 2. Keyword Search (Finds exact words)
    # Fetch all chunks for the specific document to build BM25 index
    where = {"doc_id": doc_id} if doc_id else None
    all_data = collection.get(where=where, include=["documents", "metadatas"])
    
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    if not all_docs:
        return []

    # Tokenize (convert to lowercase words) for BM25
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top K*2 results from BM25
    bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
    bm25_ids = [all_ids[i] for i in bm25_ranked_indices]

    # 3. Reciprocal Rank Fusion (RRF)
    # Formula: score = 1 / (k + rank). Standard k is 60.
    rrf_k = 60
    rrf_scores = {}

    # Add points for Semantic rank
    for rank, _id in enumerate(sem_ids):
        rrf_scores[_id] = rrf_scores.get(_id, 0) + (1.0 / (rrf_k + rank + 1))

    # Add points for BM25 rank
    for rank, _id in enumerate(bm25_ids):
        rrf_scores[_id] = rrf_scores.get(_id, 0) + (1.0 / (rrf_k + rank + 1))

    # Sort by highest RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # 4. Build the final chunks list
    # Create lookup dictionaries for fast access
    id_to_doc = {all_ids[i]: all_docs[i] for i in range(len(all_ids))}
    id_to_meta = {all_ids[i]: all_metas[i] for i in range(len(all_ids))}

    final_chunks = []
    for _id in sorted_ids:
        final_chunks.append({
            "id": _id,
            "text": id_to_doc[_id],
            "metadata": id_to_meta[_id],
            "score": rrf_scores[_id]
        })

    return final_chunks