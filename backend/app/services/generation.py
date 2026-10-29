"""
Groq LLM generation with citations.

Model: openai/gpt-oss-120b (Groq's current production model)
Every claim in the answer links back to a source chunk.
"""
from groq import Groq
from app.config import settings

client = Groq(api_key=settings.groq_api_key)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
1. Only use information from the context below.
2. If the answer is not in the context, say "I couldn't find that in the provided documents."
3. Cite sources using ONLY single numbers in square brackets, like [1], [2], [3].
   DO NOT use ranges like [1-4] or combined citations like [1][1-4].
4. Multiple citations should be separated: [1] [2] [3] — not [1][2][3].
5. Be concise. No fluff.
"""


def build_context(chunks: list[dict]) -> str:
    """Format chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        parts.append(
            f"[{i}] (source: {meta['source']}, page {meta['page_num']})\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate answer with citations."""
    if not chunks:
        return {
            "answer": "I couldn't find that in the provided documents.",
            "sources": [],
        }

    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "index": i + 1,
            "source": c["metadata"]["source"],
            "page": c["metadata"]["page_num"],
            "text": c["text"][:200] + "...",
        }
        for i, c in enumerate(chunks)
    ]

    return {"answer": answer, "sources": sources}


def stream_answer(query: str, chunks: list[dict]):
    """Stream the answer token-by-token for SSE."""
    if not chunks:
        yield "I couldn't find that in the provided documents."
        return

    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.1,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta