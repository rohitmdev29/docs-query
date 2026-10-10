# Docs-Query

RAG-powered document search with hybrid retrieval and citation-linked answers.

## Stack

- Python · FastAPI
- LangChain · ChromaDB
- OpenAI (embeddings + LLM)
- Streamlit
- Docker Compose

## Features

- Upload PDFs → auto-chunking → embeddings → vector search
- Hybrid search (BM25 + semantic via RRF)
- Streaming responses via SSE
- Citation-linked answers (file + page number)
- One command startup with Docker Compose

## Quick Start

```bash
cp .env.example .env
# Add your OPENAI_API_KEY
docker compose up
API: http://localhost:8000

Swagger: http://localhost:8000/docs

Frontend: http://localhost:8501

License
MIT
