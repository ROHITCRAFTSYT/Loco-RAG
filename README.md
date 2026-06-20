# Local LLM Chat with RAG

A privacy-first, fully-local chat application for self-hosted LLMs (**Ollama**, **LM Studio**)
with document Q&A (**RAG**), conversation memory, web search, and file uploads. React +
TypeScript frontend, FastAPI backend. Nothing leaves your machine.

## Features

- **Streaming chat** over SSE with stop / edit-and-rerun, markdown + syntax-highlighted code.
- **Multi-conversation** sidebar (create, rename via auto-title, pin, search, delete).
- **Model switcher** that auto-discovers installed Ollama / LM Studio models (vision models flagged).
- **RAG / document Q&A** — drag-and-drop PDF, DOCX, TXT, MD, CSV, code. Token-aware chunking,
  **hybrid retrieval** (dense + BM25 fused with RRF), optional cross-encoder **reranking**, and
  **inline citations** with an expandable source panel.
- **Collections / knowledge bases** — group docs and scope a chat to a collection.
- **"Talk to this doc"** — attach a file to a single message (paperclip) for ephemeral RAG over
  just that file, without adding it to a persistent collection.
- **Agentic tool-calling** — toggle **Agent** and the model decides for itself when to call the
  document-search and web-search tools, chaining multiple rounds before answering.
- **Voice in / out** — dictate with local speech-to-text (faster-whisper) and have replies read
  aloud with the browser's built-in speech synthesis.
- **Conversation memory** — older turns are summarized to stay within the context window.
- **Web search** — per-message toggle (DuckDuckGo out of the box, or self-hosted SearXNG).
- **Pluggable vector store** — ChromaDB (default) or LanceDB, switch with one env var.
- **Local embeddings** — `fastembed` (bundled) or an Ollama embedding model. No data leaves the box.
- **Themes**, per-conversation system prompt and sampling params, tokens/sec readout.

## Architecture

```
React (Vite) ──SSE──> FastAPI ──> Ollama / LM Studio   (chat, OpenAI /v1 API)
                          ├──────> SQLite               (conversations, messages, docs)
                          ├──────> Chroma / LanceDB     (vectors)
                          └──────> DuckDuckGo / SearXNG (web search)
```

Key seams: `services/llm` (provider abstraction), `services/vectorstore` (Chroma/LanceDB factory),
`services/rag.py` (hybrid retrieval + the **one** context-builder shared by RAG, web, and memory).

## Quick start (Docker)

Prereqs: Docker, and a local LLM runtime. For Ollama:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text   # only if EMBED_PROVIDER=ollama
```

Then:

```bash
cp .env.example .env
docker compose up --build
```

- UI:  http://localhost:5173
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

Seed the default collection with the bundled sample document so RAG works immediately
(ask *"How tall is the Eiffel Tower?"* with Documents enabled):

```bash
docker compose exec backend python -m scripts.seed      # or, in local dev: python -m scripts.seed
```

Enable the private SearXNG search engine instead of DuckDuckGo:

```bash
docker compose --profile websearch up --build   # then set WEBSEARCH_PROVIDER=searxng
```

## Local dev (no Docker)

**Backend**

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxies /api to :8000)
```

## Switching the vector store

Set `VECTOR_BACKEND=chroma` or `VECTOR_BACKEND=lancedb` in `.env`, restart, and re-ingest your
documents. Both implement the same `VectorStore` interface in `backend/app/services/vectorstore/`.

## Tests

```bash
cd backend
pytest          # chunking + RRF fusion unit tests (no external services needed)
```

## Configuration

All settings live in `.env` (see `.env.example`) and map to `backend/app/config.py`.

## Roadmap / ideas (deferred)

OCR for scanned PDFs · multi-modal (image) RAG · retrieval-quality analytics dashboard ·
streaming tool-call traces in the UI · single-user PIN auth.
