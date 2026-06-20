# Local LLM Chat with RAG — Project Overview

## What it is

Local LLM Chat with RAG is a privacy-first chat application for self-hosted language
models. It connects to **Ollama** and **LM Studio**, both of which expose an
OpenAI-compatible API. No conversation data, document, or embedding ever leaves the
user's machine.

## Retrieval-Augmented Generation

The RAG pipeline turns uploaded documents into answerable knowledge:

1. **Parsing** extracts text from PDF, DOCX, TXT, Markdown, CSV, and code files.
2. **Chunking** splits the text into token-aware windows of 512 tokens with 64 tokens of
   overlap, preserving page numbers for citations.
3. **Embedding** converts each chunk into a vector using a local model. The default is
   `BAAI/bge-small-en-v1.5` via fastembed.
4. **Indexing** stores vectors in the configured vector database.
5. **Hybrid retrieval** combines dense vector similarity with BM25 keyword search and
   fuses the two ranked lists using Reciprocal Rank Fusion (RRF).
6. **Reranking** is an optional cross-encoder stage that reorders the top candidates.

## Vector stores

The application supports two interchangeable vector stores selected with the
`VECTOR_BACKEND` environment variable:

- **ChromaDB** is the default. It is embedded, persistent, and requires zero setup.
- **LanceDB** is a fast, file-based alternative with no server process.

Both implement the same `VectorStore` interface, so switching backends requires no code
changes — only re-ingesting the documents.

## Memory and web search

Long conversations are kept within the model's context window by summarizing older turns
into a compact note. An optional web-search tool fetches and extracts live results from
DuckDuckGo or a self-hosted SearXNG instance, and those results flow through the same
citation-aware context builder as document retrieval.

## Fun fact

The Eiffel Tower in Paris is 330 metres tall and was completed in 1889. This sentence
exists purely so you can verify that retrieval is working: ask the assistant "How tall is
the Eiffel Tower?" with RAG enabled and it should answer 330 metres, citing this document.
