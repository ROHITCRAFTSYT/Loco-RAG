# syntax=docker/dockerfile:1

# ---- builder: resolve wheels from the hash-pinned lock into a venv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps only for any wheel that has to compile from source; they never
# reach the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
# Exact versions + hashes: the image is reproducible and a tampered wheel fails the build.
COPY requirements.txt ./
RUN pip install --require-hashes -r requirements.txt

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts
RUN pip install --no-deps .

# Bake the models into the image so the container works offline ("nothing
# leaves the box") and doesn't re-download on every container recreate. Both
# libraries otherwise cache under /tmp inside the container.
#   - the tiktoken BPE file is needed at import time (always baked);
#   - the fastembed model is baked unless PRELOAD_EMBED_MODEL=0 (e.g. when
#     EMBED_PROVIDER=ollama). A different FASTEMBED_MODEL at runtime is
#     downloaded on first use as before.
ARG PRELOAD_EMBED_MODEL=1
ARG FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
ENV TIKTOKEN_CACHE_DIR=/opt/models/tiktoken \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && if [ "$PRELOAD_EMBED_MODEL" = "1" ]; then \
         python -c "from fastembed import TextEmbedding; list(TextEmbedding('${FASTEMBED_MODEL}').embed(['warm-up']))"; \
       fi \
    && mkdir -p /opt/models/fastembed

# ---- runtime: no compilers, non-root ---------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    TIKTOKEN_CACHE_DIR=/opt/models/tiktoken \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    LOG_FORMAT=json \
    DATA_DIR=/data \
    DATABASE_URL=sqlite:////data/app.db \
    CHROMA_DIR=/data/chroma \
    LANCEDB_DIR=/data/lancedb

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data && chown appuser:appuser /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/models /opt/models

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser eval ./eval
COPY --chown=appuser:appuser samples ./samples

USER appuser
EXPOSE 8000

# Python-based probe: no curl needed in the runtime image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Keep ONE worker with the default Chroma backend: Chroma's embedded
# PersistentClient is thread-safe but not process-safe, so several worker
# processes writing the same /data/chroma directory risk corrupting it.
# (https://cookbook.chromadb.dev/core/system_constraints/) To scale out, run a
# Chroma server instead of the embedded store. Request logging is done by the
# app's middleware (with request ids), so uvicorn's access log is disabled.
ENV UVICORN_WORKERS=1
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS} --no-access-log"]
