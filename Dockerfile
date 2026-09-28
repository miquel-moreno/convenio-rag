# --- Stage 1: install dependencies with uv -------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first: this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# --- Stage 2: small runtime image, non-root user --------------------------
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app \
    # Owned by app: a named volume mounted here keeps this owner (else it is root's).
    && mkdir -p /home/app/.cache && chown app:app /home/app/.cache

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
# The official BOE XML and the loader: the first start fills an empty database.
COPY --chown=app:app data ./data
COPY --chown=app:app scripts ./scripts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_CACHE_DIR=/home/app/.cache/fastembed \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# The embedding model (~250 MB) is downloaded into /home/app/.cache on first use:
# docker-compose.yml keeps it in a volume so it is downloaded only once.

USER app
EXPOSE 8000

# start-period covers the first start: model download (~250 MB) and loading the agreements.
HEALTHCHECK --interval=30s --timeout=3s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Apply pending migrations, load the agreements if the database is empty, then start the
# API (exec: uvicorn gets the stop signals).
CMD ["sh", "-c", "alembic upgrade head && python -m scripts.ingest --if-empty && exec uvicorn convenio_rag.main:app --host 0.0.0.0 --port 8000"]
