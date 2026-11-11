# API container for hosts such as Render. Build from the repository root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install dependencies first so this layer is cached between code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Download the embedding model at build time, so the first request does not wait for it.
RUN uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# Hosts set PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn leaselens.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
