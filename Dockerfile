# syntax=docker/dockerfile:1
# Multi-stage build for the FastAPI backend. Runs as a non-root user.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install Poetry.
RUN pip install --upgrade pip && pip install "poetry==${POETRY_VERSION}"

# Install runtime deps first (better layer caching). Include the "prod" extra
# (asyncpg + qdrant-client) needed for the Dockerized Postgres/Qdrant stack.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --extras prod --no-root

# Copy application code.
COPY app ./app
COPY data ./data
COPY scripts ./scripts

# Non-root user for security.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck hits the liveness endpoint.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
