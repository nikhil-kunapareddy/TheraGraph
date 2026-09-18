# TheraGraph backend image.
#
# Multi-stage: the builder compiles wheels (RDKit, torch and friends are large), the
# runtime carries only the virtualenv and the application. The result runs as a
# non-root user with no build toolchain present.

# ---------------------------------------------------------------- builder stage
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

# libxrender/libxext are RDKit's drawing dependencies; gcc/g++ build the few wheels
# that have no manylinux build for this platform.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /build

# Dependencies from the committed lock file first. This layer is cached until the lock
# changes, which keeps ordinary rebuilds off the torch download -- and pins exact
# versions, so an image built today and one built in six months are identical.
#
# The lock pins torch's CPU build, which lives on PyTorch's own index rather than PyPI.
# That is deliberate: the default Linux wheel drags in ~2.5 GB of CUDA runtime this
# image would never load.
COPY backend/requirements.lock ./
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r requirements.lock

# Then the application itself, without touching dependencies again.
COPY backend/pyproject.toml backend/README.md ./
COPY backend/src ./src
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps .

# ---------------------------------------------------------------- runtime stage
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    ENVIRONMENT=production \
    LOG_FORMAT=json

# RDKit needs these shared libraries at runtime; libpq for psycopg where used.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libxrender1 \
        libxext6 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 theragraph \
    && useradd --uid 10001 --gid theragraph --create-home --shell /usr/sbin/nologin theragraph

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=theragraph:theragraph backend/alembic.ini ./
COPY --chown=theragraph:theragraph backend/alembic ./alembic
COPY --chown=theragraph:theragraph backend/src ./src
COPY --chown=theragraph:theragraph data/seed ./data/seed
COPY --chown=theragraph:theragraph data/raw/moleculenet ./data/raw/moleculenet

# Writable locations for model artifacts, dataset snapshots and rendered reports.
RUN mkdir -p /app/artifacts/models /app/data/raw \
    && chown -R theragraph:theragraph /app/artifacts /app/data

USER theragraph

EXPOSE 8000

# The probe hits the liveness endpoint, which deliberately touches no dependency: a
# database blip must not make the orchestrator kill a healthy process.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# uvicorn directly rather than gunicorn: one process per container, scaled by the
# orchestrator, which keeps the metrics and the log stream per-replica.
CMD ["uvicorn", "theragraph.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
