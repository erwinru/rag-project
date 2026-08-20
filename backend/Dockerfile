# syntax=docker/dockerfile:1
#
# Container image for the FastAPI retrieval service (rag.api.app).
# See docs/Docker.md for the reasoning behind the layout and for what has to
# be mounted at runtime.

# ---------------------------------------------------------------- builder --
# Base on python:3.13-slim and bring uv in as a binary, rather than using
# uv's own image: the virtualenv built here is copied wholesale into the
# runtime stage, and it hardcodes the interpreter it was created from. Same
# base image on both sides means that interpreter is actually there.
FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first, from the lockfile alone -- this layer (the slow one,
# torch and friends) is invalidated only by a pyproject.toml/uv.lock change,
# not by every edit to src/. `--locked` fails loudly on a stale lockfile
# rather than silently resolving something different from local dev.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-install-project --no-dev

# README.md is pyproject's `readme`, so hatchling needs it to build the wheel.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---------------------------------------------------------------- runtime --
FROM python:3.13-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
# uv installs the project in editable mode, so the venv's path entry points
# at /app/src -- it has to exist here at the same path.
COPY src ./src
COPY config.toml ./config.toml

# Bake the sentence-transformers model into the image. Without this, the
# ~87MB download happens on container *start*, not first request:
# rag.embedding.providers.huggingface builds its SentenceTransformer at
# import time, so `uvicorn` blocks on it before the port ever opens.
RUN python -c "\
from rag.config import config; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer(config.embedding.huggingface.model_id)"

RUN useradd --create-home --uid 10001 rag \
    && mkdir -p /app/data/index \
    && chown -R rag:rag /app/data /opt/huggingface
USER rag

EXPOSE 8000

# stdlib only -- this image has no curl or wget.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

# `config.api.host` (127.0.0.1) is right for `uv run rag-api` on a laptop and
# wrong in a container -- nothing outside could reach it. Binding 0.0.0.0
# here is the deliberate override; the port stays 8000 and is remapped with
# `docker run -p`, not by editing this.
CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
