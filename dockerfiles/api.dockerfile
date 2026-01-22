FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

EXPOSE 8080

ENTRYPOINT ["uv", "run", "uvicorn", "src.ml_ops_project.api:app", "--host", "0.0.0.0", "--port", "8080"]
