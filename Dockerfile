#Stage 1: Build
FROM ghcr.io/astral-sh/uv:0.13.6-python3.14-trixie-slim AS build

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-build --no-dev --no-install-project

#Stage 2: Run
FROM python:3.14.8-slim AS runtime

WORKDIR /app

COPY --from=build /app/.venv /app/.venv

COPY src ./src

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home --uid 10001 ocr

USER ocr

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD curl --fail http://localhost:8001/healthcheck || exit 1

EXPOSE 8001

CMD ["uvicorn", "webserver:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8001"]
