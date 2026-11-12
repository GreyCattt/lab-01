FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser index.pkl ./index.pkl
ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH="/app/index.pkl" \
    PORT="8000" \
    PYTHONUNBUFFERED="1"
USER appuser
EXPOSE 8000
CMD ["findex", "serve"]
