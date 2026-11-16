FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

ARG INSTALL_EXTRAS="--extra serve --extra serve-baseline"
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache . ${INSTALL_EXTRAS}

FROM python:3.11-slim AS runtime

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY artifacts ./artifacts

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "pidgin_senti.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]