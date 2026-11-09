FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app && mkdir /data && chown app:app /data
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=9400
WORKDIR /data
USER app
EXPOSE 9400
VOLUME /data
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
ENTRYPOINT ["agent-auth"]
CMD ["serve", "--config", "/data/agent-auth.toml"]
