FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app

# Copy workspace root files for dependecy resolution
COPY pyproject.toml uv.lock ./

# Copy package files and source
COPY apps/items_mcp_server ./apps/items_mcp_server

ENV UV_COMPILE_BYTECODE=1

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package items_mcp_server

# Enable bytecode compilation and Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/items_mcp_server/src:$PYTHONPATH"

# Create non-root user and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app

# Sweitch to non-root user
USER app

# Expose the port
EXPOSE 8000

WORKDIR /app/apps/items_mcp_server/src

# Start the application
CMD ["python", "-m", "items_mcp_server.main"]