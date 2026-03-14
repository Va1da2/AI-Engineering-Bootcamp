FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app

# Enable bytecode compilation and Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# Set Python Path to include the src directory for imports
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy the rest of the application code
COPY src/ ./src/

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app && \
    mkdir -p /home/app && \
    chown -R app:app /home/app && \
    mkdir -p /home/app/.streamlit && \
    mkdir -p /home/app/.streamlit/data && \
    mkdir -p /home/app/.streamlit/cache && \
    chown -R app:app /home/app/.streamlit

# Set home directory for the user
ENV HOME=/home/app

# Sweitch to non-root user
USER app

# Expose the Streamlit port
EXPOSE 8501

# Start the application
CMD ["uv", "run", "streamlit", "run", "./src/app.py", "--server.address=0.0.0.0"]