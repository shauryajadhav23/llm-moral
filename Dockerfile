FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-caches well)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" \
    pydantic \
    pyyaml \
    litellm \
    duckdb \
    aiohttp \
    statsmodels \
    scipy \
    numpy \
    pandas

# Copy application code
COPY src/ ./src/
COPY scenarios/ ./scenarios/
COPY analysis/ ./analysis/

# Create output directories
RUN mkdir -p outputs/cache

# Default: run the full pipeline (generate manifest then dispatch)
# Override with: docker run ... pipeline:latest python src/main.py <command> [options]
ENTRYPOINT ["python", "src/main.py"]
CMD ["run"]
