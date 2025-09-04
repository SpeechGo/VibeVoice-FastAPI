# ABOUTME: Multi-stage Dockerfile for VibeVoice-FastAPI with CUDA support
# ABOUTME: Optimized for GPU inference with model caching and minimal runtime image

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV UV_CACHE_DIR=/tmp/uv-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    wget \
    git \
    # Audio processing dependencies
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    # System utilities
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install UV package manager
RUN pip3 install uv

# Create non-root user for security
RUN groupadd --gid 1000 vibevoice \
    && useradd --uid 1000 --gid vibevoice --shell /bin/bash --create-home vibevoice

# Stage 2: Dependencies and model caching
FROM base AS dependencies

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv \
    && . /app/.venv/bin/activate \
    && uv sync --frozen --no-dev

# Pre-download and cache models (this layer will be cached)
RUN . /app/.venv/bin/activate \
    && python3 -c "
import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

# Set cache directory
cache_dir = '/app/model_cache'
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

print('Downloading VibeVoice-1.5B model...')
try:
    snapshot_download(
        repo_id='microsoft/VibeVoice-1.5B',
        cache_dir=cache_dir,
        resume_download=True
    )
    print('VibeVoice-1.5B model cached successfully')
except Exception as e:
    print(f'Warning: Could not cache VibeVoice-1.5B: {e}')

print('Downloading VibeVoice-Large model...')
try:
    snapshot_download(
        repo_id='microsoft/VibeVoice-Large',
        cache_dir=cache_dir,
        resume_download=True
    )
    print('VibeVoice-Large model cached successfully')
except Exception as e:
    print(f'Warning: Could not cache VibeVoice-Large: {e}')

print('Model caching completed')
"

# Stage 3: Runtime image
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV HF_HOME="/app/model_cache"
ENV TRANSFORMERS_CACHE="/app/model_cache"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    # Audio processing runtime libraries
    libsndfile1 \
    libasound2 \
    portaudio19-dev \
    # System utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 vibevoice \
    && useradd --uid 1000 --gid vibevoice --shell /bin/bash --create-home vibevoice

# Create app directory
WORKDIR /app

# Copy virtual environment and cached models from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv
COPY --from=dependencies /app/model_cache /app/model_cache

# Create necessary directories
RUN mkdir -p /app/voices /app/logs \
    && chown -R vibevoice:vibevoice /app

# Copy application code
COPY --chown=vibevoice:vibevoice api/ /app/api/
COPY --chown=vibevoice:vibevoice voices/ /app/voices/

# Switch to non-root user
USER vibevoice

# Health check configuration
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
