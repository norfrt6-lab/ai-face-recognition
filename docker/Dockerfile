# ============================================================
# AI Face Recognition & Face Swap - Dockerfile
# Supports CPU and CUDA GPU inference
# Multi-stage build with named targets: base, api, ui
# ============================================================
# Build (CPU):
#   docker build --target api -t ai-face-recognition-api:cpu \
#       --build-arg USE_GPU=false .
#
# Build (GPU / CUDA 12.1):
#   docker build --target api -t ai-face-recognition-api:gpu \
#       --build-arg USE_GPU=true .
#
# Run API (CPU):
#   docker run -p 8000:8000 ai-face-recognition-api:cpu
#
# Run UI (CPU):
#   docker run -p 8501:8501 ai-face-recognition-ui:cpu
#
# Run GPU:
#   docker run --gpus all -p 8000:8000 ai-face-recognition-api:gpu
# ============================================================

ARG USE_GPU=true

# ── Base image selection ─────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base-true
FROM python:3.10-slim-bookworm AS base-false

# Select the correct base depending on USE_GPU arg
# shellcheck disable=SC2140
FROM base-${USE_GPU:-true} AS base

# ── Build arguments ──────────────────────────────────────────
ARG USE_GPU=true
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER=appuser
ARG APP_UID=1001
ARG APP_GID=1001
ARG BUILD_DATE=unknown
ARG VERSION=1.0.0

# ── Labels ───────────────────────────────────────────────────
LABEL maintainer="AI Face Recognition Project"
LABEL description="AI Face Recognition & Face Swap — YOLO + InsightFace + inswapper"
LABEL version="${VERSION}"
LABEL build-date="${BUILD_DATE}"
LABEL org.opencontainers.image.source="https://github.com/your-org/ai-face-recognition"

# ============================================================
# System dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python build tools
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    # OpenCV system deps
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    # Video / FFmpeg
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Image format support
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # Fonts for watermark rendering
    fonts-liberation \
    fontconfig \
    # Networking & certificates
    ca-certificates \
    curl \
    wget \
    # C++ build tools (required by insightface)
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================
# Python — upgrade pip + install wheel
# ============================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================================
# Create non-root app user
# ============================================================
RUN groupadd --gid ${APP_GID} ${APP_USER} \
    && useradd --uid ${APP_UID} --gid ${APP_GID} \
               --shell /bin/bash \
               --create-home \
               ${APP_USER}

# ============================================================
# Working directory
# ============================================================
WORKDIR /app
RUN chown -R ${APP_USER}:${APP_USER} /app

# ============================================================
# Install Python dependencies
# (Copy requirements first — layer-cached until reqs change)
# ============================================================
COPY --chown=${APP_USER}:${APP_USER} requirements.txt .

# Install PyTorch CPU or GPU variant first to avoid pip resolver
# conflicts with CUDA wheels pulled in by onnxruntime-gpu.
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir \
            torch==2.2.0+cu121 \
            torchvision==0.17.0+cu121 \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir \
            torch==2.2.0+cpu \
            torchvision==0.17.0+cpu \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install onnxruntime (GPU or CPU variant)
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir "onnxruntime-gpu>=1.18.0"; \
    else \
        pip install --no-cache-dir "onnxruntime>=1.18.0"; \
    fi

# Install remaining requirements, skipping torch/torchvision/onnxruntime
# since they were installed above with the correct hardware variant.
RUN grep -v -E "^\s*(torch|torchvision|onnxruntime)" requirements.txt \
    | grep -v "^\s*#" \
    | grep -v "^\s*$" \
    | pip install --no-cache-dir -r /dev/stdin

# ============================================================
# Copy application source code
# ============================================================
COPY --chown=${APP_USER}:${APP_USER} . .

# ============================================================
# Create runtime directories with correct permissions
# ============================================================
RUN mkdir -p \
        /app/models \
        /app/uploads \
        /app/output \
        /app/cache \
        /app/logs \
        /app/tmp \
    && chown -R ${APP_USER}:${APP_USER} \
        /app/models \
        /app/uploads \
        /app/output \
        /app/cache \
        /app/logs \
        /app/tmp

# ============================================================
# Shared environment variables (inherited by all stages below)
# ============================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    MODELS_DIR=/app/models \
    STORAGE_UPLOAD_DIR=/app/uploads \
    STORAGE_OUTPUT_DIR=/app/output \
    STORAGE_CACHE_DIR=/app/cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ============================================================
# Switch to non-root user (applies to all subsequent stages)
# ============================================================
USER ${APP_USER}

# ============================================================
# ── Stage: api ───────────────────────────────────────────────
# FastAPI / Uvicorn backend — matched by docker-compose
#   build:
#     target: api
# ============================================================
FROM base AS api

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=1 \
    API_RELOAD=false

# Expose FastAPI port
EXPOSE 8000

# Health check — polls the /health endpoint
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: start Uvicorn
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

# ============================================================
# ── Stage: ui ────────────────────────────────────────────────
# Streamlit frontend — matched by docker-compose
#   build:
#     target: ui
# ============================================================
FROM base AS ui

ENV UI_HOST=0.0.0.0 \
    UI_PORT=8501 \
    UI_API_BASE_URL=http://api:8000

# Expose Streamlit port
EXPOSE 8501

# Health check — polls Streamlit's built-in health path
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: start Streamlit
CMD ["python", "-m", "streamlit", "run", "ui/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--server.fileWatcherType", "none"]
