# Multi-stage build: keep the final image lean by not shipping build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to build some wheels (e.g. for torch/torchvision transitive deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the ml package first (rarely changes, so this layer caches well)
COPY ml/setup.py /build/ml/setup.py
COPY ml/pulmoguard /build/ml/pulmoguard
RUN pip install --no-cache-dir --prefix=/install /build/ml

# Install backend dependencies
COPY backend/requirements.txt /build/backend/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /build/backend/requirements.txt


FROM python:3.11-slim AS runtime

# Run as a non-root user - never run a production web process as root.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY backend/app /app/app
COPY ml/configs /app/ml/configs

# Checkpoints are mounted at runtime (see docker-compose.yml), not baked
# into the image - model weights change independently of application code
# and can be large; baking them in would bloat every image build.
RUN mkdir -p /app/ml/outputs/checkpoints && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    ML_CONFIG_PATH=/app/ml/configs/config.yaml \
    ML_CHECKPOINT_PATH=/app/ml/outputs/checkpoints/pulmoguard_best.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
