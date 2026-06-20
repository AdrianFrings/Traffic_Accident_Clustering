# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build the React/Vite frontend into a static bundle.
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /frontend

# Install deps first (cached unless lockfile changes).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build (outputs to /frontend/build per vite.config.ts).
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — slim Python runtime serving the API + the static frontend.
# Only fastapi/uvicorn/duckdb here; the heavy geo/ML stack stays in the
# offline processor (src/), never in the runtime image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    FRONTEND_DIR=/app/frontend/build \
    DATA_DIR=/app/data/processed

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /frontend/build ./frontend/build

# Data (clusters.parquet + points/ partitions) is provided at runtime via a
# mounted volume at /app/data — see docker-compose.yml. Not baked into the image.

EXPOSE 8000

# Healthcheck uses Python (already present) so the image needs no extra packages.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health').getcode()==200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
