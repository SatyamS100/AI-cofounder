# Dockerfile
# =========
# Multi-stage build NOT used here because the sentence-transformers model
# download happens at runtime (first Streamlit run), not at build time.
# Downloading at build time would balloon image size by ~400MB for a model
# that only needs to be pulled once per deployment anyway.
#
# VOLUMES EXPECTED:
#   /app/chroma_db   ← ChromaDB persistence (seed docs + pitch cache)
# Map these in docker-compose.yml so data survives container restarts.

FROM python:3.11-slim

# System dependencies needed by sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer-cache friendly —
# requirements.txt changes less often than source code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Streamlit default port
EXPOSE 8501

# Disable Streamlit's browser-open-on-start behaviour (useless in a container)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
