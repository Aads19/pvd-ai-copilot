FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only FIRST (saves ~2GB vs default CUDA build)
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Restore the prebuilt SQLite DB from a Git-safe archive.
RUN if [ -f /app/chroma_db/chroma.sqlite3.xz ]; then \
    xz -dc /app/chroma_db/chroma.sqlite3.xz > /app/chroma_db/chroma.sqlite3 && \
    rm /app/chroma_db/chroma.sqlite3.xz; \
    fi

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
