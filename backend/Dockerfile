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

# Cache startup model downloads in the image so Railway boot stays fast.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# App code
COPY . .

# Restore the prebuilt SQLite DB from a Git-safe archive.
RUN if [ -f /app/chroma_db/chroma.sqlite3.xz ]; then \
    xz -dc /app/chroma_db/chroma.sqlite3.xz > /app/chroma_db/chroma.sqlite3 && \
    rm /app/chroma_db/chroma.sqlite3.xz; \
    fi

EXPOSE 8000

CMD uvicorn main:app --host :: --port ${PORT:-8000}
