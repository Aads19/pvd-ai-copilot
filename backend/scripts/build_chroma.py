"""
build_chroma.py
Run this ONCE to build the ChromaDB from the CSV dataset.
Usage: python scripts/build_chroma.py --csv path/to/PLD_CATEGORY_FINAL_DATASET.csv
"""
import argparse
import json
import os
import shutil

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_BATCH_SIZE = 256
ADD_BATCH_SIZE = 1000


def build(csv_path: str):
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['doi'].nunique()} unique papers.")

    # Clean start
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Removed existing DB at {CHROMA_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on: {device}")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="pvd_docs")

    documents, metadatas, ids = [], [], []

    for idx, row in df.iterrows():
        try:
            tag_dict = json.loads(row["tags"])
            chunk_tags = tag_dict.get("tags", [])
        except Exception:
            chunk_tags = []

        metadata = {
            "doi": str(row["doi"]),
            "title": str(row["title"]),
            "chunk_idx": int(row["chunk_start_idx"]),
            "is_Background": "Background" in chunk_tags,
            "is_Synthesis": "Synthesis" in chunk_tags,
            "is_Characterization": "Characterization" in chunk_tags,
            "is_Analysis": "Analysis" in chunk_tags,
        }

        documents.append(str(row["text_chunk"]))
        metadatas.append(metadata)
        ids.append(f"{row['doi']}_chunk_{idx}")

    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = []
    for i in range(0, len(documents), EMBED_BATCH_SIZE):
        batch = documents[i : i + EMBED_BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings.extend(emb)
        print(f"  Embedded {min(i + EMBED_BATCH_SIZE, len(documents))}/{len(documents)}")

    print("Storing in ChromaDB...")
    for i in range(0, len(documents), ADD_BATCH_SIZE):
        collection.add(
            documents=documents[i : i + ADD_BATCH_SIZE],
            metadatas=metadatas[i : i + ADD_BATCH_SIZE],
            ids=ids[i : i + ADD_BATCH_SIZE],
            embeddings=[e.tolist() for e in embeddings[i : i + ADD_BATCH_SIZE]],
        )
        print(f"  Stored {min(i + ADD_BATCH_SIZE, len(documents))}/{len(documents)}")

    print(f"\n✅ Done! Total chunks stored: {collection.count()}")
    print(f"📁 ChromaDB saved at: {CHROMA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to PLD_CATEGORY_FINAL_DATASET.csv")
    args = parser.parse_args()
    build(args.csv)
