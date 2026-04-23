"""
agents/retriever.py
Dual retrieval (HyDE + Expanded Query) with CrossEncoder reranking.
ChromaDB is loaded once at startup and reused across requests.
"""
import os
from typing import List, Optional

import chromadb
from sentence_transformers import CrossEncoder

from .state import LabState

# ─── Lazy singletons ──────────────────────────────────────────
_chroma_collection = None
_cross_encoder = None


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        _chroma_collection = client.get_collection(name="pvd_docs")
        print(f"[Retriever] Loaded ChromaDB collection 'pvd_docs' from {chroma_path}")
        print(f"[Retriever] Total chunks: {_chroma_collection.count()}")
    return _chroma_collection


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[Retriever] CrossEncoder loaded.")
    return _cross_encoder


# ─── Tag filter builder ───────────────────────────────────────
def _build_tag_filter(tags: List[str]) -> Optional[dict]:
    tag_to_field = {
        "Background": "is_Background",
        "Synthesis": "is_Synthesis",
        "Characterization": "is_Characterization",
        "Analysis": "is_Analysis",
    }
    clauses = [{tag_to_field[t]: True} for t in tags if t in tag_to_field]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


# ─── Core retrieval + rerank ──────────────────────────────────
def _retrieve_and_rerank(query_text: str, tags: List[str], top_k: int = 3) -> List[dict]:
    if not query_text:
        return []

    collection = _get_collection()
    cross_encoder = _get_cross_encoder()

    kwargs = {"query_texts": [query_text], "n_results": top_k * 3}
    where_filter = _build_tag_filter(tags)
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return []

    pairs = [[query_text, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)

    reranked = [
        {
            "document_id": doc_id,
            "text": text,
            "score": float(score),
            "doi": meta.get("doi", ""),
            "title": meta.get("title", ""),
            "chunk_idx": meta.get("chunk_idx"),
        }
        for doc_id, text, meta, score in zip(ids, documents, metadatas, scores)
    ]
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


# ─── Individual retriever nodes ───────────────────────────────
def retriever_node(state: LabState) -> dict:
    chunks = _retrieve_and_rerank(state.get("hyde_document", ""), state.get("target_tags", []))
    return {"retrieved_chunks": chunks}


def query_expander_retriever_node(state: LabState) -> dict:
    chunks = _retrieve_and_rerank(state.get("expanded_query", ""), state.get("target_tags", []))
    return {"expanded_query_chunks": chunks}


# ─── Hybrid combiner node ─────────────────────────────────────
def hybrid_retriever_node(state: LabState) -> dict:
    cross_encoder = _get_cross_encoder()
    expanded_query = state.get("expanded_query", "")
    hyde_doc = state.get("hyde_document", "")

    expanded_chunks = state.get("expanded_query_chunks", [])
    hyde_chunks = state.get("retrieved_chunks", [])

    combined = expanded_chunks + hyde_chunks
    if not combined:
        return {"final_retrieved_chunks": []}

    # Deduplicate by document_id, keep highest score
    deduped: dict[str, dict] = {}
    for chunk in combined:
        doc_id = chunk["document_id"]
        if doc_id not in deduped or chunk["score"] > deduped[doc_id]["score"]:
            deduped[doc_id] = chunk

    unique_chunks = list(deduped.values())

    # Re-score against combined query signal
    combined_query = f"{expanded_query} {hyde_doc}".strip()
    if combined_query:
        pairs = [[combined_query, c["text"]] for c in unique_chunks]
        new_scores = cross_encoder.predict(pairs)
        for chunk, score in zip(unique_chunks, new_scores):
            chunk["score"] = float(score)

    unique_chunks.sort(key=lambda x: x["score"], reverse=True)
    return {"final_retrieved_chunks": unique_chunks[:3]}
