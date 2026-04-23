"""
main.py  –  FastAPI backend for PVD AI Copilot
"""
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import run_pipeline
from agents.retriever import _get_collection, _get_cross_encoder

# ─── Lifespan: warm up heavy models at startup ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Warming up ChromaDB and CrossEncoder...")
    _get_collection()
    _get_cross_encoder()
    print("[Startup] Ready.")
    yield

app = FastAPI(
    title="PVD AI Copilot API",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────
class ChatRequest(BaseModel):
    query: str


class ChunkInfo(BaseModel):
    title: str
    doi: str
    score: float
    chunk_idx: int | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    target_tags: list[str]
    expanded_query: str
    chunks: list[ChunkInfo]
    processing_time_ms: int


# ─── Endpoints ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = time.perf_counter()
    state = run_pipeline(req.query.strip())
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    chunks = [
        ChunkInfo(
            title=c.get("title", ""),
            doi=c.get("doi", ""),
            score=round(c.get("score", 0.0), 4),
            chunk_idx=c.get("chunk_idx"),
        )
        for c in state.get("final_retrieved_chunks", [])
    ]

    return ChatResponse(
        answer=state.get("final_answer", ""),
        route=state.get("route", ""),
        target_tags=state.get("target_tags", []),
        expanded_query=state.get("expanded_query", ""),
        chunks=chunks,
        processing_time_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
