# ⬡ PVD AI Copilot — Full-Stack Deployment Guide

A 7-agent RAG chatbot for Physical Vapor Deposition research, powered by **Groq (LLaMA)**, **Gemini**, **ChromaDB**, and a **React + FastAPI** stack.

---

## 🗂 Project Structure

```
pvd-chatbot/
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py             # LabState TypedDict
│   │   ├── llm_agents.py        # Director, Query Expander, HyDE Generator
│   │   ├── retriever.py         # Dual retrieval + CrossEncoder reranking
│   │   └── answer_agents.py     # Gemini answer + Llama paraphrase
│   ├── scripts/
│   │   └── build_chroma.py      # One-time ChromaDB builder from CSV
│   ├── pipeline.py              # 7-agent orchestrator
│   ├── main.py                  # FastAPI app
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Full chat UI
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
├── docker-compose.yml
├── render.yaml
└── README.md
```

---

## 🧠 Pipeline (7 Agents)

```
Query → [1] Chief Director (route + tags)
      → [2] Query Expander (semantic enrichment)
      → [3] HyDE Generator (hypothetical document)
      → [4a] HyDE Retriever (ChromaDB)
      → [4b] Expanded Query Retriever (ChromaDB)
      → [5] Hybrid Combiner + CrossEncoder Reranker
      → [6] Gemini Final Answer
      → [7] Llama-3.2-3B Paraphrase
```

---

## ⚙️ Step 1 — Setup Environment

```bash
cd backend
cp .env.example .env
```

Edit `.env` and fill in:
```
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
HF_TOKEN=your_huggingface_token
CHROMA_PATH=./chroma_db
```

---

## 🗄️ Step 2 — Build ChromaDB (Run Once)

This reads your CSV and creates the vector database. **You only need to do this once.**

```bash
cd backend
pip install -r requirements.txt

python scripts/build_chroma.py --csv /path/to/PLD_CATEGORY_FINAL_DATASET.csv
```

This will:
- Load all 8,381 chunks from your CSV
- Embed them with `BAAI/bge-small-en-v1.5`
- Store them in ChromaDB at `./chroma_db` with metadata tags

⏱️ Takes ~5–10 minutes on CPU, ~2 minutes on GPU.

---

## 🚀 Step 3A — Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

---

## 🐳 Step 3B — Run with Docker

```bash
# First copy the built chroma_db into project root
cp -r backend/chroma_db ./chroma_db

docker-compose up --build
```

- Frontend → `http://localhost:3000`
- Backend API → `http://localhost:8000`
- API Docs → `http://localhost:8000/docs`

---

## ☁️ Step 3C — Deploy to Render.com

### Backend

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Standard (needs RAM for models)
5. Add **Environment Variables**:
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`
   - `HF_TOKEN`
   - `CHROMA_PATH` = `/opt/render/project/src/backend/chroma_db`
   - `FRONTEND_URL` = your frontend URL
6. Add a **Disk** (under Advanced):
   - Mount Path: `/opt/render/project/src/backend/chroma_db`
   - Size: 5 GB
7. **Upload ChromaDB**: After first deploy, use Render Shell to upload your `chroma_db` folder, OR run `build_chroma.py` in the shell

### Frontend

1. Go to Render → **New Static Site**
2. Connect repo → **Root Directory**: `frontend`
3. **Build Command**: `npm install && npm run build`
4. **Publish Directory**: `dist`
5. Add **Environment Variable**:
   - `VITE_API_URL` = your backend Render URL (e.g. `https://pvd-copilot-backend.onrender.com`)

---

## 🚂 Step 3D — Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Deploy backend
cd backend
railway init
railway up

# Set env vars
railway variables set GROQ_API_KEY=xxx GEMINI_API_KEY=xxx HF_TOKEN=xxx CHROMA_PATH=./chroma_db
```

---

## 📡 API Reference

### `POST /chat`

```json
// Request
{ "query": "What is the effect of substrate temperature on PZT crystallinity?" }

// Response
{
  "answer": "According to the retrieved evidence...",
  "route": "database",
  "target_tags": ["Synthesis", "Characterization"],
  "expanded_query": "PZT thin film substrate temperature crystallinity...",
  "chunks": [
    { "title": "...", "doi": "10.1016/...", "score": 0.87, "chunk_idx": 14 }
  ],
  "processing_time_ms": 12400
}
```

### `GET /health`
Returns `{"status": "ok"}` — use for uptime checks.

---

## ⚠️ Important Notes

- **Llama paraphrase node** requires ~6GB RAM. If your deployment has limited RAM, you can disable it in `pipeline.py` by commenting out `state.update(final_paraphrase_node(state))`.
- **ChromaDB must be pre-built** and accessible at the `CHROMA_PATH`. Run `build_chroma.py` once before deploying.
- **HuggingFace gated model**: Llama-3.2-3B requires you to accept Meta's license on HuggingFace before your token will work.
- **Cold starts**: The first request takes longer as models load into memory. Subsequent requests are fast.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Routing + Expansion + HyDE | Groq `llama-3.1-8b-instant` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector DB | ChromaDB (persistent) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Final Answer | Google Gemini 1.5 Flash |
| Paraphrase | Meta `Llama-3.2-3B` (HuggingFace) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Deployment | Docker / Render / Railway |
