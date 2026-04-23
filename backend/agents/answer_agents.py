"""
agents/answer_agents.py
Final answer generation (Gemini) + paraphrasing (Llama-3.2-3B via HuggingFace).
"""
import os
from typing import List

import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .state import LabState

# ─── Gemini setup ─────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ─── Llama setup (lazy) ───────────────────────────────────────
PARAPHRASE_MODEL_NAME = os.getenv("PARAPHRASE_MODEL", "meta-llama/Llama-3.2-3B")
HF_TOKEN = os.getenv("HF_TOKEN")

_paraphrase_tokenizer = None
_paraphrase_model = None


def _load_paraphrase_model():
    global _paraphrase_tokenizer, _paraphrase_model
    if _paraphrase_tokenizer is None or _paraphrase_model is None:
        print("[Paraphrase] Loading Llama tokenizer...")
        _paraphrase_tokenizer = AutoTokenizer.from_pretrained(
            PARAPHRASE_MODEL_NAME, token=HF_TOKEN
        )
        if _paraphrase_tokenizer.pad_token is None:
            _paraphrase_tokenizer.pad_token = _paraphrase_tokenizer.eos_token

        print("[Paraphrase] Loading Llama model (this may take a while)...")
        _paraphrase_model = AutoModelForCausalLM.from_pretrained(
            PARAPHRASE_MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print("[Paraphrase] Llama model ready.")
    return _paraphrase_tokenizer, _paraphrase_model


# ─── Context builder ──────────────────────────────────────────
def _build_context(chunks: List[dict]) -> str:
    if not chunks:
        return "No relevant evidence retrieved."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Unknown")
        doi = chunk.get("doi", "")
        text = chunk.get("text", "")
        parts.append(f"[Chunk {i}]\nTitle: {title}\nDOI: {doi}\n\n{text}")
    return "\n\n---\n\n".join(parts)


# ─── Final answer node (Gemini) ───────────────────────────────
def final_answer_node(state: LabState) -> dict:
    user_query = state.get("original_query", "")
    chunks = state.get("final_retrieved_chunks", [])
    evidence = _build_context(chunks)

    prompt = f"""You are a Materials Science AI Copilot for Physical Vapor Deposition (PVD) research.

Answer the user's question strictly using the retrieved chunks as the only source of truth.

Rules:
- Use ONLY the retrieved chunks.
- Do not use prior knowledge or external information.
- Cite inline using [Chunk 1], [Chunk 2], [Chunk 3].
- If chunks provide only partial information, answer only the supported portion.
- Do not infer, speculate, or generalize beyond what is explicitly stated.
- Keep the response precise, direct, and academically structured.

User Query:
{user_query}

Retrieved Evidence:
{evidence}

Return only the final answer text."""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2),
        )
        return {"final_answer": response.text.strip()}
    except Exception as e:
        print(f"final_answer_node error: {e}")
        return {"final_answer": "I could not generate a final answer from the retrieved chunks."}


# ─── Paraphrase node (Llama) ──────────────────────────────────
def final_paraphrase_node(state: LabState) -> dict:
    draft_answer = state.get("final_answer", "").strip()
    original_query = state.get("original_query", "").strip()

    if not draft_answer:
        return {}

    try:
        tokenizer, model = _load_paraphrase_model()

        prompt = (
            "You are a scientific paraphrasing assistant.\n"
            "Rewrite the following answer using completely different wording.\n"
            "Remove all inline citations like [Chunk 1], [Chunk 2], etc. from the main body.\n"
            "Keep all scientific facts, numbers, and units exactly as they are.\n"
            "After the answer, add a final section exactly titled:\n"
            "Citations :-\n"
            "------------\n"
            "In that section, reproduce only the citation labels that were present in the original draft answer, one per line.\n"
            "Do not invent new citations.\n"
            "Do not place citations inside the main paragraph text.\n\n"
            f"Query: {original_query}\n\n"
            f"Answer to paraphrase:\n{draft_answer}\n\n"
            "Paraphrased answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        paraphrased = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "final_answer_raw": draft_answer,
            "final_answer": paraphrased or draft_answer,
        }
    except Exception as e:
        print(f"final_paraphrase_node error: {e}")
        return {
            "final_answer_raw": draft_answer,
            "final_answer": draft_answer,
        }


# ─── Chat fallback (Gemini) ───────────────────────────────────
def chat_node(state: LabState) -> dict:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            f"You are a friendly PVD research assistant. Respond conversationally.\n\nUser: {state['original_query']}",
            generation_config=genai.types.GenerationConfig(temperature=0.7),
        )
        return {"final_answer": response.text.strip()}
    except Exception as e:
        print(f"chat_node error: {e}")
        return {"final_answer": "Hello! How can I help you with PVD research today?"}
