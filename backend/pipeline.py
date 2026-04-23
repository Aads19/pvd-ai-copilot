"""
pipeline.py
Orchestrates all 7 agents in sequence. Returns the final LabState.
"""
from agents import (
    LabState,
    empty_state,
    chief_director,
    query_expander,
    hyde_generator,
    retriever_node,
    query_expander_retriever_node,
    hybrid_retriever_node,
    final_answer_node,
    final_paraphrase_node,
    chat_node,
)


def run_pipeline(query: str) -> LabState:
    state: LabState = empty_state(query)

    # ── Step 1: Route + Tag ──────────────────────────────────
    state.update(chief_director(state))

    if state["route"] == "chat":
        state.update(chat_node(state))
        return state

    # ── Step 2: Expand Query ─────────────────────────────────
    state.update(query_expander(state))

    # ── Step 3: Generate HyDE Document ──────────────────────
    state.update(hyde_generator(state))

    # ── Step 4a: Retrieve via Expanded Query ─────────────────
    state.update(query_expander_retriever_node(state))

    # ── Step 4b: Retrieve via HyDE ───────────────────────────
    state.update(retriever_node(state))

    # ── Step 5: Hybrid Combine + Rerank ──────────────────────
    state.update(hybrid_retriever_node(state))

    # ── Step 6: Generate Final Answer (Gemini) ───────────────
    state.update(final_answer_node(state))

    # ── Step 7: Paraphrase (Llama) ───────────────────────────
    state.update(final_paraphrase_node(state))

    return state
