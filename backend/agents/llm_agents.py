"""
agents/llm_agents.py
Chief Director, Query Expander, and HyDE Generator nodes.
All powered by Groq llama-3.1-8b-instant with JSON mode.
"""
import json
import os

from groq import Groq

from .state import LabState

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ─────────────────────────────────────────────────────────────
# 1. Chief Director
# ─────────────────────────────────────────────────────────────
def chief_director(state: LabState) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the routing director for a Physical Vapor Deposition (PVD) AI Copilot. "
                        "Your job is to classify the user's query into a route and assign metadata tags.\n\n"
                        "STEP 1: Decide the route ('chat' or 'database').\n"
                        "- Route to 'chat' ONLY if the message is purely conversational (hello, hi, thanks, etc.).\n"
                        "- Route to 'database' for ANY query involving science, materials, physics, chemistry, "
                        "engineering, PVD, sputtering, evaporation, thin films, deposition parameters, substrates, "
                        "targets, plasma, vacuum, crystallinity, morphology, characterization tools, performance, "
                        "interpretation, or lab methods.\n"
                        "- If uncertain, prefer 'database'.\n\n"
                        "STEP 2: Assign tags (only if route='database').\n"
                        "Choose from EXACTLY these four strings:\n"
                        "  'Background' - theory, history, fundamental physics, rationale\n"
                        "  'Synthesis' - recipes, deposition parameters, hardware, fabrication steps\n"
                        "  'Characterization' - XRD, SEM, TEM, AFM, Raman, XPS, morphology, sample prep\n"
                        "  'Analysis' - EIS, performance, catalytic activity, results interpretation\n\n"
                        "RULES:\n"
                        "- If route='chat', target_tags MUST be [].\n"
                        "- A query may have multiple tags.\n"
                        "- If too general, return [].\n\n"
                        'Return strict JSON: {"reasoning": "string", "decision": "chat"|"database", "target_tags": []}'
                    ),
                },
                {"role": "user", "content": state["original_query"]},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        return {
            "route": parsed.get("decision", "database"),
            "target_tags": parsed.get("target_tags", []),
        }
    except Exception as e:
        print(f"chief_director error: {e}")
        return {"route": "database", "target_tags": []}


# ─────────────────────────────────────────────────────────────
# 2. Query Expander
# ─────────────────────────────────────────────────────────────
def query_expander(state: LabState) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Material Science semantic translator for a PVD AI Copilot. "
                        "Transform the user's raw query into a dense, keyword-rich semantic search string "
                        "optimized for a vector database.\n\n"
                        "- Preserve scientific intent.\n"
                        "- Expand informal wording to formal academic terminology.\n"
                        "- Add PVD-specific synonyms and abbreviations.\n"
                        "- Use assigned tags to bias the expansion.\n"
                        "- Do NOT answer the question.\n\n"
                        "Tag guidance:\n"
                        "- 'Background': theory, mechanisms, thermodynamics, kinetics, nucleation, growth.\n"
                        "- 'Synthesis': deposition recipes, process params, pressure, temperature, power, target.\n"
                        "- 'Characterization': XRD, SEM, TEM, AFM, Raman, XPS, EDS, morphology, grain size.\n"
                        "- 'Analysis': EIS, conductivity, catalytic activity, performance, stability.\n\n"
                        'Return strict JSON: {"optimized_query": "string"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target Tags: {state.get('target_tags', [])}\n\n"
                        f"User Query: {state['original_query']}"
                    ),
                },
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        return {"expanded_query": parsed["optimized_query"]}
    except Exception as e:
        print(f"query_expander error: {e}")
        return {"expanded_query": state["original_query"]}


# ─────────────────────────────────────────────────────────────
# 3. HyDE Generator
# ─────────────────────────────────────────────────────────────
def hyde_generator(state: LabState) -> dict:
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Material Science researcher writing a hypothetical excerpt for a peer-reviewed journal. "
                        "Write ONE dense paragraph incorporating the keywords from the query. "
                        "Focus on mimicking the structural style and vocabulary of a real PVD research paper. "
                        "Factual accuracy is not required — focus on retrieval-optimized structure.\n\n"
                        "Tag guidance:\n"
                        "- 'Background' → write like an Introduction section.\n"
                        "- 'Synthesis' → write like an Experimental Methods section.\n"
                        "- 'Characterization' → write like a Materials Characterization section.\n"
                        "- 'Analysis' → write like a Results and Discussion section.\n"
                        "- Multiple tags → blend styles smoothly.\n\n"
                        "Rules:\n"
                        "- Exactly one dense paragraph.\n"
                        "- Formal academic tone.\n"
                        "- No bullet points, headings, or meta-commentary.\n\n"
                        'Return strict JSON: {"hypothetical_document": "string"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target Tags: {state.get('target_tags', [])}\n\n"
                        f"Expanded Query: {state.get('expanded_query', '')}"
                    ),
                },
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        return {"hyde_document": parsed["hypothetical_document"]}
    except Exception as e:
        print(f"hyde_generator error: {e}")
        return {"hyde_document": state.get("expanded_query", "")}
