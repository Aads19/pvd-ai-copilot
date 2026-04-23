from typing import TypedDict, List


class LabState(TypedDict):
    original_query: str
    route: str
    target_tags: List[str]
    expanded_query: str
    hyde_document: str
    expanded_query_chunks: List[dict]
    retrieved_chunks: List[dict]
    final_retrieved_chunks: List[dict]
    final_answer: str
    final_answer_raw: str


def empty_state(query: str) -> LabState:
    return LabState(
        original_query=query,
        route="",
        target_tags=[],
        expanded_query="",
        hyde_document="",
        expanded_query_chunks=[],
        retrieved_chunks=[],
        final_retrieved_chunks=[],
        final_answer="",
        final_answer_raw="",
    )
