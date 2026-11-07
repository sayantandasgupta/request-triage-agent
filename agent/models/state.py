from typing import TypedDict

from agent.models.classification import Category, ExtractedInfo
from agent.models.result import TriageResult


class TriageState(TypedDict, total=False):
    """Shared state passed between the nodes of the triage graph."""

    request: str
    reasoning: str
    category: Category
    extracted_info: ExtractedInfo
    missing_information: list[str]
    search_results: list[dict]
    retrieval_error: str | None
    clarification_rounds: int
    recommended_action: str
    guidance: str
    policy_references: list[dict]
    policy_sufficient: bool
    result: TriageResult
