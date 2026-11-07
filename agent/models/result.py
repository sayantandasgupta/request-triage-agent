from pydantic import BaseModel, Field

from agent.models.classification import Category, ExtractedInfo
from agent.models.recommendation import Action

ADVISORY_NOTICE = (
    "This recommendation is advisory only. It does not represent a final decision "
    "and no action has been taken on any actual case."
)


class PolicyReference(BaseModel):
    """A policy excerpt cited in a recommendation."""

    policy_id: str
    title: str
    excerpt: str
    source_file: str


class TriageResult(BaseModel):
    """Structured output of the request triage agent."""

    category: Category
    extracted_info: ExtractedInfo
    missing_information: list[str] = Field(default_factory=list)
    recommended_action: Action
    guidance: str
    policy_references: list[PolicyReference] = Field(default_factory=list)
    policy_sufficient: bool = False
    retrieval_error: str | None = None
    advisory_notice: str = ADVISORY_NOTICE
