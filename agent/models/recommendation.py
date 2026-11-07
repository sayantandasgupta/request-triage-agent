from typing import Literal

from pydantic import BaseModel, Field

Action = Literal[
    "provide_policy_guidance",
    "request_missing_information",
    "follow_up_existing_case",
    "submit_new_request",
]


class Recommendation(BaseModel):
    """Recommendation written from the retrieved policy excerpts."""

    evidence_assessment: str = Field(
        description="One or two sentences on whether the excerpts answer the request, and why.")
    policy_sufficient: bool = Field(
        description="True only if the excerpts contain enough to answer the request fully.")
    recommended_action: Action = Field(
        description="The next action the excerpts support for this request.")
    guidance: str = Field(
        description="Next steps for the user, based only on the excerpts, citing policy IDs like [POL-002]. "
                    "Empty if policy_sufficient is false.")
    cited_excerpt_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the excerpts the guidance is based on, e.g. ['E1', 'E3']. Empty if policy_sufficient is false.")
