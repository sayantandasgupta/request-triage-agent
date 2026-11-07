from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["new_request", "existing_case_follow_up",
                   "policy_guidance", "other_or_unclear"]

FollowUpIntent = Literal[
    "request_update",                 # wants to know the status
    "provide_additional_information",  # has new facts for the existing case
    "ask_whether_to_resubmit",         # asks if a new request should be submitted
    "general_follow_up",               # follows up without a more specific aim
]


class ExtractedInfo(BaseModel):
    """Information stated in the request. Every field is None (or empty) unless the user actually said it."""

    @field_validator("concern_description", "incident_date_or_period", "case_reference",
                     "original_submission_date", "policy_topic", mode="before")
    @classmethod
    def blank_to_none(cls, value):
        """Convert empty strings to None."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # new_request needs both; concern_description is also filled for follow-ups
    concern_description: str | None = Field(
        None, description="What the concern is about, in the user's words. Fill this for new requests "
                          "and for follow-ups about an earlier concern. None if not described.")
    incident_date_or_period: str | None = Field(
        None, description="When the incident happened, as stated (e.g. 'last week'). None if not mentioned; do not guess.")

    # existing_case_follow_up needs at least one of these
    case_reference: str | None = Field(
        None, description="Existing case or reference number, e.g. 'CR-2026-00417'. None if not given.")
    original_submission_date: str | None = Field(
        None, description="When the original concern was submitted, as stated (e.g. 'two weeks ago'). None if not mentioned.")
    follow_up_intent: list[FollowUpIntent] = Field(
        default_factory=list,
        description="What the user wants from a follow-up. Can contain more than one value. "
                    "Empty unless the request is a follow-up on an earlier concern.")

    # policy_guidance needs this
    policy_topic: str | None = Field(
        None, description="The policy topic or question asked about (e.g. 'conflict of interest'). None if unclear.")


class Classification(BaseModel):
    reasoning: str = Field(
        description="One sentence explaining which category fits and why.")
    category: Category = Field(
        description="The request category. Use other_or_unclear if no category fits reliably.")
    extracted_info: ExtractedInfo
