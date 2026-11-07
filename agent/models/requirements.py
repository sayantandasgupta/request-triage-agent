from pydantic import BaseModel, Field


class RequirementsCheck(BaseModel):
    """Required information missing from a user request."""

    missing_information: list[str] = Field(
        default_factory=list,
        description="Short phrases naming each missing required item. Empty if nothing is missing.")
