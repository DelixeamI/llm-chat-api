from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SupportAnalysis(BaseModel):
    """What the model must return when it classifies a support ticket."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    summary: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
