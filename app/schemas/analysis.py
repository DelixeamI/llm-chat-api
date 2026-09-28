from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.chat import GenerationParams, Usage


class SupportAnalysis(BaseModel):
    """What the model must return when it classifies a support ticket."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    # Stripped before the length check: a schema-constrained model produced "\n" here,
    # which passes min_length=1 but says nothing
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=32_000)
    params: GenerationParams = Field(default_factory=GenerationParams)


class AnalysisResponse(BaseModel):
    model: str
    analysis: SupportAnalysis
    usage: Usage
