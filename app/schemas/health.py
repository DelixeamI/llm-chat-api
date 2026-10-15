from typing import Literal

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    status: Literal["ok", "unavailable"]
    # A required dependency that is down makes the instance not ready; an optional one only
    # makes it degraded
    required: bool
    latency_ms: float


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    checks: dict[str, DependencyStatus]
    # What the readiness answer deliberately does not cover
    not_checked: list[str] = ["llm_provider"]
