from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.dependencies import get_readiness_checker
from app.schemas.health import DependencyStatus, ReadinessResponse
from app.services.readiness import ReadinessChecker

router = APIRouter(tags=["health"])


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def ready(
    checker: Annotated[ReadinessChecker, Depends(get_readiness_checker)],
) -> JSONResponse:
    """Can this instance serve traffic? 200 when it can, 503 when a required dependency is down.

    Public, like /health, because a load balancer or an orchestrator has no API key. For the
    same reason the answer names only the dependencies and their state, never an error text.
    """
    report = await checker.check()
    body = ReadinessResponse(
        status=report.status,
        checks={
            name: DependencyStatus(
                status="ok" if result.ok else "unavailable",
                required=result.required,
                latency_ms=result.latency_ms,
            )
            for name, result in report.checks.items()
        },
    )
    return JSONResponse(status_code=200 if report.serving else 503, content=body.model_dump())
