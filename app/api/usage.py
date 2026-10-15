from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.api.security import AdminClient
from app.config import get_settings
from app.schemas.usage import CostReport, UsageReport
from app.services import usage

# Spending of every client together: an operator view, not a client view of its own use
router = APIRouter(prefix="/v1/usage", tags=["usage"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
DaysQuery = Annotated[int, Query(ge=1, le=365, description="Окно отчёта в днях")]


@router.get("", response_model=UsageReport)
async def get_usage(session: SessionDep, _: AdminClient, days: DaysQuery = 30) -> UsageReport:
    return await usage.usage_report(session, days)


@router.get("/cost", response_model=CostReport)
async def get_cost(session: SessionDep, _: AdminClient, days: DaysQuery = 30) -> CostReport:
    return await usage.cost_report(session, days, get_settings().usd_to_rub)
