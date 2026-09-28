from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UsageLog
from app.schemas.usage import CostReport, ModelCost, ModelUsage, UsageReport

DAYS_PER_MONTH = 30
USD_PLACES = Decimal("0.00000001")
RUB_PLACES = Decimal("0.000001")


def _since(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


async def usage_report(session: AsyncSession, days: int) -> UsageReport:
    since = _since(days)
    # Aggregation happens in the database: summing thousands of rows in Python would
    # first drag all of them over the network
    rows = (
        await session.execute(
            select(
                UsageLog.model,
                func.count(),
                func.count().filter(UsageLog.status != "success"),
                func.sum(UsageLog.input_tokens),
                func.sum(UsageLog.output_tokens),
                func.sum(UsageLog.total_tokens),
            )
            .where(UsageLog.created_at >= since)
            .group_by(UsageLog.model)
            .order_by(UsageLog.model)
        )
    ).all()

    by_model = [
        ModelUsage(
            model=model,
            requests=requests,
            failed_requests=failed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        for model, requests, failed, input_tokens, output_tokens, total_tokens in rows
    ]
    return UsageReport(
        since=since,
        days=days,
        requests=sum(m.requests for m in by_model),
        failed_requests=sum(m.failed_requests for m in by_model),
        input_tokens=sum(m.input_tokens for m in by_model),
        output_tokens=sum(m.output_tokens for m in by_model),
        total_tokens=sum(m.total_tokens for m in by_model),
        by_model=by_model,
    )


async def cost_report(session: AsyncSession, days: int, usd_to_rub: Decimal) -> CostReport:
    since = _since(days)
    rows = (
        await session.execute(
            select(
                UsageLog.model,
                func.count(),
                func.coalesce(func.sum(UsageLog.cost_usd), 0),
                func.sum(case((UsageLog.cost_usd.is_(None), 1), else_=0)),
            )
            .where(UsageLog.created_at >= since)
            .group_by(UsageLog.model)
            .order_by(UsageLog.model)
        )
    ).all()

    def rub(usd: Decimal) -> Decimal:
        return (usd * usd_to_rub).quantize(RUB_PLACES)

    by_model = [
        ModelCost(
            model=model,
            requests=requests,
            cost_usd=Decimal(cost).quantize(USD_PLACES),
            cost_rub=rub(Decimal(cost)),
            unpriced_requests=int(unpriced),
        )
        for model, requests, cost, unpriced in rows
    ]
    total_usd = sum((m.cost_usd for m in by_model), Decimal(0))
    daily_average = (total_usd / days).quantize(USD_PLACES)
    projected = (daily_average * DAYS_PER_MONTH).quantize(USD_PLACES)
    return CostReport(
        since=since,
        days=days,
        usd_to_rub=usd_to_rub,
        cost_usd=total_usd,
        cost_rub=rub(total_usd),
        daily_average_usd=daily_average,
        projected_monthly_usd=projected,
        projected_monthly_rub=rub(projected),
        unpriced_requests=sum(m.unpriced_requests for m in by_model),
        by_model=by_model,
    )
