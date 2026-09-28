from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class ModelUsage(BaseModel):
    model: str
    requests: int
    failed_requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageReport(BaseModel):
    since: datetime
    days: int
    requests: int
    failed_requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    by_model: list[ModelUsage]


class ModelCost(BaseModel):
    model: str
    requests: int
    cost_usd: Decimal
    cost_rub: Decimal
    # Requests whose model had no price when they were made
    unpriced_requests: int


class CostReport(BaseModel):
    since: datetime
    days: int
    usd_to_rub: Decimal
    cost_usd: Decimal
    cost_rub: Decimal
    daily_average_usd: Decimal
    # Daily average over the window times 30: assumes traffic stays as it was
    projected_monthly_usd: Decimal
    projected_monthly_rub: Decimal
    unpriced_requests: int
    by_model: list[ModelCost]
