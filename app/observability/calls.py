import logging
from decimal import Decimal
from typing import Literal

from app.observability import metrics

logger = logging.getLogger("app.llm")

CallStatus = Literal["success", "error", "timeout", "cached", "interrupted"]


def record_llm_call(
    *,
    model: str,
    endpoint: str,
    status: CallStatus,
    latency_seconds: float,
    retries: int = 0,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: Decimal | None = None,
    finish_reason: str | None = None,
    structured: bool = False,
    ttft_seconds: float | None = None,
) -> None:
    """One line per model interaction: what was asked of whom, how long it took, what it cost.

    Only metadata. The prompt and the reply are deliberately absent: they may hold personal
    data, secrets or whatever a user pasted in, and a log is read by far more people, and kept
    far longer, than the request that produced it.
    """
    _count(
        model=model,
        endpoint=endpoint,
        status=status,
        latency_seconds=latency_seconds,
        retries=retries,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        ttft_seconds=ttft_seconds,
    )
    logger.log(
        logging.WARNING if status in ("error", "timeout") else logging.INFO,
        "llm_call",
        extra={
            "event": "llm_call",
            "endpoint": endpoint,
            "model": model,
            "status": status,
            "latency_ms": round(latency_seconds * 1000, 1),
            "ttft_ms": None if ttft_seconds is None else round(ttft_seconds * 1000, 1),
            "retry_count": retries,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": None if cost_usd is None else str(cost_usd),
            "finish_reason": finish_reason,
            "structured": structured,
        },
    )


def _count(
    *,
    model: str,
    endpoint: str,
    status: str,
    latency_seconds: float,
    retries: int,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: Decimal | None,
    ttft_seconds: float | None,
) -> None:
    model = metrics.model_label(model)
    metrics.LLM_REQUESTS.labels(model, endpoint, status).inc()
    if retries:
        metrics.LLM_RETRIES.labels(model).inc(retries)
    if status in ("error", "timeout"):
        metrics.LLM_ERRORS.labels(
            model, "timeout" if status == "timeout" else "provider_error"
        ).inc()
    # A cached reply did not call the model and an interrupted stream never finished: neither
    # says anything about how long the model takes, and either would skew the percentiles
    if status in ("success", "error", "timeout"):
        metrics.LLM_LATENCY.labels(model, endpoint).observe(latency_seconds)
    if ttft_seconds is not None and status == "success":
        metrics.LLM_TTFT.labels(model).observe(ttft_seconds)
    if input_tokens is not None and status != "cached":
        metrics.LLM_TOKENS.labels(model, "input").inc(input_tokens)
    if output_tokens is not None and status != "cached":
        metrics.LLM_TOKENS.labels(model, "output").inc(output_tokens)
    if cost_usd is not None:
        metrics.LLM_COST.labels(model).inc(float(cost_usd))
