import logging
from decimal import Decimal
from typing import Literal

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
