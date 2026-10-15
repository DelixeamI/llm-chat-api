from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Header, Response
from pydantic import BaseModel

from app.observability import metrics
from app.services.errors import IdempotencyInProgressError, IdempotencyKeyReuseError
from app.services.idempotency import (
    IdempotencyStore,
    InProgress,
    Mismatch,
    Proceed,
    Replay,
    fingerprint,
)

IdempotencyKey = Annotated[
    str | None,
    Header(
        min_length=1,
        max_length=255,
        description="Unique key of this request: a retry with the same key returns the first "
        "response instead of running again",
    ),
]


async def run_idempotent[M: BaseModel](
    store: IdempotencyStore | None,
    *,
    client: str,
    key: str | None,
    payload: BaseModel,
    action: Callable[[], Awaitable[M]],
) -> M | Response:
    """Run `action` once per (client, key); a repeat gets the stored response."""
    if key is None or store is None:
        return await action()

    request_fingerprint = fingerprint(payload.model_dump_json())
    outcome = await store.begin(client, key, request_fingerprint)
    if isinstance(outcome, Replay):
        metrics.IDEMPOTENT_REPLAYS.inc()
        # The stored body is returned byte for byte: it is the response the first request
        # produced, not a new generation
        return Response(
            content=outcome.body,
            media_type="application/json",
            headers={"Idempotent-Replay": "true"},
        )
    if isinstance(outcome, InProgress):
        raise IdempotencyInProgressError()
    if isinstance(outcome, Mismatch):
        raise IdempotencyKeyReuseError()
    # Proceed, or Unprotected (Redis down): run it
    try:
        result = await action()
    except BaseException:
        # Includes cancellation when the client gives up. A failed attempt must not block
        # the retry that the client is about to send.
        if isinstance(outcome, Proceed):
            await store.abort(client, key)
        raise
    if isinstance(outcome, Proceed):
        await store.complete(client, key, request_fingerprint, result.model_dump_json())
    return result
