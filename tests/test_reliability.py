"""Failure scenarios through the whole HTTP stack, one test per scenario.

The list comes from the project roadmap (section 11): conditions that must be reproduced on
purpose, because code that only ever sees the happy path has never been tested at all. Scenarios
already covered elsewhere are mapped in docs/failure-scenarios.md; this module holds those that
needed the full stack, concurrency or a hand-driven ASGI connection.
"""

import asyncio
import re
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

import fakeredis
import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.api.dependencies import (
    get_chat_service,
    get_idempotency_store,
    get_rate_limiter,
    get_session,
)
from app.llm.base import Completion, LLMProviderError, StreamChunk
from app.main import app
from app.schemas.chat import GenerationParams, Message
from app.services.cache import ResponseCache
from app.services.idempotency import IdempotencyStore
from app.services.ratelimit import RateLimiter
from tests.fakes import (
    ALICE,
    CannedProvider,
    FakeProvider,
    FakeSession,
    ScriptedProvider,
    ServiceFactory,
    StreamingProvider,
    as_redis,
)

BODY = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------- provider 429 and 503


@pytest.mark.parametrize("status", [429, 503])
def test_provider_overload_is_retried_and_then_succeeds(
    client: TestClient, use_provider: Callable[..., None], status: int
) -> None:
    provider = ScriptedProvider(LLMProviderError("busy", status_code=status, retryable=True))
    use_provider(provider)

    response = client.post("/v1/chat", json=BODY)

    assert response.status_code == 200
    assert provider.calls == 2


@pytest.mark.parametrize("status", [429, 503])
def test_provider_overload_that_does_not_clear_becomes_a_502_after_bounded_retries(
    client: TestClient, use_provider: Callable[..., None], status: int
) -> None:
    errors = [LLMProviderError("busy", status_code=status, retryable=True) for _ in range(10)]
    provider = ScriptedProvider(*errors)
    use_provider(provider, max_retries=2)

    response = client.post("/v1/chat", json=BODY)

    # The provider's 429 is not passed through: to our client it is an upstream failure (502),
    # and 429 is reserved for the client exceeding its own limit
    assert response.status_code == 502
    assert response.json()["error"] == "llm_provider_error"
    assert provider.calls == 3


# ---------------------------------------------------------------- PostgreSQL unavailable


class DownSession(FakeSession):
    """A session whose database refuses every statement, like a stopped server."""

    async def get(self, entity: type[object], key: Any) -> object | None:
        raise OperationalError("SELECT", {}, ConnectionRefusedError("connection refused"))

    async def commit(self) -> None:
        raise OperationalError("COMMIT", {}, ConnectionRefusedError("connection refused"))


def use_session(session: FakeSession) -> None:
    app.dependency_overrides[get_session] = lambda: session


def test_database_down_before_the_call_is_a_503_and_the_model_is_not_called(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("never")
    use_provider(provider)
    use_session(DownSession())

    response = client.post(
        "/v1/chat", json=BODY | {"conversation_id": "5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": "database_unavailable",
        "detail": "The database is unavailable",
    }
    assert provider.calls == []


def test_database_failure_inside_the_transaction_rolls_back_and_is_a_503(
    client: TestClient,
) -> None:
    session = DownSession()
    use_session(session)

    response = client.post("/v1/chat", json=BODY)

    assert response.status_code == 503
    # The model had answered, but the exchange was not committed, and the failed transaction
    # was ended explicitly instead of being left open on the connection
    assert session.rollbacks >= 1


def test_database_down_after_a_stream_has_started_is_reported_inside_the_stream(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(FakeProvider())
    use_session(DownSession())

    response = client.post("/v1/chat/stream", json=BODY)

    # Headers were already sent with 200; the failure arrives as an event, not as silence
    assert response.status_code == 200
    assert "event: error" in response.text and "storage_unavailable" in response.text
    assert "event: done" not in response.text


def test_database_error_text_never_reaches_the_client(client: TestClient) -> None:
    use_session(DownSession())

    response = client.post("/v1/chat", json=BODY)

    assert "refused" not in response.text and "COMMIT" not in response.text


# ---------------------------------------------------------------- Redis unavailable


def test_every_redis_feature_degrades_without_failing_the_request(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    server = fakeredis.FakeServer()
    redis = fakeredis.FakeAsyncRedis(server=server)
    server.connected = False
    provider = CannedProvider("a", "b", "c")
    cache = ResponseCache(as_redis(redis), ttl_seconds=60, scope="t")  # type: ignore[arg-type]
    use_provider(provider, cache=cache)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        redis, limit=1, window_seconds=60
    )
    app.dependency_overrides[get_idempotency_store] = lambda: IdempotencyStore(
        redis, ttl_seconds=60, lock_seconds=60
    )
    body = BODY | {"params": {"temperature": 0}}

    responses = [
        client.post("/v1/chat", json=body, headers={"Idempotency-Key": "k"}) for _ in range(3)
    ]

    # Cache, rate limit and idempotency are all down, and all three requests were served.
    # The price is the loss of their guarantees, not of the service.
    assert [r.status_code for r in responses] == [200, 200, 200]
    # Nothing was deduplicated or cached: all three really ran
    assert len(provider.calls) == 3


# ---------------------------------------------------------------- about 20 concurrent requests


class SlowProvider:
    """Counts how many calls overlap, so a concurrency limit can be proven, not assumed."""

    def __init__(self, latency: float) -> None:
        self.latency = latency
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def complete(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.latency)
        finally:
            self.in_flight -= 1
        return Completion("ok", 1, 1, "stop")

    def stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Any:  # pragma: no cover
        raise NotImplementedError


def test_twenty_concurrent_requests_all_succeed_within_the_concurrency_limit(
    client: TestClient, make_service: ServiceFactory, session: FakeSession
) -> None:
    provider = SlowProvider(latency=0.05)
    # One service for all requests, as in the real app: the semaphore lives inside it
    service = make_service(provider, max_concurrency=5)
    app.dependency_overrides[get_chat_service] = lambda: service

    async def burst() -> list[int]:
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://test", headers=ALICE
        ) as http:
            responses = await asyncio.gather(*(http.post("/v1/chat", json=BODY) for _ in range(20)))
        return [r.status_code for r in responses]

    statuses = asyncio.run(burst())

    assert statuses == [200] * 20
    assert provider.calls == 20
    # 20 requests at once, never more than 5 calls to the model at the same time
    assert provider.peak == 5
    assert len(session.usage_logs) == 20


# ---------------------------------------------------------------- client disconnect while streaming


def test_client_disconnect_stops_the_generation_and_frees_the_slot(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    provider = StreamingProvider(StreamChunk(content="a"), StreamChunk(content="b"), 30.0)
    use_provider(provider, max_concurrency=1, timeout_seconds=60.0)
    first_piece_sent = asyncio.Event()

    async def drive() -> float:
        body = b'{"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}'
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/stream",
            "raw_path": b"/v1/chat/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"x-api-key", ALICE["X-API-Key"].encode()),
            ],
            "server": ("test", 80),
            "client": ("test", 1),
        }
        sent_request = False

        async def receive() -> dict[str, Any]:
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": body, "more_body": False}
            # The client reads the first piece and then goes away
            await first_piece_sent.wait()
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.body" and message.get("body"):
                first_piece_sent.set()

        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
        return loop.time() - started

    elapsed = asyncio.run(drive())

    # The server did not sit through the 30 s pause: the disconnect cancelled the stream
    assert elapsed < 5
    assert provider.opened == 1 and provider.closed == 1
    assert session.messages == [] and session.usage_logs == []


# ---------------------------------------------------------------- the scenario list stays honest


def test_every_scenario_in_the_document_points_at_a_real_test() -> None:
    document = Path("docs/failure-scenarios.md").read_text(encoding="utf-8")
    references = re.findall(r"`(tests/[\w./]+\.py)::(\w+)`", document)
    assert references, "the document lists no tests"

    missing = [
        f"{path}::{name}"
        for path, name in references
        if f"def {name}(" not in Path(path).read_text(encoding="utf-8")
    ]

    assert missing == []
