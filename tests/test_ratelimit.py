"""Rate limiting.

The Lua script runs here on fakeredis, which executes real Lua with Redis command semantics:
this verifies the algorithm. Behaviour that belongs to the real server (atomicity under
concurrent clients, key expiry) is repeated against a real Redis in test_redis_integration.py.
"""

import asyncio
from collections.abc import Callable, Coroutine

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_rate_limiter
from app.main import app
from app.services.ratelimit import RateLimiter, RateLimitResult
from tests.fakes import ALICE, BOB, CannedProvider, FakeProvider

CHAT_BODY = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
ANALYSIS_BODY = {"model": "llama3", "text": "Не могу войти"}


def make_limiter(
    *, limit: int = 3, window: int = 60, server: fakeredis.FakeServer | None = None
) -> RateLimiter:
    redis = fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer())
    return RateLimiter(redis, limit=limit, window_seconds=window)


def run[T](coroutine: Coroutine[object, object, T]) -> T:
    return asyncio.run(coroutine)


# ---------------------------------------------------------------- the algorithm


def test_requests_up_to_the_limit_are_admitted_then_rejected() -> None:
    limiter = make_limiter(limit=3)

    async def scenario() -> list[tuple[bool, int]]:
        results = [await limiter.check("alice") for _ in range(5)]
        return [(r.admitted, r.remaining) for r in results if r is not None]

    assert run(scenario()) == [(True, 2), (True, 1), (True, 0), (False, 0), (False, 0)]


def test_rejection_says_how_long_to_wait() -> None:
    limiter = make_limiter(limit=1, window=60)

    async def scenario() -> None:
        await limiter.check("alice")
        rejected = await limiter.check("alice")
        assert rejected is not None and not rejected.admitted
        assert 1 <= rejected.reset_seconds <= 60
        assert rejected.headers()["Retry-After"] == str(rejected.reset_seconds)

    run(scenario())


def test_clients_have_separate_allowances() -> None:
    limiter = make_limiter(limit=1)

    async def scenario() -> tuple[bool, bool, bool]:
        first = await limiter.check("alice")
        again = await limiter.check("alice")
        other = await limiter.check("bob")
        assert first and again and other
        return first.admitted, again.admitted, other.admitted

    assert run(scenario()) == (True, False, True)


def test_window_slides_instead_of_resetting_all_at_once() -> None:
    limiter = make_limiter(limit=2, window=1)

    async def scenario() -> list[bool]:
        outcomes = []
        for pause in (0, 0.6, 0.0, 0.5):
            await asyncio.sleep(pause)
            result = await limiter.check("alice")
            assert result is not None
            outcomes.append(result.admitted)
        return outcomes

    # t=0 admitted, t=0.6 admitted, t=0.6 rejected (two inside the window), t=1.1: the request
    # from t=0 has left the window but the one from t=0.6 has not, so exactly one slot is free
    assert run(scenario()) == [True, True, False, True]


def test_concurrent_requests_never_exceed_the_limit() -> None:
    limiter = make_limiter(limit=20)

    async def scenario() -> int:
        results = await asyncio.gather(*(limiter.check("alice") for _ in range(60)))
        return sum(1 for r in results if r is not None and r.admitted)

    assert run(scenario()) == 20


def test_two_requests_in_the_same_millisecond_are_both_counted() -> None:
    # The sorted-set members are made unique with a nonce; with the timestamp alone the
    # second request would overwrite the first and the client would get a free one
    limiter = make_limiter(limit=2)

    async def scenario() -> list[int]:
        results = await asyncio.gather(limiter.check("alice"), limiter.check("alice"))
        return sorted(r.remaining for r in results if r is not None)

    assert run(scenario()) == [0, 1]


def test_an_idle_clients_key_expires() -> None:
    server = fakeredis.FakeServer()
    limiter = make_limiter(limit=2, window=1, server=server)

    async def scenario() -> None:
        redis = fakeredis.FakeAsyncRedis(server=server)
        await limiter.check("alice")
        assert 0 < await redis.pttl("llm-chat:ratelimit:alice") <= 1000
        await asyncio.sleep(1.1)
        assert await redis.exists("llm-chat:ratelimit:alice") == 0

    run(scenario())


def test_redis_outage_means_no_limiting_not_no_service(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = fakeredis.FakeServer()
    limiter = make_limiter(server=server)
    server.connected = False

    async def scenario() -> tuple[RateLimitResult | None, RateLimitResult | None]:
        return await limiter.check("alice"), await limiter.check("alice")

    first, second = run(scenario())

    assert first is None and second is None
    # One warning for the outage, not one per request: the second call is skipped by the pause
    assert caplog.text.count("Rate limiter failed") == 1


# ---------------------------------------------------------------- over HTTP


def use_limiter(limiter: RateLimiter | None) -> None:
    app.dependency_overrides[get_rate_limiter] = lambda: limiter


def test_over_the_limit_is_a_429_with_retry_information(client: TestClient) -> None:
    use_limiter(make_limiter(limit=2))

    first = client.post("/v1/chat", json=CHAT_BODY)
    second = client.post("/v1/chat", json=CHAT_BODY)
    third = client.post("/v1/chat", json=CHAT_BODY)

    assert (first.status_code, second.status_code, third.status_code) == (200, 200, 429)
    body = third.json()
    assert (
        body["error"] == "rate_limit_exceeded" and "limit of 2 requests per 60 s" in body["detail"]
    )
    assert 1 <= int(third.headers["retry-after"]) <= 60
    assert third.headers["x-ratelimit-limit"] == "2"
    assert third.headers["x-ratelimit-remaining"] == "0"


def test_admitted_requests_report_the_remaining_allowance(client: TestClient) -> None:
    use_limiter(make_limiter(limit=3))

    first = client.post("/v1/chat", json=CHAT_BODY)
    second = client.post("/v1/chat", json=CHAT_BODY)

    assert first.headers["x-ratelimit-remaining"] == "2"
    assert second.headers["x-ratelimit-remaining"] == "1"
    assert "retry-after" not in first.headers


def test_rejected_request_never_reaches_the_model(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("one")
    use_provider(provider)
    use_limiter(make_limiter(limit=1))

    client.post("/v1/chat", json=CHAT_BODY)
    rejected = client.post("/v1/chat", json=CHAT_BODY)

    assert rejected.status_code == 429
    assert len(provider.calls) == 1


def test_one_clients_flood_does_not_block_another(client: TestClient) -> None:
    use_limiter(make_limiter(limit=1))

    client.post("/v1/chat", json=CHAT_BODY, headers=ALICE)
    flooded = client.post("/v1/chat", json=CHAT_BODY, headers=ALICE)
    other = client.post("/v1/chat", json=CHAT_BODY, headers=BOB)

    assert flooded.status_code == 429
    assert other.status_code == 200


def test_all_model_endpoints_share_one_allowance(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(FakeProvider())
    use_limiter(make_limiter(limit=2))

    chat = client.post("/v1/chat", json=CHAT_BODY)
    stream = client.post("/v1/chat/stream", json=CHAT_BODY)
    structured = client.post("/v1/chat/structured", json=ANALYSIS_BODY)

    assert (chat.status_code, stream.status_code) == (200, 200)
    assert structured.status_code == 429


def test_stream_carries_the_limit_headers(client: TestClient) -> None:
    use_limiter(make_limiter(limit=5))

    response = client.post("/v1/chat/stream", json=CHAT_BODY)

    assert response.status_code == 200
    assert response.headers["x-ratelimit-limit"] == "5"
    assert response.headers["x-ratelimit-remaining"] == "4"


def test_unauthenticated_requests_do_not_use_up_a_clients_allowance(client: TestClient) -> None:
    use_limiter(make_limiter(limit=1))

    for _ in range(5):
        assert (
            client.post("/v1/chat", json=CHAT_BODY, headers={"X-API-Key": "x" * 20}).status_code
            == 401
        )
    forbidden = client.get("/v1/usage", headers=ALICE)

    assert forbidden.status_code == 403
    assert client.post("/v1/chat", json=CHAT_BODY, headers=ALICE).status_code == 200


def test_reading_history_is_not_limited(client: TestClient) -> None:
    use_limiter(make_limiter(limit=1))

    client.post("/v1/chat", json=CHAT_BODY)

    # The limit protects what costs money: the model endpoints
    assert client.post("/v1/conversations", json={}).status_code == 201
    assert client.get("/health").status_code == 200


def test_requests_pass_when_the_limiter_is_down(client: TestClient) -> None:
    server = fakeredis.FakeServer()
    limiter = make_limiter(limit=1, server=server)
    server.connected = False
    use_limiter(limiter)

    assert [client.post("/v1/chat", json=CHAT_BODY).status_code for _ in range(3)] == [200] * 3


def test_switched_off_limiter_changes_nothing(client: TestClient) -> None:
    use_limiter(None)

    responses = [client.post("/v1/chat", json=CHAT_BODY) for _ in range(5)]

    assert all(r.status_code == 200 and "x-ratelimit-limit" not in r.headers for r in responses)
