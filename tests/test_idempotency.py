"""Idempotency keys, on fakeredis: SET NX, EX and DELETE behave as on a real server."""

import asyncio
import json
from collections.abc import Callable

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_idempotency_store, get_rate_limiter
from app.api.idempotency import run_idempotent
from app.llm.base import LLMProviderError
from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage
from app.services.errors import IdempotencyInProgressError, IdempotencyKeyReuseError
from app.services.idempotency import IdempotencyStore, fingerprint
from app.services.ratelimit import RateLimiter
from tests.fakes import ALICE, BOB, CannedProvider, FakeSession, ScriptedProvider

BODY = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
KEY = {"Idempotency-Key": "order-1"}
ANALYSIS = '{"category": "billing", "priority": "low", "summary": "x", "confidence": 0.5}'


def make_store(
    server: fakeredis.FakeServer | None = None, *, ttl: int = 3600, lock: int = 60
) -> IdempotencyStore:
    redis = fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer())
    return IdempotencyStore(redis, ttl_seconds=ttl, lock_seconds=lock)


def use_store(store: IdempotencyStore | None) -> None:
    app.dependency_overrides[get_idempotency_store] = lambda: store


# ---------------------------------------------------------------- the retry that must not repeat


def test_retry_with_the_same_key_gets_the_first_response_without_a_second_run(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    provider = CannedProvider("first answer", "second answer")
    use_provider(provider)
    use_store(make_store())

    # The client sent this, the server finished it, and the response never arrived
    first = client.post("/v1/chat", json=BODY, headers=KEY)
    retry = client.post("/v1/chat", json=BODY, headers=KEY)

    assert first.status_code == retry.status_code == 200
    assert retry.content == first.content
    assert retry.json()["message"]["content"] == "first answer"
    assert "idempotent-replay" not in first.headers
    assert retry.headers["idempotent-replay"] == "true"
    # One generation, one exchange, one charge
    assert len(provider.calls) == 1
    assert len(session.messages) == 2
    assert len(session.usage_logs) == 1


def test_without_a_key_every_request_runs(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    provider = CannedProvider("a", "b")
    use_provider(provider)
    use_store(make_store())

    client.post("/v1/chat", json=BODY)
    client.post("/v1/chat", json=BODY)

    assert len(provider.calls) == 2 and len(session.usage_logs) == 2


def test_different_keys_are_different_requests(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("a", "b")
    use_provider(provider)
    use_store(make_store())

    client.post("/v1/chat", json=BODY, headers={"Idempotency-Key": "one"})
    client.post("/v1/chat", json=BODY, headers={"Idempotency-Key": "two"})

    assert len(provider.calls) == 2


def test_structured_requests_are_protected_too(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    provider = CannedProvider(ANALYSIS, ANALYSIS)
    use_provider(provider)
    use_store(make_store())
    body = {"model": "llama3", "text": "Не могу войти"}

    first = client.post("/v1/chat/structured", json=body, headers=KEY)
    retry = client.post("/v1/chat/structured", json=body, headers=KEY)

    assert retry.content == first.content and retry.headers["idempotent-replay"] == "true"
    assert len(provider.calls) == 1 and len(session.usage_logs) == 1


def test_same_key_with_a_different_request_is_rejected(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("a", "b")
    use_provider(provider)
    use_store(make_store())

    client.post("/v1/chat", json=BODY, headers=KEY)
    other = client.post(
        "/v1/chat",
        json={"model": "llama3", "messages": [{"role": "user", "content": "something else"}]},
        headers=KEY,
    )

    # Not a retry: answering with the stored reply would hand over the answer to another question
    assert other.status_code == 422
    assert other.json()["error"] == "idempotency_key_reuse"
    assert len(provider.calls) == 1


def test_a_key_belongs_to_one_client(client: TestClient, use_provider: Callable[..., None]) -> None:
    provider = CannedProvider("for alice", "for bob")
    use_provider(provider)
    use_store(make_store())

    alice = client.post("/v1/chat", json=BODY, headers=ALICE | KEY)
    bob = client.post("/v1/chat", json=BODY, headers=BOB | KEY)

    assert alice.json()["message"]["content"] == "for alice"
    assert bob.json()["message"]["content"] == "for bob"
    assert "idempotent-replay" not in bob.headers


# ---------------------------------------------------------------- failures and retries


def test_a_failed_attempt_does_not_block_the_retry(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    provider = ScriptedProvider(LLMProviderError("bad", status_code=400))
    use_provider(provider, max_retries=0)
    use_store(make_store())

    failed = client.post("/v1/chat", json=BODY, headers=KEY)
    retry = client.post("/v1/chat", json=BODY, headers=KEY)
    replay = client.post("/v1/chat", json=BODY, headers=KEY)

    assert failed.status_code == 502
    # The key was freed on failure, so the retry really ran and succeeded
    assert retry.status_code == 200 and "idempotent-replay" not in retry.headers
    assert replay.headers["idempotent-replay"] == "true"
    assert provider.calls == 2 and len(session.usage_logs) == 1


def test_errors_are_not_stored_for_replay(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(ScriptedProvider(LLMProviderError("bad", status_code=400)), max_retries=0)
    store = make_store()
    use_store(store)

    client.post("/v1/chat", json=BODY, headers=KEY)

    async def stored() -> object:
        return await store._redis.keys("llm-chat:idem:*")

    assert asyncio.run(stored()) == []


def test_request_still_running_makes_the_retry_wait_with_409() -> None:
    store = make_store()
    started = asyncio.Event()
    release = asyncio.Event()
    runs = 0

    async def slow_action() -> ChatResponse:
        nonlocal runs
        runs += 1
        started.set()
        await release.wait()
        return ChatResponse(
            conversation_id="5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1",
            model="m",
            message=Message(role="assistant", content="done"),
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    request = ChatRequest(model="m", messages=[Message(role="user", content="hi")])

    async def scenario() -> None:
        first = asyncio.create_task(
            run_idempotent(store, client="alice", key="k", payload=request, action=slow_action)
        )
        await started.wait()
        # The retry arrives while the first request is still running
        with pytest.raises(IdempotencyInProgressError):
            await run_idempotent(
                store, client="alice", key="k", payload=request, action=slow_action
            )
        release.set()
        await first
        # Once it has finished, the same key replays the stored response
        replay = await run_idempotent(
            store, client="alice", key="k", payload=request, action=slow_action
        )
        assert replay.headers["idempotent-replay"] == "true"  # type: ignore[union-attr]

    asyncio.run(scenario())

    assert runs == 1


def test_simultaneous_requests_with_one_key_run_exactly_once() -> None:
    store = make_store()
    runs = 0

    async def action() -> ChatResponse:
        nonlocal runs
        runs += 1
        await asyncio.sleep(0.05)
        return ChatResponse(
            conversation_id="5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1",
            model="m",
            message=Message(role="assistant", content="done"),
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    request = ChatRequest(model="m", messages=[Message(role="user", content="hi")])

    async def scenario() -> list[object]:
        return await asyncio.gather(
            *(
                run_idempotent(store, client="alice", key="k", payload=request, action=action)
                for _ in range(10)
            ),
            return_exceptions=True,
        )

    outcomes = asyncio.run(scenario())

    # SET NX lets exactly one through; the other nine are told it is in progress
    assert runs == 1
    assert sum(isinstance(o, IdempotencyInProgressError) for o in outcomes) == 9


def test_mismatch_is_detected_even_while_the_first_request_runs() -> None:
    store = make_store()

    async def scenario() -> None:
        await store.begin("alice", "k", "fingerprint-a")
        outcome = await store.begin("alice", "k", "fingerprint-b")
        assert type(outcome).__name__ == "Mismatch"

    asyncio.run(scenario())


def test_reusing_a_key_for_another_request_raises_the_dedicated_error() -> None:
    store = make_store()
    first = ChatRequest(model="m", messages=[Message(role="user", content="one")])
    second = ChatRequest(model="m", messages=[Message(role="user", content="two")])

    async def action() -> ChatResponse:
        return ChatResponse(
            conversation_id="5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1",
            model="m",
            message=Message(role="assistant", content="x"),
            usage=Usage(input_tokens=1, output_tokens=1),
        )

    async def scenario() -> None:
        await run_idempotent(store, client="a", key="k", payload=first, action=action)
        with pytest.raises(IdempotencyKeyReuseError):
            await run_idempotent(store, client="a", key="k", payload=second, action=action)

    asyncio.run(scenario())


# ---------------------------------------------------------------- storage details


def test_markers_and_responses_expire() -> None:
    server = fakeredis.FakeServer()
    store = make_store(server, ttl=3600, lock=60)

    async def scenario() -> None:
        redis = fakeredis.FakeAsyncRedis(server=server)
        await store.begin("alice", "k", "fp")
        [key] = await redis.keys("llm-chat:idem:alice:*")
        # A running request holds its key only briefly, so a crash cannot block it for long
        assert 0 < await redis.ttl(key) <= 60
        await store.complete("alice", "k", "fp", "{}")
        # A finished one is kept long enough to recognise any realistic retry
        assert 60 < await redis.ttl(key) <= 3600

    asyncio.run(scenario())


def test_the_stored_key_does_not_contain_what_the_client_sent() -> None:
    server = fakeredis.FakeServer()
    store = make_store(server)

    async def scenario() -> list[bytes | str]:
        await store.begin("alice", "user@example.com/order 7", "fp")
        return await fakeredis.FakeAsyncRedis(server=server).keys("*")

    [stored] = asyncio.run(scenario())

    assert b"example.com" not in stored and b"order" not in stored


def test_fingerprint_ignores_key_order_and_spacing() -> None:
    assert fingerprint('{"a": 1, "b": [1, 2]}') == fingerprint('{"b":[1,2],"a":1}')
    assert fingerprint('{"a": 1}') != fingerprint('{"a": 2}')


# ---------------------------------------------------------------- around the edges


def test_redis_outage_runs_requests_unprotected(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    server = fakeredis.FakeServer()
    store = make_store(server)
    server.connected = False
    provider = CannedProvider("a", "b")
    use_provider(provider)
    use_store(store)

    first = client.post("/v1/chat", json=BODY, headers=KEY)
    second = client.post("/v1/chat", json=BODY, headers=KEY)

    # Availability over the guarantee: the requests are served, the retry is not deduplicated
    assert (first.status_code, second.status_code) == (200, 200)
    assert len(provider.calls) == 2


def test_stream_refuses_the_key_instead_of_ignoring_it(client: TestClient) -> None:
    response = client.post("/v1/chat/stream", json=BODY, headers=KEY)

    assert response.status_code == 400
    assert response.json()["error"] == "idempotency_not_supported"


@pytest.mark.parametrize("value", ["", "x" * 256])
def test_malformed_key_is_rejected(client: TestClient, value: str) -> None:
    response = client.post("/v1/chat", json=BODY, headers={"Idempotency-Key": value})

    assert response.status_code == 422


def test_switched_off_idempotency_ignores_the_header(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("a", "b")
    use_provider(provider)
    use_store(None)

    client.post("/v1/chat", json=BODY, headers=KEY)
    client.post("/v1/chat", json=BODY, headers=KEY)

    assert len(provider.calls) == 2


def test_a_replay_still_counts_against_the_rate_limit(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider("a"))
    use_store(make_store())
    redis = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        redis, limit=2, window_seconds=60
    )

    codes = [client.post("/v1/chat", json=BODY, headers=KEY).status_code for _ in range(3)]

    # A retry loop must not be a way around the limit
    assert codes == [200, 200, 429]


def test_replayed_body_is_valid_json_with_the_original_fields(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider("kept"))
    use_store(make_store())

    client.post("/v1/chat", json=BODY, headers=KEY)
    replay = client.post("/v1/chat", json=BODY, headers=KEY)

    body = json.loads(replay.content)
    assert body["message"]["content"] == "kept"
    assert body["usage"]["total_tokens"] == 60
    assert body["cached"] is False
