import asyncio
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_response_cache
from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse, GenerationParams, Message
from app.services.cache import (
    COOLDOWN_SECONDS,
    HITS_KEY,
    MISSES_KEY,
    ResponseCache,
    is_cacheable,
)
from app.services.chat import ChatService
from tests.fakes import (
    CannedProvider,
    FakeRedis,
    FakeSession,
    ServiceFactory,
    as_redis,
    as_session,
)

TTL = 600
GREEDY = GenerationParams(temperature=0, max_tokens=100)


def make_cache(redis: FakeRedis, *, scope: str = "test") -> ResponseCache:
    return ResponseCache(as_redis(redis), ttl_seconds=TTL, scope=scope)


def request(content: str = "Столица Франции?", **params: object) -> ChatRequest:
    return ChatRequest(
        model="llama3",
        messages=[Message(role="user", content=content)],
        params=GenerationParams.model_validate({"temperature": 0, "max_tokens": 100} | params),
    )


def ask(service: ChatService, chat_request: ChatRequest, session: FakeSession) -> ChatResponse:
    return asyncio.run(service.generate_reply(chat_request, as_session(session)))


def cache_entries(redis: FakeRedis) -> list[str]:
    return [key for key in redis.data if key.startswith("llm-chat:cache:")]


def test_only_greedy_decoding_is_cacheable() -> None:
    assert is_cacheable(GenerationParams(temperature=0))
    assert not is_cacheable(GenerationParams(temperature=0.1))
    assert not is_cacheable(GenerationParams())


def test_equal_requests_get_equal_keys() -> None:
    cache = make_cache(FakeRedis())
    first = [Message(role="user", content="Привет")]
    # Validation strips surrounding whitespace, so both requests reach the model identically
    second = [Message(role="user", content="  Привет \n")]

    assert cache.key_for("llama3", first, GREEDY) == cache.key_for("llama3", second, GREEDY)


def test_every_input_that_shapes_the_reply_changes_the_key() -> None:
    cache = make_cache(FakeRedis())
    user = Message(role="user", content="Привет")
    base = cache.key_for("llama3", [user], GREEDY)

    variants = [
        cache.key_for("mistral", [user], GREEDY),
        cache.key_for("llama3", [Message(role="user", content="Привет!")], GREEDY),
        cache.key_for("llama3", [Message(role="assistant", content="Привет")], GREEDY),
        cache.key_for("llama3", [Message(role="system", content="Отвечай кратко"), user], GREEDY),
        cache.key_for("llama3", [user], GenerationParams(temperature=0, max_tokens=101)),
        make_cache(FakeRedis(), scope="other").key_for("llama3", [user], GREEDY),
    ]

    assert len({base, *variants}) == len(variants) + 1
    assert base.startswith("llm-chat:cache:v1:")


def test_repeated_request_is_served_from_cache(make_service: ServiceFactory) -> None:
    redis = FakeRedis()
    provider = CannedProvider("Париж")
    service = make_service(provider, cache=make_cache(redis))
    session = FakeSession()

    first = ask(service, request(), session)
    second = ask(service, request(), session)

    assert len(provider.calls) == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.message.content == "Париж"
    # Usage describes the generation that produced the reply
    assert second.usage == first.usage
    # Both exchanges are part of the conversation history, but only one was paid for
    assert len(session.messages) == 4
    assert len(session.usage_logs) == 1


def test_entry_is_stored_with_ttl(make_service: ServiceFactory) -> None:
    redis = FakeRedis()
    service = make_service(CannedProvider("Париж"), cache=make_cache(redis))

    ask(service, request(), FakeSession())

    [key] = cache_entries(redis)
    assert redis.ttls[key] == TTL


def test_sampled_request_bypasses_cache(make_service: ServiceFactory) -> None:
    redis = FakeRedis()
    provider = CannedProvider("Париж", "Конечно, Париж")
    service = make_service(provider, cache=make_cache(redis))

    ask(service, request(temperature=0.7), FakeSession())
    second = ask(service, request(temperature=0.7), FakeSession())

    assert len(provider.calls) == 2
    assert second.cached is False
    assert redis.commands == 0


def test_truncated_reply_is_not_cached(make_service: ServiceFactory) -> None:
    redis = FakeRedis()
    service = make_service(
        CannedProvider("Пари", "Пари", finish_reason="length"), cache=make_cache(redis)
    )

    ask(service, request(), FakeSession())
    second = ask(service, request(), FakeSession())

    assert cache_entries(redis) == []
    assert second.cached is False


def test_unreadable_entry_counts_as_miss(make_service: ServiceFactory) -> None:
    redis = FakeRedis()
    cache = make_cache(redis)
    service = make_service(CannedProvider("Париж"), cache=cache)
    key = cache.key_for("llama3", request().messages, GREEDY)
    redis.data[key] = b"{not json"

    response = ask(service, request(), FakeSession())

    assert response.cached is False
    assert response.message.content == "Париж"


def test_redis_outage_falls_back_to_model_and_pauses_cache(
    make_service: ServiceFactory,
) -> None:
    redis = FakeRedis(down=True)
    provider = CannedProvider("Париж", "Париж")
    service = make_service(provider, cache=make_cache(redis))

    first = ask(service, request(), FakeSession())
    second = ask(service, request(), FakeSession())

    assert first.message.content == "Париж"
    assert second.cached is False
    assert len(provider.calls) == 2
    # One failed lookup, then the cache is skipped instead of waiting on Redis each time
    assert redis.commands == 1


def test_cache_is_retried_after_cooldown(
    make_service: ServiceFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr("app.services.cache.time.monotonic", lambda: clock[0])
    redis = FakeRedis(down=True)
    provider = CannedProvider("Париж", "Париж", "Париж")
    service = make_service(provider, cache=make_cache(redis))

    ask(service, request(), FakeSession())
    redis.down = False
    clock[0] += COOLDOWN_SECONDS + 1
    ask(service, request(), FakeSession())
    third = ask(service, request(), FakeSession())

    assert third.cached is True
    assert len(provider.calls) == 2


def test_hits_and_misses_are_counted(make_service: ServiceFactory) -> None:
    redis = FakeRedis()
    cache = make_cache(redis)
    service = make_service(CannedProvider("Париж", "Берлин"), cache=cache)

    ask(service, request(), FakeSession())
    ask(service, request(), FakeSession())
    ask(service, request("Столица Германии?"), FakeSession())

    assert (redis.data[HITS_KEY], redis.data[MISSES_KEY]) == (b"1", b"2")
    stats = asyncio.run(cache.stats())
    assert (stats.hits, stats.misses, stats.hit_rate) == (1, 2, 0.3333)


def test_chat_endpoint_marks_cached_reply(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider("Париж"), cache=make_cache(FakeRedis()))
    body = {
        "model": "llama3",
        "messages": [{"role": "user", "content": "Столица Франции?"}],
        "params": {"temperature": 0},
    }

    first = client.post("/v1/chat", json=body)
    second = client.post("/v1/chat", json=body)

    assert (first.json()["cached"], second.json()["cached"]) == (False, True)
    assert second.json()["message"]["content"] == "Париж"


@pytest.mark.parametrize(
    ("counters", "expected"),
    [
        ({}, {"enabled": True, "hits": 0, "misses": 0, "hit_rate": None}),
        (
            {HITS_KEY: b"3", MISSES_KEY: b"1"},
            {"enabled": True, "hits": 3, "misses": 1, "hit_rate": 0.75},
        ),
    ],
)
def test_stats_endpoint(
    client: TestClient, counters: dict[str, bytes], expected: dict[str, object]
) -> None:
    redis = FakeRedis()
    redis.data.update(counters)
    app.dependency_overrides[get_response_cache] = lambda: make_cache(redis)

    response = client.get("/v1/cache/stats")

    assert response.status_code == 200
    assert response.json() == expected


def test_stats_endpoint_when_cache_is_disabled(client: TestClient) -> None:
    app.dependency_overrides[get_response_cache] = lambda: None

    response = client.get("/v1/cache/stats")

    assert response.json() == {"enabled": False, "hits": 0, "misses": 0, "hit_rate": None}


def test_stats_endpoint_returns_503_when_redis_is_down(client: TestClient) -> None:
    app.dependency_overrides[get_response_cache] = lambda: make_cache(FakeRedis(down=True))

    response = client.get("/v1/cache/stats")

    assert response.status_code == 503
    assert response.json()["error"] == "redis_unavailable"
