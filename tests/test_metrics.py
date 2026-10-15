import asyncio
from collections.abc import Callable
from typing import Any

import fakeredis
import httpx2
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.api.dependencies import get_chat_service, get_rate_limiter
from app.llm.base import Completion, LLMProviderError, StreamChunk
from app.main import app
from app.observability import metrics
from app.schemas.chat import GenerationParams, Message
from app.services.cache import ResponseCache
from app.services.ratelimit import RateLimiter
from tests.fakes import (
    ALICE,
    BOB,
    OPS,
    ROOT,
    CannedProvider,
    FakeRedis,
    ScriptedProvider,
    ServiceFactory,
    StreamingProvider,
    as_redis,
)

BODY = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
ANALYSIS = '{"category": "billing", "priority": "low", "summary": "x", "confidence": 0.5}'


def value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def known_model(client: TestClient) -> None:
    # The lifespan allows the configured model names; the tests use another one
    metrics.allow_models({"llama3"})


# ---------------------------------------------------------------- HTTP


def test_requests_are_counted_by_route_template_and_status(client: TestClient) -> None:
    labels = {"method": "GET", "path": "/v1/conversations/{conversation_id}", "status": "404"}
    before = value("http_requests_total", **labels)

    client.get("/v1/conversations/5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1")
    client.get("/v1/conversations/6e3dec63-8f10-4b33-b4d1-7f1ee4eae9f2")

    # Two different ids, one series
    assert value("http_requests_total", **labels) == before + 2


def test_unknown_paths_share_one_series(client: TestClient) -> None:
    labels = {"method": "GET", "path": "unmatched", "status": "404"}
    before = value("http_requests_total", **labels)

    for path in ("/wp-login.php", "/.env", "/admin", "/a/b/c"):
        client.get(path)

    assert value("http_requests_total", **labels) == before + 4


def test_duration_is_recorded_in_a_histogram(client: TestClient) -> None:
    labels = {"method": "GET", "path": "/health"}
    before = value("http_request_duration_seconds_count", **labels)

    client.get("/health")

    assert value("http_request_duration_seconds_count", **labels) == before + 1
    assert value("http_request_duration_seconds_sum", **labels) > 0
    # Cumulative buckets: a fast request is inside every bucket from some point on
    assert value("http_request_duration_seconds_bucket", le="+Inf", **labels) >= before + 1


def test_in_flight_gauge_returns_to_where_it_started(client: TestClient) -> None:
    before = value("http_requests_in_flight")

    client.get("/health")
    client.post("/v1/chat", json=BODY)
    client.post("/v1/chat", json={"bad": 1})

    assert value("http_requests_in_flight") == before


# ---------------------------------------------------------------- the model


def test_a_successful_call_counts_requests_tokens_cost_and_latency(
    client: TestClient, known_model: None
) -> None:
    ok = {"model": "llama3", "endpoint": "chat", "status": "success"}
    before = {
        "requests": value("llm_requests_total", **ok),
        "input": value("llm_tokens_total", model="llama3", direction="input"),
        "output": value("llm_tokens_total", model="llama3", direction="output"),
        "cost": value("llm_cost_usd_total", model="llama3"),
        "latency": value("llm_latency_seconds_count", model="llama3", endpoint="chat"),
    }

    client.post("/v1/chat", json=BODY)

    assert value("llm_requests_total", **ok) == before["requests"] + 1
    assert value("llm_tokens_total", model="llama3", direction="input") == before["input"] + 3
    assert value("llm_tokens_total", model="llama3", direction="output") == before["output"] + 2
    assert value("llm_cost_usd_total", model="llama3") == pytest.approx(before["cost"] + 0.000007)
    assert (
        value("llm_latency_seconds_count", model="llama3", endpoint="chat") == before["latency"] + 1
    )


def test_errors_and_timeouts_are_counted_separately(
    client: TestClient, use_provider: Callable[..., None], known_model: None
) -> None:
    provider_error = value("llm_errors_total", model="llama3", kind="provider_error")
    use_provider(ScriptedProvider(LLMProviderError("bad", status_code=400)))

    client.post("/v1/chat", json=BODY)

    assert value("llm_errors_total", model="llama3", kind="provider_error") == provider_error + 1
    assert value("llm_requests_total", model="llama3", endpoint="chat", status="error") >= 1


def test_a_timeout_is_a_kind_of_its_own(
    client: TestClient, use_provider: Callable[..., None], known_model: None
) -> None:
    class Slow(CannedProvider):
        async def complete(self, *args: Any, **kwargs: Any) -> Completion:
            await asyncio.sleep(1)
            raise AssertionError("the timeout should have fired first")

    before = value("llm_errors_total", model="llama3", kind="timeout")
    use_provider(Slow(), timeout_seconds=0.01, max_retries=0)

    client.post("/v1/chat", json=BODY)

    assert value("llm_errors_total", model="llama3", kind="timeout") == before + 1


def test_retries_are_counted(
    client: TestClient, use_provider: Callable[..., None], known_model: None
) -> None:
    before = value("llm_retries_total", model="llama3")
    use_provider(ScriptedProvider(LLMProviderError("busy", status_code=503, retryable=True)))

    client.post("/v1/chat", json=BODY)

    assert value("llm_retries_total", model="llama3") == before + 1


def test_a_cached_reply_is_counted_but_does_not_count_as_model_latency_or_tokens(
    client: TestClient, use_provider: Callable[..., None], known_model: None
) -> None:
    cache = ResponseCache(as_redis(FakeRedis()), ttl_seconds=60, scope="t")
    use_provider(CannedProvider("Paris"), cache=cache)
    body = BODY | {"params": {"temperature": 0}}
    client.post("/v1/chat", json=body)
    hits = value("cache_hits_total")
    misses = value("cache_misses_total")
    latency = value("llm_latency_seconds_count", model="llama3", endpoint="chat")
    tokens = value("llm_tokens_total", model="llama3", direction="input")

    client.post("/v1/chat", json=body)

    assert value("cache_hits_total") == hits + 1
    assert value("cache_misses_total") == misses
    assert value("llm_requests_total", model="llama3", endpoint="chat", status="cached") >= 1
    # A hit did not call the model: counting it would drag the percentiles down and bill nothing
    assert value("llm_latency_seconds_count", model="llama3", endpoint="chat") == latency
    assert value("llm_tokens_total", model="llama3", direction="input") == tokens


def test_cache_misses_and_errors_are_counted(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    cache = ResponseCache(as_redis(FakeRedis()), ttl_seconds=60, scope="t")
    use_provider(CannedProvider("a"), cache=cache)
    misses = value("cache_misses_total")

    client.post("/v1/chat", json=BODY | {"params": {"temperature": 0}})

    assert value("cache_misses_total") == misses + 1
    errors = value("cache_errors_total")
    use_provider(
        CannedProvider("b"),
        cache=ResponseCache(as_redis(FakeRedis(down=True)), ttl_seconds=60, scope="t"),
    )
    client.post("/v1/chat", json=BODY | {"params": {"temperature": 0}})
    assert value("cache_errors_total") > errors


def test_time_to_first_piece_is_observed_for_streams(
    client: TestClient, use_provider: Callable[..., None], known_model: None
) -> None:
    before = value("llm_ttft_seconds_count", model="llama3")
    use_provider(
        StreamingProvider(
            StreamChunk(content="Par"),
            StreamChunk(content="is"),
            StreamChunk(finish_reason="stop"),
            StreamChunk(input_tokens=7, output_tokens=2),
        )
    )

    client.post("/v1/chat/stream", json=BODY)

    assert value("llm_ttft_seconds_count", model="llama3") == before + 1
    assert value("llm_requests_total", model="llama3", endpoint="stream", status="success") >= 1


def test_structured_failures_are_counted_by_kind_and_finality(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    bad = '{"category": "billing", "priority": "low", "summary": "x", "confidence": 150}'
    retried = value("structured_output_failures_total", kind="schema_mismatch", final="false")
    gave_up = value("structured_output_failures_total", kind="schema_mismatch", final="true")
    use_provider(CannedProvider(bad, bad))

    response = client.post("/v1/chat/structured", json={"model": "llama3", "text": "Не могу войти"})

    assert response.status_code == 502
    # The first rejection led to a retry, the second ended the request
    assert (
        value("structured_output_failures_total", kind="schema_mismatch", final="false")
        == retried + 1
    )
    assert (
        value("structured_output_failures_total", kind="schema_mismatch", final="true")
        == gave_up + 1
    )


def test_truncated_output_counts_as_a_final_failure_at_once(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    before = value("structured_output_failures_total", kind="truncated", final="true")
    use_provider(CannedProvider('{"category": "bil', finish_reason="length"))

    client.post("/v1/chat/structured", json={"model": "llama3", "text": "Не могу войти"})

    assert value("structured_output_failures_total", kind="truncated", final="true") == before + 1


def test_the_in_flight_gauge_shows_calls_running_at_once(
    client: TestClient, make_service: ServiceFactory
) -> None:
    seen: list[float] = []

    class Watching:
        async def complete(
            self,
            model: str,
            messages: list[Message],
            params: GenerationParams,
            *,
            json_schema: dict[str, Any] | None = None,
        ) -> Completion:
            await asyncio.sleep(0.02)
            seen.append(value("llm_calls_in_flight"))
            await asyncio.sleep(0.02)
            return Completion("ok", 1, 1, "stop")

        def stream(self, *args: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    service = make_service(Watching(), max_concurrency=3)
    app.dependency_overrides[get_chat_service] = lambda: service

    async def burst() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://t", headers=ALICE
        ) as http:
            await asyncio.gather(*(http.post("/v1/chat", json=BODY) for _ in range(9)))

    before = value("llm_calls_in_flight")
    asyncio.run(burst())

    # Nine requests, a limit of three: the gauge rose to the limit, never above, and fell back
    assert max(seen) == before + 3
    assert value("llm_calls_in_flight") == before


# ---------------------------------------------------------------- protection and extras


def test_rate_limit_rejections_are_counted(client: TestClient) -> None:
    redis = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(
        redis, limit=1, window_seconds=60
    )
    before = value("rate_limit_rejections_total")

    client.post("/v1/chat", json=BODY)
    client.post("/v1/chat", json=BODY)

    assert value("rate_limit_rejections_total") == before + 1


def test_authentication_failures_are_counted_by_reason(client: TestClient) -> None:
    invalid = value("auth_failures_total", reason="invalid_key")
    scope = value("auth_failures_total", reason="insufficient_scope")

    client.post("/v1/chat", json=BODY, headers={"X-API-Key": "nope"})
    client.get("/v1/usage", headers=ALICE)

    assert value("auth_failures_total", reason="invalid_key") == invalid + 1
    assert value("auth_failures_total", reason="insufficient_scope") == scope + 1


def test_unknown_model_names_do_not_create_new_series(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    metrics.allow_models({"llama3"})
    use_provider(CannedProvider("a", "b", "c"))
    other = value("llm_requests_total", model="other", endpoint="chat", status="success")

    for name in ("made-up-1", "made-up-2", "made-up-3"):
        client.post("/v1/chat", json={"model": name, "messages": BODY["messages"]})

    # The model name comes from the client; used as a label it would let anyone mint series
    assert (
        value("llm_requests_total", model="other", endpoint="chat", status="success") == other + 3
    )
    assert value("llm_requests_total", model="made-up-1", endpoint="chat", status="success") == 0


# ---------------------------------------------------------------- the endpoint


def test_metrics_endpoint_needs_the_admin_scope(client: TestClient) -> None:
    assert client.get("/metrics", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/metrics", headers=ALICE).status_code == 403
    assert client.get("/metrics", headers=BOB).status_code == 403
    assert client.get("/metrics", headers=ROOT).status_code == 200
    assert client.get("/metrics", headers=OPS).status_code == 200


def test_a_scraper_can_authenticate_with_a_bearer_token(client: TestClient) -> None:
    key = ROOT["X-API-Key"]

    # The test client sends alice's X-API-Key by default; blank it so the Bearer token is used
    response = client.get("/metrics", headers={"X-API-Key": "", "Authorization": f"Bearer {key}"})

    assert response.status_code == 200


def test_the_endpoint_exposes_the_prometheus_text_format_with_the_planned_metrics(
    client: TestClient,
) -> None:
    client.post("/v1/chat", json=BODY)

    response = client.get("/metrics", headers=ROOT)

    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    for name in (
        "http_requests_total",
        "http_request_duration_seconds_bucket",
        "llm_requests_total",
        "llm_latency_seconds_bucket",
        "llm_errors_total",
        "llm_tokens_total",
        "llm_cost_usd_total",
        "structured_output_failures_total",
        "cache_hits_total",
    ):
        assert f"# TYPE {name.removesuffix('_bucket')}" in text, name
    assert "# TYPE http_requests_in_flight gauge" in text
    assert "# TYPE llm_latency_seconds histogram" in text
    assert "# TYPE llm_requests_total counter" in text


def test_metrics_never_contain_prompts_keys_or_client_names(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider("REPLY-SECRET-7731"))
    client.post(
        "/v1/chat",
        json={"model": "llama3", "messages": [{"role": "user", "content": "PROMPT-SECRET-4419"}]},
        headers={"Idempotency-Key": "IDEM-SECRET-2208"},
    )

    text = client.get("/metrics", headers=ROOT).text

    for secret in (
        "PROMPT-SECRET-4419",
        "REPLY-SECRET-7731",
        "IDEM-SECRET-2208",
        ROOT["X-API-Key"],
    ):
        assert secret not in text
    assert 'client="alice"' not in text
