import asyncio
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app.api.dependencies import get_chat_service
from app.llm.base import Completion, LLMProviderError, StreamChunk
from app.main import app
from app.observability.tracing import configure_tracing, span, use_provider
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.services.cache import ResponseCache
from tests.fakes import (
    ALICE,
    CannedProvider,
    FakeRedis,
    FakeSession,
    ScriptedProvider,
    ServiceFactory,
    StreamingProvider,
    as_redis,
    as_session,
    make_conversation,
)

BODY = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
ANALYSIS = '{"category": "billing", "priority": "low", "summary": "x", "confidence": 0.5}'


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    use_provider(provider)
    yield memory
    use_provider(None)


def by_name(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def one(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    [found] = by_name(exporter, name)
    return found


def attributes(found: ReadableSpan) -> dict[str, Any]:
    return dict(found.attributes or {})


# ---------------------------------------------------------------- the shape of a chat trace


def test_one_request_is_one_trace_with_the_steps_nested_under_the_root(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    client.post("/v1/chat", json=BODY)

    spans = exporter.get_finished_spans()
    assert {s.name for s in spans} == {
        "http.request",
        "prompt.build",
        "llm.request",
        "llm.attempt",
        "db.save",
    }
    assert len({s.context.trace_id for s in spans}) == 1
    root = one(exporter, "http.request")
    assert root.parent is None
    assert one(exporter, "prompt.build").parent.span_id == root.context.span_id  # type: ignore[union-attr]
    assert one(exporter, "db.save").parent.span_id == root.context.span_id  # type: ignore[union-attr]
    request_span = one(exporter, "llm.request")
    assert request_span.parent.span_id == root.context.span_id  # type: ignore[union-attr]
    assert one(exporter, "llm.attempt").parent.span_id == request_span.context.span_id  # type: ignore[union-attr]


def test_loading_an_existing_conversation_is_a_step_of_its_own(
    client: TestClient, exporter: InMemorySpanExporter, session: FakeSession
) -> None:
    conversation = make_conversation(owner="alice")
    session.conversations[conversation.id] = conversation

    client.post("/v1/chat", json=BODY | {"conversation_id": str(conversation.id)})

    assert len(by_name(exporter, "db.load_conversation")) == 1


def test_the_root_span_describes_the_request(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    response = client.post("/v1/chat", json=BODY)

    root = attributes(one(exporter, "http.request"))
    assert root["http.method"] == "POST" and root["http.route"] == "/v1/chat"
    assert root["http.status_code"] == 200
    assert root["request.id"] == response.headers["x-request-id"]
    assert root["client.name"] == "alice"


def test_the_model_span_carries_model_tokens_and_cost(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    client.post("/v1/chat", json=BODY)

    model = attributes(one(exporter, "llm.request"))
    assert model["llm.model"] == "llama3" and model["llm.endpoint"] == "chat"
    assert (model["llm.input_tokens"], model["llm.output_tokens"]) == (3, 2)
    assert model["llm.retry_count"] == 0 and model["llm.status"] == "success"
    assert model["llm.cost_usd"] == pytest.approx(0.000007)
    prompt = attributes(one(exporter, "prompt.build"))
    assert prompt["prompt.messages"] == 1 and prompt["prompt.estimated_tokens"] > 0


def test_a_retry_shows_as_a_failed_attempt_and_an_event(
    client: TestClient, exporter: InMemorySpanExporter, use_provider: Callable[..., None]
) -> None:
    use_provider(ScriptedProvider(LLMProviderError("busy", status_code=503, retryable=True)))

    client.post("/v1/chat", json=BODY)

    attempts = sorted(by_name(exporter, "llm.attempt"), key=lambda s: s.start_time or 0)
    assert [attributes(a)["llm.attempt"] for a in attempts] == [1, 2]
    assert attempts[0].status.status_code == StatusCode.ERROR
    assert attempts[1].status.status_code != StatusCode.ERROR
    request_span = one(exporter, "llm.request")
    assert attributes(request_span)["llm.retry_count"] == 1
    assert [e.name for e in request_span.events] == ["retry"]


def test_time_spent_waiting_for_a_model_slot_is_visible(
    client: TestClient, exporter: InMemorySpanExporter, make_service: ServiceFactory
) -> None:
    class Slow:
        async def complete(
            self,
            model: str,
            messages: list[Message],
            params: GenerationParams,
            *,
            json_schema: dict[str, Any] | None = None,
        ) -> Completion:
            await asyncio.sleep(0.1)
            return Completion("ok", 1, 1, "stop")

        def stream(self, *args: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    service = make_service(Slow(), max_concurrency=1)
    app.dependency_overrides[get_chat_service] = lambda: service

    async def burst() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://t", headers=ALICE
        ) as http:
            await asyncio.gather(*(http.post("/v1/chat", json=BODY) for _ in range(2)))

    asyncio.run(burst())

    waits = sorted(attributes(a)["llm.slot_wait_ms"] for a in by_name(exporter, "llm.attempt"))
    # One request went straight in; the other stood in the queue behind a 100 ms call
    assert waits[0] < 50 and waits[1] >= 80


def test_a_cache_hit_has_a_lookup_span_and_no_model_span(
    client: TestClient, exporter: InMemorySpanExporter, use_provider: Callable[..., None]
) -> None:
    cache = ResponseCache(as_redis(FakeRedis()), ttl_seconds=60, scope="t")
    use_provider(CannedProvider("Paris"), cache=cache)
    body = BODY | {"params": {"temperature": 0}}
    client.post("/v1/chat", json=body)
    exporter.clear()

    client.post("/v1/chat", json=body)

    assert attributes(one(exporter, "cache.lookup"))["cache.hit"] is True
    assert by_name(exporter, "llm.request") == []


def test_validation_attempts_are_spans_and_failures_name_their_kind(
    client: TestClient, exporter: InMemorySpanExporter, use_provider: Callable[..., None]
) -> None:
    bad = '{"category": "billing", "priority": "low", "summary": "x", "confidence": 150}'
    use_provider(CannedProvider(bad, ANALYSIS))

    client.post("/v1/chat/structured", json={"model": "llama3", "text": "Не могу войти"})

    checks = sorted(by_name(exporter, "output.validate"), key=lambda s: s.start_time or 0)
    assert [attributes(c)["validate.attempt"] for c in checks] == [1, 2]
    assert attributes(checks[0])["validate.failure_kind"] == "schema_mismatch"
    assert checks[0].status.status_code == StatusCode.ERROR
    assert checks[1].status.status_code != StatusCode.ERROR
    # Two generations, one model span each, and the ledger write is a step of its own
    assert len(by_name(exporter, "llm.request")) == 2
    assert len(by_name(exporter, "db.save_usage")) == 1


def test_a_stream_is_one_span_with_time_to_first_piece(
    client: TestClient, exporter: InMemorySpanExporter, use_provider: Callable[..., None]
) -> None:
    use_provider(
        StreamingProvider(
            StreamChunk(content="Par"),
            StreamChunk(content="is"),
            StreamChunk(finish_reason="stop"),
            StreamChunk(input_tokens=7, output_tokens=2),
        )
    )

    client.post("/v1/chat/stream", json=BODY)

    stream = one(exporter, "llm.stream")
    assert attributes(stream)["llm.pieces"] == 2 and attributes(stream)["llm.ttft_ms"] >= 0
    assert stream.end_time is not None
    assert stream.parent.span_id == one(exporter, "http.request").context.span_id  # type: ignore[union-attr]
    assert len(by_name(exporter, "db.save")) == 1


def test_a_stream_span_is_closed_when_the_client_disconnects(
    make_service: ServiceFactory, exporter: InMemorySpanExporter
) -> None:
    provider = StreamingProvider(StreamChunk(content="a"), StreamChunk(content="b"), 5.0)
    service = make_service(provider, timeout_seconds=10.0)
    request = ChatRequest(model="llama3", messages=[Message(role="user", content="hi")])

    async def scenario() -> None:
        stream = service.stream_reply(request, as_session(FakeSession()), owner="alice")
        await anext(stream)
        await stream.aclose()

    asyncio.run(scenario())

    # An unfinished span would never be exported: the abandoned request would vanish from tracing
    assert one(exporter, "llm.stream").end_time is not None


# ---------------------------------------------------------------- joining other traces


def test_a_traceparent_header_continues_the_callers_trace(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    trace_id, parent_id = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"

    client.get("/health", headers={"traceparent": f"00-{trace_id}-{parent_id}-01"})

    root = one(exporter, "http.request")
    assert f"{root.context.trace_id:032x}" == trace_id
    assert root.parent is not None and f"{root.parent.span_id:016x}" == parent_id


def test_a_log_line_names_the_trace_it_belongs_to(
    client: TestClient, exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    client.post("/v1/chat", json=BODY)

    root = one(exporter, "http.request")
    ids = {getattr(r, "trace_id", None) for r in caplog.records if r.name == "app.llm"}
    assert ids == {f"{root.context.trace_id:032x}"}


def test_a_server_error_marks_the_root_span_as_failed(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    def broken() -> None:
        raise RuntimeError("TOP-SECRET internal detail")

    app.dependency_overrides[get_chat_service] = broken

    client.post("/v1/chat", json=BODY)

    root = one(exporter, "http.request")
    assert root.status.status_code == StatusCode.ERROR
    assert attributes(root)["http.status_code"] == 500


# ---------------------------------------------------------------- what never goes into a trace


def test_prompts_replies_and_keys_never_appear_in_a_span(
    client: TestClient, exporter: InMemorySpanExporter, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider("REPLY-SECRET-7731"))
    prompt = {"model": "llama3", "messages": [{"role": "user", "content": "PROMPT-SECRET-4419"}]}

    client.post("/v1/chat", json=prompt, headers={"Idempotency-Key": "IDEM-SECRET-2208"})
    client.post("/v1/chat/stream", json=prompt)
    client.post("/v1/chat", json=prompt, headers={"X-API-Key": "KEY-SECRET-9053"})

    finished = exporter.get_finished_spans()
    everything = repr(
        [
            (
                s.name,
                dict(s.attributes or {}),
                [(e.name, dict(e.attributes or {})) for e in s.events],
                s.status.description,
            )
            for s in finished
        ]
    )
    assert exporter.get_finished_spans()
    for secret in (
        "PROMPT-SECRET-4419",
        "REPLY-SECRET-7731",
        "IDEM-SECRET-2208",
        "KEY-SECRET-9053",
    ):
        assert secret not in everything


def test_a_failed_span_records_the_error_type_not_the_message(
    exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(RuntimeError), span("step"):
        raise RuntimeError("TOP-SECRET internal detail")

    failed = one(exporter, "step")
    assert failed.status.status_code == StatusCode.ERROR
    assert failed.status.description == "RuntimeError"
    assert "TOP-SECRET" not in repr(failed.events) and "TOP-SECRET" not in repr(failed.attributes)


# ---------------------------------------------------------------- configuration


def test_tracing_is_off_by_default_and_requests_still_work(client: TestClient) -> None:
    use_provider(None)

    assert client.post("/v1/chat", json=BODY).status_code == 200


def test_disabled_configuration_installs_no_provider() -> None:
    assert configure_tracing(enabled=False) is None


def test_a_sampling_ratio_of_zero_traces_nothing(client: TestClient) -> None:
    provider = configure_tracing(enabled=True, console=False, sample_ratio=0.0)
    assert provider is not None
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    try:
        assert client.post("/v1/chat", json=BODY).status_code == 200
        assert memory.get_finished_spans() == ()
    finally:
        use_provider(None)
        provider.shutdown()


def test_a_sampled_callers_decision_is_respected_even_at_ratio_zero(client: TestClient) -> None:
    provider = configure_tracing(enabled=True, console=False, sample_ratio=0.0)
    assert provider is not None
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    try:
        client.get(
            "/health",
            headers={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        )
        # The trace was already sampled upstream: cutting it here would leave a hole in it
        assert len(memory.get_finished_spans()) == 1
    finally:
        use_provider(None)
        provider.shutdown()
