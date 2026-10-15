import asyncio
import json
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMProviderError, StreamChunk
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.schemas.stream import StreamDelta, StreamDone, StreamError
from app.services.cache import ResponseCache
from tests.fakes import (
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
GOOD = [
    StreamChunk(content="Par"),
    StreamChunk(content="is"),
    StreamChunk(finish_reason="stop"),
    StreamChunk(input_tokens=7, output_tokens=2),
]


def parse_sse(text: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        assert name.startswith("event: ") and data.startswith("data: ")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def stream_request(**params: object) -> ChatRequest:
    return ChatRequest(
        model="llama3",
        messages=[Message(role="user", content="hi")],
        params=GenerationParams.model_validate(params),
    )


def collect(service_stream) -> list[object]:  # type: ignore[no-untyped-def]
    async def run() -> list[object]:
        return [event async for event in service_stream]

    return asyncio.run(run())


def test_reply_arrives_as_pieces_then_a_closing_event(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(StreamingProvider(*GOOD))

    response = client.post("/v1/chat/stream", json=BODY)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["delta", "delta", "done"]
    assert "".join(str(data["content"]) for name, data in events if name == "delta") == "Paris"
    done = events[-1][1]
    assert done["usage"] == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}
    assert done["finish_reason"] == "stop"
    assert done["cached"] is False
    assert 0 <= float(str(done["ttft_ms"])) <= float(str(done["duration_ms"]))


def test_finished_stream_is_stored_like_a_normal_exchange(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    use_provider(StreamingProvider(*GOOD))

    client.post("/v1/chat/stream", json=BODY)

    assert [(m.role, m.content) for m in session.messages] == [
        ("user", "hi"),
        ("assistant", "Paris"),
    ]
    [usage] = session.usage_logs
    assert (usage.endpoint, usage.input_tokens, usage.output_tokens) == ("stream", 7, 2)
    assert session.commits == 1


def test_unknown_conversation_is_a_404_before_any_text(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = StreamingProvider(*GOOD)
    use_provider(provider)

    response = client.post(
        "/v1/chat/stream", json=BODY | {"conversation_id": "5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1"}
    )

    assert response.status_code == 404
    assert provider.opened == 0


def test_oversized_prompt_is_a_400_before_any_text(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = StreamingProvider(*GOOD)
    use_provider(provider, context_limit=50)

    response = client.post(
        "/v1/chat/stream",
        json={"model": "llama3", "messages": [{"role": "user", "content": "x" * 500}]},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "context_length_exceeded"
    assert provider.opened == 0


def test_provider_failure_before_the_first_piece_is_a_normal_http_error(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(StreamingProvider(LLMProviderError("bad request", status_code=400)))

    response = client.post("/v1/chat/stream", json=BODY)

    assert response.status_code == 502
    assert response.json()["error"] == "llm_provider_error"


def test_failure_before_the_first_piece_is_retried(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = ScriptedProvider(LLMProviderError("unavailable", status_code=503, retryable=True))
    use_provider(provider)

    response = client.post("/v1/chat/stream", json=BODY)

    assert response.status_code == 200
    assert provider.calls == 2
    assert "recovered" in response.text


def test_silence_before_the_first_piece_is_a_504(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(StreamingProvider(5.0), timeout_seconds=0.05, max_retries=0)

    response = client.post("/v1/chat/stream", json=BODY)

    assert response.status_code == 504
    assert response.json()["error"] == "llm_timeout"


def test_failure_after_text_started_becomes_an_error_event_and_stores_nothing(
    make_service: ServiceFactory, session: FakeSession
) -> None:
    provider = StreamingProvider(
        StreamChunk(content="Par"), LLMProviderError("connection reset", retryable=True)
    )
    service = make_service(provider)

    events = collect(service.stream_reply(stream_request(), as_session(session), owner="alice"))

    assert isinstance(events[0], StreamDelta)
    assert isinstance(events[-1], StreamError)
    assert events[-1].error == "llm_provider_error"
    # Text already reached the client, so starting over would send it twice: no retry
    assert provider.opened == 1
    assert session.messages == [] and session.usage_logs == []


def test_silence_after_text_started_becomes_a_timeout_event(
    make_service: ServiceFactory, session: FakeSession
) -> None:
    service = make_service(StreamingProvider(StreamChunk(content="Par"), 5.0), timeout_seconds=0.05)

    events = collect(service.stream_reply(stream_request(), as_session(session), owner="alice"))

    assert isinstance(events[-1], StreamError) and events[-1].error == "llm_timeout"


def test_client_disconnect_closes_the_generation_and_frees_the_slot(
    make_service: ServiceFactory, session: FakeSession
) -> None:
    provider = StreamingProvider(StreamChunk(content="a"), StreamChunk(content="b"), 5.0)
    service = make_service(provider, max_concurrency=1, timeout_seconds=1.0)

    async def scenario() -> None:
        stream = service.stream_reply(stream_request(), as_session(session), owner="alice")
        first = await anext(stream)
        assert isinstance(first, StreamDelta)
        await stream.aclose()  # what the server does when the client goes away

        assert provider.closed == 1
        # The only slot is free again: a second stream can start at once
        again = service.stream_reply(stream_request(), as_session(session), owner="alice")
        assert isinstance(await asyncio.wait_for(anext(again), 0.5), StreamDelta)
        await again.aclose()

    asyncio.run(scenario())

    # A half-delivered reply is not an exchange
    assert session.messages == [] and session.usage_logs == []


def test_cache_hit_is_replayed_as_one_piece_without_the_model(
    make_service: ServiceFactory,
) -> None:
    redis = FakeRedis()
    cache = ResponseCache(as_redis(redis), ttl_seconds=60, scope="t")
    provider = CannedProvider("Paris")
    service = make_service(provider, cache=cache)
    request = stream_request(temperature=0)

    first = collect(service.stream_reply(request, as_session(FakeSession()), owner="alice"))
    second = collect(service.stream_reply(request, as_session(FakeSession()), owner="alice"))

    assert len(provider.calls) == 1
    assert isinstance(first[-1], StreamDone) and first[-1].cached is False
    assert [type(e) for e in second] == [StreamDelta, StreamDone]
    assert isinstance(second[-1], StreamDone) and second[-1].cached is True
    assert isinstance(second[0], StreamDelta) and second[0].content == "Paris"


def test_sampled_stream_is_not_cached(make_service: ServiceFactory) -> None:
    redis = FakeRedis()
    service = make_service(
        CannedProvider("a", "b"), cache=ResponseCache(as_redis(redis), ttl_seconds=60, scope="t")
    )

    collect(
        service.stream_reply(
            stream_request(temperature=0.7), as_session(FakeSession()), owner="alice"
        )
    )

    assert redis.commands == 0


def test_missing_usage_falls_back_to_an_estimate(
    make_service: ServiceFactory, session: FakeSession
) -> None:
    service = make_service(
        StreamingProvider(StreamChunk(content="Paris"), StreamChunk(finish_reason="stop"))
    )

    events = collect(service.stream_reply(stream_request(), as_session(session), owner="alice"))

    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.usage.input_tokens > 0 and done.usage.output_tokens > 0


def test_existing_conversation_is_appended_to(
    make_service: ServiceFactory,
) -> None:
    conversation = make_conversation()
    session = FakeSession({conversation.id: conversation})
    service = make_service(StreamingProvider(*GOOD))
    request = ChatRequest(
        model="llama3",
        conversation_id=conversation.id,
        messages=[Message(role="user", content="hi")],
    )

    events = collect(service.stream_reply(request, as_session(session), owner="alice"))

    done = events[-1]
    assert isinstance(done, StreamDone) and done.conversation_id == conversation.id


def test_storage_failure_after_the_text_is_reported_inside_the_stream(
    make_service: ServiceFactory, session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_commit() -> None:
        raise ConnectionRefusedError("database is down")

    monkeypatch.setattr(session, "commit", broken_commit)
    service = make_service(StreamingProvider(*GOOD))

    events = collect(service.stream_reply(stream_request(), as_session(session), owner="alice"))

    assert [type(e) for e in events] == [StreamDelta, StreamDelta, StreamError]
    assert isinstance(events[-1], StreamError) and events[-1].error == "storage_unavailable"
    # The failed transaction was rolled back, not left open
    assert session.rollbacks >= 1
