import asyncio
import random

import pytest

from app.llm.base import Completion, LLMProviderError, LLMTimeoutError
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.services.chat import ChatService
from tests.fakes import ScriptedProvider, ServiceFactory

REQUEST = ChatRequest(model="fake", messages=[Message(role="user", content="hi")])


def transient(status: int) -> LLMProviderError:
    return LLMProviderError(f"HTTP {status}", status_code=status, retryable=True)


class InFlightCounter:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.in_flight -= 1
        return Completion(content="ok", input_tokens=1, output_tokens=1)


class SlowThenFastProvider:
    """Hangs on the first `slow_calls` calls, then answers immediately."""

    def __init__(self, slow_calls: int) -> None:
        self._slow_calls = slow_calls
        self.calls = 0

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        self.calls += 1
        if self.calls <= self._slow_calls:
            await asyncio.sleep(10)
        return Completion(content="fast", input_tokens=1, output_tokens=1)


def test_success_on_first_attempt(make_service: ServiceFactory) -> None:
    provider = ScriptedProvider()

    response = asyncio.run(make_service(provider).generate_reply(REQUEST))

    assert response.message.content == "recovered"
    assert response.usage.input_tokens == 1
    assert provider.calls == 1


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_transient_error_is_retried_until_success(
    make_service: ServiceFactory, status: int
) -> None:
    provider = ScriptedProvider(transient(status), transient(status))

    response = asyncio.run(make_service(provider, max_retries=2).generate_reply(REQUEST))

    assert response.message.content == "recovered"
    assert provider.calls == 3


@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_error_is_not_retried(make_service: ServiceFactory, status: int) -> None:
    provider = ScriptedProvider(LLMProviderError(f"HTTP {status}", status_code=status))

    with pytest.raises(LLMProviderError) as exc_info:
        asyncio.run(make_service(provider, max_retries=2).generate_reply(REQUEST))

    assert exc_info.value.status_code == status
    assert provider.calls == 1


def test_retries_are_bounded_and_last_error_is_preserved(make_service: ServiceFactory) -> None:
    provider = ScriptedProvider(transient(503), transient(502), transient(429), transient(504))

    with pytest.raises(LLMProviderError) as exc_info:
        asyncio.run(make_service(provider, max_retries=2).generate_reply(REQUEST))

    assert provider.calls == 3
    assert exc_info.value.status_code == 429


def test_timeout_is_retried_until_success(make_service: ServiceFactory) -> None:
    provider = SlowThenFastProvider(slow_calls=1)
    service = make_service(provider, timeout_seconds=0.05, max_retries=1)

    response = asyncio.run(service.generate_reply(REQUEST))

    assert response.message.content == "fast"
    assert provider.calls == 2


def test_timeout_exhausted_raises_llm_timeout(make_service: ServiceFactory) -> None:
    provider = SlowThenFastProvider(slow_calls=99)
    service = make_service(provider, timeout_seconds=0.05, max_retries=1)

    with pytest.raises(LLMTimeoutError):
        asyncio.run(service.generate_reply(REQUEST))

    assert provider.calls == 2


def test_backoff_delay_grows_but_stays_within_cap() -> None:
    service = ChatService(
        ScriptedProvider(),
        timeout_seconds=1,
        max_retries=5,
        retry_base_delay_seconds=0.5,
        retry_max_delay_seconds=4.0,
        max_concurrency=1,
    )
    random.seed(0)

    for attempt in range(1, 8):
        ceiling = min(4.0, 0.5 * 2**attempt)
        delays = [service._backoff_delay(attempt) for _ in range(200)]
        assert all(0 <= d <= ceiling for d in delays)
        # Jitter: delays are spread out, not a single fixed value
        assert max(delays) - min(delays) > ceiling / 2


def test_concurrent_provider_calls_never_exceed_limit(make_service: ServiceFactory) -> None:
    provider = InFlightCounter()
    service = make_service(provider, max_concurrency=3)

    async def run() -> None:
        await asyncio.gather(*(service.generate_reply(REQUEST) for _ in range(20)))

    asyncio.run(run())

    assert provider.max_in_flight == 3
