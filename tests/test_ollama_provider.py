"""The provider against a fake HTTP transport.

The real OpenAI SDK parses real HTTP responses here; only the network is replaced. This tests
how vendor errors are translated without depending on the SDK's internal exception classes.
"""

import asyncio
from collections.abc import Callable

import httpx2
import pytest

from app.llm.base import Completion, LLMProviderError, LLMTimeoutError
from app.llm.ollama import OllamaProvider
from app.schemas.chat import GenerationParams, Message

Handler = Callable[[httpx2.Request], httpx2.Response]

MESSAGES = [Message(role="user", content="hi")]


def completion_body(content: str | None) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen3:8b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }


def call(handler: Handler) -> Completion:
    async def run() -> Completion:
        from openai import AsyncOpenAI

        async with AsyncOpenAI(
            base_url="http://ollama.test/v1",
            api_key="test",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        ) as client:
            provider = OllamaProvider(client, reasoning_effort="none")
            return await provider.complete("qwen3:8b", MESSAGES, GenerationParams())

    return asyncio.run(run())


def test_successful_response_is_mapped_to_completion() -> None:
    completion = call(lambda request: httpx2.Response(200, json=completion_body("  Paris  ")))

    assert completion == Completion(content="Paris", input_tokens=11, output_tokens=4)


def test_request_carries_messages_and_generation_params() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion_body("ok"))

    call(handler)

    body = seen[0].read().decode()
    assert '"messages":[{"role":"user","content":"hi"}]' in body
    assert '"temperature":0.7' in body
    assert '"reasoning_effort":"none"' in body


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (502, True), (503, True), (504, True), (400, False), (401, False), (404, False)],
)
def test_http_error_is_translated_with_retryability(status: int, retryable: bool) -> None:
    with pytest.raises(LLMProviderError) as exc_info:
        call(lambda request: httpx2.Response(status, json={"error": {"message": "nope"}}))

    assert exc_info.value.status_code == status
    assert exc_info.value.retryable is retryable


def test_connection_error_is_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(LLMProviderError) as exc_info:
        call(handler)

    assert exc_info.value.retryable is True
    assert exc_info.value.status_code is None


def test_sdk_timeout_becomes_llm_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("read timed out", request=request)

    with pytest.raises(LLMTimeoutError):
        call(handler)


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_completion_is_a_non_retryable_error(content: str | None) -> None:
    with pytest.raises(LLMProviderError) as exc_info:
        call(lambda request: httpx2.Response(200, json=completion_body(content)))

    assert exc_info.value.retryable is False
