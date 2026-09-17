import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.llm.base import Completion, LLMProviderError
from app.main import app
from app.services.chat import ChatService

USER_MESSAGE = {"role": "user", "content": "hi"}


def payload(**overrides: object) -> dict[str, object]:
    return {"model": "llama3", "messages": [USER_MESSAGE]} | overrides


def test_valid_request_returns_assistant_reply(client: TestClient) -> None:
    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "llama3"
    assert body["message"] == {"role": "assistant", "content": "fake reply"}
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_provider_error_returns_502(client: TestClient) -> None:
    class FailingProvider:
        async def complete(self, *args: object) -> Completion:
            raise LLMProviderError("boom", status_code=500)

    app.dependency_overrides[get_chat_service] = lambda: ChatService(
        FailingProvider(), timeout_seconds=5
    )

    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 502
    assert response.json() == {
        "error": "llm_provider_error",
        "detail": "LLM provider request failed",
    }


def test_slow_provider_returns_504(client: TestClient) -> None:
    class SlowProvider:
        async def complete(self, *args: object) -> Completion:
            await asyncio.sleep(1)
            raise AssertionError("timeout should have cancelled the call")

    app.dependency_overrides[get_chat_service] = lambda: ChatService(
        SlowProvider(), timeout_seconds=0.05
    )

    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 504
    assert response.json()["error"] == "llm_timeout"


@pytest.mark.parametrize(
    ("body", "error_type"),
    [
        pytest.param(payload(messages=[]), "too_short", id="empty-messages"),
        pytest.param(
            payload(messages=[{"role": "banana", "content": "hi"}]),
            "literal_error",
            id="invalid-role",
        ),
        pytest.param(
            payload(params={"temperature": 100}), "less_than_equal", id="temperature-too-high"
        ),
        pytest.param(
            payload(params={"temperature": -0.1}), "greater_than_equal", id="temperature-negative"
        ),
        pytest.param(payload(params={"max_tokens": 0}), "greater_than", id="max-tokens-zero"),
        pytest.param(payload(randon_parametr=1), "extra_forbidden", id="extra-field"),
        pytest.param(payload(params={"top_p": 0.9}), "extra_forbidden", id="extra-nested-field"),
        pytest.param(
            payload(messages=[{"role": "user", "content": ""}]),
            "string_too_short",
            id="empty-content",
        ),
        pytest.param(
            payload(messages=[{"role": "user", "content": "   "}]),
            "value_error",
            id="blank-content",
        ),
        pytest.param(
            payload(messages=[{"role": "system", "content": "be nice"}]),
            "value_error",
            id="no-user-message",
        ),
        pytest.param({"messages": [USER_MESSAGE]}, "missing", id="missing-model"),
    ],
)
def test_invalid_request_returns_422(
    client: TestClient, body: dict[str, object], error_type: str
) -> None:
    response = client.post("/v1/chat", json=body)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == error_type


def test_malformed_json_returns_422(client: TestClient) -> None:
    response = client.post(
        "/v1/chat", content='{"model": ', headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"
