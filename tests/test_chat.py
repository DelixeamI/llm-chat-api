import asyncio
import uuid
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.db.models import Conversation
from app.llm.base import Completion, LLMProviderError
from tests.fakes import FakeSession, ScriptedProvider

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


def test_non_retryable_provider_error_returns_502_without_retry(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = ScriptedProvider(LLMProviderError("bad request", status_code=400))
    use_provider(provider)

    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 502
    assert response.json() == {
        "error": "llm_provider_error",
        "detail": "LLM provider request failed",
    }
    assert provider.calls == 1


def test_transient_provider_error_is_retried(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = ScriptedProvider(LLMProviderError("unavailable", status_code=503, retryable=True))
    use_provider(provider)

    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 200
    assert response.json()["message"]["content"] == "recovered"
    assert provider.calls == 2


def test_slow_provider_returns_504(client: TestClient, use_provider: Callable[..., None]) -> None:
    class SlowProvider:
        async def complete(self, *args: object) -> Completion:
            await asyncio.sleep(1)
            raise AssertionError("timeout should have cancelled the call")

    use_provider(SlowProvider(), timeout_seconds=0.05, max_retries=0)

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


def test_reply_is_persisted_with_new_conversation(client: TestClient, session: FakeSession) -> None:
    response = client.post("/v1/chat", json=payload())

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    assert session.commits == 1

    roles = [(m.role, m.content, m.model) for m in session.messages]
    assert roles == [
        ("user", "hi", None),
        ("assistant", "fake reply", "llama3"),
    ]
    assert str(session.messages[0].conversation_id) == conversation_id
    assert (session.messages[1].input_tokens, session.messages[1].output_tokens) == (3, 2)


def test_existing_conversation_is_reused(client: TestClient, session: FakeSession) -> None:
    conversation = Conversation(id=uuid.uuid4())
    session.conversations[conversation.id] = conversation

    response = client.post("/v1/chat", json=payload(conversation_id=str(conversation.id)))

    assert response.status_code == 200
    assert response.json()["conversation_id"] == str(conversation.id)
    assert all(m.conversation_id == conversation.id for m in session.messages)


def test_unknown_conversation_returns_404(client: TestClient, session: FakeSession) -> None:
    missing = uuid.uuid4()

    response = client.post("/v1/chat", json=payload(conversation_id=str(missing)))

    assert response.status_code == 404
    assert response.json()["error"] == "conversation_not_found"
    assert session.messages == []
    assert session.commits == 0


def test_malformed_conversation_id_returns_422(client: TestClient) -> None:
    response = client.post("/v1/chat", json=payload(conversation_id="not-a-uuid"))

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "conversation_id"]
