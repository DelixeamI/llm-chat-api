import json
from collections.abc import Callable

from fastapi.testclient import TestClient

from app.schemas.analysis import SupportAnalysis
from tests.fakes import CannedProvider

VALID = {
    "category": "account",
    "priority": "high",
    "summary": "Не входит в аккаунт",
    "confidence": 0.9,
}
REQUEST = {"model": "qwen3:8b", "text": "Не могу войти в аккаунт после смены пароля."}


def test_ticket_is_analyzed_with_schema_constrained_call(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider(json.dumps(VALID, ensure_ascii=False))
    use_provider(provider)

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["analysis"] == VALID
    assert body["usage"] == {"input_tokens": 40, "output_tokens": 20}

    call = provider.calls[0]
    assert call["json_schema"] == SupportAnalysis.model_json_schema()
    # Instructions and user data travel in separate messages
    assert [m.role for m in call["messages"]] == ["system", "user"]
    assert call["messages"][1].content == REQUEST["text"]


def test_invalid_json_from_model_returns_502(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider('{"category": "account", "priority":'))

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502
    assert response.json()["error"] == "invalid_model_output:invalid_json"


def test_schema_violation_returns_502_without_values(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider(json.dumps(VALID | {"confidence": 150})))

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "invalid_model_output:schema_mismatch"
    assert "confidence" in body["detail"]
    assert "150" not in body["detail"]


def test_blank_summary_is_rejected(client: TestClient, use_provider: Callable[..., None]) -> None:
    use_provider(CannedProvider(json.dumps(VALID | {"summary": "\n"})))

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502


def test_empty_text_is_rejected_before_calling_the_model(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider()
    use_provider(provider)

    response = client.post("/v1/chat/structured", json={"model": "qwen3:8b", "text": ""})

    assert response.status_code == 422
    assert provider.calls == []
