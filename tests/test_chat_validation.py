from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def valid_payload(**overrides: object) -> dict[str, object]:
    return {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]} | overrides


def test_valid_request_is_accepted() -> None:
    assert client.post("/v1/chat", json=valid_payload()).status_code == 200


def test_unknown_field_is_rejected() -> None:
    response = client.post("/v1/chat", json=valid_payload(randon_parametr=999))
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_request_without_user_message_is_rejected() -> None:
    payload = valid_payload(messages=[{"role": "system", "content": "you are helpful"}])
    assert client.post("/v1/chat", json=payload).status_code == 422


def test_whitespace_only_content_is_rejected() -> None:
    payload = valid_payload(messages=[{"role": "user", "content": "   "}])
    assert client.post("/v1/chat", json=payload).status_code == 422
