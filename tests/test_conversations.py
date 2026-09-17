import uuid

from fastapi.testclient import TestClient

from tests.fakes import FakeSession, make_conversation


def test_create_conversation_returns_201(client: TestClient, session: FakeSession) -> None:
    response = client.post("/v1/conversations", json={"title": "Про Python"})

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Про Python"
    assert uuid.UUID(body["id"])
    assert session.commits == 1


def test_create_conversation_rejects_unknown_field(client: TestClient) -> None:
    response = client.post("/v1/conversations", json={"titel": "опечатка"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_get_conversation_returns_it(client: TestClient, session: FakeSession) -> None:
    conversation = make_conversation(title="Сохранённый")
    session.conversations[conversation.id] = conversation

    response = client.get(f"/v1/conversations/{conversation.id}")

    assert response.status_code == 200
    assert response.json()["title"] == "Сохранённый"


def test_get_unknown_conversation_returns_404(client: TestClient) -> None:
    response = client.get(f"/v1/conversations/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"] == "conversation_not_found"


def test_messages_of_unknown_conversation_return_404(client: TestClient) -> None:
    response = client.get(f"/v1/conversations/{uuid.uuid4()}/messages")

    assert response.status_code == 404


def test_messages_are_returned_after_a_chat_call(client: TestClient, session: FakeSession) -> None:
    conversation_id = client.post(
        "/v1/chat", json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
    ).json()["conversation_id"]

    response = client.get(f"/v1/conversations/{conversation_id}/messages")

    assert response.status_code == 200
    body = response.json()
    assert [m["role"] for m in body] == ["user", "assistant"]
    assert body[1]["model"] == "llama3"
    assert body[1]["output_tokens"] == 2


def test_malformed_conversation_id_returns_422(client: TestClient) -> None:
    assert client.get("/v1/conversations/not-a-uuid").status_code == 422
