from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_identifies_service(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "llm-chat-api"


def test_unknown_path_returns_404(client: TestClient) -> None:
    assert client.get("/does-not-exist").status_code == 404


def test_wrong_method_returns_405(client: TestClient) -> None:
    assert client.post("/health").status_code == 405
