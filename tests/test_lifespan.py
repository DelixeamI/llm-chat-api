from fastapi.testclient import TestClient

from app.main import app
from app.services.chat import ChatService


def test_lifespan_creates_shared_service_and_closes_client() -> None:
    with TestClient(app) as client:
        service = app.state.chat_service
        llm_client = app.state.llm_client
        assert isinstance(service, ChatService)
        assert not llm_client.is_closed()

        client.get("/health")
        assert app.state.chat_service is service

    assert llm_client.is_closed()
