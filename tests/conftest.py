from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.llm.base import Completion
from app.main import app
from app.schemas.chat import GenerationParams, Message
from app.services.chat import ChatService


class FakeProvider:
    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        return Completion(content="fake reply", input_tokens=3, output_tokens=2)


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_chat_service] = lambda: ChatService(
        FakeProvider(), timeout_seconds=5
    )
    yield TestClient(app)
    app.dependency_overrides.clear()
