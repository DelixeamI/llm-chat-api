from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.llm.base import LLMProvider
from app.main import app
from app.services.chat import ChatService
from tests.fakes import FakeProvider, ServiceFactory


@pytest.fixture
def make_service() -> ServiceFactory:
    def factory(
        provider: LLMProvider,
        *,
        timeout_seconds: float = 5.0,
        max_retries: int = 2,
        max_concurrency: int = 5,
    ) -> ChatService:
        # Zero backoff keeps retry tests instant
        return ChatService(
            provider,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_base_delay_seconds=0,
            retry_max_delay_seconds=0,
            max_concurrency=max_concurrency,
        )

    return factory


@pytest.fixture
def use_provider(make_service: ServiceFactory) -> Callable[..., None]:
    def install(provider: LLMProvider, **options: float) -> None:
        app.dependency_overrides[get_chat_service] = lambda: make_service(provider, **options)

    return install


@pytest.fixture
def client(use_provider: Callable[..., None]) -> Iterator[TestClient]:
    use_provider(FakeProvider())
    # The context manager runs the app lifespan, exactly as a real server start would
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
