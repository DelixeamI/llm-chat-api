from functools import lru_cache

from app.config import get_settings
from app.llm.ollama import OllamaProvider
from app.services.chat import ChatService


# ponytail: process-wide singleton; replaced by app lifespan on day 13
@lru_cache
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        OllamaProvider(settings.ollama_base_url, settings.ollama_reasoning_effort),
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        retry_max_delay_seconds=settings.llm_retry_max_delay_seconds,
        max_concurrency=settings.llm_max_concurrency,
    )
