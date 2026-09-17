from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_reasoning_effort: ReasoningEffort = "none"

    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_retry_base_delay_seconds: float = 0.5
    llm_retry_max_delay_seconds: float = 8.0
    llm_max_concurrency: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
