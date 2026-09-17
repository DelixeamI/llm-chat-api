from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_reasoning_effort: str = "none"

    llm_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
