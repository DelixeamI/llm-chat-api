from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.services.pricing import ModelPrice

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
    # Extra generations allowed when the model returns unusable structured output
    structured_max_retries: int = 1

    database_url: str = "postgresql+asyncpg://llm_chat:llm_chat@127.0.0.1:5433/llm_chat"

    # Illustrative prices for demonstrating cost accounting, not a real tariff: a local
    # model costs GPU time and electricity, not a per-token bill. Override via MODEL_PRICES.
    model_prices: dict[str, ModelPrice] = Field(
        default_factory=lambda: {
            "qwen3:8b": ModelPrice(
                input_per_million_usd=Decimal("0.05"), output_per_million_usd=Decimal("0.40")
            )
        }
    )
    # Conversion happens when a report is built, so changing the rate re-prices history
    usd_to_rub: Decimal = Field(default=Decimal("95"), gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
