from dataclasses import dataclass
from typing import Protocol

from app.schemas.chat import GenerationParams, Message


@dataclass(frozen=True)
class Completion:
    content: str
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    """Base error for any LLM provider failure, independent of the vendor SDK."""


class LLMProviderError(LLMError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMProvider(Protocol):
    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion: ...
