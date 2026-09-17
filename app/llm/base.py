from dataclasses import dataclass
from typing import Protocol

from app.schemas.chat import GenerationParams, Message

# Rate limiting and gateway/overload errors are usually transient; 400/401/404 are not
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class Completion:
    content: str
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    """Base error for any LLM provider failure, independent of the vendor SDK."""

    retryable = False


class LLMTimeoutError(LLMError):
    retryable = True


class LLMProviderError(LLMError):
    def __init__(
        self, message: str, status_code: int | None = None, *, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class LLMProvider(Protocol):
    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion: ...
