from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Protocol

from app.schemas.chat import GenerationParams, Message

# Rate limiting and gateway/overload errors are usually transient; 400/401/404 are not
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class Completion:
    content: str
    input_tokens: int
    output_tokens: int
    # "length" means the output was cut by max_tokens rather than finished by the model
    finish_reason: str | None = None
    # Served from the response cache: no provider call was made and nothing was spent
    cached: bool = False


@dataclass(frozen=True)
class StreamChunk:
    """One piece of a streamed reply, as the provider delivers it.

    A chunk is a network unit, not a token: a provider may pack several tokens into one
    chunk or split a multi-byte character across several. Deltas carry text; the closing
    chunks carry the finish reason and the token usage, so any field may be empty.
    """

    content: str = ""
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


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
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """json_schema, when given, asks the provider to constrain output to that schema."""
        ...

    def stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> AsyncGenerator[StreamChunk, None]:
        """The reply as it is generated. Closing the generator stops the generation."""
        ...
