from collections.abc import AsyncGenerator
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    omit,
)
from openai.types.chat import ChatCompletionMessageParam

from app.config import ReasoningEffort
from app.llm.base import (
    RETRYABLE_STATUS_CODES,
    Completion,
    LLMError,
    LLMProviderError,
    LLMTimeoutError,
    StreamChunk,
)
from app.schemas.chat import GenerationParams, Message


def _translate(exc: APIError) -> LLMError:
    """Vendor errors become ours, with the one fact the service needs: can a retry help."""
    if isinstance(exc, APITimeoutError):
        return LLMTimeoutError(str(exc))
    if isinstance(exc, APIConnectionError):
        return LLMProviderError(str(exc), retryable=True)
    if isinstance(exc, APIStatusError):
        return LLMProviderError(
            str(exc),
            status_code=exc.status_code,
            retryable=exc.status_code in RETRYABLE_STATUS_CODES,
        )
    return LLMProviderError(str(exc))


class OllamaProvider:
    def __init__(self, client: AsyncOpenAI, reasoning_effort: ReasoningEffort) -> None:
        # The client is owned by the app lifespan: the provider uses it but never closes it
        self._client = client
        self._reasoning_effort = reasoning_effort

    async def complete(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        # Structured output: Ollama turns the schema into a grammar that constrains which
        # tokens may be generated, so the syntax and shape are enforced during decoding
        response_format = (
            {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
            if json_schema is not None
            else omit
        )
        try:
            response = await self._client.chat.completions.create(
                model=model,
                # Message has exactly the role/content shape of an OpenAI chat message
                messages=cast(list[ChatCompletionMessageParam], [m.model_dump() for m in messages]),
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                reasoning_effort=self._reasoning_effort,
                response_format=response_format,  # type: ignore[arg-type]
            )
        except APIError as exc:
            raise _translate(exc) from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise LLMProviderError(
                f"empty completion, finish_reason={response.choices[0].finish_reason}"
            )

        usage = response.usage
        return Completion(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=response.choices[0].finish_reason,
        )

    async def stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> AsyncGenerator[StreamChunk, None]:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], [m.model_dump() for m in messages]),
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                reasoning_effort=self._reasoning_effort,
                stream=True,
                # Without this the stream carries no token counts, and the usage log would
                # have nothing to record: the counts arrive in one extra closing chunk
                stream_options={"include_usage": True},
            )
        except APIError as exc:
            raise _translate(exc) from exc

        try:
            async for chunk in response:
                usage = chunk.usage
                choice = chunk.choices[0] if chunk.choices else None
                yield StreamChunk(
                    content=(choice.delta.content or "") if choice else "",
                    finish_reason=choice.finish_reason if choice else None,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                )
        except APIError as exc:
            raise _translate(exc) from exc
        finally:
            # Also runs when the consumer stops early (client disconnect): closing the HTTP
            # response tells Ollama to stop generating tokens nobody will read
            await response.close()
