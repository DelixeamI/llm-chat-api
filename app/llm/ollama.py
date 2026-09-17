from typing import cast

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.config import ReasoningEffort
from app.llm.base import RETRYABLE_STATUS_CODES, Completion, LLMProviderError, LLMTimeoutError
from app.schemas.chat import GenerationParams, Message


class OllamaProvider:
    def __init__(self, client: AsyncOpenAI, reasoning_effort: ReasoningEffort) -> None:
        # The client is owned by the app lifespan: the provider uses it but never closes it
        self._client = client
        self._reasoning_effort = reasoning_effort

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                # Message has exactly the role/content shape of an OpenAI chat message
                messages=cast(list[ChatCompletionMessageParam], [m.model_dump() for m in messages]),
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                reasoning_effort=self._reasoning_effort,
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except APIConnectionError as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc
        except APIStatusError as exc:
            raise LLMProviderError(
                str(exc),
                status_code=exc.status_code,
                retryable=exc.status_code in RETRYABLE_STATUS_CODES,
            ) from exc
        except APIError as exc:
            raise LLMProviderError(str(exc)) from exc

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
        )
