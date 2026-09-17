from openai import APIError, APIStatusError, OpenAI

from app.llm.base import Completion, LLMProviderError
from app.schemas.chat import GenerationParams, Message


class OllamaProvider:
    def __init__(self, base_url: str, reasoning_effort: str) -> None:
        # Ollama ignores the key, but the SDK requires a non-empty one
        self._client = OpenAI(base_url=base_url, api_key="ollama", max_retries=0)
        self._reasoning_effort = reasoning_effort

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        # ponytail: sync client blocks the event loop; switched to AsyncOpenAI on day 9
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in messages],
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                reasoning_effort=self._reasoning_effort,
            )
        except APIStatusError as exc:
            raise LLMProviderError(str(exc), status_code=exc.status_code) from exc
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
