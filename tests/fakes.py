from app.llm.base import Completion, LLMError
from app.schemas.chat import GenerationParams, Message


class FakeProvider:
    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        return Completion(content="fake reply", input_tokens=3, output_tokens=2)


class ScriptedProvider:
    """Raises the given errors one per call, then succeeds. Counts calls."""

    def __init__(self, *errors: LLMError) -> None:
        self._errors = list(errors)
        self.calls = 0

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return Completion(content="recovered", input_tokens=1, output_tokens=1)
