import asyncio
import logging

from app.llm.base import LLMProvider, LLMTimeoutError
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, provider: LLMProvider, *, timeout_seconds: float) -> None:
        self._provider = provider
        self._timeout_seconds = timeout_seconds

    async def generate_reply(self, request: ChatRequest) -> ChatResponse:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                completion = await self._provider.complete(
                    request.model, request.messages, request.params
                )
        except TimeoutError as exc:
            # Metadata only: prompts may contain personal data and must not reach logs
            logger.warning(
                "LLM call timed out: model=%s timeout_s=%s messages=%d",
                request.model,
                self._timeout_seconds,
                len(request.messages),
            )
            raise LLMTimeoutError(f"LLM call exceeded {self._timeout_seconds}s") from exc

        return ChatResponse(
            model=request.model,
            message=Message(role="assistant", content=completion.content),
            usage=Usage(
                input_tokens=completion.input_tokens, output_tokens=completion.output_tokens
            ),
        )
