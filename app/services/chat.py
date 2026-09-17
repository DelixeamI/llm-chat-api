import asyncio
import logging
import random

from app.llm.base import Completion, LLMError, LLMProvider, LLMTimeoutError
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_seconds: float,
        max_retries: int,
        retry_base_delay_seconds: float,
        retry_max_delay_seconds: float,
        max_concurrency: int,
    ) -> None:
        self._provider = provider
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        # Caps in-flight provider calls across all requests handled by this service
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_reply(self, request: ChatRequest) -> ChatResponse:
        completion = await self._complete_with_retries(request)
        return ChatResponse(
            model=request.model,
            message=Message(role="assistant", content=completion.content),
            usage=Usage(
                input_tokens=completion.input_tokens, output_tokens=completion.output_tokens
            ),
        )

    async def _complete_with_retries(self, request: ChatRequest) -> Completion:
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._complete_once(request)
            except LLMError as exc:
                if not exc.retryable or attempt == attempts:
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM call failed, retrying: model=%s attempt=%d/%d error=%s delay_s=%.2f",
                    request.model,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _complete_once(self, request: ChatRequest) -> Completion:
        # The slot is held per attempt, not across backoff sleeps, so a retrying request
        # does not block others. The timeout starts after the slot is acquired: it bounds
        # the provider, not the queue.
        # ponytail: waiting for a slot is unbounded; add an acquire timeout that returns 503
        # if queues grow under sustained overload
        async with self._semaphore:
            return await self._call_provider(request)

    async def _call_provider(self, request: ChatRequest) -> Completion:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await self._provider.complete(
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

    def _backoff_delay(self, attempt: int) -> float:
        # Full jitter: a random point in [0, capped exponential] spreads retries of many
        # clients over time instead of letting them hit a recovering provider in sync
        ceiling = min(self._retry_max_delay_seconds, self._retry_base_delay_seconds * 2**attempt)
        return random.uniform(0, ceiling)
