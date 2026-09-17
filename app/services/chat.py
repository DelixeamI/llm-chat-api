import asyncio
import logging
import random
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repository
from app.db.models import Conversation
from app.llm.base import Completion, LLMError, LLMProvider, LLMTimeoutError
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage
from app.services.errors import ConversationNotFoundError

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

    async def generate_reply(self, request: ChatRequest, session: AsyncSession) -> ChatResponse:
        # Existence is checked before the call so a wrong id does not cost a generation
        if request.conversation_id is not None:
            await self._require_conversation(session, request.conversation_id)

        # No transaction is open while the model works. A call can take minutes with
        # retries, and an open transaction would hold a pooled connection and block
        # vacuum for all that time.
        completion = await self._complete_with_retries(request)

        conversation_id = await self._persist_exchange(session, request, completion)
        return ChatResponse(
            conversation_id=conversation_id,
            model=request.model,
            message=Message(role="assistant", content=completion.content),
            usage=Usage(
                input_tokens=completion.input_tokens, output_tokens=completion.output_tokens
            ),
        )

    async def _require_conversation(
        self, session: AsyncSession, conversation_id: uuid.UUID
    ) -> Conversation:
        conversation = await repository.get_conversation(session, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        # The read opened a transaction; end it so the connection is not held idle
        # in transaction while the model generates
        await session.rollback()
        return conversation

    async def _persist_exchange(
        self, session: AsyncSession, request: ChatRequest, completion: Completion
    ) -> uuid.UUID:
        """One transaction: the conversation and both messages appear together or not at all.

        A half-written exchange is worse than none: an assistant reply without the question
        it answers, or a question with no reply and no error, cannot be interpreted later.
        """
        # Only the new exchange is stored, not the whole history the client sent.
        # ponytail: the client is trusted to send history consistent with what is stored;
        # reconciling the two belongs with server-side history assembly
        last_user_message = next(m for m in reversed(request.messages) if m.role == "user")
        try:
            conversation_id = request.conversation_id
            if conversation_id is None:
                conversation = await repository.create_conversation(session)
                conversation_id = conversation.id

            repository.add_message(
                session, conversation_id, role="user", content=last_user_message.content
            )
            repository.add_message(
                session,
                conversation_id,
                role="assistant",
                content=completion.content,
                model=request.model,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
            await session.commit()
        except Exception:
            # Without an explicit rollback the failed transaction stays open and every
            # later statement on this connection fails too
            await session.rollback()
            raise
        return conversation_id

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
