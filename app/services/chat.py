import asyncio
import logging
import random
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repository
from app.db.models import Conversation
from app.llm.base import (
    Completion,
    LLMError,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
    StreamChunk,
)
from app.schemas.chat import ChatRequest, ChatResponse, GenerationParams, Message, Usage
from app.schemas.stream import StreamDelta, StreamDone, StreamError, StreamEvent
from app.services.cache import ResponseCache, is_cacheable
from app.services.context import ContextBudget
from app.services.errors import ConversationNotFoundError
from app.services.pricing import PriceList

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
        prices: PriceList,
        context: ContextBudget,
        cache: ResponseCache | None = None,
    ) -> None:
        self._provider = provider
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        # Caps in-flight provider calls across all requests handled by this service
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.prices = prices
        self._context = context
        self._cache = cache

    async def generate_reply(
        self, request: ChatRequest, session: AsyncSession, *, owner: str
    ) -> ChatResponse:
        # Existence is checked before the call so a wrong id does not cost a generation
        if request.conversation_id is not None:
            await self._require_conversation(session, request.conversation_id, owner)

        # No transaction is open while the model works. A call can take minutes with
        # retries, and an open transaction would hold a pooled connection and block
        # vacuum for all that time.
        messages, dropped = self._context.fit(
            request.model, request.messages, request.params.max_tokens
        )
        if dropped:
            logger.info("Context trimmed: model=%s dropped_messages=%d", request.model, dropped)
        completion = await self._complete_cached(request.model, messages, request.params, owner)

        conversation_id = await self._persist_exchange(session, request, completion, owner=owner)
        return ChatResponse(
            conversation_id=conversation_id,
            model=request.model,
            message=Message(role="assistant", content=completion.content),
            usage=Usage(
                input_tokens=completion.input_tokens, output_tokens=completion.output_tokens
            ),
            dropped_messages=dropped,
            cached=completion.cached,
        )

    async def stream_reply(
        self, request: ChatRequest, session: AsyncSession, *, owner: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """The reply as events: text pieces as they arrive, then one closing event.

        Everything that can fail before the model has produced its first piece raises as an
        ordinary exception, so the caller can still answer with a proper HTTP status. After
        the first piece a failure can only be reported inside the stream, as an error event.
        """
        started = time.perf_counter()
        if request.conversation_id is not None:
            await self._require_conversation(session, request.conversation_id, owner)
        messages, dropped = self._context.fit(
            request.model, request.messages, request.params.max_tokens
        )

        key = None
        if self._cache is not None and is_cacheable(request.params):
            key = self._cache.key_for(request.model, messages, request.params, owner)
            cached = await self._cache.get(key)
            if cached is not None:
                # A cached reply is replayed as one piece: there is no generation to wait for
                logger.info("Cache hit (stream): model=%s", request.model)
                ttft_ms = _elapsed_ms(started)
                yield StreamDelta(content=cached.content)
                async for done in self._finish_stream(
                    session, request, cached, dropped, started, ttft_ms, owner
                ):
                    yield done
                return

        parts: list[str] = []
        finish_reason: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        ttft_ms = 0.0
        chunks = self._provider_stream(request.model, messages, request.params)
        try:
            async for chunk in chunks:
                if chunk.content:
                    if not parts:
                        ttft_ms = _elapsed_ms(started)
                    parts.append(chunk.content)
                    yield StreamDelta(content=chunk.content)
                finish_reason = chunk.finish_reason or finish_reason
                if chunk.input_tokens is not None:
                    input_tokens, output_tokens = chunk.input_tokens, chunk.output_tokens
            if not parts:
                raise LLMProviderError(f"empty stream, finish_reason={finish_reason}")
        except LLMError as exc:
            if not parts:
                raise
            logger.warning("Stream broke after %d pieces: %s", len(parts), type(exc).__name__)
            yield StreamError(error=_stream_error_code(exc), detail=str(exc))
            return
        except GeneratorExit:
            # The client went away: closing `chunks` below stops the generation upstream.
            # Nothing is stored: half a reply is not an exchange, and no token count exists.
            logger.info(
                "Client disconnected mid-stream: model=%s pieces=%d", request.model, len(parts)
            )
            raise
        finally:
            await chunks.aclose()

        content = "".join(parts)
        if input_tokens is None or output_tokens is None:
            # Some servers ignore stream_options; fall back to the estimate and say so
            logger.warning("Provider sent no usage in the stream: model=%s", request.model)
            input_tokens = self._context.estimate(messages)
            output_tokens = self._context.estimate([Message(role="assistant", content=content)])
        completion = Completion(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )
        if key is not None and self._cache is not None and finish_reason == "stop":
            await self._cache.put(key, completion)
        async for done in self._finish_stream(
            session, request, completion, dropped, started, ttft_ms, owner
        ):
            yield done

    async def _finish_stream(
        self,
        session: AsyncSession,
        request: ChatRequest,
        completion: Completion,
        dropped: int,
        started: float,
        ttft_ms: float,
        owner: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        try:
            conversation_id = await self._persist_exchange(
                session, request, completion, owner=owner, endpoint="stream"
            )
        except Exception:
            # The text has already been delivered, so the status cannot change and raising
            # would only cut the stream off without a word. Say what happened instead.
            logger.exception("Streamed reply could not be stored: model=%s", request.model)
            yield StreamError(
                error="storage_unavailable",
                detail="the reply was generated but could not be saved",
            )
            return
        yield StreamDone(
            conversation_id=conversation_id,
            model=request.model,
            usage=Usage(
                input_tokens=completion.input_tokens, output_tokens=completion.output_tokens
            ),
            finish_reason=completion.finish_reason,
            dropped_messages=dropped,
            cached=completion.cached,
            ttft_ms=ttft_ms,
            duration_ms=_elapsed_ms(started),
        )

    async def _provider_stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> AsyncGenerator[StreamChunk, None]:
        """Provider stream under the same rules as complete(), adapted to streaming.

        The concurrency slot is held for the whole stream. The timeout is per piece: a
        stream may legitimately run for minutes, but it must not go silent. Retries are
        possible only until the first piece: after text has reached the client, starting
        over would send it twice.
        """
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            async with self._semaphore:
                stream = self._provider.stream(model, messages, params)
                try:
                    first = await self._next_chunk(stream)
                    if first is None:
                        raise LLMProviderError("empty stream")
                except LLMError as exc:
                    await stream.aclose()
                    if not exc.retryable or attempt == attempts:
                        raise
                    failure = exc
                else:
                    try:
                        yield first
                        while (chunk := await self._next_chunk(stream)) is not None:
                            yield chunk
                    finally:
                        await stream.aclose()
                    return
            # Outside the slot: a request that is waiting to retry must not block others
            delay = self._backoff_delay(attempt)
            logger.warning(
                "LLM stream failed before the first piece, retrying: model=%s attempt=%d/%d "
                "error=%s delay_s=%.2f",
                model,
                attempt,
                attempts,
                type(failure).__name__,
                delay,
            )
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _next_chunk(self, stream: AsyncGenerator[StreamChunk, None]) -> StreamChunk | None:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await anext(stream)
        except StopAsyncIteration:
            return None
        except TimeoutError as exc:
            logger.warning("LLM stream went silent: timeout_s=%s", self._timeout_seconds)
            raise LLMTimeoutError(f"no stream data for {self._timeout_seconds}s") from exc

    async def _complete_cached(
        self, model: str, messages: list[Message], params: GenerationParams, owner: str
    ) -> Completion:
        # Keyed on the messages after context fitting: that is what the model would see
        if self._cache is None or not is_cacheable(params):
            return await self.complete(model, messages, params)

        key = self._cache.key_for(model, messages, params, owner)
        cached = await self._cache.get(key)
        if cached is not None:
            logger.info("Cache hit: model=%s", model)
            return cached

        completion = await self.complete(model, messages, params)
        # Only replies the model finished itself: one cut by max_tokens would be served
        # truncated to everyone for the whole TTL
        if completion.finish_reason == "stop":
            await self._cache.put(key, completion)
        return completion

    async def _require_conversation(
        self, session: AsyncSession, conversation_id: uuid.UUID, owner: str
    ) -> Conversation:
        conversation = await repository.get_conversation(session, conversation_id)
        # Someone else's conversation answers exactly like a missing one: 404, not 403.
        # A 403 would confirm that the id exists, and ids are what an attacker is guessing.
        if conversation is None or conversation.owner != owner:
            raise ConversationNotFoundError(conversation_id)
        # The read opened a transaction; end it so the connection is not held idle
        # in transaction while the model generates
        await session.rollback()
        return conversation

    async def _persist_exchange(
        self,
        session: AsyncSession,
        request: ChatRequest,
        completion: Completion,
        *,
        owner: str,
        endpoint: str = "chat",
    ) -> uuid.UUID:
        """One transaction: conversation, both messages and the usage record, or nothing.

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
                conversation = await repository.create_conversation(session, owner=owner)
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
            # The usage log records spending; a cached reply called no model and spent nothing
            if not completion.cached:
                repository.add_usage_log(
                    session,
                    endpoint=endpoint,
                    model=request.model,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    status="success",
                    conversation_id=conversation_id,
                    client=owner,
                    cost_usd=self.prices.cost_usd(
                        request.model, completion.input_tokens, completion.output_tokens
                    ),
                )
            await session.commit()
        except Exception:
            # Without an explicit rollback the failed transaction stays open and every
            # later statement on this connection fails too
            await session.rollback()
            raise
        return conversation_id

    async def complete(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """One reliable model call: concurrency limit, timeout per attempt, bounded retries."""
        # Every call passes through here, including structured-output recovery, whose
        # conversation grows with each attempt. Checked before taking a concurrency slot.
        self._context.check(model, messages, params.max_tokens)
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._complete_once(model, messages, params, json_schema)
            except LLMError as exc:
                if not exc.retryable or attempt == attempts:
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM call failed, retrying: model=%s attempt=%d/%d error=%s delay_s=%.2f",
                    model,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _complete_once(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        json_schema: dict[str, Any] | None,
    ) -> Completion:
        # The slot is held per attempt, not across backoff sleeps, so a retrying request
        # does not block others. The timeout starts after the slot is acquired: it bounds
        # the provider, not the queue.
        # ponytail: waiting for a slot is unbounded; add an acquire timeout that returns 503
        # if queues grow under sustained overload
        async with self._semaphore:
            return await self._call_provider(model, messages, params, json_schema)

    async def _call_provider(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        json_schema: dict[str, Any] | None,
    ) -> Completion:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await self._provider.complete(
                    model, messages, params, json_schema=json_schema
                )
        except TimeoutError as exc:
            # Metadata only: prompts may contain personal data and must not reach logs
            logger.warning(
                "LLM call timed out: model=%s timeout_s=%s messages=%d",
                model,
                self._timeout_seconds,
                len(messages),
            )
            raise LLMTimeoutError(f"LLM call exceeded {self._timeout_seconds}s") from exc

    def _backoff_delay(self, attempt: int) -> float:
        # Full jitter: a random point in [0, capped exponential] spreads retries of many
        # clients over time instead of letting them hit a recovering provider in sync
        ceiling = min(self._retry_max_delay_seconds, self._retry_base_delay_seconds * 2**attempt)
        return random.uniform(0, ceiling)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _stream_error_code(exc: LLMError) -> str:
    return "llm_timeout" if isinstance(exc, LLMTimeoutError) else "llm_provider_error"
