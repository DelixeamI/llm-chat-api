from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.services.analysis import AnalysisService
from app.services.cache import ResponseCache
from app.services.chat import ChatService
from app.services.idempotency import IdempotencyStore
from app.services.ratelimit import RateLimiter
from app.services.readiness import ReadinessChecker


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


def get_analysis_service(
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> AnalysisService:
    # Built per request on top of the shared ChatService: it holds no state of its own,
    # and tests that replace the chat service get a matching analysis service for free
    return AnalysisService(chat_service, max_output_retries=get_settings().structured_max_retries)


def get_readiness_checker(request: Request) -> ReadinessChecker:
    checker: ReadinessChecker = request.app.state.readiness
    return checker


def get_idempotency_store(request: Request) -> IdempotencyStore | None:
    # None when idempotency is switched off in settings
    store: IdempotencyStore | None = request.app.state.idempotency
    return store


def get_rate_limiter(request: Request) -> RateLimiter | None:
    # None when rate limiting is switched off in settings
    limiter: RateLimiter | None = request.app.state.rate_limiter
    return limiter


def get_response_cache(request: Request) -> ResponseCache | None:
    # None when the cache is switched off in settings
    cache: ResponseCache | None = request.app.state.response_cache
    return cache


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    # One session per request, opened here and closed when the response is done.
    # A single shared session would mix unrelated requests in one transaction and
    # is not safe to use from several tasks at once.
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session
