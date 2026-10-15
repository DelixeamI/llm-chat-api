import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.exceptions import RedisError

from app.api.cache import router as cache_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.usage import router as usage_router
from app.auth import Authenticator
from app.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.kv.client import create_redis
from app.llm.base import LLMError, LLMTimeoutError
from app.llm.ollama import OllamaProvider
from app.llm.structured import StructuredOutputError
from app.schemas.errors import ErrorResponse
from app.services.cache import ResponseCache
from app.services.chat import ChatService
from app.services.context import ContextBudget
from app.services.errors import (
    AuthenticationError,
    ContextOverflowError,
    ConversationNotFoundError,
    InsufficientScopeError,
    RateLimitExceededError,
)
from app.services.pricing import PriceList
from app.services.ratelimit import RateLimiter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Created once per process and shared by all requests: the client keeps a pool of
    # open connections, and the service owns the concurrency semaphore
    settings = get_settings()
    authenticator = Authenticator(settings.api_keys)
    app.state.authenticator = authenticator
    if not authenticator.configured:
        logger.warning("No API keys configured (API_KEYS): every protected endpoint answers 401")
    # Ollama ignores the key, but the SDK requires a non-empty one.
    # SDK retries are disabled: retry policy lives in one place, the service.
    llm_client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama", max_retries=0)
    app.state.llm_client = llm_client
    redis = create_redis(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        timeout_seconds=settings.redis_timeout_seconds,
    )
    app.state.redis = redis
    response_cache = None
    if settings.response_cache_enabled:
        response_cache = ResponseCache(
            redis,
            ttl_seconds=settings.response_cache_ttl_seconds,
            # Provider settings change the answer, so they are part of every cache key
            scope=f"ollama:reasoning_effort={settings.ollama_reasoning_effort}",
        )
    app.state.response_cache = response_cache
    app.state.rate_limiter = (
        RateLimiter(
            redis,
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if settings.rate_limit_enabled
        else None
    )
    app.state.chat_service = ChatService(
        OllamaProvider(llm_client, settings.ollama_reasoning_effort),
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        retry_max_delay_seconds=settings.llm_retry_max_delay_seconds,
        max_concurrency=settings.llm_max_concurrency,
        prices=PriceList(settings.model_prices),
        context=ContextBudget(
            limits=settings.model_context_limits,
            default_limit=settings.default_context_limit,
            strategy=settings.context_overflow_strategy,
        ),
        cache=response_cache,
    )
    db_engine = create_engine(settings.database_url)
    app.state.db_engine = db_engine
    app.state.session_factory = create_session_factory(db_engine)
    yield
    await llm_client.close()
    await db_engine.dispose()
    await redis.aclose()


app = FastAPI(
    title="LLM Chat API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(usage_router)
app.include_router(cache_router)


@app.exception_handler(AuthenticationError)
async def authentication_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    # The same answer for a missing key and a wrong one: it must not tell a guesser which
    body = ErrorResponse(error="invalid_api_key", detail="Missing or invalid API key")
    return JSONResponse(
        status_code=401, content=body.model_dump(), headers={"WWW-Authenticate": "ApiKey"}
    )


@app.exception_handler(InsufficientScopeError)
async def scope_handler(request: Request, exc: InsufficientScopeError) -> JSONResponse:
    logger.warning("Forbidden: client=%s needs=%s", exc.client, exc.scope)
    body = ErrorResponse(error="insufficient_scope", detail=str(exc))
    return JSONResponse(status_code=403, content=body.model_dump())


@app.exception_handler(RateLimitExceededError)
async def rate_limit_handler(request: Request, exc: RateLimitExceededError) -> JSONResponse:
    logger.warning("Rate limit exceeded: client=%s", exc.client)
    body = ErrorResponse(error="rate_limit_exceeded", detail=str(exc))
    # Retry-After tells a well-behaved client when to come back; X-RateLimit-* say where it stands
    return JSONResponse(status_code=429, content=body.model_dump(), headers=exc.headers)


@app.exception_handler(ConversationNotFoundError)
async def conversation_not_found_handler(
    request: Request, exc: ConversationNotFoundError
) -> JSONResponse:
    body = ErrorResponse(error="conversation_not_found", detail=str(exc))
    return JSONResponse(status_code=404, content=body.model_dump())


@app.exception_handler(ContextOverflowError)
async def context_overflow_handler(request: Request, exc: ContextOverflowError) -> JSONResponse:
    # 400 like OpenAI's context_length_exceeded: the request is well-formed, but this model
    # cannot take it; the client has to shorten the input or lower max_tokens
    body = ErrorResponse(error="context_length_exceeded", detail=str(exc))
    return JSONResponse(status_code=400, content=body.model_dump())


@app.exception_handler(StructuredOutputError)
async def structured_output_handler(request: Request, exc: StructuredOutputError) -> JSONResponse:
    # The message names the broken field and rule, never the values the model produced
    body = ErrorResponse(
        error=f"invalid_model_output:{exc.kind}", detail=f"{exc} (attempts: {exc.attempts})"
    )
    return JSONResponse(status_code=502, content=body.model_dump())


@app.exception_handler(RedisError)
async def redis_error_handler(request: Request, exc: RedisError) -> JSONResponse:
    # Reached only where Redis is required; the response cache bypasses Redis errors itself
    logger.error("Redis call failed: %s", type(exc).__name__)
    body = ErrorResponse(error="redis_unavailable", detail="Redis is unavailable")
    return JSONResponse(status_code=503, content=body.model_dump())


@app.exception_handler(LLMTimeoutError)
async def llm_timeout_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
    body = ErrorResponse(error="llm_timeout", detail="LLM provider did not respond in time")
    return JSONResponse(status_code=504, content=body.model_dump())


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    logger.error("LLM call failed: %s", exc)
    body = ErrorResponse(error="llm_provider_error", detail="LLM provider request failed")
    return JSONResponse(status_code=502, content=body.model_dump())


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "llm-chat-api", "version": app.version}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
