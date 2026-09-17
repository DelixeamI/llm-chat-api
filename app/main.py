import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.api.chat import router as chat_router
from app.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.llm.base import LLMError, LLMTimeoutError
from app.llm.ollama import OllamaProvider
from app.schemas.errors import ErrorResponse
from app.services.chat import ChatService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Created once per process and shared by all requests: the client keeps a pool of
    # open connections, and the service owns the concurrency semaphore
    settings = get_settings()
    # Ollama ignores the key, but the SDK requires a non-empty one.
    # SDK retries are disabled: retry policy lives in one place, the service.
    llm_client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama", max_retries=0)
    app.state.llm_client = llm_client
    app.state.chat_service = ChatService(
        OllamaProvider(llm_client, settings.ollama_reasoning_effort),
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        retry_max_delay_seconds=settings.llm_retry_max_delay_seconds,
        max_concurrency=settings.llm_max_concurrency,
    )
    db_engine = create_engine(settings.database_url)
    app.state.db_engine = db_engine
    app.state.session_factory = create_session_factory(db_engine)
    yield
    await llm_client.close()
    await db_engine.dispose()


app = FastAPI(
    title="LLM Chat API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)


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
