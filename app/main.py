import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.llm.base import LLMError
from app.schemas.errors import ErrorResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Chat API",
    version="0.1.0",
)

app.include_router(chat_router)


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
