from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_analysis_service, get_chat_service, get_session
from app.api.security import ChatClient, RateLimited
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.errors import ErrorResponse
from app.schemas.stream import StreamEvent
from app.services.analysis import AnalysisService
from app.services.chat import ChatService

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat(
    request: ChatRequest,
    client: ChatClient,
    _: RateLimited,
    service: Annotated[ChatService, Depends(get_chat_service)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatResponse:
    return await service.generate_reply(request, session, owner=client.name)


@router.post(
    "/chat/structured",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat_structured(
    request: AnalysisRequest,
    client: ChatClient,
    _: RateLimited,
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AnalysisResponse:
    """Classify a support ticket; the answer is guaranteed to match SupportAnalysis."""
    return await service.analyze(request, session, owner=client.name)


def _sse(event: StreamEvent) -> bytes:
    # One JSON object per event, so text containing newlines cannot break the framing
    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n".encode()


async def _event_stream(
    first: StreamEvent, events: AsyncGenerator[StreamEvent, None]
) -> AsyncGenerator[bytes, None]:
    try:
        yield _sse(first)
        async for event in events:
            yield _sse(event)
    finally:
        # Runs on normal end and on client disconnect alike. Closing the service generator
        # closes the provider stream and frees the concurrency slot.
        await events.aclose()


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events"},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat_stream(
    request: ChatRequest,
    client: ChatClient,
    rate: RateLimited,
    service: Annotated[ChatService, Depends(get_chat_service)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> StreamingResponse:
    """Stream the reply as Server-Sent Events: `delta` pieces, then `done` (or `error`)."""
    events = service.stream_reply(request, session, owner=client.name)
    try:
        # Waiting for the first event here, before the response starts, lets every failure
        # up to that point (unknown conversation, oversized prompt, provider down) produce a
        # normal HTTP error. Once the headers are sent the status can no longer change.
        first = await anext(events)
    except BaseException:
        await events.aclose()
        raise
    return StreamingResponse(
        _event_stream(first, events),
        media_type="text/event-stream",
        # Headers set on the injected Response are lost when a Response is returned directly
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            **(rate.headers() if rate else {}),
        },
    )
