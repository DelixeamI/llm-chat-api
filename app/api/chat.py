from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_analysis_service, get_chat_service, get_session
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.errors import ErrorResponse
from app.services.analysis import AnalysisService
from app.services.chat import ChatService

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatResponse:
    return await service.generate_reply(request, session)


@router.post(
    "/chat/structured",
    response_model=AnalysisResponse,
    responses={502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def chat_structured(
    request: AnalysisRequest,
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
) -> AnalysisResponse:
    """Classify a support ticket; the answer is guaranteed to match SupportAnalysis."""
    return await service.analyze(request)
