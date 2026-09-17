from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.errors import ErrorResponse
from app.services.chat import ChatService

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={502: {"model": ErrorResponse}},
)
async def chat(
    request: ChatRequest, service: Annotated[ChatService, Depends(get_chat_service)]
) -> ChatResponse:
    return await service.generate_reply(request)
