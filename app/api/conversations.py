import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.api.security import ChatClient, ConversationClient
from app.schemas.conversation import ConversationCreate, ConversationResponse, MessageResponse
from app.schemas.errors import ErrorResponse
from app.services import conversations

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
NOT_FOUND: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
}


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: ConversationCreate, session: SessionDep, client: ChatClient
) -> ConversationResponse:
    conversation = await conversations.create(session, request.title, client)
    return ConversationResponse.model_validate(conversation)


@router.get("/{conversation_id}", response_model=ConversationResponse, responses=NOT_FOUND)
async def get_conversation(
    conversation_id: uuid.UUID, session: SessionDep, client: ConversationClient
) -> ConversationResponse:
    conversation = await conversations.get(session, conversation_id, client)
    return ConversationResponse.model_validate(conversation)


@router.get(
    "/{conversation_id}/messages", response_model=list[MessageResponse], responses=NOT_FOUND
)
async def list_conversation_messages(
    conversation_id: uuid.UUID, session: SessionDep, client: ConversationClient
) -> list[MessageResponse]:
    messages = await conversations.list_messages(session, conversation_id, client)
    return [MessageResponse.model_validate(message) for message in messages]
