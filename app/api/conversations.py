import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_session
from app.schemas.conversation import ConversationCreate, ConversationResponse, MessageResponse
from app.schemas.errors import ErrorResponse
from app.services import conversations

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse}}


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: ConversationCreate, session: SessionDep
) -> ConversationResponse:
    conversation = await conversations.create(session, request.title)
    return ConversationResponse.model_validate(conversation)


@router.get("/{conversation_id}", response_model=ConversationResponse, responses=NOT_FOUND)
async def get_conversation(conversation_id: uuid.UUID, session: SessionDep) -> ConversationResponse:
    conversation = await conversations.get(session, conversation_id)
    return ConversationResponse.model_validate(conversation)


@router.get(
    "/{conversation_id}/messages", response_model=list[MessageResponse], responses=NOT_FOUND
)
async def list_conversation_messages(
    conversation_id: uuid.UUID, session: SessionDep
) -> list[MessageResponse]:
    messages = await conversations.list_messages(session, conversation_id)
    return [MessageResponse.model_validate(message) for message in messages]
