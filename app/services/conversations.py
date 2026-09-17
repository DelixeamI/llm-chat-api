import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repository
from app.db.models import Conversation, Message
from app.services.errors import ConversationNotFoundError


async def create(session: AsyncSession, title: str | None) -> Conversation:
    conversation = await repository.create_conversation(session, title)
    await session.commit()
    return conversation


async def get(session: AsyncSession, conversation_id: uuid.UUID) -> Conversation:
    conversation = await repository.get_conversation(session, conversation_id)
    if conversation is None:
        raise ConversationNotFoundError(conversation_id)
    return conversation


async def list_messages(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    # Checked first so an unknown conversation gives 404 instead of an empty list
    await get(session, conversation_id)
    return await repository.list_messages(session, conversation_id)
