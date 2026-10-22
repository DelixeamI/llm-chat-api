import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import Client
from app.db import repository
from app.db.models import Conversation, Message
from app.services.errors import ConversationNotFoundError


async def create(session: AsyncSession, title: str | None, client: Client) -> Conversation:
    conversation = await repository.create_conversation(session, title, owner=client.name)
    await session.commit()
    return conversation


async def get(session: AsyncSession, conversation_id: uuid.UUID, client: Client) -> Conversation:
    conversation = await repository.get_conversation(session, conversation_id)
    # Not yours is reported as not found, so ids cannot be probed for existence. An admin
    # may read any conversation, including those from before owners were recorded.
    if conversation is None or not (client.has("admin") or conversation.owner == client.name):
        raise ConversationNotFoundError(conversation_id)
    return conversation


async def list_messages(
    session: AsyncSession, conversation_id: uuid.UUID, client: Client
) -> list[Message]:
    # Checked first so an unknown conversation gives 404 instead of an empty list
    await get(session, conversation_id, client)
    return await repository.list_messages(session, conversation_id)
