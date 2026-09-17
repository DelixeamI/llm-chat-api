import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message


async def create_conversation(session: AsyncSession, title: str | None = None) -> Conversation:
    conversation = Conversation(title=title)
    session.add(conversation)
    # flush sends the INSERT and fills server-generated fields without ending the transaction
    await session.flush()
    return conversation


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID
) -> Conversation | None:
    return await session.get(Conversation, conversation_id)


async def list_messages(session: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    result = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(result)


def add_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    role: str,
    content: str,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    session.add(message)
    return message
