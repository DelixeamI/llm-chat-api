from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.services.chat import ChatService


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    # One session per request, opened here and closed when the response is done.
    # A single shared session would mix unrelated requests in one transaction and
    # is not safe to use from several tasks at once.
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session
