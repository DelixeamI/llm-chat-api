"""A session that stores nothing.

The experiments measure provider latency and concurrency; writing to PostgreSQL would add
unrelated noise to the numbers. Only the slice of AsyncSession used by the chat flow exists.
"""

import uuid
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation


class NullSession:
    def add(self, instance: object) -> None:
        if isinstance(instance, Conversation) and instance.id is None:
            instance.id = uuid.uuid4()

    async def flush(self) -> None:
        return None

    async def get(self, entity: type[object], key: uuid.UUID) -> object | None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def null_session() -> AsyncSession:
    return cast(AsyncSession, NullSession())
