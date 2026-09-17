import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation
from app.db.models import Message as MessageRow
from app.llm.base import Completion, LLMError
from app.schemas.chat import GenerationParams, Message
from app.services.chat import ChatService

ServiceFactory = Callable[..., ChatService]


class FakeProvider:
    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        return Completion(content="fake reply", input_tokens=3, output_tokens=2)


class ScriptedProvider:
    """Raises the given errors one per call, then succeeds. Counts calls."""

    def __init__(self, *errors: LLMError) -> None:
        self._errors = list(errors)
        self.calls = 0

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return Completion(content="recovered", input_tokens=1, output_tokens=1)


class FakeSession:
    """Minimal stand-in for AsyncSession: enough for the chat flow, nothing more.

    Contract tests use it so they stay independent of a running database. Real database
    behaviour is covered by integration tests against PostgreSQL.
    """

    def __init__(self, conversations: dict[uuid.UUID, Conversation] | None = None) -> None:
        self.conversations = conversations or {}
        self.added: list[object] = []
        self.commits = 0
        self.rollbacks = 0

    def add(self, instance: object) -> None:
        self.added.append(instance)
        # A real database fills the primary key and created_at itself; the fake does the
        # same so responses built from these rows are complete
        if isinstance(instance, Conversation | MessageRow):
            if instance.id is None:
                instance.id = uuid.uuid4()
            if instance.created_at is None:
                instance.created_at = datetime.now(UTC)
        if isinstance(instance, MessageRow) and instance.seq is None:
            instance.seq = len(self.messages)
        if isinstance(instance, Conversation):
            self.conversations[instance.id] = instance

    async def flush(self) -> None:
        return None

    async def get(self, entity: type[object], key: uuid.UUID) -> object | None:
        return self.conversations.get(key)

    async def scalars(self, statement: object) -> list[MessageRow]:
        return self.messages

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    @property
    def messages(self) -> list[MessageRow]:
        return [row for row in self.added if isinstance(row, MessageRow)]


def as_session(fake: FakeSession) -> AsyncSession:
    """FakeSession implements only the slice of AsyncSession the chat flow uses."""
    return cast(AsyncSession, fake)


def make_conversation(title: str | None = None) -> Conversation:
    """A conversation row as it would look after the database filled its defaults."""
    return Conversation(id=uuid.uuid4(), title=title, created_at=datetime.now(UTC))
