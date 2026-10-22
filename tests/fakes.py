import asyncio
import uuid
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

from pydantic import SecretStr
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import ApiKeyConfig
from app.db.models import Conversation, UsageLog
from app.db.models import Message as MessageRow
from app.llm.base import Completion, LLMError, StreamChunk
from app.schemas.chat import GenerationParams, Message
from app.services.chat import ChatService
from app.services.pricing import ModelPrice, PriceList

ServiceFactory = Callable[..., ChatService]

ALICE_KEY = "alice-test-key-0123456789"
BOB_KEY = "bob-test-key-0123456789ab"
OPS_KEY = "ops-test-key-0123456789ab"
ROOT_KEY = "root-test-key-0123456789a"
TEST_KEYS = [
    ApiKeyConfig(client="alice", key=SecretStr(ALICE_KEY), scopes=frozenset({"chat"})),
    ApiKeyConfig(client="bob", key=SecretStr(BOB_KEY), scopes=frozenset({"chat"})),
    # Operator without the right to use the model: proves that 403 is about scope, not identity
    ApiKeyConfig(client="ops", key=SecretStr(OPS_KEY), scopes=frozenset({"admin"})),
    ApiKeyConfig(client="root", key=SecretStr(ROOT_KEY), scopes=frozenset({"chat", "admin"})),
]
ALICE = {"X-API-Key": ALICE_KEY}
BOB = {"X-API-Key": BOB_KEY}
OPS = {"X-API-Key": OPS_KEY}
ROOT = {"X-API-Key": ROOT_KEY}

# Round numbers so expected costs are easy to check by hand
TEST_PRICES = PriceList(
    {"llama3": ModelPrice(input_per_million_usd=Decimal("1"), output_per_million_usd=Decimal("2"))}
)


def chunks_from(completion: Completion) -> list[StreamChunk]:
    """The stream a real provider would produce for this completion: words, then usage."""
    words = completion.content.split(" ")
    pieces = [
        StreamChunk(content=word + (" " if i < len(words) - 1 else ""))
        for i, word in enumerate(words)
    ]
    closing = StreamChunk(
        finish_reason=completion.finish_reason or "stop",
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )
    return [*pieces, closing]


class FakeProvider:
    async def stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> AsyncGenerator[StreamChunk, None]:
        for chunk in chunks_from(Completion("fake reply", 3, 2, "stop")):
            yield chunk

    async def complete(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        return Completion(content="fake reply", input_tokens=3, output_tokens=2)


class ScriptedProvider:
    """Raises the given errors one per call, then succeeds. Counts calls."""

    async def stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> AsyncGenerator[StreamChunk, None]:
        completion = await self.complete(model, messages, params)
        for chunk in chunks_from(completion):
            yield chunk

    def __init__(self, *errors: LLMError) -> None:
        self._errors = list(errors)
        self.calls = 0

    async def complete(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return Completion(content="recovered", input_tokens=1, output_tokens=1)


class CannedProvider:
    """Returns the given texts in order and records every call."""

    async def stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> AsyncGenerator[StreamChunk, None]:
        completion = await self.complete(model, messages, params)
        for chunk in chunks_from(completion):
            yield chunk

    def __init__(self, *contents: str, finish_reason: str = "stop") -> None:
        self._contents = list(contents)
        self._finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        self.calls.append({"messages": messages, "json_schema": json_schema})
        return Completion(
            content=self._contents.pop(0),
            input_tokens=40,
            output_tokens=20,
            finish_reason=self._finish_reason,
        )


class StreamingProvider:
    """Streams a fixed script. Each step is a chunk, an error to raise, or a pause in seconds.

    Records how many streams were opened and how many were closed, so tests can prove that a
    consumer which walks away does not leave a generation running.
    """

    def __init__(self, *script: StreamChunk | Exception | float) -> None:
        self._script = list(script)
        self.opened = 0
        self.closed = 0
        self.delivered = 0

    async def complete(
        self,
        model: str,
        messages: list[Message],
        params: GenerationParams,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        raise AssertionError("streaming tests never call complete()")

    async def stream(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> AsyncGenerator[StreamChunk, None]:
        self.opened += 1
        try:
            for step in self._script:
                if isinstance(step, Exception):
                    raise step
                if isinstance(step, float):
                    await asyncio.sleep(step)
                    continue
                self.delivered += 1
                yield step
        finally:
            self.closed += 1


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

    @property
    def usage_logs(self) -> list[UsageLog]:
        return [row for row in self.added if isinstance(row, UsageLog)]


class FakeRedis:
    """In-memory stand-in for the few Redis commands the cache uses.

    TTLs are recorded, not enforced: expiry is Redis behaviour and is tested against the
    real server. `down=True` makes every command fail like an unreachable Redis.
    """

    def __init__(self, *, down: bool = False) -> None:
        self.data: dict[str, bytes] = {}
        self.ttls: dict[str, int | None] = {}
        self.down = down
        self.commands = 0

    def _command(self) -> None:
        self.commands += 1
        if self.down:
            raise RedisConnectionError("Redis is down")

    async def get(self, key: str) -> bytes | None:
        self._command()
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._command()
        self.data[key] = value.encode()
        self.ttls[key] = ex
        return True

    async def incr(self, key: str) -> int:
        self._command()
        value = int(self.data.get(key, b"0")) + 1
        self.data[key] = str(value).encode()
        return value

    async def mget(self, *keys: str) -> list[bytes | None]:
        self._command()
        return [self.data.get(key) for key in keys]


def as_redis(fake: FakeRedis) -> Redis:
    return cast(Redis, fake)


def as_session(fake: FakeSession) -> AsyncSession:
    """FakeSession implements only the slice of AsyncSession the chat flow uses."""
    return cast(AsyncSession, fake)


def make_conversation(title: str | None = None, owner: str | None = "alice") -> Conversation:
    """A conversation row as it would look after the database filled its defaults."""
    return Conversation(id=uuid.uuid4(), title=title, owner=owner, created_at=datetime.now(UTC))
