"""Тесты против настоящей PostgreSQL: мок сессии здесь не помог бы.

Внешние ключи, каскадное удаление, значения по умолчанию на стороне сервера, откат
транзакции и сортировка — это поведение самой базы, а не нашего кода.
"""

import uuid

import pytest
from sqlalchemy import URL, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repository
from app.llm.base import Completion
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.services.chat import ChatService
from app.services.errors import ConversationNotFoundError
from tests.db_support import run_with_session
from tests.fakes import FakeProvider

pytestmark = pytest.mark.integration

REQUEST = ChatRequest(model="fake", messages=[Message(role="user", content="вопрос")])


def make_service() -> ChatService:
    return ChatService(
        FakeProvider(),
        timeout_seconds=5,
        max_retries=0,
        retry_base_delay_seconds=0,
        retry_max_delay_seconds=0,
        max_concurrency=1,
    )


def test_conversation_round_trip(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        created = await repository.create_conversation(session, title="Заголовок")
        await session.commit()

        stored = await repository.get_conversation(session, created.id)
        assert stored is not None
        assert stored.title == "Заголовок"
        # created_at заполняет сама база, а не приложение
        assert stored.created_at is not None
        assert stored.created_at.tzinfo is not None

    run_with_session(scenario, database_url)


def test_chat_exchange_is_persisted_in_order(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        response = await make_service().generate_reply(REQUEST, session)

        messages = await repository.list_messages(session, response.conversation_id)
        assert [(m.role, m.content) for m in messages] == [
            ("user", "вопрос"),
            ("assistant", "fake reply"),
        ]
        assert messages[1].model == "fake"
        assert (messages[1].input_tokens, messages[1].output_tokens) == (3, 2)

    run_with_session(scenario, database_url)


def test_failure_between_writes_leaves_nothing_behind(
    database_url: URL, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario(session: AsyncSession) -> None:
        original = repository.add_message
        calls = {"n": 0}

        def failing(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("сбой между записями")
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(repository, "add_message", failing)

        with pytest.raises(RuntimeError):
            await make_service().generate_reply(REQUEST, session)

        monkeypatch.undo()
        conversations = await session.scalar(text("select count(*) from conversations"))
        messages = await session.scalar(text("select count(*) from messages"))
        assert (conversations, messages) == (0, 0)

    run_with_session(scenario, database_url)


def test_unknown_conversation_is_rejected_before_any_write(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        request = REQUEST.model_copy(update={"conversation_id": uuid.uuid4()})

        with pytest.raises(ConversationNotFoundError):
            await make_service().generate_reply(request, session)

        assert await session.scalar(text("select count(*) from messages")) == 0

    run_with_session(scenario, database_url)


def test_message_requires_existing_conversation(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        repository.add_message(session, uuid.uuid4(), role="user", content="сирота")

        # Внешний ключ не даст сохранить сообщение без диалога
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

    run_with_session(scenario, database_url)


def test_deleting_conversation_cascades_to_messages(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        response = await make_service().generate_reply(REQUEST, session)
        await session.execute(
            text("delete from conversations where id = :id"), {"id": response.conversation_id}
        )
        await session.commit()

        assert await session.scalar(text("select count(*) from messages")) == 0

    run_with_session(scenario, database_url)


def test_generation_params_do_not_leak_into_storage(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        request = REQUEST.model_copy(
            update={"params": GenerationParams(temperature=0.1, max_tokens=10)}
        )
        response = await make_service().generate_reply(request, session)

        messages = await repository.list_messages(session, response.conversation_id)
        # Параметры генерации пока не сохраняются: в таблице только текст, модель и токены
        assert all(not hasattr(m, "temperature") for m in messages)
        assert isinstance(messages[1].content, str)

    run_with_session(scenario, database_url)


def test_completion_shape_matches_stored_tokens(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        completion = await FakeProvider().complete("fake", REQUEST.messages, GenerationParams())
        assert completion == Completion(content="fake reply", input_tokens=3, output_tokens=2)

        response = await make_service().generate_reply(REQUEST, session)
        messages = await repository.list_messages(session, response.conversation_id)
        assert messages[1].input_tokens == completion.input_tokens

    run_with_session(scenario, database_url)
