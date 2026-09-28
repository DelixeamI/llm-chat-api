"""Тесты против настоящей PostgreSQL: мок сессии здесь не помог бы.

Внешние ключи, каскадное удаление, значения по умолчанию на стороне сервера, откат
транзакции и сортировка — это поведение самой базы, а не нашего кода.
"""

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import URL, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import repository
from app.llm.base import Completion
from app.schemas.chat import ChatRequest, GenerationParams, Message
from app.services import usage as usage_service
from app.services.chat import ChatService
from app.services.context import ContextBudget
from app.services.errors import ConversationNotFoundError
from app.services.pricing import ModelPrice, PriceList
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
        prices=PriceList({}),
        context=ContextBudget(limits={}, default_limit=32_768),
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
        usage = await session.scalar(text("select count(*) from usage_logs"))
        assert (conversations, messages, usage) == (0, 0, 0)

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


def test_usage_log_total_is_computed_by_the_database(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        response = await make_service().generate_reply(REQUEST, session)

        row = (
            await session.execute(
                text(
                    "select input_tokens, output_tokens, total_tokens, conversation_id "
                    "from usage_logs"
                )
            )
        ).one()
        assert tuple(row) == (3, 2, 5, response.conversation_id)

    run_with_session(scenario, database_url)


def test_deleting_conversation_keeps_its_usage(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        response = await make_service().generate_reply(REQUEST, session)
        await session.execute(
            text("delete from conversations where id = :id"), {"id": response.conversation_id}
        )
        await session.commit()

        # The cost happened; deleting the conversation must not erase it from the ledger
        rows = (await session.scalars(text("select conversation_id from usage_logs"))).all()
        assert list(rows) == [None]

    run_with_session(scenario, database_url)


def test_cost_is_stored_at_call_time(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        service = make_service()
        service.prices = PriceList(
            {
                "fake": ModelPrice(
                    input_per_million_usd=Decimal("1"), output_per_million_usd=Decimal("2")
                )
            }
        )
        await service.generate_reply(REQUEST, session)

        # 3 input * 1 + 2 output * 2 = 7 per million tokens
        assert await session.scalar(text("select cost_usd from usage_logs")) == Decimal(
            "0.00000700"
        )

    run_with_session(scenario, database_url)


def test_usage_and_cost_reports_aggregate_in_the_database(database_url: URL) -> None:
    async def scenario(session: AsyncSession) -> None:
        prices = {
            "priced": ModelPrice(
                input_per_million_usd=Decimal("1"), output_per_million_usd=Decimal("3")
            )
        }
        list_ = PriceList(prices)
        for model, status, tokens in [
            ("priced", "success", (1000, 500)),
            ("priced", "failed", (200, 100)),
            ("unpriced", "success", (50, 50)),
        ]:
            repository.add_usage_log(
                session,
                endpoint="structured",
                model=model,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                status=status,
                cost_usd=list_.cost_usd(model, *tokens),
            )
        # A row outside the window must not count
        await session.execute(
            text(
                "insert into usage_logs (id, created_at, endpoint, model, input_tokens, "
                "output_tokens, status, cost_usd) values (gen_random_uuid(), "
                "now() - interval '40 days', 'chat', 'priced', 999, 999, 'success', 1)"
            )
        )
        await session.commit()

        usage = await usage_service.usage_report(session, days=30)
        assert (usage.requests, usage.failed_requests, usage.total_tokens) == (3, 1, 1900)
        assert [(m.model, m.requests) for m in usage.by_model] == [("priced", 2), ("unpriced", 1)]

        cost = await usage_service.cost_report(session, days=30, usd_to_rub=Decimal("100"))
        # priced: (1000*1 + 500*3 + 200*1 + 100*3) / 1e6 = 0.003
        assert cost.cost_usd == Decimal("0.003")
        assert cost.cost_rub == Decimal("0.3")
        assert cost.daily_average_usd == Decimal("0.0001")
        assert cost.projected_monthly_usd == Decimal("0.003")
        assert cost.unpriced_requests == 1

    run_with_session(scenario, database_url)
