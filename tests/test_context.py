import math
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from app.schemas.chat import Message
from app.services.context import CHARS_PER_TOKEN, ContextBudget
from app.services.errors import ContextOverflowError
from tests.fakes import CannedProvider

BUDGET = ContextBudget(limits={"small": 200}, default_limit=4096)


def text_of_tokens(tokens: int) -> str:
    return "я" * math.ceil(tokens * CHARS_PER_TOKEN)


def user(tokens: int) -> Message:
    return Message(role="user", content=text_of_tokens(tokens))


def test_estimate_overestimates_measured_russian_text() -> None:
    # 43 characters of Russian measured at 16 tokens on qwen3:8b
    content = "Не могу войти в аккаунт после смены пароля."
    per_message = BUDGET.estimate([Message(role="user", content=content)])

    assert per_message - 10 - 8 >= 16


def test_output_budget_is_reserved() -> None:
    messages = [user(100)]
    estimated = BUDGET.estimate(messages)

    BUDGET.check("small", messages, max_tokens=200 - estimated)
    with pytest.raises(ContextOverflowError) as exc_info:
        BUDGET.check("small", messages, max_tokens=200 - estimated + 1)

    assert exc_info.value.limit == 200


def test_unknown_model_uses_default_limit() -> None:
    assert BUDGET.limit_for("anything") == 4096


def test_truncation_drops_oldest_turns_and_keeps_essentials() -> None:
    budget = ContextBudget(limits={}, default_limit=200, strategy="truncate_oldest")
    system = Message(role="system", content="инструкция")
    old_q, old_a, question = (
        user(60),
        Message(role="assistant", content=text_of_tokens(60)),
        user(60),
    )

    kept, dropped = budget.fit("m", [system, old_q, old_a, question], max_tokens=50)

    assert dropped == 2
    assert kept == [system, question]


def test_truncation_stops_as_soon_as_it_fits() -> None:
    budget = ContextBudget(limits={}, default_limit=200, strategy="truncate_oldest")
    oldest, recent_answer, question = user(150), Message(role="assistant", content="ok"), user(20)

    kept, dropped = budget.fit("m", [oldest, recent_answer, question], max_tokens=50)

    assert dropped == 1
    assert kept == [recent_answer, question]


def test_truncation_never_cuts_the_question_itself() -> None:
    budget = ContextBudget(limits={}, default_limit=200, strategy="truncate_oldest")

    with pytest.raises(ContextOverflowError):
        budget.fit("m", [user(20), user(500)], max_tokens=50)


def test_reject_strategy_changes_nothing_when_it_fits() -> None:
    messages = [user(10)]

    assert BUDGET.fit("small", messages, max_tokens=50) == (messages, 0)


def payload(messages: list[dict[str, str]], max_tokens: int = 100) -> dict[str, object]:
    return {"model": "llama3", "messages": messages, "params": {"max_tokens": max_tokens}}


def test_oversized_chat_is_rejected_before_calling_the_model(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("never used")
    use_provider(provider, context_limit=300)

    response = client.post(
        "/v1/chat", json=payload([{"role": "user", "content": text_of_tokens(400)}])
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "context_length_exceeded"
    assert "300" in body["detail"]
    assert provider.calls == []


def test_truncate_strategy_drops_history_and_reports_it(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("ответ")
    use_provider(provider, context_limit=200, overflow="truncate_oldest")
    history = [
        {"role": "user", "content": text_of_tokens(150)},
        {"role": "assistant", "content": text_of_tokens(150)},
        {"role": "user", "content": "последний вопрос"},
    ]

    response = client.post("/v1/chat", json=payload(history))

    assert response.status_code == 200
    assert response.json()["dropped_messages"] == 2
    assert [m.content for m in provider.calls[0]["messages"]] == ["последний вопрос"]


def test_oversized_structured_request_is_rejected(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("never used")
    use_provider(provider, context_limit=300)

    response = client.post(
        "/v1/chat/structured", json={"model": "llama3", "text": text_of_tokens(400)}
    )

    assert response.status_code == 400
    assert provider.calls == []
