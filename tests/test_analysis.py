import json
from collections.abc import Callable

from fastapi.testclient import TestClient

from app.schemas.analysis import SupportAnalysis
from tests.fakes import CannedProvider, FakeSession

VALID = {
    "category": "account",
    "priority": "high",
    "summary": "Не входит в аккаунт",
    "confidence": 0.9,
}
GOOD = json.dumps(VALID, ensure_ascii=False)
OUT_OF_RANGE = json.dumps(VALID | {"confidence": 150})
CUT_OFF = '{"category": "account", "priority":'
REQUEST = {"model": "qwen3:8b", "text": "Не могу войти в аккаунт после смены пароля."}


def test_ticket_is_analyzed_with_schema_constrained_call(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider(GOOD)
    use_provider(provider)

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["analysis"] == VALID
    assert body["usage"] == {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    assert body["attempts"] == 1

    call = provider.calls[0]
    assert call["json_schema"] == SupportAnalysis.model_json_schema()
    # Instructions and user data travel in separate messages
    assert [m.role for m in call["messages"]] == ["system", "user"]
    assert call["messages"][1].content == REQUEST["text"]


def test_invalid_output_is_recovered_with_feedback(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider(OUT_OF_RANGE, GOOD)
    use_provider(provider)

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["attempts"] == 2
    # Both generations were paid for
    assert body["usage"] == {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120}

    retry = provider.calls[1]["messages"]
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == OUT_OF_RANGE
    assert "confidence" in retry[3].content


def test_retries_are_bounded_and_last_reason_is_reported(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("не json", OUT_OF_RANGE)
    use_provider(provider)

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502
    body = response.json()
    # The first failure was invalid JSON; the reported one is the final attempt's
    assert body["error"] == "invalid_model_output:schema_mismatch"
    assert "attempts: 2" in body["detail"]
    assert len(provider.calls) == 2


def test_truncated_output_fails_fast(client: TestClient, use_provider: Callable[..., None]) -> None:
    provider = CannedProvider(CUT_OFF, GOOD, finish_reason="length")
    use_provider(provider)

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502
    assert response.json()["error"] == "invalid_model_output:truncated"
    # The same token limit would cut the next answer too: no second generation
    assert len(provider.calls) == 1


def test_schema_violation_detail_carries_no_values(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    use_provider(CannedProvider(OUT_OF_RANGE, OUT_OF_RANGE))

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "confidence" in detail
    assert "150" not in detail


def test_blank_summary_is_rejected(client: TestClient, use_provider: Callable[..., None]) -> None:
    blank = json.dumps(VALID | {"summary": "\n"})
    use_provider(CannedProvider(blank, blank))

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502


def test_empty_text_is_rejected_before_calling_the_model(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider()
    use_provider(provider)

    response = client.post("/v1/chat/structured", json={"model": "qwen3:8b", "text": ""})

    assert response.status_code == 422
    assert provider.calls == []


def test_success_is_logged_with_all_attempts(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    use_provider(CannedProvider(OUT_OF_RANGE, GOOD))

    client.post("/v1/chat/structured", json=REQUEST)

    [log] = session.usage_logs
    assert (log.endpoint, log.status, log.attempts) == ("structured", "success", 2)
    assert (log.input_tokens, log.output_tokens) == (80, 40)
    assert log.conversation_id is None


def test_failed_generations_are_logged_too(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    use_provider(CannedProvider(OUT_OF_RANGE, OUT_OF_RANGE))

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502
    [log] = session.usage_logs
    assert (log.status, log.attempts) == ("failed", 2)
    assert (log.input_tokens, log.output_tokens) == (80, 40)


def test_nothing_is_logged_when_the_model_never_answered(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    class Down:
        async def complete(self, *args: object, **kwargs: object) -> None:
            from app.llm.base import LLMProviderError

            raise LLMProviderError("connection refused", status_code=None)

    use_provider(Down())

    response = client.post("/v1/chat/structured", json=REQUEST)

    assert response.status_code == 502
    assert session.usage_logs == []
