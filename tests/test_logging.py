import asyncio
import json
import logging
import re
import sys
from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.llm.base import Completion, LLMProviderError, StreamChunk
from app.main import app
from app.observability.logs import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    install_context_in_records,
)
from app.services.cache import ResponseCache
from tests.fakes import (
    ALICE,
    CannedProvider,
    FakeRedis,
    FakeSession,
    ScriptedProvider,
    StreamingProvider,
    as_redis,
)

BODY = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
ANALYSIS = '{"category": "billing", "priority": "low", "summary": "x", "confidence": 0.5}'


class SlowProvider(CannedProvider):
    async def complete(self, *args: Any, **kwargs: Any) -> Completion:
        await asyncio.sleep(1)
        raise AssertionError("the timeout should have fired first")


@pytest.fixture(autouse=True)
def records_carry_the_request_context() -> None:
    install_context_in_records()


@pytest.fixture
def caplogger(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


def events(caplog: pytest.LogCaptureFixture, name: str) -> list[Any]:
    return [r for r in caplog.records if getattr(r, "event", None) == name]


def make_record(message: str = "hello", **extra: object) -> logging.LogRecord:
    log = logging.LogRecord("app.test", logging.INFO, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(log, key, value)
    return log


# ---------------------------------------------------------------- the formatter


def test_json_line_has_the_standard_fields_and_the_extras() -> None:
    line = JsonFormatter().format(make_record("done", model="llama3", latency_ms=12.5))

    payload = json.loads(line)
    assert payload["message"] == "done" and payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["model"] == "llama3" and payload["latency_ms"] == 12.5
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", payload["ts"])


def test_empty_fields_are_left_out_and_unusual_types_are_written_as_text() -> None:
    payload = json.loads(
        JsonFormatter().format(make_record(request_id=None, cost=Decimal("0.000007")))
    )

    assert "request_id" not in payload
    assert payload["cost"] == "0.000007"


def test_exception_is_written_with_its_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        log = make_record("failed")
        log.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(log))

    assert "ValueError: boom" in payload["exception"]


@pytest.mark.parametrize(
    "field",
    ["prompt", "messages", "content", "completion", "api_key", "X-API-Key", "Authorization",
     "password", "token", "idempotency_key", "cookie"],
)  # fmt: skip
def test_sensitive_fields_are_redacted_even_when_passed_by_mistake(field: str) -> None:
    line = JsonFormatter().format(make_record("oops", **{field.replace("-", "_"): "TOP-SECRET"}))

    assert "TOP-SECRET" not in line
    assert "[redacted]" in line


def test_redaction_reaches_into_nested_values() -> None:
    line = JsonFormatter().format(
        make_record("oops", payload={"model": "m", "messages": [{"content": "TOP-SECRET"}]})
    )

    payload = json.loads(line)["payload"]
    assert payload["model"] == "m" and payload["messages"] == "[redacted]"
    assert "TOP-SECRET" not in line


def test_token_counts_are_not_mistaken_for_secrets() -> None:
    payload = json.loads(JsonFormatter().format(make_record(input_tokens=5, output_tokens=7)))

    assert payload["input_tokens"] == 5 and payload["output_tokens"] == 7


def test_text_format_shows_the_same_fields_for_a_terminal() -> None:
    line = TextFormatter().format(make_record("done", model="llama3", prompt="TOP-SECRET"))

    assert "done" in line and "model=llama3" in line
    assert "TOP-SECRET" not in line


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:], root.level = handlers, level


def test_configured_logging_prints_info_lines_as_json(
    capsys: pytest.CaptureFixture[str], restore_root_logger: None
) -> None:
    configure_logging("INFO", "json")

    logging.getLogger("app.anything").info("visible", extra={"model": "llama3"})
    logging.getLogger("app.anything").debug("hidden")

    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "visible"


# ---------------------------------------------------------------- request identity


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_two_requests_get_two_ids(client: TestClient) -> None:
    ids = {client.get("/health").headers["x-request-id"] for _ in range(3)}

    assert len(ids) == 3


def test_a_caller_may_supply_its_own_id(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-12345"})

    assert response.headers["x-request-id"] == "trace-abc-12345"


@pytest.mark.parametrize("supplied", ["short", "x" * 65, "has space in it", "semi;colon;12345"])
def test_a_malformed_id_is_replaced_not_trusted(client: TestClient, supplied: str) -> None:
    response = client.get("/health", headers={"X-Request-ID": supplied})

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


@pytest.mark.parametrize(
    ("method", "path", "headers", "status"),
    [
        ("POST", "/v1/chat", {"X-API-Key": "wrong"}, 401),
        ("POST", "/v1/chat", ALICE, 422),
        ("GET", "/no-such-path", {}, 404),
    ],
)
def test_the_id_is_on_error_responses_too(
    client: TestClient, method: str, path: str, headers: dict[str, str], status: int
) -> None:
    response = client.request(method, path, json={} if method == "POST" else None, headers=headers)

    assert response.status_code == status
    assert "x-request-id" in response.headers


def test_log_lines_of_a_request_share_its_id(
    client: TestClient, caplogger: pytest.LogCaptureFixture
) -> None:
    response = client.post("/v1/chat", json=BODY)

    request_id = response.headers["x-request-id"]
    records: list[Any] = caplogger.records
    ids = {r.request_id for r in records if r.name in ("app.request", "app.llm")}
    assert ids == {request_id}


def test_one_request_line_per_request_with_the_route_not_the_path(
    client: TestClient, caplogger: pytest.LogCaptureFixture
) -> None:
    client.get("/v1/conversations/5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1")

    [line] = events(caplogger, "request")
    assert (line.method, line.status) == ("GET", 404)
    # The template: with the real id every conversation would be a separate series in a chart
    assert line.path == "/v1/conversations/{conversation_id}"
    assert line.duration_ms >= 0
    assert line.client == "alice"


def test_unknown_paths_are_logged_as_unmatched(
    client: TestClient, caplogger: pytest.LogCaptureFixture
) -> None:
    client.get("/wp-login.php?x=<script>")

    [line] = events(caplogger, "request")
    assert line.path == "unmatched" and line.status == 404


def test_unauthenticated_request_has_no_client(
    client: TestClient, caplogger: pytest.LogCaptureFixture
) -> None:
    client.post("/v1/chat", json=BODY, headers={"X-API-Key": "wrong"})

    [line] = events(caplogger, "request")
    assert line.status == 401 and line.client is None


def test_client_name_does_not_leak_between_requests(
    client: TestClient, caplogger: pytest.LogCaptureFixture
) -> None:
    client.post("/v1/chat", json=BODY)
    client.get("/health", headers={"X-API-Key": ""})

    lines = events(caplogger, "request")
    assert [line.client for line in lines] == ["alice", None]


def test_an_unhandled_error_answers_with_the_request_id_and_is_logged(
    client: TestClient, caplogger: pytest.LogCaptureFixture
) -> None:
    def broken() -> None:
        raise RuntimeError("TOP-SECRET internal detail")

    app.dependency_overrides[get_chat_service] = broken

    response = client.post("/v1/chat", json=BODY)

    assert response.status_code == 500
    request_id = response.headers["x-request-id"]
    assert response.json() == {
        "error": "internal_error",
        "detail": f"Unexpected error, request id {request_id}",
    }
    # The client gets the ID to quote; the cause stays in the log
    assert "TOP-SECRET" not in response.text
    [line] = events(caplogger, "request")
    assert line.status == 500 and line.levelno == logging.ERROR
    assert any(r.exc_info for r in caplogger.records if r.name == "app.main" or r.exc_info)


# ---------------------------------------------------------------- one line per model call


def test_a_model_call_is_logged_with_latency_tokens_cost_and_status(
    client: TestClient, caplogger: pytest.LogCaptureFixture
) -> None:
    client.post("/v1/chat", json=BODY)

    [call] = events(caplogger, "llm_call")
    assert (call.endpoint, call.model, call.status) == ("chat", "llama3", "success")
    assert (call.input_tokens, call.output_tokens) == (3, 2)
    assert Decimal(call.cost_usd) == Decimal("0.000007")
    assert call.latency_ms >= 0 and call.retry_count == 0 and call.structured is False
    assert call.finish_reason is None or isinstance(call.finish_reason, str)


def test_retries_are_counted(
    client: TestClient, use_provider: Callable[..., None], caplogger: pytest.LogCaptureFixture
) -> None:
    use_provider(ScriptedProvider(LLMProviderError("busy", status_code=503, retryable=True)))

    client.post("/v1/chat", json=BODY)

    [call] = events(caplogger, "llm_call")
    assert call.status == "success" and call.retry_count == 1


def test_a_failed_call_is_logged_as_a_warning(
    client: TestClient, use_provider: Callable[..., None], caplogger: pytest.LogCaptureFixture
) -> None:
    use_provider(ScriptedProvider(LLMProviderError("bad", status_code=400)))

    client.post("/v1/chat", json=BODY)

    [call] = events(caplogger, "llm_call")
    assert call.status == "error" and call.levelno == logging.WARNING
    assert call.input_tokens is None and call.cost_usd is None


def test_a_timeout_is_told_apart_from_other_errors(
    client: TestClient, use_provider: Callable[..., None], caplogger: pytest.LogCaptureFixture
) -> None:
    use_provider(SlowProvider(), timeout_seconds=0.01, max_retries=0)

    client.post("/v1/chat", json=BODY)

    [call] = events(caplogger, "llm_call")
    assert call.status == "timeout"


def test_a_cache_hit_is_logged_as_cached_and_free(
    client: TestClient, use_provider: Callable[..., None], caplogger: pytest.LogCaptureFixture
) -> None:
    cache = ResponseCache(as_redis(FakeRedis()), ttl_seconds=60, scope="t")
    use_provider(CannedProvider("Paris"), cache=cache)
    body = BODY | {"params": {"temperature": 0}}

    client.post("/v1/chat", json=body)
    client.post("/v1/chat", json=body)

    statuses = [c.status for c in events(caplogger, "llm_call")]
    assert statuses == ["success", "cached"]
    assert events(caplogger, "llm_call")[1].cost_usd is None


def test_structured_calls_are_marked(
    client: TestClient, use_provider: Callable[..., None], caplogger: pytest.LogCaptureFixture
) -> None:
    use_provider(CannedProvider(ANALYSIS))

    client.post("/v1/chat/structured", json={"model": "llama3", "text": "Не могу войти"})

    [call] = events(caplogger, "llm_call")
    assert call.endpoint == "structured" and call.structured is True


def test_a_stream_is_logged_with_time_to_first_piece(
    client: TestClient, use_provider: Callable[..., None], caplogger: pytest.LogCaptureFixture
) -> None:
    use_provider(
        StreamingProvider(
            StreamChunk(content="Par"),
            StreamChunk(content="is"),
            StreamChunk(finish_reason="stop"),
            StreamChunk(input_tokens=7, output_tokens=2),
        )
    )

    client.post("/v1/chat/stream", json=BODY)

    [call] = events(caplogger, "llm_call")
    assert (call.endpoint, call.status) == ("stream", "success")
    assert call.ttft_ms is not None and 0 <= call.ttft_ms <= call.latency_ms
    assert (call.input_tokens, call.output_tokens) == (7, 2)


# ---------------------------------------------------------------- what is never logged


def test_prompts_replies_and_keys_never_reach_any_log_line(
    client: TestClient,
    use_provider: Callable[..., None],
    session: FakeSession,
    caplogger: pytest.LogCaptureFixture,
) -> None:
    use_provider(CannedProvider("REPLY-SECRET-7731"))
    prompt = {"model": "llama3", "messages": [{"role": "user", "content": "PROMPT-SECRET-4419"}]}

    client.post("/v1/chat", json=prompt, headers={"Idempotency-Key": "IDEM-SECRET-2208"})
    client.post("/v1/chat/stream", json=prompt)
    client.post("/v1/chat", json=prompt, headers={"X-API-Key": "KEY-SECRET-9053"})
    client.post("/v1/chat", json={"model": "llama3", "messages": "PROMPT-SECRET-4419"})

    formatter = JsonFormatter()
    written = "\n".join(formatter.format(r) for r in caplogger.records)
    assert caplogger.records, "nothing was logged, so this test proves nothing"
    for secret in (
        "PROMPT-SECRET-4419",
        "REPLY-SECRET-7731",
        "IDEM-SECRET-2208",
        "KEY-SECRET-9053",
    ):
        assert secret not in written
