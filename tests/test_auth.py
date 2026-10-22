import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.api.dependencies import get_response_cache
from app.api.security import get_authenticator
from app.auth import ApiKeyConfig, Authenticator
from app.config import Settings
from app.main import app
from app.schemas.usage import CostReport, UsageReport
from app.services.cache import ResponseCache
from tests.fakes import (
    ALICE,
    ALICE_KEY,
    BOB,
    OPS,
    ROOT,
    TEST_KEYS,
    CannedProvider,
    FakeProvider,
    FakeRedis,
    FakeSession,
    StreamingProvider,
    as_redis,
    make_conversation,
)

CHAT_BODY = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
ANALYSIS_BODY = {"model": "llama3", "text": "Не могу войти в аккаунт"}

PROTECTED = [
    ("POST", "/v1/chat", CHAT_BODY),
    ("POST", "/v1/chat/stream", CHAT_BODY),
    ("POST", "/v1/chat/structured", ANALYSIS_BODY),
    ("POST", "/v1/conversations", {}),
    ("GET", "/v1/conversations/5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1", None),
    ("GET", "/v1/conversations/5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1/messages", None),
    ("GET", "/v1/usage", None),
    ("GET", "/v1/usage/cost", None),
    ("GET", "/v1/cache/stats", None),
]


def call(client: TestClient, method: str, path: str, body: object, headers: dict[str, str]):  # type: ignore[no-untyped-def]
    return client.request(method, path, json=body, headers=headers)


# ---------------------------------------------------------------- the authenticator itself


def test_known_key_identifies_its_client() -> None:
    authenticator = Authenticator(TEST_KEYS)

    client = authenticator.authenticate(ALICE_KEY)

    assert client is not None and client.name == "alice" and client.scopes == {"chat"}


@pytest.mark.parametrize(
    "presented", [None, "", "wrong-key-0123456789", ALICE_KEY + "x", ALICE_KEY[:-1]]
)
def test_unknown_or_missing_key_identifies_nobody(presented: str | None) -> None:
    assert Authenticator(TEST_KEYS).authenticate(presented) is None


def test_a_client_may_have_several_keys_for_rotation() -> None:
    old = ApiKeyConfig(client="alice", key=SecretStr("old-key-0123456789ab"))
    new = ApiKeyConfig(client="alice", key=SecretStr("new-key-0123456789ab"))
    authenticator = Authenticator([old, new])

    assert authenticator.authenticate("old-key-0123456789ab").name == "alice"  # type: ignore[union-attr]
    assert authenticator.authenticate("new-key-0123456789ab").name == "alice"  # type: ignore[union-attr]


def test_plain_keys_are_not_kept_by_the_authenticator() -> None:
    authenticator = Authenticator(TEST_KEYS)

    assert (
        ALICE_KEY not in repr(vars(authenticator))
        and ALICE_KEY.encode() not in repr(vars(authenticator)).encode()
    )


def test_key_configuration_never_prints_the_key() -> None:
    config = ApiKeyConfig(client="alice", key=SecretStr(ALICE_KEY))

    assert ALICE_KEY not in repr(config) and ALICE_KEY not in str(config.model_dump())


def test_short_key_is_rejected_at_configuration_time() -> None:
    with pytest.raises(ValidationError):
        ApiKeyConfig(client="alice", key=SecretStr("short"))


def test_client_name_must_be_simple() -> None:
    with pytest.raises(ValidationError):
        ApiKeyConfig(client="Alice Smith", key=SecretStr(ALICE_KEY))


def test_keys_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "API_KEYS",
        json.dumps(
            [{"client": "svc", "key": "service-key-0123456789", "scopes": ["chat", "admin"]}]
        ),
    )

    settings = Settings(_env_file=None)

    [config] = settings.api_keys
    assert config.client == "svc" and config.scopes == {"chat", "admin"}
    assert config.key.get_secret_value() == "service-key-0123456789"


def test_no_configured_keys_means_closed_not_open(client: TestClient) -> None:
    app.dependency_overrides[get_authenticator] = lambda: Authenticator([])

    response = client.post("/v1/chat", json=CHAT_BODY, headers=ALICE)

    assert response.status_code == 401


# ---------------------------------------------------------------- authentication: 401


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_every_protected_endpoint_requires_a_key(
    client: TestClient, method: str, path: str, body: object
) -> None:
    response = call(client, method, path, body, headers={"X-API-Key": ""})

    assert response.status_code == 401
    assert response.json() == {"error": "invalid_api_key", "detail": "Missing or invalid API key"}
    assert response.headers["www-authenticate"] == "ApiKey"


def test_wrong_key_answers_exactly_like_a_missing_one(client: TestClient) -> None:
    missing = client.post("/v1/chat", json=CHAT_BODY, headers={"X-API-Key": ""})
    wrong = client.post("/v1/chat", json=CHAT_BODY, headers={"X-API-Key": "guess-0123456789abcdef"})

    assert (wrong.status_code, wrong.json()) == (missing.status_code, missing.json())


def test_invalid_key_is_rejected_before_the_request_body_is_judged(client: TestClient) -> None:
    # Otherwise an anonymous caller could learn the request format from validation errors
    response = client.post("/v1/chat", json={"nonsense": True}, headers={"X-API-Key": "nope"})

    assert response.status_code == 401


@pytest.mark.parametrize("path", ["/", "/health"])
def test_liveness_and_identity_stay_public(client: TestClient, path: str) -> None:
    assert client.get(path, headers={"X-API-Key": ""}).status_code == 200


def test_rejected_key_is_not_logged(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    secret = "guess-0123456789abcdef"
    with caplog.at_level(logging.DEBUG):
        client.post("/v1/chat", json=CHAT_BODY, headers={"X-API-Key": secret})
        client.post("/v1/chat", json=CHAT_BODY, headers=ALICE)

    assert secret not in caplog.text and ALICE_KEY not in caplog.text


# ---------------------------------------------------------------- authorization: 403


def test_known_client_without_the_scope_gets_403_not_401(client: TestClient) -> None:
    response = client.get("/v1/usage", headers=ALICE)

    assert response.status_code == 403
    assert response.json()["error"] == "insufficient_scope"


@pytest.mark.parametrize("path", ["/v1/usage", "/v1/usage/cost", "/v1/cache/stats"])
def test_reports_are_closed_to_chat_clients(client: TestClient, path: str) -> None:
    assert client.get(path, headers=ALICE).status_code == 403
    assert client.get(path, headers=BOB).status_code == 403


def test_reports_are_open_to_admins(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    usage_report = UsageReport(
        since=now, days=30, requests=0, failed_requests=0, input_tokens=0, output_tokens=0,
        total_tokens=0, by_model=[],
    )  # fmt: skip
    cost_report = CostReport(
        since=now, days=30, usd_to_rub=Decimal(95), cost_usd=Decimal(0), cost_rub=Decimal(0),
        daily_average_usd=Decimal(0), projected_monthly_usd=Decimal(0),
        projected_monthly_rub=Decimal(0), unpriced_requests=0, by_model=[],
    )  # fmt: skip

    async def fake_usage(*args: object) -> UsageReport:
        return usage_report

    async def fake_cost(*args: object) -> CostReport:
        return cost_report

    monkeypatch.setattr("app.api.usage.usage.usage_report", fake_usage)
    monkeypatch.setattr("app.api.usage.usage.cost_report", fake_cost)
    app.dependency_overrides[get_response_cache] = lambda: None

    for headers in (ROOT, OPS):
        for path in ("/v1/usage", "/v1/usage/cost", "/v1/cache/stats"):
            assert client.get(path, headers=headers).status_code == 200


def test_model_endpoints_need_the_chat_scope(client: TestClient) -> None:
    # An operator account may read reports but may not spend the model
    for path, body in [("/v1/chat", CHAT_BODY), ("/v1/chat/stream", CHAT_BODY)]:
        assert client.post(path, json=body, headers=OPS).status_code == 403
    assert client.post("/v1/chat/structured", json=ANALYSIS_BODY, headers=OPS).status_code == 403


def test_forbidden_request_never_reaches_the_model(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("never")
    use_provider(provider)

    client.post("/v1/chat", json=CHAT_BODY, headers=OPS)

    assert provider.calls == []


# ---------------------------------------------------------------- ownership


def test_conversation_and_spending_are_attributed_to_the_caller(
    client: TestClient, session: FakeSession
) -> None:
    response = client.post("/v1/chat", json=CHAT_BODY)

    assert response.status_code == 200
    conversation = next(iter(session.conversations.values()))
    assert conversation.owner == "alice"
    assert [log.client for log in session.usage_logs] == ["alice"]


def test_created_conversation_belongs_to_the_creator(
    client: TestClient, session: FakeSession
) -> None:
    client.post("/v1/conversations", json={"title": "mine"}, headers=BOB)

    assert [c.owner for c in session.conversations.values()] == ["bob"]


def test_other_clients_conversation_looks_like_a_missing_one(
    client: TestClient, session: FakeSession
) -> None:
    conversation = make_conversation(owner="alice")
    session.conversations[conversation.id] = conversation

    mine = client.get(f"/v1/conversations/{conversation.id}", headers=ALICE)
    theirs = client.get(f"/v1/conversations/{conversation.id}", headers=BOB)
    absent = client.get("/v1/conversations/5d2cdb52-7f0f-4a22-a3c0-6f0dd3d9d8f1", headers=BOB)

    assert mine.status_code == 200
    # 404, not 403: a 403 would confirm that the id exists
    assert theirs.status_code == absent.status_code == 404
    assert theirs.json()["error"] == absent.json()["error"] == "conversation_not_found"


def test_other_clients_history_is_not_readable(client: TestClient, session: FakeSession) -> None:
    conversation = make_conversation(owner="alice")
    session.conversations[conversation.id] = conversation

    assert (
        client.get(f"/v1/conversations/{conversation.id}/messages", headers=BOB).status_code == 404
    )


def test_cannot_continue_someone_elses_conversation(
    client: TestClient, session: FakeSession, use_provider: Callable[..., None]
) -> None:
    conversation = make_conversation(owner="alice")
    session.conversations[conversation.id] = conversation
    provider = CannedProvider("never")
    use_provider(provider)
    body = CHAT_BODY | {"conversation_id": str(conversation.id)}

    chat = client.post("/v1/chat", json=body, headers=BOB)
    stream = client.post("/v1/chat/stream", json=body, headers=BOB)

    assert chat.status_code == stream.status_code == 404
    # Refused before the model was called: nobody pays to answer in a conversation not theirs
    assert provider.calls == []


def test_admin_may_read_any_conversation_but_not_write_to_it(
    client: TestClient, session: FakeSession, use_provider: Callable[..., None]
) -> None:
    conversation = make_conversation(owner="alice")
    session.conversations[conversation.id] = conversation
    use_provider(FakeProvider())

    read = client.get(f"/v1/conversations/{conversation.id}", headers=ROOT)
    write = client.post(
        "/v1/chat", json=CHAT_BODY | {"conversation_id": str(conversation.id)}, headers=ROOT
    )

    assert read.status_code == 200
    assert write.status_code == 404


def test_conversation_from_before_owners_existed_is_for_admins_only(
    client: TestClient, session: FakeSession
) -> None:
    legacy = make_conversation(owner=None)
    session.conversations[legacy.id] = legacy

    assert client.get(f"/v1/conversations/{legacy.id}", headers=ALICE).status_code == 404
    assert client.get(f"/v1/conversations/{legacy.id}", headers=ROOT).status_code == 200


def test_cache_entries_are_private_to_the_client_that_made_them(
    client: TestClient, use_provider: Callable[..., None]
) -> None:
    provider = CannedProvider("Paris", "Paris")
    cache = ResponseCache(as_redis(FakeRedis()), ttl_seconds=60, scope="t")
    use_provider(provider, cache=cache)
    body = CHAT_BODY | {"params": {"temperature": 0}}

    first = client.post("/v1/chat", json=body, headers=ALICE)
    again = client.post("/v1/chat", json=body, headers=ALICE)
    other = client.post("/v1/chat", json=body, headers=BOB)

    assert (first.json()["cached"], again.json()["cached"]) == (False, True)
    # Bob sends the same conversation and still waits for the model: a fast answer would
    # tell him that someone else had already asked exactly this
    assert other.json()["cached"] is False
    assert len(provider.calls) == 2


def test_streaming_is_attributed_to_the_caller(
    client: TestClient, use_provider: Callable[..., None], session: FakeSession
) -> None:
    use_provider(StreamingProvider())
    # No script: an empty stream is a provider error, which is not what this test is about
    provider = FakeProvider()
    use_provider(provider)

    response = client.post("/v1/chat/stream", json=CHAT_BODY, headers=BOB)

    assert response.status_code == 200
    assert [log.client for log in session.usage_logs] == ["bob"]
    assert next(iter(session.conversations.values())).owner == "bob"
