import asyncio
import logging
import time

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.api.dependencies import get_readiness_checker
from app.main import app
from app.services.readiness import Probe, ReadinessChecker


class Counter:
    """A probe that counts how often it is called, and can be told to fail or to hang."""

    def __init__(self, *, fail: bool = False, hang: bool = False, secret: str = "") -> None:
        self.calls = 0
        self.fail = fail
        self.hang = hang
        self.secret = secret

    async def __call__(self) -> None:
        self.calls += 1
        if self.hang:
            await asyncio.sleep(30)
        if self.fail:
            raise ConnectionRefusedError(f"cannot connect to {self.secret}")


def make_checker(
    postgres: Probe,
    redis: Probe,
    *,
    redis_required: bool = False,
    timeout: float = 1.0,
    cache: float = 0.0,
) -> ReadinessChecker:
    return ReadinessChecker(
        {"postgres": postgres, "redis": redis},
        required=frozenset({"postgres", "redis"} if redis_required else {"postgres"}),
        timeout_seconds=timeout,
        cache_seconds=cache,
    )


def use_checker(checker: ReadinessChecker) -> None:
    app.dependency_overrides[get_readiness_checker] = lambda: checker


def gauge(dependency: str) -> float | None:
    return REGISTRY.get_sample_value("dependency_up", {"dependency": dependency})


# ---------------------------------------------------------------- the verdicts


def test_everything_reachable_is_ready(client: TestClient) -> None:
    use_checker(make_checker(Counter(), Counter()))

    response = client.get("/ready", headers={"X-API-Key": ""})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["postgres"]["status"] == "ok" and body["checks"]["postgres"]["required"]
    assert body["checks"]["redis"]["status"] == "ok" and not body["checks"]["redis"]["required"]


def test_a_dead_database_makes_the_instance_not_ready(client: TestClient) -> None:
    use_checker(make_checker(Counter(fail=True), Counter()))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["postgres"]["status"] == "unavailable"


def test_a_dead_redis_only_makes_it_degraded_and_it_keeps_serving(client: TestClient) -> None:
    use_checker(make_checker(Counter(), Counter(fail=True)))

    response = client.get("/ready")

    # Redis speeds things up and guards the budget; without it requests are still answered
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["redis"]["status"] == "unavailable"


def test_redis_can_be_declared_required(client: TestClient) -> None:
    use_checker(make_checker(Counter(), Counter(fail=True), redis_required=True))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["redis"]["required"] is True


def test_the_llm_provider_is_deliberately_not_part_of_readiness(client: TestClient) -> None:
    use_checker(make_checker(Counter(), Counter()))

    body = client.get("/ready").json()

    assert set(body["checks"]) == {"postgres", "redis"}
    assert body["not_checked"] == ["llm_provider"]


# ---------------------------------------------------------------- bounded and safe


def test_a_hanging_dependency_is_reported_within_the_timeout(client: TestClient) -> None:
    use_checker(make_checker(Counter(hang=True), Counter(), timeout=0.05))

    started = time.perf_counter()
    response = client.get("/ready")

    # The probe answered in a fraction of a second instead of hanging with the dependency
    assert time.perf_counter() - started < 2
    assert response.status_code == 503
    assert response.json()["checks"]["postgres"]["status"] == "unavailable"


def test_the_response_and_the_log_never_carry_the_error_text(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    use_checker(
        make_checker(Counter(fail=True, secret="postgres://user:HUNTER2@db/app"), Counter())
    )

    response = client.get("/ready")

    assert "HUNTER2" not in response.text and "cannot connect" not in response.text
    assert "HUNTER2" not in caplog.text
    assert "ConnectionRefusedError" in caplog.text  # the type is enough to diagnose


def test_the_probe_is_public_because_orchestrators_have_no_key(client: TestClient) -> None:
    use_checker(make_checker(Counter(), Counter()))

    assert client.get("/ready", headers={"X-API-Key": ""}).status_code == 200
    assert client.get("/health", headers={"X-API-Key": ""}).status_code == 200


# ---------------------------------------------------------------- the probe must not become load


def test_frequent_probes_share_one_check(client: TestClient) -> None:
    postgres, redis = Counter(), Counter()
    use_checker(make_checker(postgres, redis, cache=60))

    for _ in range(20):
        client.get("/ready")

    assert (postgres.calls, redis.calls) == (1, 1)


def test_the_answer_is_refreshed_after_the_cache_expires(client: TestClient) -> None:
    postgres = Counter()
    use_checker(make_checker(postgres, Counter(), cache=0.05))

    client.get("/ready")
    time.sleep(0.1)
    client.get("/ready")

    assert postgres.calls == 2


def test_probes_arriving_together_trigger_a_single_check() -> None:
    postgres, redis = Counter(), Counter()
    checker = make_checker(postgres, redis, cache=60)

    async def scenario() -> None:
        await asyncio.gather(*(checker.check() for _ in range(50)))

    asyncio.run(scenario())

    assert (postgres.calls, redis.calls) == (1, 1)


def test_a_recovered_dependency_is_noticed_once_the_cache_expires() -> None:
    postgres = Counter(fail=True)
    checker = make_checker(postgres, Counter(), cache=0.05)

    async def scenario() -> tuple[str, str]:
        first = await checker.check()
        postgres.fail = False
        await asyncio.sleep(0.1)
        second = await checker.check()
        return first.status, second.status

    assert asyncio.run(scenario()) == ("not_ready", "ready")


def test_liveness_touches_no_dependency_at_all(client: TestClient) -> None:
    postgres, redis = Counter(fail=True), Counter(fail=True)
    use_checker(make_checker(postgres, redis))

    response = client.get("/health")

    # Both dependencies are down and the process is still alive: restarting it would not help
    assert response.status_code == 200
    assert (postgres.calls, redis.calls) == (0, 0)


# ---------------------------------------------------------------- metrics and wiring


def test_the_result_is_exported_as_a_gauge_per_dependency() -> None:
    checker = make_checker(Counter(), Counter(fail=True))

    asyncio.run(checker.check())

    assert gauge("postgres") == 1.0
    assert gauge("redis") == 0.0


def test_the_app_checks_postgres_as_required_and_redis_as_optional_by_default(
    client: TestClient,
) -> None:
    checker = app.state.readiness

    assert isinstance(checker, ReadinessChecker)
    assert checker._required == frozenset({"postgres"})
    assert set(checker._probes) == {"postgres", "redis"}
