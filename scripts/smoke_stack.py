"""End-to-end smoke test of a running stack: the real app, PostgreSQL, Redis and a model.

Run against `docker compose --profile fake-llm up` (no GPU needed), or against any running
instance. It goes through the features a user relies on and fails loudly on the first broken one.

    API_KEY=<key with chat and admin scopes> python -m scripts.smoke_stack

Environment: BASE_URL (default http://127.0.0.1:8000), API_KEY (required), MODEL (default fake).
"""

import json
import os
import sys
import time
import uuid

import httpx2 as httpx

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "fake")
WAIT_SECONDS = float(os.environ.get("SMOKE_WAIT_SECONDS", "90"))

steps: list[str] = []


def check(condition: bool, description: str, detail: object = "") -> None:
    if not condition:
        print(f"FAIL  {description}  {detail}")
        sys.exit(1)
    steps.append(description)
    print(f"ok    {description}")


def wait_until_ready(client: httpx.Client) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    last = ""
    while time.monotonic() < deadline:
        try:
            response = client.get("/ready")
            last = response.text
            if response.status_code == 200:
                return
        except httpx.TransportError as exc:
            last = type(exc).__name__
        time.sleep(1)
    print(f"FAIL  service did not become ready in {WAIT_SECONDS:.0f}s: {last}")
    sys.exit(1)


def events(text: str) -> list[tuple[str, dict[str, object]]]:
    parsed = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        parsed.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return parsed


def main() -> None:
    if not API_KEY:
        print("API_KEY is required")
        sys.exit(2)
    auth = {"X-API-Key": API_KEY}
    # Unique per run: the cache and the idempotency keys live in Redis between runs
    tag = uuid.uuid4().hex[:8]

    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        wait_until_ready(client)

        health = client.get("/health")
        check(health.status_code == 200, "liveness answers", health.text)
        ready = client.get("/ready").json()
        check(
            ready["status"] == "ready" and ready["checks"]["postgres"]["status"] == "ok",
            "readiness sees PostgreSQL and Redis",
            ready,
        )
        check("x-request-id" in health.headers, "every response carries a request id")

        body = {"model": MODEL, "messages": [{"role": "user", "content": f"hello {tag}"}]}
        check(
            client.post("/v1/chat", json=body).status_code == 401,
            "a request without a key is refused with 401",
        )

        chat = client.post("/v1/chat", json=body, headers=auth)
        check(chat.status_code == 200, "chat answers", chat.text)
        reply = chat.json()
        check(reply["message"]["content"].startswith("echo:"), "chat returns the model's reply")
        check("x-ratelimit-limit" in chat.headers, "rate limit headers are present")

        history = client.get(f"/v1/conversations/{reply['conversation_id']}/messages", headers=auth)
        roles = [m["role"] for m in history.json()]
        check(roles == ["user", "assistant"], "the exchange is stored in PostgreSQL", roles)

        cached_body = body | {
            "messages": [{"role": "user", "content": f"cache {tag}"}],
            "params": {"temperature": 0},
        }
        first = client.post("/v1/chat", json=cached_body, headers=auth).json()
        second = client.post("/v1/chat", json=cached_body, headers=auth).json()
        check(
            (first["cached"], second["cached"]) == (False, True),
            "a repeated temperature-0 request is served from the Redis cache",
        )

        key = {"Idempotency-Key": f"smoke-{tag}", **auth}
        original = client.post("/v1/chat", json=body, headers=key)
        retry = client.post("/v1/chat", json=body, headers=key)
        check(
            retry.content == original.content and retry.headers.get("idempotent-replay") == "true",
            "a retry with the same Idempotency-Key replays the first response",
        )

        stream = client.post("/v1/chat/stream", json=body, headers=auth)
        names = [name for name, _ in events(stream.text)]
        check(
            stream.status_code == 200 and names[0] == "delta" and names[-1] == "done",
            "streaming delivers pieces and a closing event",
            names,
        )

        analysis = client.post(
            "/v1/chat/structured",
            json={"model": MODEL, "text": "Не могу войти в аккаунт"},
            headers=auth,
        )
        check(
            analysis.status_code == 200
            and analysis.json()["analysis"]["category"]
            in {"billing", "technical", "account", "other"},
            "structured output is validated against the schema",
            analysis.text,
        )

        failing = client.post(
            "/v1/chat",
            json={"model": MODEL, "messages": [{"role": "user", "content": "__503__"}]},
            headers=auth,
        )
        check(
            failing.status_code == 502,
            "a failing model becomes a 502 after bounded retries",
            failing.status_code,
        )

        usage = client.get("/v1/usage", params={"days": 1}, headers=auth)
        check(
            usage.status_code == 200 and usage.json()["requests"] >= 3,
            "usage is recorded and reported (admin scope)",
            usage.text,
        )

        metrics = client.get("/metrics", headers=auth)
        check(
            metrics.status_code == 200 and "llm_requests_total" in metrics.text,
            "Prometheus metrics are exposed",
        )

    print(f"\n{len(steps)} checks passed")


if __name__ == "__main__":
    main()
