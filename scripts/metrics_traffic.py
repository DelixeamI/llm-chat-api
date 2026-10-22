"""Generate mixed traffic against the real app and a real Ollama, then read /metrics.

Run: python -m scripts.metrics_traffic
The database is replaced by a session that stores nothing, so only the model is real; Redis is
used if it runs, and the features that depend on it step aside if it does not.
"""

import re

from fastapi.testclient import TestClient

from app.api.dependencies import get_session
from app.config import get_settings
from app.main import app
from scripts._null_session import null_session

MODEL = "qwen3:8b"
SHOW = (
    "http_requests_total",
    "llm_requests_total",
    "llm_tokens_total",
    "llm_errors_total",
    "auth_failures_total",
    "structured_output_failures_total",
)


def chat(content: str, **params: object) -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "params": {"max_tokens": 40, "temperature": 0.7} | params,
    }


def main() -> None:
    keys = {k.client: k.key.get_secret_value() for k in get_settings().api_keys}
    admin = {"X-API-Key": keys["dev"]}
    user = {"X-API-Key": keys["friend"]}
    app.dependency_overrides[get_session] = lambda: null_session()

    with TestClient(app) as client:
        for i in range(12):
            client.post(
                "/v1/chat",
                json=chat(f"Назови одну страну на букву {'АБВГДЕЖЗИКЛМ'[i]}."),
                headers=user,
            )
        for _ in range(4):  # repeated: cache hits when Redis runs
            client.post("/v1/chat", json=chat("Столица Франции?", temperature=0), headers=user)
        for i in range(3):
            client.post(
                "/v1/chat/structured",
                json={"model": MODEL, "text": f"Не приходит письмо с кодом подтверждения #{i}"},
                headers=user,
            )
        client.post("/v1/chat/stream", json=chat("Скажи привет."), headers=user)
        client.post("/v1/chat", json=chat("x"), headers={"X-API-Key": "wrong"})  # 401
        client.post("/v1/chat", json={"nonsense": 1}, headers=user)  # 422
        client.get("/v1/usage", headers=user)  # 403
        client.get("/nothing-here")  # 404

        text = client.get("/metrics", headers=admin).text

    print("selected series:")
    for line in text.splitlines():
        if line.startswith(SHOW) and not line.startswith("#"):
            print("  ", line)
    for name in ("llm_latency_seconds", "http_request_duration_seconds"):
        count = re.findall(rf"^{name}_count\{{([^}}]*)\}} (\S+)", text, flags=re.M)
        total = re.findall(rf"^{name}_sum\{{([^}}]*)\}} (\S+)", text, flags=re.M)
        print(f"{name}:")
        for (labels, n), (_, seconds) in zip(count, total, strict=True):
            if float(n):
                print(
                    f"   {labels}: n={float(n):.0f} mean={float(seconds) / float(n) * 1000:.0f} ms"
                )


if __name__ == "__main__":
    main()
