"""Print the span tree of real requests: a chat, a structured analysis and a stream.

Run: python -m scripts.tracing_demo
Real Ollama; the database is replaced by a session that stores nothing, so the "db.save" step
is nearly free here: against PostgreSQL it would take its own share of the request.
"""

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.api.dependencies import get_session
from app.config import get_settings
from app.main import app
from app.observability.tracing import use_provider
from scripts._null_session import null_session

MODEL = "qwen3:8b"
SHOWN = ("llm.input_tokens", "llm.output_tokens", "llm.slot_wait_ms", "llm.ttft_ms", "cache.hit")


def tree(spans: tuple[ReadableSpan, ...]) -> list[str]:
    children: dict[int | None, list[ReadableSpan]] = {}
    for item in spans:
        parent = item.parent.span_id if item.parent else None
        children.setdefault(parent, []).append(item)
    lines: list[str] = []

    def walk(parent: int | None, depth: int) -> None:
        for item in sorted(children.get(parent, []), key=lambda s: s.start_time or 0):
            ms = ((item.end_time or 0) - (item.start_time or 0)) / 1e6
            attrs = {k: v for k, v in (item.attributes or {}).items() if k in SHOWN}
            extra = "  " + " ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
            lines.append(f"{'  ' * depth}{item.name:<22s}{ms:9.1f} ms{extra}")
            walk(item.context.span_id, depth + 1)

    walk(None, 0)
    return lines


def main() -> None:
    keys = {k.client: k.key.get_secret_value() for k in get_settings().api_keys}
    headers = {"X-API-Key": keys["friend"]}
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    app.dependency_overrides[get_session] = lambda: null_session()

    requests = [
        (
            "POST /v1/chat",
            "/v1/chat",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Назови столицу Франции."}],
                "params": {"max_tokens": 30},
            },
        ),
        (
            "POST /v1/chat/structured",
            "/v1/chat/structured",
            {"model": MODEL, "text": "Не могу войти в аккаунт после смены пароля, срочно нужно."},
        ),
        (
            "POST /v1/chat/stream",
            "/v1/chat/stream",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Скажи привет."}],
                "params": {"max_tokens": 20},
            },
        ),
    ]
    with TestClient(app) as client:
        # After startup: the app's own lifespan configures tracing from the settings (off here)
        use_provider(provider)
        client.post("/v1/chat", json=requests[0][2], headers=headers)  # warm-up: model loading
        memory.clear()
        for title, path, body in requests:
            client.post(path, json=body, headers=headers)
            print(f"\n{title}")
            print("\n".join(tree(memory.get_finished_spans())))
            memory.clear()


if __name__ == "__main__":
    main()
